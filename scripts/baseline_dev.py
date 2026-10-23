"""Single-shot Nemotron baseline on DEV tasks only (never held-out, never writes to
the library). For each template, cycles through the dev tasks for that template to
fill N generations, each a fresh `pipeline.solve(n_candidates=1, do_repair=False,
use_library=False, save_to_library=False)` call at the default temperature.

Reports, per template: pass rate with a Wilson 95% interval, a failure-mode
breakdown, and tokens/cost/seconds (both per-generation and totals), plus the
GENERATE_SYSTEM prompt version these numbers belong to. Writes the full raw
record set to eval/results/baseline_<ts>.json.

Usage:
  SIMSMITH_BUDGET_USD=2 python scripts/baseline_dev.py --n-per-template 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.llm import TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.prompts import PROMPT_VERSION  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SIM_PYTHON = os.path.expanduser("~/nw/bin/python")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def classify(candidate) -> str:
    """One of: truncation / no_simulation / hallucinated_api / crash / wrong_logic /
    pass, plus two fallbacks (preflight_other, task_param_mismatch) for anything that
    doesn't cleanly fit, reported honestly rather than forced into the wrong bucket."""
    if not candidate.preflight_ok:
        text = " ".join(candidate.preflight_problems).lower()
        if "llm error" in text and ("truncat" in text or "empty content" in text):
            return "truncation"
        if "must import mujoco and call mj_step" in text:
            return "no_simulation"
        return "preflight_other"
    score = candidate.score
    if score is None:
        return "preflight_other"
    text = score.failure_text.lower()
    if score.crashes > 0:
        if "need >=" in text and "step" in text:
            return "no_simulation"
        if ("modulenotfounderror" in text or "no module named" in text or "importerror" in text
                or ("attributeerror" in text and "module" in text)):
            return "hallucinated_api"
        if "differs from task" in text:
            return "task_param_mismatch"
        return "crash"
    if not score.ok:
        return "wrong_logic"
    return "pass"


def read_new_usage(path: str, start_line: int) -> tuple:
    if not os.path.exists(path):
        return [], start_line
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    new = [json.loads(ln) for ln in lines[start_line:]]
    return new, len(lines)


GENERATION_WALL_TIMEOUT_S = 240  # a single intake+plan+generate(+sim) cycle this slow is stuck, not thinking


def _load_progress(progress_path: str) -> dict:
    """(template, i) -> row, for rows already completed in a previous (interrupted) run."""
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for ln in f:
                row = json.loads(ln)
                done[(row["template"], row["i"])] = row
    return done


def run_template(template: str, tasks: list, n: int, llm, runner, docs, cfg,
                 usage_path: str, progress_path: str, done: dict) -> list:
    rows = []
    line_ptr = 0
    if os.path.exists(usage_path):
        with open(usage_path, encoding="utf-8") as f:
            line_ptr = sum(1 for _ in f)
    lib = Library(os.path.join(HERE, f"_baseline_unused_{template}.jsonl"))
    pipe = Pipeline(llm, runner, lib, docs, cfg)
    pool = ThreadPoolExecutor(max_workers=1)
    for i in range(n):
        if (template, i) in done:
            row = done[(template, i)]
            rows.append(row)
            print(f"  [{template} {i + 1}/{n}] task={row['task_id']} (resumed from progress file) "
                  f"status={row['status']} class={row['classification']}", flush=True)
            continue
        task = tasks[i % len(tasks)]
        t0 = time.time()
        try:
            future = pool.submit(pipe.solve, task["task"], n_candidates=1, use_library=False,
                                 do_repair=False, save_to_library=False)
            res = future.result(timeout=GENERATION_WALL_TIMEOUT_S)
        except FutureTimeoutError:
            wall = time.time() - t0
            usage_rows, line_ptr = read_new_usage(usage_path, line_ptr)
            row = {
                "template": template, "i": i, "task_id": task["id"], "task": task["task"],
                "intake_template": None, "exact_match": None,
                "status": "TIMEOUT", "classification": "wall_clock_timeout",
                "preflight_ok": None, "preflight_problems": [],
                "score_ok": None, "success_rate": None, "crashes": None,
                "failure_text": f"generation exceeded {GENERATION_WALL_TIMEOUT_S}s wall clock, abandoned",
                "prompt_tokens": sum(r["prompt_tokens"] for r in usage_rows),
                "completion_tokens": sum(r["completion_tokens"] for r in usage_rows),
                "cost_usd": sum(r["cost_usd"] for r in usage_rows),
                "wall_seconds": wall,
            }
            rows.append(row)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  [{template} {i + 1}/{n}] task={task['id']} TIMEOUT after {wall:.1f}s -- abandoned", flush=True)
            # the stuck future is leaked (daemon-ish, process-local); a fresh pool avoids it
            # blocking future submits in this same run.
            pool = ThreadPoolExecutor(max_workers=1)
            continue
        wall = time.time() - t0
        c = res.candidates[0]
        usage_rows, line_ptr = read_new_usage(usage_path, line_ptr)
        row = {
            "template": template, "i": i, "task_id": task["id"], "task": task["task"],
            "intake_template": res.template, "exact_match": res.exact_match,
            "status": res.status, "classification": classify(c),
            "preflight_ok": c.preflight_ok,
            "preflight_problems": c.preflight_problems,
            "score_ok": c.score.ok if c.score else None,
            "success_rate": c.score.success_rate if c.score else None,
            "crashes": c.score.crashes if c.score else None,
            "failure_text": (c.score.failure_text if c.score else "")[:1000],
            "prompt_tokens": sum(r["prompt_tokens"] for r in usage_rows),
            "completion_tokens": sum(r["completion_tokens"] for r in usage_rows),
            "cost_usd": sum(r["cost_usd"] for r in usage_rows),
            "wall_seconds": wall,
        }
        rows.append(row)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{template} {i + 1}/{n}] task={task['id']} status={res.status} "
              f"class={row['classification']} cost=${row['cost_usd']:.5f} {wall:.1f}s", flush=True)
    pool.shutdown(wait=False)
    return rows


def summarize(template: str, rows: list) -> dict:
    n = len(rows)
    passes = sum(1 for r in rows if r["classification"] == "pass")
    lo, hi = wilson_interval(passes, n)
    from collections import Counter
    breakdown = Counter(r["classification"] for r in rows)
    return {
        "template": template, "n": n, "passes": passes,
        "pass_rate": passes / n if n else 0.0,
        "wilson_95": [lo, hi],
        "failure_breakdown": dict(breakdown),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "mean_cost_usd": sum(r["cost_usd"] for r in rows) / n if n else 0.0,
        "mean_wall_seconds": sum(r["wall_seconds"] for r in rows) / n if n else 0.0,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n if n else 0.0,
        "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / n if n else 0.0,
    }


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['template']} ===")
    print(f"pass rate: {summary['passes']}/{summary['n']} = {summary['pass_rate']:.2%} "
          f"(Wilson 95% CI: [{summary['wilson_95'][0]:.2%}, {summary['wilson_95'][1]:.2%}])")
    print("failure breakdown:")
    for k, v in sorted(summary["failure_breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}")
    print(f"mean tokens: {summary['mean_prompt_tokens']:.0f} in / {summary['mean_completion_tokens']:.0f} out")
    print(f"mean cost: ${summary['mean_cost_usd']:.5f}/generation  "
          f"(total ${summary['total_cost_usd']:.5f} over {summary['n']} generations)")
    print(f"mean wall time: {summary['mean_wall_seconds']:.1f}s/generation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-template", type=int, default=20)
    ap.add_argument("--sim-python", default="")
    ap.add_argument("--progress-file", default=os.path.join(ROOT, "eval", "results", "baseline_progress.jsonl"),
                    help="resumable per-generation log; rerunning with the same file skips completed rows")
    ap.add_argument("--fresh", action="store_true", help="ignore/delete any existing --progress-file first")
    args = ap.parse_args()

    if args.fresh and os.path.exists(args.progress_file):
        os.remove(args.progress_file)
    done = _load_progress(args.progress_file)
    if done:
        print(f"resuming: {len(done)} generation(s) already completed in {args.progress_file}")

    cfg = Config.from_env()
    sim_python = args.sim_python or cfg.sim_python or DEFAULT_SIM_PYTHON
    runner = SubprocessRunner(sim_python, cfg.run_timeout_s)
    docs = DocIndex(os.path.join(ROOT, "docs_index"))

    ts = int(time.time())
    usage_path = os.path.join(ROOT, "eval", "results", f"baseline_usage_{ts}.jsonl")
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key, usage_log_path=usage_path)

    with open(os.path.join(ROOT, "eval", "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    dev_by_template = {"reach": [], "pick_place": []}
    for t in tasks["dev"]:
        dev_by_template.setdefault(t["template"], []).append(t)

    print(f"prompt_version={PROMPT_VERSION}  model={cfg.model_generate}  sim_python={sim_python}", flush=True)
    print(f"sim seeds per generation: {cfg.seeds}  budget_usd={llm.budget_usd}", flush=True)

    all_rows = {}
    summaries = {}
    for template in ("reach", "pick_place"):
        print(f"\n--- {template}: {args.n_per_template} single-shot generations, dev tasks only ---", flush=True)
        rows = run_template(template, dev_by_template[template], args.n_per_template,
                            llm, runner, docs, cfg, usage_path, args.progress_file, done)
        all_rows[template] = rows
        summaries[template] = summarize(template, rows)

    for template in ("reach", "pick_place"):
        print_summary(summaries[template])

    report = {
        "prompt_version": PROMPT_VERSION,
        "model": cfg.model_generate,
        "sim_python": sim_python,
        "sim_seeds": list(cfg.seeds),
        "timestamp": ts,
        "summaries": summaries,
        "rows": all_rows,
    }
    out_path = os.path.join(ROOT, "eval", "results", f"baseline_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {out_path}")
    print(f"Usage log: {usage_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
