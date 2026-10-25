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
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.llm import TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Pipeline, candidate_log  # noqa: E402
from simsmith.prompts import PROMPT_VERSION  # noqa: E402
from simsmith.run_meta import build_run_meta  # noqa: E402
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


# A real module-attribute hallucination renders as e.g. "AttributeError: module 'mujoco'
# has no attribute 'MjSim'" -- this is the ONLY form of AttributeError that means "the
# model invented an API name." A plain substring check for "attributeerror" + "module"
# (the previous version of this function) also fires on ANY AttributeError, because
# every Python traceback frame line is literally rendered "in <module>" regardless of
# what the error actually is -- e.g. "AttributeError: 'list' object has no attribute
# 'tolist'" matched too, mislabeling an ordinary bug as a hallucinated API. See
# docs/diagnosis_2026-10-24.md Part 1 (found: 14/14 hallucinated_api labels in one run
# were false positives this way).
_MODULE_ATTR_ERROR = re.compile(r"module '[\w.]+' has no attribute '[\w]+'")


def classify_failure(preflight_ok: bool, preflight_problems: list, score_ok, crashes, failure_text: str) -> str:
    """Pure classification logic (no Candidate/Score objects) -- one of: truncation /
    no_simulation / hallucinated_api / crash / wrong_logic / pass, plus two fallbacks
    (preflight_other, task_param_mismatch) for anything that doesn't cleanly fit,
    reported honestly rather than forced into the wrong bucket.

    Used both live by classify(candidate) below and offline by
    scripts/reclassify_results.py against previously-stored result rows, so the two
    can never drift apart -- a fix here (like the hallucinated_api regex) applies to
    both without being written twice."""
    if not preflight_ok:
        text = " ".join(preflight_problems).lower()
        if "llm error" in text and ("truncat" in text or "empty content" in text):
            return "truncation"
        if "must import mujoco and call mj_step" in text:
            return "no_simulation"
        if "unknown attribute" in text:
            return "hallucinated_api"  # caught statically now (Part 1), not just at runtime
        return "preflight_other"
    if score_ok is None:
        return "preflight_other"
    text = (failure_text or "").lower()
    if crashes and crashes > 0:
        if "need >=" in text and "step" in text:
            return "no_simulation"
        if ("modulenotfounderror" in text or "no module named" in text or "importerror" in text
                or _MODULE_ATTR_ERROR.search(text)):
            return "hallucinated_api"
        if "differs from task" in text:
            return "task_param_mismatch"
        return "crash"
    if not score_ok:
        return "wrong_logic"
    return "pass"


def classify(candidate) -> str:
    """classify_failure(), fed from a live Candidate/Score pair."""
    if not candidate.preflight_ok:
        return classify_failure(False, candidate.preflight_problems, None, None, "")
    score = candidate.score
    if score is None:
        return "preflight_other"
    return classify_failure(True, [], score.ok, score.crashes, score.failure_text)


def cost_summary(usage_rows: list) -> tuple:
    """(total_cost_usd, price_known) for a set of usage rows (see TokenFactoryLLM's
    per-model SIMSMITH_PRICE_<MODEL>_IN/_OUT in simsmith/llm.py). price_known is False,
    and total_cost_usd is None (never a misleadingly-partial number), if ANY
    contributing call used a model with no price configured -- "cost only for models
    with a price set" means the caller gets an honest "unknown," not a partial sum."""
    known = bool(usage_rows) and all(r.get("price_known") for r in usage_rows)
    if not known:
        return None, False
    return sum(r["cost_usd"] for r in usage_rows), True


def usage_rows_for_tag(usage_path: str, tag: str) -> list:
    """Under concurrent generations, usage rows interleave in the shared log by
    arrival order, not by generation -- so attribution is by the tag llm.set_tag()
    stamped on each row (see simsmith/llm.py), not by line position."""
    if not os.path.exists(usage_path):
        return []
    rows = []
    with open(usage_path, encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            if rec.get("tag") == tag:
                rows.append(rec)
    return rows


def _load_progress(progress_path: str) -> dict:
    """(template, i) -> row, for rows already completed in a previous (interrupted) run."""
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for ln in f:
                row = json.loads(ln)
                done[(row["template"], row["i"])] = row
    return done


_progress_lock = threading.Lock()


def _do_one(template: str, task: dict, i: int, n: int, llm, runner, docs, cfg,
           usage_path: str, progress_path: str, workers: int = 1) -> dict:
    tag = f"{template}:{i}"
    llm.set_tag(tag)
    lib = Library(os.path.join(HERE, f"_baseline_unused_{template}.jsonl"))
    pipe = Pipeline(llm, runner, lib, docs, cfg)
    t0 = time.time()
    res = pipe.solve(task["task"], n_candidates=1, use_library=False,
                     do_repair=False, save_to_library=False)
    wall = time.time() - t0
    c = res.candidates[0]
    usage_rows = usage_rows_for_tag(usage_path, tag)
    cost_usd, price_known = cost_summary(usage_rows)
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
        "candidate_log": candidate_log(c),
        "prompt_tokens": sum(r["prompt_tokens"] for r in usage_rows),
        "completion_tokens": sum(r["completion_tokens"] for r in usage_rows),
        "cost_usd": cost_usd, "price_known": price_known,
        "wall_seconds": wall,
        **build_run_meta(condition="single_shot", cfg=cfg, extra_payload=llm.extra_payload,
                         n_candidates=1, max_repairs=0, workers=workers),
    }
    with _progress_lock:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    cost_s = f"${row['cost_usd']:.5f}" if row["cost_usd"] is not None else "unknown (no price set)"
    print(f"  [{template} {i + 1}/{n}] task={task['id']} status={res.status} "
          f"class={row['classification']} cost={cost_s} {wall:.1f}s", flush=True)
    return row


def run_all(dev_by_template: dict, n: int, llm, runner, docs, cfg,
           usage_path: str, progress_path: str, done: dict, workers: int, templates=("reach", "pick_place")) -> dict:
    jobs = []  # (template, i, task)
    for template in templates:
        tasks = dev_by_template[template]
        for i in range(n):
            if (template, i) not in done:
                jobs.append((template, tasks[i % len(tasks)], i))

    all_rows = {t: [] for t in templates}
    for template in templates:
        for i in range(n):
            if (template, i) in done:
                row = done[(template, i)]
                all_rows[template].append(row)
                print(f"  [{template} {i + 1}/{n}] task={row['task_id']} (resumed) "
                      f"status={row['status']} class={row['classification']}", flush=True)

    if jobs:
        print(f"\nrunning {len(jobs)} generation(s) with {workers} parallel workers...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_one, template, task, i, n, llm, runner, docs, cfg,
                                   usage_path, progress_path, workers): template
                      for template, task, i in jobs}
            for future in as_completed(futures):
                row = future.result()  # exceptions surface here, deliberately not swallowed
                all_rows[row["template"]].append(row)

    for template in templates:
        all_rows[template].sort(key=lambda r: r["i"])
    return all_rows


def summarize(template: str, rows: list) -> dict:
    n = len(rows)
    passes = sum(1 for r in rows if r["classification"] == "pass")
    lo, hi = wilson_interval(passes, n)
    from collections import Counter
    breakdown = Counter(r["classification"] for r in rows)
    priced = [r for r in rows if r.get("price_known")]
    total_cost = sum(r["cost_usd"] for r in priced) if priced else None
    all_priced = bool(rows) and len(priced) == len(rows)
    return {
        "template": template, "n": n, "passes": passes,
        "pass_rate": passes / n if n else 0.0,
        "wilson_95": [lo, hi],
        "failure_breakdown": dict(breakdown),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_cost_usd": total_cost if all_priced else None,
        "priced_rows": len(priced), "all_priced": all_priced,
        "mean_cost_usd": (total_cost / len(priced)) if priced else None,
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
    if summary["mean_cost_usd"] is None:
        print("mean cost: unknown (no SIMSMITH_PRICE_* set for this model)")
    elif summary["all_priced"]:
        print(f"mean cost: ${summary['mean_cost_usd']:.5f}/generation "
              f"(total ${summary['total_cost_usd']:.5f} over {summary['n']} generations)")
    else:
        print(f"mean cost (priced rows only, {summary['priced_rows']}/{summary['n']}): "
              f"${summary['mean_cost_usd']:.5f}/generation")
    print(f"mean wall time: {summary['mean_wall_seconds']:.1f}s/generation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-template", type=int, default=20)
    ap.add_argument("--sim-python", default="")
    ap.add_argument("--progress-file", default=os.path.join(ROOT, "eval", "results", "baseline_progress.jsonl"),
                    help="resumable per-generation log; rerunning with the same file skips completed rows")
    ap.add_argument("--fresh", action="store_true", help="ignore/delete any existing --progress-file first")
    ap.add_argument("--extra-payload", default="", help="JSON dict merged into every LLM request, "
                    'e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    ap.add_argument("--tag", default="", help="label stamped into the output report (e.g. a mode name)")
    ap.add_argument("--workers", type=int, default=1, help="parallel generations (API calls); sim runs "
                    "within a generation stay sequential")
    ap.add_argument("--model-generate", default="", help="override SIMSMITH_MODEL_GENERATE for this run")
    ap.add_argument("--templates", default="reach,pick_place")
    args = ap.parse_args()
    extra_payload = json.loads(args.extra_payload) if args.extra_payload else {}
    templates = args.templates.split(",")

    if args.fresh and os.path.exists(args.progress_file):
        os.remove(args.progress_file)
    done = _load_progress(args.progress_file)
    if done:
        print(f"resuming: {len(done)} generation(s) already completed in {args.progress_file}")

    cfg = Config.from_env()
    if args.model_generate:
        cfg.model_generate = args.model_generate
    sim_python = args.sim_python or cfg.sim_python or DEFAULT_SIM_PYTHON
    runner = SubprocessRunner(sim_python, cfg.run_timeout_s)
    docs = DocIndex(os.path.join(ROOT, "docs_index"))

    ts = int(time.time())
    usage_path = os.path.join(ROOT, "eval", "results", f"baseline_usage_{ts}.jsonl")
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key, usage_log_path=usage_path, extra_payload=extra_payload)
    if extra_payload:
        print(f"extra_payload={extra_payload}")

    with open(os.path.join(ROOT, "eval", "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    dev_by_template = {"reach": [], "pick_place": []}
    for t in tasks["dev"]:
        dev_by_template.setdefault(t["template"], []).append(t)
    dev_by_template = {t: v for t, v in dev_by_template.items() if t in templates}

    print(f"prompt_version={PROMPT_VERSION}  model={cfg.model_generate}  sim_python={sim_python}", flush=True)
    print(f"sim seeds per generation: {cfg.seeds}  budget_usd={llm.budget_usd}", flush=True)

    print(f"\n--- {args.n_per_template} single-shot generations per template ({templates}), dev tasks only, "
          f"{args.workers} worker(s) ---", flush=True)
    all_rows = run_all(dev_by_template, args.n_per_template, llm, runner, docs, cfg,
                       usage_path, args.progress_file, done, args.workers, templates)
    summaries = {template: summarize(template, all_rows[template]) for template in templates}

    for template in templates:
        print_summary(summaries[template])

    report = {
        "prompt_version": PROMPT_VERSION,
        "tag": args.tag,
        "condition": "single_shot",
        "extra_payload": extra_payload,
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
