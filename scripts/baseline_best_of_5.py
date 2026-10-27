"""best_of_5 + up to 2 repair rounds, on DEV tasks only, N runs per task.

For each dev task, repeats `pipeline.solve(n_candidates=5, do_repair=True,
max_repairs=2, use_library=False, save_to_library=False)` --runs-per-task times
(a fresh task-run each time; repair "rounds" are the existing max_repairs loop,
which repairs up to 2 of the failing candidates once each, in descending
success-rate order -- round 1 = the first (best) failing candidate repaired,
round 2 = the second).

Reports per template: VERIFIED-rate with a Wilson 95% interval, how many VERIFIED
task-runs won via a generated vs. repaired (round 1/2) candidate, a
hallucinated_api count broken down by round across ALL candidates (not just the
winner), a failure-mode breakdown of the non-VERIFIED task-runs (by their best
candidate), and tokens/cost/seconds. Writes eval/results/best_of_5_<ts>.json.

Reuses classify()/wilson_interval()/usage_rows_for_tag() from baseline_dev.py so
the failure taxonomy stays identical to the single-shot baseline.

Usage:
  SIMSMITH_BUDGET_USD=2 python scripts/baseline_best_of_5.py --runs-per-task 4 --workers 8 \\
      --extra-payload '{"chat_template_kwargs": {"enable_thinking": false}}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
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
from simsmith.tavily import client_from_env as tavily_client_from_env  # noqa: E402

# SIMSMITH_TAVILY=1 (+ TAVILY_API_KEY) opts a dev baseline run into the plan-step
# Tavily hook; off by default. Built once, not per-task-run -- see
# simsmith.tavily.client_from_env()'s own docstring for what "off" means here.
_TAVILY_CLIENT = tavily_client_from_env()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_dev import classify, wilson_interval, usage_rows_for_tag, cost_summary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SIM_PYTHON = os.path.expanduser("~/nw/bin/python")
MAX_REPAIRS = 2

_progress_lock = threading.Lock()


def candidate_round(res, c) -> int:
    """0 = original generation, 1/2 = which repair attempt (in success-rate order)."""
    if c.stage != "repaired":
        return 0
    repaired = [x for x in res.candidates if x.stage == "repaired"]
    return repaired.index(c) + 1


def _load_progress(progress_path: str) -> dict:
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for ln in f:
                row = json.loads(ln)
                done[(row["template"], row["task_id"], row["run_idx"])] = row
    return done


def _do_one(template: str, task: dict, run_idx: int, n_runs: int, llm, runner, docs, cfg,
           usage_path: str, progress_path: str, max_repairs: int, workers: int = 1) -> dict:
    tag = f"{template}:{task['id']}:{run_idx}"
    llm.set_tag(tag)
    lib = Library(os.path.join(HERE, f"_baseline_unused_{template}.jsonl"))
    pipe = Pipeline(llm, runner, lib, docs, cfg, tavily_client=_TAVILY_CLIENT)
    t0 = time.time()
    res = pipe.solve(task["task"], n_candidates=5, use_library=False,
                     do_repair=max_repairs > 0, max_repairs=max_repairs, save_to_library=False)
    wall = time.time() - t0

    per_candidate = []
    hallucinated_by_round = Counter()
    for c in res.candidates:
        rnd = candidate_round(res, c)
        cls = classify(c)
        per_candidate.append({"idx": c.idx, "stage": c.stage, "round": rnd,
                              "classification": cls, "score_ok": c.score.ok if c.score else None,
                              **candidate_log(c)})
        if cls == "hallucinated_api":
            hallucinated_by_round[rnd] += 1

    winner = res.winner
    winner_round = candidate_round(res, winner) if winner else None
    best_for_classification = winner or (res.candidates[0] if res.candidates else None)
    failure_class = classify(best_for_classification) if best_for_classification else "no_candidates"

    usage_rows = usage_rows_for_tag(usage_path, tag)
    cost_usd, price_known = cost_summary(usage_rows)
    row = {
        "template": template, "task_id": task["id"], "run_idx": run_idx, "task": task["task"],
        "status": res.status,
        "passed": res.status == "VERIFIED",
        "winner_stage": winner.stage if winner else None,
        "winner_round": winner_round,
        "failure_classification": None if res.status == "VERIFIED" else failure_class,
        "n_candidates_generated": len(res.candidates),
        "hallucinated_by_round": dict(hallucinated_by_round),
        "candidates": per_candidate,
        "prompt_tokens": sum(r["prompt_tokens"] for r in usage_rows),
        "completion_tokens": sum(r["completion_tokens"] for r in usage_rows),
        "cost_usd": cost_usd, "price_known": price_known,
        "wall_seconds": wall,
        **build_run_meta(condition=("best_of_5_repair" if max_repairs > 0 else "best_of_5"),
                         cfg=cfg, extra_payload=llm.extra_payload, n_candidates=5,
                         max_repairs=max_repairs, workers=workers),
    }
    with _progress_lock:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    cost_s = f"${row['cost_usd']:.5f}" if row["cost_usd"] is not None else "unknown (no price set)"
    print(f"  [{template} {task['id']} run {run_idx + 1}/{n_runs}] status={res.status} "
          f"winner_stage={row['winner_stage']} round={winner_round} "
          f"cost={cost_s} {wall:.1f}s", flush=True)
    return row


def run_all(jobs_by_template: dict, llm, runner, docs, cfg,
           usage_path: str, progress_path: str, done: dict, workers: int, max_repairs: int) -> dict:
    """jobs_by_template: template -> ordered list of (task_dict, run_idx)."""
    jobs = []
    for template, task_runs in jobs_by_template.items():
        for task, run_idx in task_runs:
            if (template, task["id"], run_idx) not in done:
                jobs.append((template, task, run_idx))

    all_rows = {t: [] for t in jobs_by_template}
    for template, task_runs in jobs_by_template.items():
        n_runs = len(task_runs)
        for task, run_idx in task_runs:
            key = (template, task["id"], run_idx)
            if key in done:
                all_rows[template].append(done[key])
                print(f"  [{template} {task['id']} run {run_idx + 1}/{n_runs}] (resumed) "
                      f"status={done[key]['status']}", flush=True)

    runs_per_task = {t: len(task_runs) for t, task_runs in jobs_by_template.items()}
    if jobs:
        print(f"\nrunning {len(jobs)} task-run(s) with {workers} parallel workers...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_do_one, template, task, run_idx, runs_per_task[template],
                                   llm, runner, docs, cfg, usage_path, progress_path, max_repairs, workers)
                      for template, task, run_idx in jobs]
            for future in as_completed(futures):
                row = future.result()
                all_rows[row["template"]].append(row)

    for template in all_rows:
        all_rows[template].sort(key=lambda r: (r["task_id"], r["run_idx"]))
    return all_rows


def summarize(template: str, rows: list) -> dict:
    n = len(rows)
    passes = sum(1 for r in rows if r["passed"])
    lo, hi = wilson_interval(passes, n)
    win_by_round = Counter((r["winner_round"] if r["passed"] else None) for r in rows if r["passed"])
    hallu_by_round = Counter()
    for r in rows:
        for rnd, cnt in r["hallucinated_by_round"].items():
            hallu_by_round[int(rnd)] += cnt
    fail_breakdown = Counter(r["failure_classification"] for r in rows if not r["passed"])
    priced = [r for r in rows if r.get("price_known")]
    total_cost = sum(r["cost_usd"] for r in priced) if priced else None
    all_priced = bool(rows) and len(priced) == len(rows)
    return {
        "template": template, "n": n, "passes": passes,
        "pass_rate": passes / n if n else 0.0,
        "wilson_95": [lo, hi],
        "passes_by_winner_round": dict(win_by_round),
        "hallucinated_api_by_round": dict(hallu_by_round),
        "failure_breakdown": dict(fail_breakdown),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_cost_usd": total_cost if all_priced else None,
        "priced_rows": len(priced), "all_priced": all_priced,
        "mean_cost_usd": (total_cost / len(priced)) if priced else None,
        "mean_wall_seconds": sum(r["wall_seconds"] for r in rows) / n if n else 0.0,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n if n else 0.0,
        "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / n if n else 0.0,
    }


def print_summary(s: dict, max_repairs: int) -> None:
    print(f"\n=== {s['template']} (best_of_5 + up to {max_repairs} repairs) ===")
    print(f"VERIFIED rate: {s['passes']}/{s['n']} = {s['pass_rate']:.2%} "
          f"(Wilson 95% CI: [{s['wilson_95'][0]:.2%}, {s['wilson_95'][1]:.2%}])")
    print(f"passes by winner round (0=generated, 1/2=repair round): {s['passes_by_winner_round']}")
    print(f"hallucinated_api count by round: {s['hallucinated_api_by_round']}")
    print("failure breakdown (non-VERIFIED task-runs, by best candidate):")
    for k, v in sorted(s["failure_breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}")
    print(f"mean tokens: {s['mean_prompt_tokens']:.0f} in / {s['mean_completion_tokens']:.0f} out")
    if s["mean_cost_usd"] is None:
        print("mean cost: unknown (no SIMSMITH_PRICE_* set for this model)")
    elif s["all_priced"]:
        print(f"mean cost: ${s['mean_cost_usd']:.5f}/task-run (total ${s['total_cost_usd']:.5f} over {s['n']})")
    else:
        print(f"mean cost (priced rows only, {s['priced_rows']}/{s['n']}): ${s['mean_cost_usd']:.5f}/task-run")
    print(f"mean wall time: {s['mean_wall_seconds']:.1f}s/task-run")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-task", type=int, default=4,
                    help="ignored if --n-per-template is given")
    ap.add_argument("--n-per-template", type=int, default=0,
                    help="if set, cycle dev tasks (by index) to produce exactly this many "
                    "task-runs per template, instead of --runs-per-task * (all dev tasks)")
    ap.add_argument("--templates", default="reach,pick_place")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-repairs", type=int, default=MAX_REPAIRS,
                    help="0 disables repair entirely (plain best-of-5, single generation round)")
    ap.add_argument("--model-generate", default="", help="override SIMSMITH_MODEL_GENERATE for this run")
    ap.add_argument("--sim-python", default="")
    ap.add_argument("--progress-file", default=os.path.join(ROOT, "eval", "results", "best_of_5_progress.jsonl"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--extra-payload", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    extra_payload = json.loads(args.extra_payload) if args.extra_payload else {}
    templates = args.templates.split(",")

    if args.fresh and os.path.exists(args.progress_file):
        os.remove(args.progress_file)
    done = _load_progress(args.progress_file)
    if done:
        print(f"resuming: {len(done)} task-run(s) already completed in {args.progress_file}")

    cfg = Config.from_env()
    if args.model_generate:
        cfg.model_generate = args.model_generate
    sim_python = args.sim_python or cfg.sim_python or DEFAULT_SIM_PYTHON
    runner = SubprocessRunner(sim_python, cfg.run_timeout_s)
    docs = DocIndex(os.path.join(ROOT, "docs_index"))

    ts = int(time.time())
    usage_path = os.path.join(ROOT, "eval", "results", f"best_of_5_usage_{ts}.jsonl")
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key, usage_log_path=usage_path, extra_payload=extra_payload)
    if extra_payload:
        print(f"extra_payload={extra_payload}")

    with open(os.path.join(ROOT, "eval", "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    dev_by_template = {"reach": [], "pick_place": []}
    for t in tasks["dev"]:
        dev_by_template.setdefault(t["template"], []).append(t)
    dev_by_template = {t: v for t, v in dev_by_template.items() if t in templates}

    jobs_by_template = {}
    for template, dev_tasks in dev_by_template.items():
        if args.n_per_template:
            jobs_by_template[template] = [(dev_tasks[i % len(dev_tasks)], i) for i in range(args.n_per_template)]
        else:
            jobs_by_template[template] = [(task, run_idx) for task in dev_tasks
                                          for run_idx in range(args.runs_per_task)]

    print(f"prompt_version={PROMPT_VERSION}  model={cfg.model_generate}  sim_python={sim_python}", flush=True)
    print(f"sim seeds per generation: {cfg.seeds}  budget_usd={llm.budget_usd}  max_repairs={args.max_repairs}", flush=True)
    for template in templates:
        print(f"  {template}: {len(jobs_by_template[template])} task-runs over dev tasks "
              f"{[t['id'] for t in dev_by_template[template]]}", flush=True)

    all_rows = run_all(jobs_by_template, llm, runner, docs, cfg,
                       usage_path, args.progress_file, done, args.workers, args.max_repairs)
    summaries = {t: summarize(t, all_rows[t]) for t in templates}
    for t in templates:
        print_summary(summaries[t], args.max_repairs)

    report = {
        "prompt_version": PROMPT_VERSION, "tag": args.tag, "extra_payload": extra_payload,
        "condition": ("best_of_5_repair" if args.max_repairs > 0 else "best_of_5"),
        "max_repairs": args.max_repairs,
        "model": cfg.model_generate, "sim_python": sim_python, "sim_seeds": list(cfg.seeds),
        "timestamp": ts, "summaries": summaries, "rows": all_rows,
    }
    out_path = os.path.join(ROOT, "eval", "results", f"best_of_5_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {out_path}")
    print(f"Usage log: {usage_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
