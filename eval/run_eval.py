"""Three-condition evaluation on held-out tasks.

Conditions
  single_shot               : 1 candidate, no repair, no library
  best_of_5_repair          : 5 candidates + repair, no library
  best_of_5_repair_library  : 5 candidates + repair + library of VERIFIED DEV scripts

Honesty rules baked in
  * The library is built only from dev tasks, then frozen. Held-out runs never
    write to it.
  * A hard cap on total sim executions (--max-runs) and total $ spend
    (--budget-usd, checked against TokenFactoryLLM.spent_usd/budget_usd -- see
    simsmith/llm.py) protect the credit budget; if either is hit, the report says
    it was truncated, and says which cap. Task-runs are submitted in a bounded
    sliding window of at most --workers in flight, re-checking both caps as each
    one finishes, so a cap is never overshot by more than one window's worth of
    in-flight work.
  * All results are written to eval/results/, wins and losses alike.

Safety features (step D, 2026-10-27)
  * --repeats N (default 5): each heldout task is run N times per condition, not
    just once -- a single VERIFIED/FAILED draw from a sampled LLM is not a rate.
  * Prints an estimated task-run count and $ cost BEFORE any network call, and
    refuses to proceed without --yes. --dry-run prints the estimate and exits,
    always, regardless of --yes.
  * --progress-file checkpoints after every task-run (same pattern as
    scripts/baseline_best_of_5.py) -- a crash or Ctrl+C can be resumed by rerunning
    the exact same command, without re-paying for any task-run already completed.
    --library-path defaults to a name derived from --progress-file, so a resume
    reuses the same frozen library instead of silently rebuilding it from scratch.
  * --workers parallelizes task-runs across a thread pool; TokenFactoryLLM's own
    retry logic already handles 429/5xx (simsmith/llm.py's _RETRYABLE_STATUS).
  * config/frozen.json (simsmith/frozen_config.py) is loaded and enforced before
    any of this runs -- see that module for what "conditions differ in the
    intended variable only" means here.

Usage:  python -m eval.run_eval --repeats 5 --workers 4 --yes
        python -m eval.run_eval --dry-run           # estimate only, no network calls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.frozen_config import (  # noqa: E402
    ConfigDrift, config_hash as compute_config_hash, load_frozen_config,
    require_config_matches_frozen,
)
from simsmith.library import Library  # noqa: E402
from simsmith.llm import TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.preflight import load_allowlist  # noqa: E402
from simsmith.run_meta import CLASSIFIER_VERSION, build_run_meta  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    "single_shot": dict(n_candidates=1, use_library=False, do_repair=False),
    "best_of_5_repair": dict(n_candidates=5, use_library=False, do_repair=True),
    "best_of_5_repair_library": dict(n_candidates=5, use_library=True, do_repair=True),
}
_MAX_REPAIRS_BY_CONDITION = {"single_shot": 0, "best_of_5_repair": 2, "best_of_5_repair_library": 2}

# Empirical mean cost_usd/task-run, Super reasoning off (config/frozen.json), from
# this project's own real measured runs -- used ONLY to print a pre-run ESTIMATE; the
# real number is always measured and reported after the run regardless. A condition/
# template this project hasn't actually measured yet falls back to
# _FALLBACK_COST_PER_CANDIDATE * n_candidates.
_EMPIRICAL_MEAN_COST_PER_TASK_RUN = {
    ("single_shot", "reach"): 0.0025,            # eval/results/baseline_1792933477.json
    ("best_of_5_repair", "reach"): 0.0169,       # eval/results/best_of_5_1792933674.json
    ("best_of_5_repair", "pick_place"): 0.0273,  # eval/results/best_of_5_1792933958.json, _1792936413.json
}
_LIBRARY_COST_MULTIPLIER = 1.15  # best_of_5_repair_library ~= best_of_5_repair + a couple extra prompt examples
_FALLBACK_COST_PER_CANDIDATE = 0.0025  # ~= single_shot/reach's own single-candidate cost


def estimate_cost_per_task_run(condition: str, template: str, n_candidates: int) -> float:
    known = _EMPIRICAL_MEAN_COST_PER_TASK_RUN.get((condition, template))
    if known is not None:
        return known
    if condition == "best_of_5_repair_library":
        base = _EMPIRICAL_MEAN_COST_PER_TASK_RUN.get(("best_of_5_repair", template))
        if base is not None:
            return base * _LIBRARY_COST_MULTIPLIER
    return _FALLBACK_COST_PER_CANDIDATE * max(n_candidates, 1)


def build_jobs(tasks: dict, conditions: list, repeats: int) -> list:
    """[{"condition", "template", "task_id", "task", "repeat_idx"}, ...] for every
    heldout task-run this invocation needs to measure (library-build jobs are
    tracked separately -- see _lib_build_jobs)."""
    jobs = []
    for cond in conditions:
        for t in tasks["heldout"]:
            for i in range(repeats):
                jobs.append({"condition": cond, "template": t["template"],
                            "task_id": t["id"], "task": t["task"], "repeat_idx": i})
    return jobs


def print_cost_estimate(jobs: list, lib_build_tasks: list) -> float:
    by_key = Counter()
    for j in jobs:
        n_candidates = CONDITIONS[j["condition"]]["n_candidates"]
        by_key[(j["condition"], j["template"])] += 1
    total = 0.0
    print(f"\n{'condition':<26}{'template':<14}{'task-runs':>10}{'est. $/run':>12}{'est. total':>12}")
    for (cond, template), n in sorted(by_key.items()):
        n_candidates = CONDITIONS[cond]["n_candidates"]
        per_run = estimate_cost_per_task_run(cond, template, n_candidates)
        subtotal = per_run * n
        total += subtotal
        print(f"{cond:<26}{template:<14}{n:>10}{per_run:>12.4f}{subtotal:>12.4f}")
    lib_cost = 0.0
    if lib_build_tasks:
        for t in lib_build_tasks:
            lib_cost += estimate_cost_per_task_run("best_of_5_repair_library", t["template"], 5)
        print(f"{'(library build, dev tasks)':<26}{'':<14}{len(lib_build_tasks):>10}{'':>12}{lib_cost:>12.4f}")
        total += lib_cost
    print(f"\nestimated total task-runs: {len(jobs) + len(lib_build_tasks)}  "
         f"estimated total cost: ${total:.4f}")
    print("Estimates are historical means, not a guarantee -- the real per-run cost "
         "varies with how many repairs a task-run needs.")
    return total


def _load_progress(progress_path: str) -> dict:
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                row = json.loads(ln)
                done[tuple(row["_job_key"])] = row
    return done


_progress_lock = threading.Lock()


def _append_progress(progress_path: str, row: dict) -> None:
    with _progress_lock:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _run_bounded(pool: ThreadPoolExecutor, jobs: list, workers: int, submit_fn, over_budget_fn,
                 on_result) -> None:
    """Keeps at most `workers` jobs in flight at once, re-checking over_budget_fn()
    before each new submission -- so a cap is overshot by at most one window's worth
    of already-in-flight work, not by the whole remaining job list. Submitting the
    full job list up front (the simpler pattern scripts/baseline_best_of_5.py uses)
    would check the cap once, before any of them had actually run, and then run every
    one of them regardless once workers>1 -- fine for a sim-run cap with a generous
    margin, not fine for a tight $ budget cap meant to stop spend close to exactly
    where it's set."""
    pending = iter(jobs)
    in_flight = {}

    def _fill():
        while len(in_flight) < workers and not over_budget_fn():
            try:
                job = next(pending)
            except StopIteration:
                return
            in_flight[submit_fn(job)] = job

    _fill()
    while in_flight:
        done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
        for f in done:
            job = in_flight.pop(f)
            on_result(job, f.result())
        _fill()


def _usage_rows_for_tag(usage_path: str, tag: str) -> list:
    if not os.path.exists(usage_path):
        return []
    rows = []
    with open(usage_path, encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            if rec.get("tag") == tag:
                rows.append(rec)
    return rows


def _cost_for_tag(usage_path: str, tag: str) -> tuple:
    rows = _usage_rows_for_tag(usage_path, tag)
    if rows and all(r.get("price_known") for r in rows):
        return sum(r["cost_usd"] for r in rows), True
    return None, False


def _budget_exceeded(llm) -> bool:
    """True once this run's $ cap is hit. NOT detected by catching BudgetExceeded
    around a solve() call -- Pipeline.intake()/plan()/generate()/repair() each catch
    LLMError (BudgetExceeded's base class) internally and degrade to a fallback/failed
    candidate instead of letting it propagate, so a caller never sees the exception.
    llm.spent_usd/llm.budget_usd (simsmith/llm.py) are the real signal."""
    return llm.budget_usd is not None and llm.spent_usd >= llm.budget_usd


def _do_lib_build_one(task: dict, llm, runner, docs, cfg, lib: Library, usage_path: str,
                      progress_path: str) -> dict:
    job_key = ["lib", task["id"], 0]
    tag = f"lib:{task['id']}"
    row = {"_job_key": job_key, "task_id": task["id"], "template": task["template"]}
    if _budget_exceeded(llm):
        # Not persisted to progress_path: an un-attempted job (same as one this
        # invocation never submitted at all) should be retried on a later resume,
        # e.g. after the budget is raised -- not permanently skipped.
        row["skipped"] = "budget_exceeded"
        return row
    llm.set_tag(tag)
    pipe = Pipeline(llm, runner, lib, docs, cfg)
    res = pipe.solve(task["task"], save_to_library=True, **CONDITIONS["best_of_5_repair_library"])
    row["status"] = res.status
    cost, known = _cost_for_tag(usage_path, tag)
    row["cost_usd"], row["price_known"] = cost, known
    _append_progress(progress_path, row)
    return row


def _do_measure_one(job: dict, llm, runner, docs, cfg, lib_for_condition: dict,
                    usage_path: str, progress_path: str, run_meta_by_condition: dict) -> dict:
    cond, task_id, repeat_idx = job["condition"], job["task_id"], job["repeat_idx"]
    job_key = ["measure", cond, task_id, repeat_idx]
    tag = f"{cond}:{task_id}:{repeat_idx}"
    if _budget_exceeded(llm):
        # Not persisted -- see _do_lib_build_one's comment; retried on a later resume.
        return {"_job_key": job_key, "skipped": "budget_exceeded", "condition": cond,
               "template": job["template"], "id": task_id}
    llm.set_tag(tag)
    lib = lib_for_condition[cond]
    pipe = Pipeline(llm, runner, lib, docs, cfg)
    t0 = time.time()
    res = pipe.solve(job["task"], save_to_library=False, **CONDITIONS[cond])
    elapsed_s = time.time() - t0
    w = res.winner
    row = {
        "_job_key": job_key, "id": task_id, "template": res.template, "status": res.status,
        "success_rate": w.score.success_rate if w and w.score else 0.0,
        "winner_stage": w.stage if w else None,
        "n_candidates_generated": len(res.candidates),
        "preflight_rejects": sum(1 for c in res.candidates if not c.preflight_ok),
        "elapsed_s": elapsed_s,
    }
    cost, known = _cost_for_tag(usage_path, tag)
    row["cost_usd"], row["price_known"] = cost, known
    row.update(run_meta_by_condition[cond])
    _append_progress(progress_path, row)
    return row


def run_eval(make_pipeline, tasks: dict, conditions: list, max_runs: int, out_dir: str,
             library_path: str, extra_payload: dict | None = None, workers: int | None = None,
             config_hash: str | None = None, repeats: int = 1, progress_path: str | None = None,
             model: str = "unknown", prompt_version: str = "unknown") -> dict:
    """make_pipeline(library) -> (Pipeline). Returns the result dict (also written to disk).

    Resumable: pass the same progress_path across invocations and already-completed
    task-runs (by (kind, condition, task_id, repeat_idx)) are loaded from disk instead
    of re-executed, so a crash/Ctrl+C never re-pays for a completed call."""
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or 1
    progress_path = progress_path or os.path.join(out_dir, "eval_progress.jsonl")
    done = _load_progress(progress_path)

    report = {"started": time.time(), "max_runs": max_runs, "truncated": False,
              "truncated_reason": None, "model": model, "prompt_version": prompt_version,
              "classifier_version": CLASSIFIER_VERSION, "config_hash": config_hash,
              "repeats": repeats, "conditions": {}, "library_size_at_heldout": 0}
    lib = Library(library_path)
    pipe_for_runs = make_pipeline(lib)
    runner = pipe_for_runs.runner
    extra_payload = extra_payload or {}
    # The single source of truth for where usage/cost rows actually land is the LLM
    # client itself (TokenFactoryLLM.usage_log_path, default eval/results/usage.jsonl
    # -- see simsmith/llm.py) -- NOT a path invented here, which would silently read
    # back nothing if it didn't happen to match. MockLLM (tests) has no such
    # attribute unless given one; cost tracking is simply unavailable then.
    usage_path = (getattr(pipe_for_runs.llm, "usage_log_path", None)
                 or os.path.join(out_dir, f"usage_{int(time.time())}.jsonl"))

    def over_budget() -> bool:
        return runner.runs >= max_runs or _budget_exceeded(pipe_for_runs.llm)

    # Build the library from DEV tasks (only needed for the library condition),
    # checkpointed the same way as the measured jobs -- a resume never re-solves a
    # dev task whose library entry is already on disk.
    if "best_of_5_repair_library" in conditions:
        lib_jobs = [t for t in tasks["dev"] if ("lib", t["id"], 0) not in done]
        if lib_jobs and not over_budget():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                _run_bounded(
                    pool, lib_jobs, workers,
                    submit_fn=lambda t: pool.submit(_do_lib_build_one, t, pipe_for_runs.llm, runner,
                                                    pipe_for_runs.docs, pipe_for_runs.cfg, lib,
                                                    usage_path, progress_path),
                    over_budget_fn=over_budget,
                    on_result=lambda job, row: None,
                )
        report["library_size_at_heldout"] = len(lib)

    # A single Library instance shared across all best_of_5_repair_library jobs (read
    # after the build above, never written to again); single_shot/best_of_5_repair
    # jobs get an empty, never-saved-to Library so use_library=False is enforced at
    # the data level too, not just by the flag.
    lib_for_condition = {c: (lib if c == "best_of_5_repair_library"
                            else Library(os.path.join(out_dir, "_unused_heldout_lib.jsonl")))
                        for c in conditions}

    run_meta_by_condition = {
        cond: build_run_meta(condition=cond, cfg=pipe_for_runs.cfg, extra_payload=extra_payload,
                             n_candidates=CONDITIONS[cond]["n_candidates"],
                             max_repairs=_MAX_REPAIRS_BY_CONDITION[cond], workers=workers,
                             config_hash=config_hash)
        for cond in conditions
    }

    all_jobs = build_jobs(tasks, conditions, repeats)
    pending = [j for j in all_jobs
              if ("measure", j["condition"], j["task_id"], j["repeat_idx"]) not in done]
    rows_by_condition = {c: [] for c in conditions}
    for j in all_jobs:
        key = ("measure", j["condition"], j["task_id"], j["repeat_idx"])
        if key in done:
            rows_by_condition[j["condition"]].append(done[key])

    if pending and not over_budget():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            _run_bounded(
                pool, pending, workers,
                submit_fn=lambda j: pool.submit(_do_measure_one, j, pipe_for_runs.llm, runner,
                                               pipe_for_runs.docs, pipe_for_runs.cfg, lib_for_condition,
                                               usage_path, progress_path, run_meta_by_condition),
                over_budget_fn=over_budget,
                on_result=lambda job, row: rows_by_condition[job["condition"]].append(row),
            )

    if over_budget():
        report["truncated"] = True
        report["truncated_reason"] = "budget" if _budget_exceeded(pipe_for_runs.llm) else "max_runs"

    for cond in conditions:
        rows = [r for r in rows_by_condition[cond] if not r.get("skipped")]
        n = len(rows)
        verified = sum(1 for r in rows if r.get("status") == "VERIFIED")
        cost_vals = [r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), (int, float))]
        report["conditions"][cond] = {
            "tasks_run": n, "verified": verified,
            "verified_rate": verified / n if n else None,
            "mean_success_rate": sum(r["success_rate"] for r in rows) / n if n else None,
            "total_cost_usd": sum(cost_vals) if cost_vals else None,
            "rows": rows,
        }
    report["total_sim_runs"] = runner.runs
    report["total_cost_usd"] = sum(c["total_cost_usd"] for c in report["conditions"].values()
                                   if c["total_cost_usd"] is not None) or None
    report["finished"] = time.time()
    path = os.path.join(out_dir, f"eval_{int(report['started'])}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["path"] = path
    return report


def print_summary(report: dict) -> None:
    reason = f"  ** TRUNCATED ({report['truncated_reason']}): numbers are partial **" if report['truncated'] else ""
    print(f"\nTotal sim executions: {report['total_sim_runs']} (cap {report['max_runs']}){reason}")
    print(f"Library size when held-out started: {report['library_size_at_heldout']}")
    print(f"Repeats per task: {report['repeats']}")
    print(f"{'condition':<26}{'tasks':>6}{'verified':>10}{'verified %':>12}{'mean succ.':>12}{'cost':>12}")
    for name, c in report["conditions"].items():
        vr = "n/a" if c["verified_rate"] is None else f"{100 * c['verified_rate']:.0f}%"
        ms = "n/a" if c["mean_success_rate"] is None else f"{c['mean_success_rate']:.2f}"
        cost = "n/a" if c["total_cost_usd"] is None else f"${c['total_cost_usd']:.4f}"
        print(f"{name:<26}{c['tasks_run']:>6}{c['verified']:>10}{vr:>12}{ms:>12}{cost:>12}")
    total_cost = "n/a" if report.get("total_cost_usd") is None else f"${report['total_cost_usd']:.4f}"
    print(f"\nActual total cost: {total_cost}")
    print(f"Full results: {report.get('path')}")
    print("Small n: treat differences below ~2 tasks as noise. Report all conditions, including losses.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=300, help="hard cap on sim executions")
    ap.add_argument("--budget-usd", type=float, default=None, help="hard cap on $ spend "
                    "(default: SIMSMITH_BUDGET_USD env, or uncapped if neither is set)")
    ap.add_argument("--repeats", type=int, default=5, help="times each heldout task is "
                    "run per condition")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--allowlist", default="docs_index/isaac_api_allowlist.txt")
    ap.add_argument("--isaac-version", default=os.environ.get("SIMSMITH_ISAAC_VERSION", "unset"))
    ap.add_argument("--model-generate", default="", help="override SIMSMITH_MODEL_GENERATE; "
                    "must match config/frozen.json's model or the run is refused")
    ap.add_argument("--extra-payload", default="", help="JSON dict merged into every LLM request "
                    "-- usually left unset: reasoning on/off is derived from config/frozen.json's "
                    "\"reasoning\" field automatically. An explicit value here is still checked "
                    "against frozen.json and the run is refused if it disagrees.")
    ap.add_argument("--workers", type=int, default=None, help="defaults to config/frozen.json's "
                    "\"workers\" if unset")
    ap.add_argument("--frozen-config", default=None, help="path to the frozen config "
                    "(default: config/frozen.json)")
    ap.add_argument("--progress-file", default=None, help="checkpoint file for resuming "
                    "(default: eval/results/eval_progress.jsonl)")
    ap.add_argument("--library-path", default=None, help="default: derived from --progress-file, "
                    "so a resume reuses the same frozen library instead of rebuilding it")
    ap.add_argument("--fresh", action="store_true", help="ignore/delete any existing "
                    "--progress-file and --library-path first")
    ap.add_argument("--yes", action="store_true", help="actually run (otherwise only the "
                    "cost estimate is printed)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and exit; "
                    "never makes a network call, regardless of --yes")
    ap.add_argument("--tasks-file", default=None, help="override eval/tasks.json -- for "
                    "verifying this script against a small task set without touching the "
                    "real held-out tasks (e.g. a copy with 2 dev tasks per template moved "
                    "into \"heldout\"). Never point this at the real tasks.json's heldout "
                    "list for anything other than the real eval.")
    args = ap.parse_args()

    frozen = load_frozen_config(args.frozen_config)
    cfg = Config.from_env()
    if args.model_generate:
        cfg.model_generate = args.model_generate
    workers = args.workers if args.workers is not None else frozen["workers"]
    if args.extra_payload:
        extra_payload = json.loads(args.extra_payload)
    else:
        extra_payload = ({} if frozen["reasoning"] == "on"
                        else {"chat_template_kwargs": {"enable_thinking": False}})

    try:
        require_config_matches_frozen(cfg, extra_payload, workers, frozen)
    except ConfigDrift as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    frozen_hash = compute_config_hash(frozen)

    tasks_path = args.tasks_file or os.path.join(HERE, "tasks.json")
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    jobs = build_jobs(tasks, args.conditions, args.repeats)
    lib_build_tasks = tasks["dev"] if "best_of_5_repair_library" in args.conditions else []
    print_cost_estimate(jobs, lib_build_tasks)
    if args.dry_run:
        print("\n--dry-run: exiting before any network call.")
        return
    if not args.yes:
        print("\nPass --yes to actually run this (or --dry-run to only estimate).")
        sys.exit(1)

    results_dir = os.path.join(HERE, "results")
    progress_path = args.progress_file or os.path.join(results_dir, "eval_progress.jsonl")
    library_path = args.library_path or (progress_path + ".library.jsonl")
    if args.fresh:
        for p in (progress_path, library_path):
            if os.path.exists(p):
                os.remove(p)

    usage_log_path = os.path.join(results_dir, f"eval_usage_{int(time.time())}.jsonl")
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key, extra_payload=extra_payload,
                          budget_usd=args.budget_usd, usage_log_path=usage_log_path)
    runner = SubprocessRunner(cfg.sim_python, cfg.run_timeout_s)
    docs = DocIndex("docs_index")
    allow = load_allowlist(args.allowlist)

    def make_pipeline(lib):
        return Pipeline(llm, runner, lib, docs, cfg, allow or None, args.isaac_version)

    report = run_eval(make_pipeline, tasks, args.conditions, args.max_runs, results_dir,
                      library_path, extra_payload, workers, frozen_hash, args.repeats,
                      progress_path, cfg.model_generate, frozen["prompt_version"])
    print_summary(report)


if __name__ == "__main__":
    main()
