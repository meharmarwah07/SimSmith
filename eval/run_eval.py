"""Three-condition evaluation on held-out tasks.

Conditions
  single_shot               : 1 candidate, no repair, no library
  best_of_5_repair          : 5 candidates + repair, no library
  best_of_5_repair_library  : 5 candidates + repair + library of VERIFIED DEV scripts

Honesty rules baked in
  * The library is built only from dev tasks, then frozen. Held-out runs never
    write to it.
  * A hard cap on total sim executions (--max-runs) protects the credit budget;
    if hit, the report says it was truncated.
  * All results are written to eval/results/, wins and losses alike.

Usage:  python -m eval.run_eval --max-runs 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.llm import TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.frozen_config import (  # noqa: E402
    ConfigDrift, config_hash as compute_config_hash, load_frozen_config,
    require_config_matches_frozen,
)
from simsmith.preflight import load_allowlist  # noqa: E402
from simsmith.run_meta import build_run_meta  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    "single_shot": dict(n_candidates=1, use_library=False, do_repair=False),
    "best_of_5_repair": dict(n_candidates=5, use_library=False, do_repair=True),
    "best_of_5_repair_library": dict(n_candidates=5, use_library=True, do_repair=True),
}
_MAX_REPAIRS_BY_CONDITION = {"single_shot": 0, "best_of_5_repair": 2, "best_of_5_repair_library": 2}


def run_eval(make_pipeline, tasks: dict, conditions: list, max_runs: int, out_dir: str,
             library_path: str, extra_payload: dict | None = None, workers: int | None = None,
             config_hash: str | None = None) -> dict:
    """make_pipeline(library) -> (Pipeline). Returns the result dict (also written to disk)."""
    os.makedirs(out_dir, exist_ok=True)
    report = {"started": time.time(), "max_runs": max_runs, "truncated": False,
              "config_hash": config_hash, "conditions": {}, "library_size_at_heldout": 0}
    lib = Library(library_path)
    pipe_for_runs = make_pipeline(lib)
    runner = pipe_for_runs.runner
    extra_payload = extra_payload or {}

    def over_budget() -> bool:
        return runner.runs >= max_runs

    # Build the library from DEV tasks (only needed for the library condition).
    if "best_of_5_repair_library" in conditions:
        for t in tasks["dev"]:
            if over_budget():
                report["truncated"] = True
                break
            pipe_for_runs.solve(t["task"], save_to_library=True, **CONDITIONS["best_of_5_repair_library"])
        report["library_size_at_heldout"] = len(lib)

    for cond in conditions:
        rows = []
        run_meta = build_run_meta(condition=cond, cfg=pipe_for_runs.cfg, extra_payload=extra_payload,
                                  n_candidates=CONDITIONS[cond]["n_candidates"],
                                  max_repairs=_MAX_REPAIRS_BY_CONDITION[cond], workers=workers,
                                  config_hash=config_hash)
        for t in tasks["heldout"]:
            if over_budget():
                report["truncated"] = True
                break
            # held-out runs never save to the library
            res = pipe_for_runs.solve(t["task"], save_to_library=False, **CONDITIONS[cond])
            w = res.winner
            rows.append({
                "id": t["id"], "template": res.template, "status": res.status,
                "success_rate": w.score.success_rate if w and w.score else 0.0,
                "winner_stage": w.stage if w else None,
                "n_candidates_generated": len(res.candidates),
                "preflight_rejects": sum(1 for c in res.candidates if not c.preflight_ok),
                **run_meta,
            })
        verified = sum(1 for r in rows if r["status"] == "VERIFIED")
        report["conditions"][cond] = {
            "tasks_run": len(rows), "verified": verified,
            "verified_rate": verified / len(rows) if rows else None,
            "mean_success_rate": sum(r["success_rate"] for r in rows) / len(rows) if rows else None,
            "rows": rows,
        }
    report["total_sim_runs"] = runner.runs
    path = os.path.join(out_dir, f"eval_{int(report['started'])}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["path"] = path
    return report


def print_summary(report: dict) -> None:
    print(f"\nTotal sim executions: {report['total_sim_runs']} (cap {report['max_runs']})"
          f"{'  ** TRUNCATED: budget cap hit, numbers are partial **' if report['truncated'] else ''}")
    print(f"Library size when held-out started: {report['library_size_at_heldout']}")
    print(f"{'condition':<22}{'tasks':>6}{'verified':>10}{'verified %':>12}{'mean succ.':>12}")
    for name, c in report["conditions"].items():
        vr = "n/a" if c["verified_rate"] is None else f"{100 * c['verified_rate']:.0f}%"
        ms = "n/a" if c["mean_success_rate"] is None else f"{c['mean_success_rate']:.2f}"
        print(f"{name:<22}{c['tasks_run']:>6}{c['verified']:>10}{vr:>12}{ms:>12}")
    print(f"\nFull results: {report.get('path')}")
    print("Small n: treat differences below ~2 tasks as noise. Report all conditions, including losses.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=300, help="hard cap on sim executions")
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

    with open(os.path.join(HERE, "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key, extra_payload=extra_payload)
    runner = SubprocessRunner(cfg.sim_python, cfg.run_timeout_s)
    docs = DocIndex("docs_index")
    allow = load_allowlist(args.allowlist)

    def make_pipeline(lib):
        return Pipeline(llm, runner, lib, docs, cfg, allow or None, args.isaac_version)

    lib_path = os.path.join(HERE, "results", f"library_{int(time.time())}.jsonl")
    report = run_eval(make_pipeline, tasks, args.conditions, args.max_runs,
                      os.path.join(HERE, "results"), lib_path, extra_payload, workers, frozen_hash)
    print_summary(report)


if __name__ == "__main__":
    main()
