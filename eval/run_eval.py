"""Three-condition evaluation on held-out tasks.

Conditions
  single_shot       : 1 candidate, no repair, no library
  best_of_5         : 5 candidates + repair, no library
  best_of_5_library : 5 candidates + repair + library of VERIFIED DEV scripts

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
from simsmith.preflight import load_allowlist  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    "single_shot": dict(n_candidates=1, use_library=False, do_repair=False),
    "best_of_5": dict(n_candidates=5, use_library=False, do_repair=True),
    "best_of_5_library": dict(n_candidates=5, use_library=True, do_repair=True),
}


def run_eval(make_pipeline, tasks: dict, conditions: list, max_runs: int, out_dir: str,
             library_path: str) -> dict:
    """make_pipeline(library) -> (Pipeline). Returns the result dict (also written to disk)."""
    os.makedirs(out_dir, exist_ok=True)
    report = {"started": time.time(), "max_runs": max_runs, "truncated": False,
              "conditions": {}, "library_size_at_heldout": 0}
    lib = Library(library_path)
    pipe_for_runs = make_pipeline(lib)
    runner = pipe_for_runs.runner

    def over_budget() -> bool:
        return runner.runs >= max_runs

    # Build the library from DEV tasks (only needed for the library condition).
    if "best_of_5_library" in conditions:
        for t in tasks["dev"]:
            if over_budget():
                report["truncated"] = True
                break
            pipe_for_runs.solve(t["task"], save_to_library=True, **CONDITIONS["best_of_5_library"])
        report["library_size_at_heldout"] = len(lib)

    for cond in conditions:
        rows = []
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
                "n_candidates": len(res.candidates),
                "preflight_rejects": sum(1 for c in res.candidates if not c.preflight_ok),
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
    args = ap.parse_args()

    cfg = Config.from_env()
    with open(os.path.join(HERE, "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key)
    runner = SubprocessRunner(cfg.sim_python, cfg.run_timeout_s)
    docs = DocIndex("docs_index")
    allow = load_allowlist(args.allowlist)

    def make_pipeline(lib):
        return Pipeline(llm, runner, lib, docs, cfg, allow or None, args.isaac_version)

    lib_path = os.path.join(HERE, "results", f"library_{int(time.time())}.jsonl")
    report = run_eval(make_pipeline, tasks, args.conditions, args.max_runs,
                      os.path.join(HERE, "results"), lib_path)
    print_summary(report)


if __name__ == "__main__":
    main()
