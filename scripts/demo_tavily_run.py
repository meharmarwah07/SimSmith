"""Runs ONE real (non-MOCK) demo task through the live pipeline with Tavily's
plan-step failure-mode research turned on, and saves the result so the demo UI
can load it with --replay -- for the Tavily bonus prize.

Usage:
  set -a; source .env; set +a   # needs TOKEN_FACTORY_* and TAVILY_API_KEY
  python scripts/demo_tavily_run.py [--max-usd 0.30] [--max-tavily-credits 4]

Always runs the "reach" example task from app/server.py's EXAMPLE_TASKS
("Move the arm's gripper to the target point."), with Tavily unconditionally
on (a TavilyClient is built directly here, not gated behind SIMSMITH_TAVILY --
this script's whole point is to demo it).

Hard caps, both fail CLOSED (raise rather than silently undercounting):
  --max-usd             LLM spend cap, passed straight to TokenFactoryLLM's
                         budget_usd (simsmith/llm.py). Every model this run uses
                         must have a SIMSMITH_PRICE_* configured, or the very
                         first call is blocked before it runs.
  --max-tavily-credits  Tavily search cap, passed to TavilyClient's max_searches
                         (simsmith/tavily.py). search_depth="basic" costs 1
                         credit/search, so this is also the Tavily credit cap.

Writes two files under eval/results/, both stamped <ts> = unix time at start:
  demo_tavily_run_<ts>.json      -- the UI-loadable run, same shape /api/run
                                     produces (serialize_result(), source="live").
                                     Open it with:
                                       python -m app.server --replay eval/results/demo_tavily_run_<ts>.json
  demo_tavily_results_<ts>.json  -- the raw Tavily research alone (queries used,
                                     how many sources were dropped and why, and
                                     the failure modes with their source URLs).

Run by hand, not from CI or any test suite: this makes real Token Factory and
Tavily calls and spends real credits.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.server import serialize_result, tavily_failure_modes_for_result  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.llm import LLMError, TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.prompts import PROMPT_VERSION  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.tavily import TavilyClient, TavilyError  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "eval", "results")
DEFAULT_SIM_PYTHON = os.path.expanduser("~/nw/bin/python")

# The reach dev task the demo UI itself offers as its first example (see
# app/server.py's EXAMPLE_TASKS) -- keeps this script's run identical to what a
# person clicking that example button in the UI and pressing Run would get.
TASK = "Move the arm's gripper to the target point."


def _usage_totals(usage_path: str) -> tuple:
    """(prompt+completion tokens, cost_usd) summed from a fresh usage log file."""
    tokens, cost = 0, 0.0
    if not os.path.exists(usage_path):
        return tokens, cost
    with open(usage_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tokens += int(rec.get("prompt_tokens", 0)) + int(rec.get("completion_tokens", 0))
            if rec.get("cost_usd") is not None:
                cost += float(rec["cost_usd"])
    return tokens, cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-usd", type=float, default=0.30, help="hard LLM spend cap (fail closed)")
    ap.add_argument("--max-tavily-credits", type=int, default=4, help="hard Tavily search cap")
    ap.add_argument("--sim-python", default="", help="override SIMSMITH_SIM_PYTHON")
    args = ap.parse_args()

    cfg = Config.from_env()  # loads .env (if present) before the TAVILY_API_KEY check below
    if not os.environ.get("TAVILY_API_KEY"):
        print("Set TAVILY_API_KEY first (see .env.example) -- this script always runs with "
              "Tavily on.", file=sys.stderr)
        return 1

    sim_python = args.sim_python or cfg.sim_python or DEFAULT_SIM_PYTHON
    ts = int(time.time())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    usage_path = os.path.join(RESULTS_DIR, f"demo_tavily_usage_{ts}.jsonl")

    try:
        llm = TokenFactoryLLM(cfg.base_url, cfg.api_key, usage_log_path=usage_path,
                              budget_usd=args.max_usd)
    except LLMError as e:
        print(f"Cannot start: {e}", file=sys.stderr)
        return 1
    try:
        tavily_client = TavilyClient(max_searches=args.max_tavily_credits)
    except TavilyError as e:
        print(f"Cannot start: {e}", file=sys.stderr)
        return 1

    runner = SubprocessRunner(sim_python, cfg.run_timeout_s)
    docs = DocIndex(os.path.join(ROOT, "docs_index"))
    library = Library(os.path.join(ROOT, "app", "ui_library.jsonl"))
    pipe = Pipeline(llm, runner, library, docs, cfg, tavily_client=tavily_client)

    print(f"model={cfg.model_generate}  sim_python={sim_python}  "
          f"max_usd={args.max_usd}  max_tavily_credits={args.max_tavily_credits}", flush=True)
    print(f"task: {TASK!r}", flush=True)

    t0 = time.time()
    try:
        res = pipe.solve(TASK, save_to_library=False)
    except Exception as e:  # noqa: BLE001 -- report, don't retry (see module docstring)
        print(f"Run failed: {e}", file=sys.stderr)
        return 1
    elapsed_s = time.time() - t0

    tokens, cost_usd = _usage_totals(usage_path)
    tavily = tavily_failure_modes_for_result(tavily_client, res)

    payload = serialize_result(
        res, library_size=len(library), elapsed_s=elapsed_s, source="live",
        tokens=tokens, cost_usd=cost_usd, sim_runs=runner.runs,
        model=cfg.model_generate or "unknown-model", prompt_version=PROMPT_VERSION,
        date=datetime.date.today().isoformat(), tavily=tavily,
    )

    run_path = os.path.join(RESULTS_DIR, f"demo_tavily_run_{ts}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump({"result": payload}, f, indent=2)

    tavily_results_path = os.path.join(RESULTS_DIR, f"demo_tavily_results_{ts}.json")
    with open(tavily_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "date": datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z",
            "task": TASK,
            "template": res.template,
            "tavily_on": tavily is not None,
            "searches_made": tavily_client.searches_made,
            **(tavily or {"queries_used": [], "dropped_count": 0, "items": []}),
        }, f, indent=2)

    print(f"\nstatus={res.status}  elapsed={elapsed_s:.1f}s  tokens={tokens}  "
          f"cost_usd={cost_usd:.5f}  tavily_credits_used={tavily_client.searches_made}")
    if tavily:
        print(f"tavily queries used: {tavily['queries_used']}")
        print(f"{len(tavily['items'])} failure mode(s) found:")
        for item in tavily["items"]:
            print(f"  - {item['mode']}  (source: {item['source_url']})")
    else:
        print("tavily: off or unavailable for this run")
    print(f"\nRun file: {run_path}")
    print(f"Tavily results file: {tavily_results_path}")
    print(f"Usage log: {usage_path}")
    print(f"Replay with: python -m app.server --replay {run_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
