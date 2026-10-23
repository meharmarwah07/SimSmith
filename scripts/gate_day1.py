"""Day-1 gate: proves the end-to-end machinery works for real before building on it.

Checks, in order (stops at first failure):
  1. Token Factory reachable and the plan/generate/fast models answer.
  2. The sim Python launcher exists and can import the contract.
  3. A HAND-WRITTEN script runs end to end and produces a verifiable result.
  4. A Nemotron-GENERATED script for the same task runs (pass or fail is
     recorded; what matters is that the machinery works and the error is captured).
Also prints seconds per run, which sets the size of your final eval.

Usage:
  python scripts/gate_day1.py
Defaults to examples/reach_mujoco.py (real MuJoCo physics, headless, no GPU
needed) as the hand-written reference, run through SIMSMITH_SIM_PYTHON if set
or ~/nw/bin/python otherwise. Pass --handwritten to point at a different
script (e.g. examples/reach_newton.py), or --sim-python to override the
interpreter.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.llm import LLMError, TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.verifier import verify  # noqa: E402

DEFAULT_SIM_PYTHON = os.path.expanduser("~/nw/bin/python")


def step(n, msg):
    print(f"\n[{n}] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handwritten", default="examples/reach_mujoco.py")
    ap.add_argument("--sim-python", default="", help="override SIMSMITH_SIM_PYTHON / the ~/nw/bin/python default")
    args = ap.parse_args()
    cfg = Config.from_env()
    sim_python = args.sim_python or cfg.sim_python or DEFAULT_SIM_PYTHON

    step(1, "Token Factory models")
    try:
        llm = TokenFactoryLLM(cfg.base_url, cfg.api_key)
        for label, model in (("plan", cfg.model_plan), ("generate", cfg.model_generate), ("fast", cfg.model_fast)):
            out = llm.complete("Reply with the single word: ok", "ping", model, 0.0, 300, kind="intake")
            print(f"  {label:<9} {model}: {out.strip()[:40]!r}")
    except LLMError as e:
        print(f"  FAIL: {e}")
        return 1

    step(2, "Sim Python launcher")
    runner = SubprocessRunner(sim_python, cfg.run_timeout_s)
    print(f"  command: {runner.cmd}")
    if not sim_python:
        print("  WARNING: no sim python resolved, so this uses plain Python. "
              "A pass below does NOT show MuJoCo/Newton works.")

    step(3, f"Hand-written script: {args.handwritten}")
    with open(args.handwritten, encoding="utf-8") as f:
        script = f.read()
    results = [runner.run(script, seed) for seed in cfg.seeds]
    for r in results:
        print(f"  seed {r.seed}: ok={r.ok} {r.seconds:.1f}s {r.error[:200]}")
    score = verify("reach", results)
    print(f"  verified success rate: {score.success_rate:.2f}, crashes: {score.crashes}")
    if score.crashes or score.success_rate == 0:
        print("  FAIL: the hand-written script must run and succeed before anything else matters.")
        for line in score.details:
            print("   ", line)
        return 1
    secs = sum(r.seconds for r in results) / len(results)
    print(f"  mean seconds per run: {secs:.1f}  -> estimate eval cost before choosing sizes")

    step(4, "Nemotron-generated script, same task")
    pipe = Pipeline(llm, runner, Library("runs/gate_library.jsonl"), DocIndex("docs_index"), cfg)
    res = pipe.solve("Move the arm's gripper to the target point.", n_candidates=1,
                     use_library=False, do_repair=False, save_to_library=False)
    c = res.candidates[0]
    print(f"  status: {res.status}  preflight_ok={c.preflight_ok}")
    for p in c.preflight_problems:
        print("   preflight:", p)
    if c.score:
        for line in c.score.details:
            print("   ", line)
    print("\nGate result: machinery works. A failing generated script here is EXPECTED and is "
          "the baseline you improve on; record its error type (hallucinated API vs logic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
