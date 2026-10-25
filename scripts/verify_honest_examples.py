"""Run the hand-written honest examples (examples/reach_mujoco.py,
examples/pick_place_mujoco.py) through the real, hardened harness
(SubprocessRunner -> verify()) across N fresh seeds each, to confirm hardening
(stripped env, network isolation, stricter preflight) hasn't regressed the scripts
the project's own honesty claims rest on. Local only, no LLM calls, $0 cost.

Usage:
  python scripts/verify_honest_examples.py --n 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.preflight import check  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.verifier import verify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = {"reach": "reach_mujoco.py", "pick_place": "pick_place_mujoco.py"}


def run_one(script: str, template: str, seed: int) -> tuple:
    r = SubprocessRunner("", 60).run(script, seed=seed, template=template)
    if not r.ok:
        return seed, False, f"crashed: {r.error[:300]}"
    s = verify(template, [r])
    return seed, s.ok, (s.failure_text[:300] if not s.ok else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    preflight_ok = True
    for template, fname in EXAMPLES.items():
        with open(os.path.join(ROOT, "examples", fname), encoding="utf-8") as f:
            script = f.read()
        pf = check(script)
        print(f"{template} ({fname}): preflight ok={pf.ok} {pf.problems if not pf.ok else ''}")
        preflight_ok = preflight_ok and pf.ok

    for template, fname in EXAMPLES.items():
        with open(os.path.join(ROOT, "examples", fname), encoding="utf-8") as f:
            script = f.read()
        t0 = time.time()
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, script, template, seed) for seed in range(args.n)]
            results = [f.result() for f in as_completed(futures)]
        passed = sum(1 for _, ok, _ in results if ok)
        failures = [(seed, msg) for seed, ok, msg in results if not ok]
        dt = time.time() - t0
        print(f"\n{template}: {passed}/{args.n} passed ({dt:.1f}s, {args.workers} workers)")
        for seed, msg in sorted(failures):
            print(f"  seed {seed} FAILED: {msg}")

    return 0 if preflight_ok else 1


if __name__ == "__main__":
    sys.exit(main())
