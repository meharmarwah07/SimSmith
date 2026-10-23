"""Execution-based verification. The LLM never decides who wins."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .runner import CONTRACT_DIR, RunResult
from .templates import TEMPLATES

if CONTRACT_DIR not in sys.path:
    sys.path.insert(0, CONTRACT_DIR)
import simsmith_contract  # noqa: E402  (runtime/ module, path-injected above)

# Short names for the clear-message wording the task-mismatch check reports.
_TASK_KEY_LABEL = {"target_pos": "target", "cube_initial_pos": "cube_initial", "bin_center": "bin"}

# A script that touched mujoco but stepped it fewer times than this never actually
# simulated -- it ticked the pre-flight "calls mj_step" box without doing anything with
# it. "_sim_steps" is written by the contract's own measurement (runtime/
# simsmith_contract.py), never by the script, so this can't be gamed by reporting a
# bigger number.
MIN_MJ_STEPS = 10


def _simulation_too_short(result: RunResult) -> str:
    steps = result.final_state.get("_sim_steps")
    if steps is None or steps >= MIN_MJ_STEPS:
        return ""
    return f"script only stepped the simulation {steps} time(s), need >= {MIN_MJ_STEPS}"


def _task_mismatch(template_name: str, result: RunResult, tol: float = 1e-6) -> str:
    """Recompute this run's harness-owned task parameters from its seed + perturb and
    compare against what the script reported. The script's own target/cube/bin values
    are NEVER trusted as ground truth -- only this independent recomputation is."""
    if template_name not in simsmith_contract.TASK_KEYS:
        return ""
    expected = simsmith_contract.task_for(template_name, result.seed, result.perturb)
    for key, exp_vec in expected.items():
        label = _TASK_KEY_LABEL.get(key, key)
        rep_vec = result.final_state.get(key)
        if not isinstance(rep_vec, (list, tuple)) or len(rep_vec) < len(exp_vec):
            return f"reported {label} is missing or malformed"
        if any(abs(float(rep_vec[i]) - float(exp_vec[i])) > tol for i in range(len(exp_vec))):
            return f"reported {label} differs from task {label}"
    return ""


@dataclass
class Score:
    ok: bool                        # passes the bar (success_rate >= pass_threshold, no crashes)
    success_rate: float             # fraction of seeds whose final state passed the template check
    crashes: int                    # seeds with no usable result
    details: list = field(default_factory=list)   # per-seed human-readable lines
    failure_text: str = ""          # fed to the repair step
    mean_seconds: float = 0.0


def verify(template_name: str, results: list, pass_threshold: float = 0.8) -> Score:
    tmpl = TEMPLATES[template_name]
    passed, crashes, lines, fail_lines = 0, 0, [], []
    for r in results:
        if not r.ok:
            crashes += 1
            lines.append(f"seed {r.seed}: CRASH {r.error[:200]}")
            fail_lines.append(f"seed {r.seed}: crashed: {r.error[-600:]}")
            continue
        missing = [k for k in tmpl.required_keys if k not in r.final_state]
        if missing:
            crashes += 1
            lines.append(f"seed {r.seed}: missing keys {missing}")
            fail_lines.append(f"seed {r.seed}: result missing keys {missing}. Expected {tmpl.state_help}")
            continue
        too_short = _simulation_too_short(r)
        if too_short:
            crashes += 1
            lines.append(f"seed {r.seed}: {too_short}")
            fail_lines.append(f"seed {r.seed}: {too_short} (it must actually run physics, not just report positions)")
            continue
        mismatch = _task_mismatch(template_name, r)
        if mismatch:
            crashes += 1
            lines.append(f"seed {r.seed}: {mismatch}")
            fail_lines.append(f"seed {r.seed}: {mismatch} (self-reported task parameters are never trusted)")
            continue
        ok, detail = tmpl.check(r.final_state)
        lines.append(f"seed {r.seed}: {'PASS' if ok else 'FAIL'} {detail}")
        if ok:
            passed += 1
        else:
            fail_lines.append(f"seed {r.seed}: task not achieved: {detail}")
    n = max(len(results), 1)
    rate = passed / n
    secs = sum(r.seconds for r in results) / n
    return Score(ok=(crashes == 0 and rate >= pass_threshold), success_rate=rate,
                 crashes=crashes, details=lines, failure_text="\n".join(fail_lines),
                 mean_seconds=secs)
