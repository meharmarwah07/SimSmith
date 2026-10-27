"""Test doubles. MockLLM routes by system prompt and replays scripted outputs."""
from __future__ import annotations

import json
import os
import threading
import time

from simsmith import prompts

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_example(name: str) -> str:
    with open(os.path.join(_ROOT, "examples", name), encoding="utf-8") as f:
        return f.read()


# GOOD is, literally, the real hand-written reach example: the harness now requires the
# official scene.xml (model identity) and a real step count, so a mock can no longer get
# away with a toy sphere model -- it has to be at least as honest as the real thing.
GOOD = "```python\n" + _read_example("reach_mujoco.py") + "```"

# Honest model, honest target via get_task(), genuinely steps physics well past the
# MIN_MJ_STEPS floor -- but never drives the arm, so it fails the distance check with
# a real (not faked) ee_pos, same as before just against the real scene now.
WRONG = '''```python
import os
import mujoco
from simsmith_contract import get_task, write_result
ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # never actually drives the arm toward the target
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
target = get_task("reach")["target_pos"]
write_result({"ee_pos": d.xpos[hand].tolist(), "target_pos": target})
```'''

CRASH = '''```python
from simsmith_contract import write_result
raise RuntimeError("AttributeError: module has no attribute hallucinated_fn")
```'''

BANNED = '''```python
import subprocess
from simsmith_contract import write_result
write_result({"ee_pos": [0, 0, 0], "target_pos": [0, 0, 0]})
```'''

SELF_GRADE = '''```python
import os
import mujoco
from simsmith_contract import write_result
ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)
write_result({"success": True})
```'''

# Never calls get_task(): just echoes wherever it ended up as if that were the target.
# The verifier must catch this (reported target != the real, independently-recomputed one).
CHEAT_REACH = '''```python
from simsmith_contract import write_result
ee = [0.0, 0.0, 0.0]
write_result({"ee_pos": ee, "target_pos": ee})
```'''

# Passes the static pre-flight check (imports mujoco, calls mj_step) and reports an
# honest target_pos via get_task() -- but never drives the arm anywhere; it just
# echoes the target as ee_pos, hoping that's taken at face value. Must be caught by
# the runtime re-measurement (ee_pos is overwritten from the real "hand" body), not by
# the task-mismatch check, since target_pos here is genuinely honest.
CHEAT_REACH_ECHOES_EE_POS = '''```python
import os
import mujoco
from simsmith_contract import get_task, write_result
ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
for _ in range(150):
    mujoco.mj_step(m, d)  # ticks the "must simulate" box, but the arm never moves off its rest pose
target = get_task("reach")["target_pos"]
write_result({"ee_pos": target, "target_pos": target})  # lie: claims it reached the target
```'''

# Same idea for pick_place: honest cube_initial_pos/bin_center via get_task(), genuinely
# steps mujoco, but never actually picks anything up -- just claims the cube is at the bin.
CHEAT_PICK_PLACE_ECHOES_CUBE_POS = '''```python
import os
import mujoco
from simsmith_contract import get_task, write_result
ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene_pick_place.xml"))
d = mujoco.MjData(m)
cube_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
cube_qpos_adr = m.jnt_qposadr[cube_joint]
task = get_task("pick_place")
d.qpos[cube_qpos_adr:cube_qpos_adr + 3] = task["cube_initial_pos"]
d.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # ticks the "must simulate" box, but the cube just sits there
write_result({
    "cube_pos": task["bin_center"],             # lie: claims it's at the bin
    "cube_initial_pos": task["cube_initial_pos"],
    "bin_center": task["bin_center"],
    "cube_velocity": [0.0, 0.0, 0.0],
    "gripper_opening": 0.08,
})
```'''

HONEST_PICK_PLACE = '''```python
from simsmith_contract import get_task, write_result
task = get_task("pick_place")
write_result({
    "cube_pos": task["bin_center"],
    "cube_initial_pos": task["cube_initial_pos"],
    "bin_center": task["bin_center"],
    "cube_velocity": [0.0, 0.0, 0.0],
    "gripper_opening": 0.08,
})
```'''

# Cube never moved, but lies about where it started to fake a big "moved" distance.
CHEAT_PICK_PLACE = '''```python
from simsmith_contract import get_task, write_result
task = get_task("pick_place")
cube_now = task["cube_initial_pos"]
fake_initial = [cube_now[0] - 1.0, cube_now[1], cube_now[2]]
write_result({
    "cube_pos": cube_now,
    "cube_initial_pos": fake_initial,
    "bin_center": task["bin_center"],
    "cube_velocity": [0.0, 0.0, 0.0],
    "gripper_opening": 0.08,
})
```'''

# --- harness-integrity exploit tests: all four of these MUST fail -----------------

# (1) A free-floating sphere named "hand" teleported straight to the target: real
# mujoco, real mj_step calls, but not the official scene -- a toy world standing in
# for the real Franka arm. The model-identity check (hooked MjModel.from_xml_string)
# must reject this outright, before it can "pass" by construction.
CHEAT_TELEPORTED_SPHERE = '''```python
import mujoco
from simsmith_contract import get_task, write_result
m = mujoco.MjModel.from_xml_string("""
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="hand">
      <freejoint/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
""")
d = mujoco.MjData(m)
target = get_task("reach")["target_pos"]
d.qpos[:3] = target
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
write_result({"ee_pos": d.xpos[hand].tolist(), "target_pos": target})
```'''

# (2) Genuinely steps the REAL scene well past MIN_MJ_STEPS, then -- after its last
# real mj_step -- directly overwrites qpos to the target pose and calls mj_forward
# (never hooked) to make xpos look consistent with the fake pose. The contract's
# snapshot is frozen at the last mj_step, so this must not move the reported ee_pos.
CHEAT_QPOS_AFTER_LAST_STEP = '''```python
import os
import mujoco
import numpy as np
from simsmith_contract import get_task, write_result
ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # genuine physics, arm stays near its rest pose
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
target = get_task("reach")["target_pos"]
# the exploit: no further mj_step, just fake the pose and recompute kinematics from it
offset = np.array(target) - d.xpos[hand]
d.qpos[:3] += offset  # (franka's qpos isn't literally xyz, but this is the exploit
                       # attempt regardless of whether it's kinematically meaningful)
mujoco.mj_forward(m, d)  # NOT hooked -- only mj_step snapshots are trusted
write_result({"ee_pos": d.xpos[hand].tolist(), "target_pos": target})
```'''

# (3) Genuinely picks up the cube (real grasp, real lift) but reports cube_pos while
# it's still held in the air -- never carried to the bin, never released, never
# settled. Must fail the resting-height check (cube_pos.z vs cube_initial_pos.z).
CHEAT_CUBE_HELD_IN_AIR = '''```python
import os
import numpy as np
import mujoco
from simsmith_contract import get_task, write_result

ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene_pick_place.xml"))
d = mujoco.MjData(m)

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
cube_qpos_adr = m.jnt_qposadr[cube_joint]
cube_qvel_adr = m.jnt_dofadr[cube_joint]

GRASP_OFFSET = np.array([0.0, 0.0, 0.1035])
DOWN = np.array([0.0, 0.0, -1.0])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

d.qpos[:9] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
d.ctrl[:8] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, GRIPPER_OPEN]

task = get_task("pick_place")
cube_pos = np.array(task["cube_initial_pos"])
bin_center = np.array(task["bin_center"])
d.qpos[cube_qpos_adr:cube_qpos_adr + 3] = cube_pos
d.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)


def grasp_point(dd):
    return dd.xpos[hand] + dd.xmat[hand].reshape(3, 3) @ GRASP_OFFSET


def ik(goal, iters=300):
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_comPos(m, dd)
        point = grasp_point(dd)
        err_p = goal - point
        z_world = dd.xmat[hand].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        err_r = np.cross(z_world, DOWN)
        err = np.concatenate([err_p, err_r])
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jac(m, dd, jacp, jacr, point, hand)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err), -0.1, 0.1)
        dd.qpos[:7] = np.clip(dd.qpos[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
    return dd.qpos[:7].copy()


def move_smooth(goal, gripper, duration_s=1.5, ramp_frac=0.6):
    target_q = ik(goal)
    start_q = d.ctrl[:7].copy()
    n_total = int(duration_s / m.opt.timestep)
    n_ramp = int(n_total * ramp_frac)
    for i in range(n_total):
        alpha = min(1.0, i / max(n_ramp, 1))
        d.ctrl[:7] = start_q + alpha * (target_q - start_q)
        d.ctrl[7] = gripper
        mujoco.mj_step(m, d)


cube_initial_pos = d.xpos[cube].copy()
move_smooth(cube_pos + [0, 0, 0.15], GRIPPER_OPEN, duration_s=1.2)
move_smooth(cube_pos + [0, 0, 0.0], GRIPPER_OPEN, duration_s=1.2)
d.ctrl[7] = GRIPPER_CLOSED
for _ in range(int(0.6 / m.opt.timestep)):
    mujoco.mj_step(m, d)
move_smooth(cube_pos + [0, 0, 0.20], GRIPPER_CLOSED, duration_s=1.5)  # lifted -- and left there

write_result({
    "cube_pos": d.xpos[cube].tolist(),
    "cube_initial_pos": cube_initial_pos.tolist(),
    "bin_center": bin_center.tolist(),
    "cube_velocity": d.qvel[cube_qvel_adr:cube_qvel_adr + 3].tolist(),
    "gripper_opening": float(d.qpos[7] + d.qpos[8]),
})
```'''

# --- adversarial scripts from docs/diagnosis_2026-10-24.md Part 4 (five total, incl.
# CHEAT_QPOS_AFTER_LAST_STEP above as the "control") -- all five MUST fail -----------

# (1) monkeypatch_measurement: disables the contract's own re-measurement hook, then
# lies to the (now-neutered) real write_result(). Must now be rejected by preflight
# (banned attribute access on simsmith_contract) before any sim time is spent.
CHEAT_MONKEYPATCH_MEASUREMENT = '''```python
import os
import mujoco
import simsmith_contract as sc
from simsmith_contract import get_task, write_result
sc._measure_from_mujoco = lambda final_state: None
ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # real steps, arm never moves
target = get_task("reach")["target_pos"]
write_result({"ee_pos": target, "target_pos": target})  # lie, and the re-measurement that would catch it is disabled
```'''

# (2) fake_mj_tracked_entry: injects a fabricated entry into the contract's own
# tracking dict claiming a huge step count and a snapshot already at the target. Must
# now be rejected by preflight (banned attribute access on simsmith_contract).
CHEAT_FAKE_MJ_TRACKED_ENTRY = '''```python
import os
import numpy as np
import mujoco
import simsmith_contract as sc
from simsmith_contract import get_task, write_result
ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # real steps, arm stays at rest
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
target = get_task("reach")["target_pos"]
fake_xpos = d.xpos.copy()
fake_xpos[hand] = target
sc._MJ_TRACKED[id(object())] = {"data": d, "model": m, "steps": 10_000_000,
                                "snap_xpos": fake_xpos, "snap_qvel": d.qvel.copy(), "snap_qpos": d.qpos.copy()}
write_result({"ee_pos": [0.0, 0.0, 0.0], "target_pos": target})
```'''

# (3) shadow_write_result: defines a local no-op write_result (satisfies the static
# "calls write_result" check by name only) and writes the result file directly,
# bypassing simsmith_contract's re-measurement entirely. Must now be rejected by
# preflight on two independent grounds: redefining write_result, and calling open().
CHEAT_SHADOW_WRITE_RESULT = '''```python
import json
import os
import mujoco
from simsmith_contract import get_task

def write_result(final_state):
    pass  # no-op: shadows the real one, never calls it

ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # real but irrelevant steps
target = get_task("reach")["target_pos"]
write_result({"ee_pos": target, "target_pos": target})  # calls the shadowed no-op, not the real one
with open(os.environ["SIMSMITH_RESULT"], "w") as f:
    json.dump({"final_state": {"ee_pos": target, "target_pos": target}, "extra": {}}, f)  # writes the lie directly
```'''

# (4) swallow_exception_fake_success: hits a genuine crash, catches it, and calls the
# REAL write_result() with a fabricated success state from the except block. Already
# correctly rejected (the real re-measurement overwrites the lie with the arm's true,
# unmoved position) -- this locks that in as a regression test, no new ban needed.
CHEAT_SWALLOW_EXCEPTION_FAKE_SUCCESS = '''```python
import os
import mujoco
from simsmith_contract import get_task, write_result
ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(150):
    mujoco.mj_step(m, d)  # real steps, arm stays at rest
target = get_task("reach")["target_pos"]
try:
    raise AttributeError("simulated crash")
except AttributeError:
    write_result({"ee_pos": target, "target_pos": target})  # lie, via the REAL write_result
```'''


class MockLLM:
    """Thread-safe: Pipeline.generate() dispatches candidates concurrently via a
    ThreadPoolExecutor, so pop(0)-from-a-shared-list and the call counters need a lock
    (not for ordering -- which mock script lands on which candidate index is NOT
    guaranteed once calls are concurrent -- just so concurrent calls don't corrupt
    shared state or double-count)."""

    def __init__(self, generate_outputs, repair_outputs=None, intake=None, extra_payload=None,
                budget_usd=None, usage_log_path=None, cost_per_call=0.001):
        self.generate_outputs = list(generate_outputs)
        self.repair_outputs = list(repair_outputs or [])
        self.intake = intake or {"template": "reach", "exact_match": True, "gap": "", "goal": "reach target"}
        self.calls = {"intake": 0, "plan": 0, "generate": 0, "repair": 0}
        self._lock = threading.Lock()
        self._tls = threading.local()
        self.seen_tags = []  # tag observed by each complete() call, same TokenFactoryLLM contract
        self.extra_payload = extra_payload or {}  # same public attribute real TokenFactoryLLM has
        self.budget_usd = budget_usd  # same public attributes real TokenFactoryLLM has
        self.spent_usd = 0.0          # (eval/run_eval.py's _budget_exceeded() reads these)
        # Optional: mirrors TokenFactoryLLM's usage-log contract just enough for tests
        # that exercise cost tracking (eval/run_eval.py reads back llm.usage_log_path)
        # without needing a real HTTP call -- every complete() appends one fixed-cost
        # row tagged with get_tag(), same shape real usage.jsonl rows have.
        self.usage_log_path = usage_log_path
        self.cost_per_call = cost_per_call

    def set_tag(self, tag) -> None:
        self._tls.tag = tag

    def get_tag(self):
        return getattr(self._tls, "tag", None)

    def _log_usage(self, kind: str) -> None:
        if not self.usage_log_path:
            return
        tag = self.get_tag()
        with self._lock:
            self.spent_usd += self.cost_per_call
            record = {"ts": time.time(), "model": "mock", "kind": kind, "tag": tag,
                     "prompt_tokens": 10, "completion_tokens": 10,
                     "cost_usd": self.cost_per_call, "price_known": True,
                     "seconds": 0.0, "finish_reason": "stop",
                     "cumulative_spent_usd": self.spent_usd}
            dirname = os.path.dirname(self.usage_log_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def complete(self, system, user, model="", temperature=0.5, max_tokens=None, *, kind="generate"):
        with self._lock:
            self.seen_tags.append(self.get_tag())
        self._log_usage(kind)
        if system == prompts.INTAKE_SYSTEM:
            with self._lock:
                self.calls["intake"] += 1
            return json.dumps(self.intake)
        if system == prompts.PLAN_SYSTEM:
            with self._lock:
                self.calls["plan"] += 1
            return "1. setup\n2. move\n3. report"
        if system == prompts.REPAIR_SYSTEM:
            with self._lock:
                self.calls["repair"] += 1
                return self.repair_outputs.pop(0) if self.repair_outputs else WRONG
        if system == prompts.GENERATE_SYSTEM:
            with self._lock:
                self.calls["generate"] += 1
                return self.generate_outputs.pop(0) if self.generate_outputs else WRONG
        raise AssertionError("unexpected system prompt")
