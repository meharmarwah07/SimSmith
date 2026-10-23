"""Test doubles. MockLLM routes by system prompt and replays scripted outputs."""
from __future__ import annotations

import json
import threading

from simsmith import prompts

# A tiny, self-contained scene (no Franka assets needed): one free body named "hand",
# gravity off so whatever we set its pose to is exactly where it stays once stepped.
# Real mujoco.MjData + real mj_step calls, so these pass the "must actually simulate"
# pre-flight rule and the contract's own ee_pos re-measurement -- same bar a real
# generated script has to clear.
_MJCF_HAND = '''"""
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="hand">
      <freejoint/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""'''

# GOOD/WRONG read target_pos from get_task(), same as a real script must: the verifier
# recomputes it independently from the seed and rejects a report that doesn't match.
GOOD = f'''```python
import mujoco
from simsmith_contract import get_task, write_result
m = mujoco.MjModel.from_xml_string({_MJCF_HAND})
d = mujoco.MjData(m)
target = get_task("reach")["target_pos"]
d.qpos[:3] = target
mujoco.mj_forward(m, d)
for _ in range(20):
    mujoco.mj_step(m, d)
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
write_result({{"ee_pos": d.xpos[hand].tolist(), "target_pos": target}})
```'''

WRONG = f'''```python
import mujoco
from simsmith_contract import get_task, write_result
m = mujoco.MjModel.from_xml_string({_MJCF_HAND})
d = mujoco.MjData(m)
target = get_task("reach")["target_pos"]
d.qpos[:3] = [target[0] + 0.5, target[1], target[2]]  # deliberately placed 0.5m off
mujoco.mj_forward(m, d)
for _ in range(20):
    mujoco.mj_step(m, d)
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
write_result({{"ee_pos": d.xpos[hand].tolist(), "target_pos": target}})
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

SELF_GRADE = f'''```python
import mujoco
from simsmith_contract import write_result
m = mujoco.MjModel.from_xml_string({_MJCF_HAND})
d = mujoco.MjData(m)
for _ in range(20):
    mujoco.mj_step(m, d)
write_result({{"success": True}})
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
for _ in range(20):
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
for _ in range(20):
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


class MockLLM:
    """Thread-safe: Pipeline.generate() dispatches candidates concurrently via a
    ThreadPoolExecutor, so pop(0)-from-a-shared-list and the call counters need a lock
    (not for ordering -- which mock script lands on which candidate index is NOT
    guaranteed once calls are concurrent -- just so concurrent calls don't corrupt
    shared state or double-count)."""

    def __init__(self, generate_outputs, repair_outputs=None, intake=None):
        self.generate_outputs = list(generate_outputs)
        self.repair_outputs = list(repair_outputs or [])
        self.intake = intake or {"template": "reach", "exact_match": True, "gap": "", "goal": "reach target"}
        self.calls = {"intake": 0, "plan": 0, "generate": 0, "repair": 0}
        self._lock = threading.Lock()

    def complete(self, system, user, model="", temperature=0.5, max_tokens=None, *, kind="generate"):
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
