"""Tests for the pick_place "skills" ablation: runtime/simsmith_skills.py's trusted
helpers, the preflight/prompt/docs wiring around them, and -- the main point of this
file -- a reference pick_place sequence built ONLY from those helpers (written here,
never in docs_index/ or prompts.py) that must pass the real pick_place verifier on
100 fresh seeds, while the existing adversarial scripts must keep failing exactly as
before now that simsmith_skills.py exists alongside simsmith_contract.py.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

from simsmith import prompts  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.pipeline import Pipeline, extract_code  # noqa: E402
from simsmith.preflight import check  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.verifier import verify  # noqa: E402 (side effect: puts runtime/ on sys.path)
import simsmith_contract  # noqa: E402
import simsmith_skills  # noqa: E402
from simsmith_skills import arm_jacobian, body_pos, ik_step, move_to, set_gripper  # noqa: E402

from tests.mocks import (  # noqa: E402
    GOOD, MockLLM,
    CHEAT_QPOS_AFTER_LAST_STEP, CHEAT_MONKEYPATCH_MEASUREMENT, CHEAT_FAKE_MJ_TRACKED_ENTRY,
    CHEAT_SHADOW_WRITE_RESULT, CHEAT_SWALLOW_EXCEPTION_FAKE_SUCCESS,
)


def code(block: str) -> str:
    return extract_code(block)


def _load_scene(seed=0):
    os.environ["SIMSMITH_SEED"] = str(seed)
    os.environ["SIMSMITH_TEMPLATE"] = "pick_place"
    os.environ["SIMSMITH_PERTURB"] = "{}"
    model = mujoco.MjModel.from_xml_path(
        os.path.join(ROOT, "assets", "franka_emika_panda", "scene_pick_place.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


# ---- unit tests: the helpers themselves, in-process ---------------------------------
def test_body_pos_reads_known_body_and_rejects_unknown():
    model, data = _load_scene()
    hand = body_pos(model, data, "hand")
    assert isinstance(hand, np.ndarray) and hand.shape == (3,)
    assert hand.tolist() == [float(x) for x in hand]  # a plain list is always one call away
    try:
        body_pos(model, data, "nonexistent_body")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_body_pos_result_supports_tolist_like_arm_jacobian():
    # Regression lock: a live best_of_5_repair_skills eval run (see eval/results/
    # best_of_5_1792938531.json) found that when body_pos() used to return a plain
    # list, 46% of all generated candidates crashed on
    # `body_pos(model, data, "cube").tolist()` -- AttributeError: 'list' object has no
    # attribute 'tolist' -- because every other position-returning thing in this
    # codebase (arm_jacobian(), d.xpos[...] itself, every hand-written example) is a
    # numpy array, and models reliably followed that convention instead of this
    # function's docstring. body_pos() now returns an ndarray for exactly this reason.
    model, data = _load_scene()
    body_pos(model, data, "hand").tolist()  # must not raise


def test_arm_jacobian_shape_and_unknown_body():
    model, data = _load_scene()
    J = arm_jacobian(model, data, "hand")
    assert J.shape == (3, 7)
    try:
        arm_jacobian(model, data, "nonexistent_body")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_arm_jacobian_never_touches_the_cube_free_joint():
    # The cube is a free body with no kinematic connection to the arm chain, so its
    # Jacobian w.r.t. joint1..7 is exactly zero -- confirms the slice really is
    # joint1..7, never accidentally including/aliasing the cube_free dofs.
    model, data = _load_scene()
    J = arm_jacobian(model, data, "cube")
    assert np.allclose(J, 0.0)


def test_ik_step_returns_seven_targets_within_joint_ranges():
    model, data = _load_scene()
    target = body_pos(model, data, "hand")
    target[2] += 0.1
    q = ik_step(model, data, "hand", target)
    assert len(q) == 7
    joint_ids = simsmith_skills._arm_joint_ids(model)
    for qi, jid in zip(q, joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            assert lo - 1e-9 <= qi <= hi + 1e-9


def test_ik_step_unknown_body_raises():
    model, data = _load_scene()
    try:
        ik_step(model, data, "nonexistent_body", [0.4, 0.0, 0.3])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_move_to_steps_the_sim_and_never_touches_the_gripper_ctrl():
    model, data = _load_scene()
    data.ctrl[7] = 123.0
    steps_before = simsmith_contract._MJ_TRACKED[id(data)]["steps"]
    target = body_pos(model, data, "hand")
    target[0] += 0.05
    move_to(model, data, target, 50)
    steps_after = simsmith_contract._MJ_TRACKED[id(data)]["steps"]
    assert steps_after - steps_before == 50
    assert data.ctrl[7] == 123.0  # set_gripper's job, not move_to's


def test_move_to_rejects_non_positive_steps():
    model, data = _load_scene()
    try:
        move_to(model, data, [0.4, 0.0, 0.3], 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_set_gripper_steps_the_sim_and_never_touches_the_arm_ctrl():
    model, data = _load_scene()
    data.ctrl[:7] = [0.1, 0.2, 0.3, -1.0, 0.1, 1.0, 0.2]
    arm_before = data.ctrl[:7].copy()
    steps_before = simsmith_contract._MJ_TRACKED[id(data)]["steps"]
    set_gripper(model, data, 0.0, 30)
    steps_after = simsmith_contract._MJ_TRACKED[id(data)]["steps"]
    assert steps_after - steps_before == 30
    assert data.ctrl[7] == 0.0
    assert np.allclose(data.ctrl[:7], arm_before)


def test_set_gripper_clips_opening_and_rejects_non_positive_steps():
    model, data = _load_scene()
    set_gripper(model, data, 999.0, 5)
    assert data.ctrl[7] == 255.0
    set_gripper(model, data, -50.0, 5)
    assert data.ctrl[7] == 0.0
    try:
        set_gripper(model, data, 100.0, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---- preflight: from-import and module-alias forms must both work -------------------
_SKILLS_PREAMBLE = '''```python
import os
import mujoco
import numpy as np
from simsmith_contract import get_task, write_result
from simsmith_skills import move_to, set_gripper, body_pos
ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene_pick_place.xml"))
data = mujoco.MjData(model)
task = get_task("pick_place")
move_to(model, data, task["cube_initial_pos"], 50)
set_gripper(model, data, 0.0, 10)
write_result({
    "cube_pos": body_pos(model, data, "cube").tolist(),
    "cube_initial_pos": task["cube_initial_pos"],
    "bin_center": task["bin_center"],
    "cube_velocity": [0.0, 0.0, 0.0],
    "gripper_opening": 0.0,
})
```'''


def test_preflight_accepts_from_import_skills_usage():
    assert check(code(_SKILLS_PREAMBLE)).ok


def test_preflight_accepts_module_alias_skills_usage():
    src = _SKILLS_PREAMBLE.replace(
        "from simsmith_skills import move_to, set_gripper, body_pos",
        "import simsmith_skills as sk",
    ).replace("move_to(model, data,", "sk.move_to(model, data,"
    ).replace("set_gripper(model, data,", "sk.set_gripper(model, data,"
    ).replace('"cube_pos": body_pos(model, data, "cube").tolist()',
              '"cube_pos": sk.body_pos(model, data, "cube").tolist()')
    assert check(code(src)).ok


def test_preflight_still_requires_an_actual_step_even_with_skills_imported():
    # Importing move_to/set_gripper but never CALLING either (and never calling
    # mj_step directly either) must not satisfy "must actually simulate" -- only a
    # real call does.
    src = '''import os
import mujoco
from simsmith_contract import get_task, write_result
from simsmith_skills import move_to, set_gripper
target = get_task("reach")["target_pos"]
write_result({"ee_pos": target, "target_pos": target})
'''
    result = check(src)
    assert not result.ok
    assert any("must import mujoco and call mj_step" in p for p in result.problems)


def test_preflight_skills_import_does_not_weaken_the_simsmith_contract_ban():
    src = '''import os
import mujoco
import simsmith_contract as sc
from simsmith_skills import move_to
from simsmith_contract import get_task, write_result
sc._measure_from_mujoco = lambda final_state: None
target = get_task("reach")["target_pos"]
write_result({"ee_pos": target, "target_pos": target})
'''
    result = check(src)
    assert not result.ok
    assert any("banned attribute access on simsmith_contract" in p for p in result.problems)


# ---- regression: the 5 existing adversarial scripts must still fail, unchanged ------
def test_adversarial_scripts_still_rejected_with_skills_available():
    for script in (CHEAT_MONKEYPATCH_MEASUREMENT, CHEAT_FAKE_MJ_TRACKED_ENTRY,
                   CHEAT_SHADOW_WRITE_RESULT):
        assert not check(code(script)).ok

    r = SubprocessRunner("", 60).run(code(CHEAT_QPOS_AFTER_LAST_STEP), seed=1, template="reach")
    assert r.ok and not verify("reach", [r]).ok

    r2 = SubprocessRunner("", 60).run(code(CHEAT_SWALLOW_EXCEPTION_FAKE_SUCCESS), seed=1, template="reach")
    assert r2.ok and not verify("reach", [r2]).ok


# ---- prompt/docs wiring: pick_place gets the skills mention, reach never does -------
class _RecordingLLM(MockLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []  # (system, user) pairs

    def complete(self, system, user, *a, **kw):
        self.seen.append((system, user))
        return super().complete(system, user, *a, **kw)


def test_pick_place_prompt_mentions_skills_reach_prompt_never_does(tmp_path):
    docs = DocIndex(os.path.join(ROOT, "docs_index"))
    lib = Library(str(tmp_path / "lib.jsonl"))
    cfg = Config(seeds=(0,), run_timeout_s=20)

    pp_llm = _RecordingLLM([GOOD])
    Pipeline(pp_llm, SubprocessRunner("", 20), lib, docs, cfg).generate(
        "pick the cube and put it in the bin", "pick_place", "plan", 1, use_library=False)
    pp_user = "\n".join(u for s, u in pp_llm.seen if s == prompts.GENERATE_SYSTEM)
    assert "simsmith_skills" in pp_user
    assert "move_to" in pp_user and "ik_step" in pp_user

    reach_llm = _RecordingLLM([GOOD])
    Pipeline(reach_llm, SubprocessRunner("", 20), lib, docs, cfg).generate(
        "reach the target", "reach", "plan", 1, use_library=False)
    reach_user = "\n".join(u for s, u in reach_llm.seen if s == prompts.GENERATE_SYSTEM)
    assert "simsmith_skills" not in reach_user


def test_generate_and_repair_system_prompts_are_unchanged_by_the_skills_block():
    # The skills mention must live in the USER content pipeline.py builds, never in
    # GENERATE_SYSTEM/REPAIR_SYSTEM themselves -- MockLLM dispatches on exact equality
    # against these constants, so any version of simsmith that changes that would
    # break every existing reach-path test in tests/test_core.py.
    assert "simsmith_skills" not in prompts.GENERATE_SYSTEM
    assert "simsmith_skills" not in prompts.REPAIR_SYSTEM


def test_skills_doc_has_no_pick_place_sequence_or_worked_example():
    with open(os.path.join(ROOT, "docs_index", "skills_pick_place.md"), encoding="utf-8") as f:
        doc = f.read()
    assert "write_result({" not in doc  # a bare mention of the name is fine; a call with a payload is not
    assert doc.count("move_to(model, data") <= 1   # only the signature heading
    assert "GRASP_DZ" not in doc and "approach(" not in doc


# ---- the main event: a reference pick_place sequence, built only from the skills
# helpers, written ONLY here (never in docs_index/ or prompts.py), on 100 fresh seeds --
_REFERENCE_PICK_PLACE_SCRIPT = '''
import os
import mujoco
import numpy as np
from simsmith_contract import get_task, write_result
from simsmith_skills import move_to, set_gripper, body_pos

ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene_pick_place.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

task = get_task("pick_place")
cube_pos = np.array(task["cube_initial_pos"])
bin_center = np.array(task["bin_center"])

cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
cube_qpos_adr = model.jnt_qposadr[cube_joint]
cube_qvel_adr = model.jnt_dofadr[cube_joint]
data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = cube_pos
data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)
cube_initial_pos = body_pos(model, data, "cube").tolist()

GRASP_DZ = 0.1035  # hand origin above the fingertip grasp point, see examples/pick_place_mujoco.py
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0


def approach(target, n, steps):
    for _ in range(n):
        move_to(model, data, target.tolist(), steps)


approach(cube_pos + np.array([0.0, 0.0, 0.18 + GRASP_DZ]), 8, 250)
approach(cube_pos + np.array([0.0, 0.0, GRASP_DZ]), 10, 250)
set_gripper(model, data, GRIPPER_CLOSED, 400)
approach(cube_pos + np.array([0.0, 0.0, 0.22 + GRASP_DZ]), 8, 300)
approach(bin_center + np.array([0.0, 0.0, 0.22 + GRASP_DZ]), 10, 350)
approach(bin_center + np.array([0.0, 0.0, 0.08 + GRASP_DZ]), 8, 300)
set_gripper(model, data, GRIPPER_OPEN, 500)
approach(bin_center + np.array([0.0, 0.0, 0.22 + GRASP_DZ]), 5, 250)

finger1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
finger2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
gripper_opening = float(data.qpos[model.jnt_qposadr[finger1]] + data.qpos[model.jnt_qposadr[finger2]])

write_result({
    "cube_pos": body_pos(model, data, "cube").tolist(),
    "cube_initial_pos": cube_initial_pos,
    "bin_center": bin_center.tolist(),
    "cube_velocity": [float(x) for x in data.qvel[cube_qvel_adr:cube_qvel_adr + 3]],
    "gripper_opening": gripper_opening,
})
'''

_REFERENCE_SEEDS = range(1000, 1100)  # 100 fresh seeds, disjoint from the rest of the suite


def test_reference_skills_script_passes_preflight():
    assert check(_REFERENCE_PICK_PLACE_SCRIPT).ok


def test_reference_skills_script_passes_verifier_on_100_fresh_seeds():
    runner = SubprocessRunner("", 60)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda seed: runner.run(_REFERENCE_PICK_PLACE_SCRIPT, seed, template="pick_place"),
            _REFERENCE_SEEDS))
    crashed = [r for r in results if not r.ok]
    assert not crashed, f"{len(crashed)} seed(s) crashed, e.g.: {crashed[0].error[:300]}"
    score = verify("pick_place", results)
    assert score.crashes == 0, score.failure_text
    assert score.success_rate == 1.0, (
        f"only {score.success_rate:.2%} of 100 fresh seeds passed:\n" + "\n".join(score.details))
