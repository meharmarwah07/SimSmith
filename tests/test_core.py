import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.preflight import check  # noqa: E402
from simsmith.runner import RunResult, SubprocessRunner  # noqa: E402
from simsmith.templates import TEMPLATES, closest_template  # noqa: E402
from simsmith.verifier import verify  # noqa: E402 (side effect: puts runtime/ on sys.path)
import simsmith_contract  # noqa: E402
from simsmith.pipeline import Pipeline, extract_code  # noqa: E402
from tests.mocks import (  # noqa: E402
    BANNED, CRASH, GOOD, SELF_GRADE, WRONG, MockLLM,
    CHEAT_REACH, CHEAT_PICK_PLACE, HONEST_PICK_PLACE,
    CHEAT_REACH_ECHOES_EE_POS, CHEAT_PICK_PLACE_ECHOES_CUBE_POS,
)


def code(block: str) -> str:
    return extract_code(block)


# ---- templates / verifier -------------------------------------------------
def test_reach_check_boundaries():
    ok, _ = TEMPLATES["reach"].check({"ee_pos": [0, 0, 0], "target_pos": [0.019, 0, 0]})
    bad, _ = TEMPLATES["reach"].check({"ee_pos": [0, 0, 0], "target_pos": [0.021, 0, 0]})
    assert ok and not bad


def _pick_place_state(**overrides):
    s = {
        "cube_pos": [0.5, 0, 0.02],
        "cube_initial_pos": [0.2, 0.2, 0.02],
        "bin_center": [0.5, 0, 0.02],
        "cube_velocity": [0.0, 0.0, 0.0],
        "gripper_opening": 0.08,
    }
    s.update(overrides)
    return s


def test_pick_place_requires_cube_to_have_moved():
    s = {"cube_pos": [0.5, 0, 0.1], "cube_initial_pos": [0.5, 0, 0.1], "bin_center": [0.5, 0, 0.1],
         "cube_velocity": [0, 0, 0], "gripper_opening": 0.08}
    ok, detail = TEMPLATES["pick_place"].check(s)
    assert not ok and "moved" in detail   # starting inside the bin must not count
    s["cube_initial_pos"] = [0.2, 0.2, 0.05]
    s["cube_pos"] = [0.5, 0, 0.05]
    assert TEMPLATES["pick_place"].check(s)[0]


def test_pick_place_passes_when_settled_and_released():
    assert TEMPLATES["pick_place"].check(_pick_place_state())[0]


def test_pick_place_rejects_cube_held_in_the_air():
    # Position and xy-to-bin look right, but the cube is still up at drop height, not
    # resting on the floor: a script could report this mid-carry and must not pass.
    s = _pick_place_state(cube_pos=[0.5, 0, 0.22], gripper_opening=0.0)
    ok, detail = TEMPLATES["pick_place"].check(s)
    assert not ok and "resting height" in detail


def test_pick_place_rejects_moving_cube():
    s = _pick_place_state(cube_velocity=[0.2, 0.0, 0.0])
    ok, detail = TEMPLATES["pick_place"].check(s)
    assert not ok and "speed" in detail


def test_pick_place_rejects_closed_gripper_even_if_cube_settled():
    # Cube at rest at the bin with zero velocity, but the gripper never let go.
    s = _pick_place_state(gripper_opening=0.01)
    ok, detail = TEMPLATES["pick_place"].check(s)
    assert not ok and "gripper_opening" in detail


def test_pick_place_requires_new_fields():
    s = _pick_place_state()
    del s["cube_velocity"]
    assert not TEMPLATES["pick_place"].check(s)[0]
    s = _pick_place_state()
    del s["gripper_opening"]
    assert not TEMPLATES["pick_place"].check(s)[0]


def test_verifier_rejects_reach_script_that_self_reports_target():
    # Never calls get_task(); reports target_pos = ee_pos, i.e. "whatever I ended up
    # at was the target." The verifier must catch this against its own recomputation.
    r = SubprocessRunner("", 20).run(code(CHEAT_REACH), seed=5)
    assert r.ok  # the script itself runs fine and writes a result
    s = verify("reach", [r])
    assert not s.ok and s.crashes == 1
    assert "differs from task target" in s.failure_text


def test_verifier_accepts_honest_reach_script():
    r = SubprocessRunner("", 20).run(code(GOOD), seed=5)
    s = verify("reach", [r])
    assert s.ok and s.success_rate == 1.0


def test_verifier_rejects_pick_place_script_that_fakes_cube_initial_pos():
    r = SubprocessRunner("", 20).run(code(CHEAT_PICK_PLACE), seed=7)
    assert r.ok
    s = verify("pick_place", [r])
    assert not s.ok and s.crashes == 1
    assert "differs from task cube_initial" in s.failure_text


def test_verifier_accepts_honest_pick_place_script():
    r = SubprocessRunner("", 20).run(code(HONEST_PICK_PLACE), seed=0)
    s = verify("pick_place", [r])
    assert s.ok and s.success_rate == 1.0


# ---- the mujoco echo exploit: measured outcome, never the script's own report --------
def test_mujoco_echo_of_ee_pos_as_target_is_caught_and_fails():
    # Passes pre-flight (real mujoco import + mj_step) and honestly reports target_pos
    # via get_task() -- the lie is purely "ee_pos = target_pos" with no real motion.
    # The contract's own re-measurement of "hand" must override that and fail it.
    r = SubprocessRunner("", 60).run(code(CHEAT_REACH_ECHOES_EE_POS), seed=3)
    assert r.ok
    assert r.final_state.get("_sim_steps", 0) >= 20          # contract measured real steps
    assert r.final_state["ee_pos"] != r.final_state["target_pos"]  # the lie was overwritten
    s = verify("reach", [r])
    assert not s.ok
    assert "task not achieved" in s.failure_text or "ee-target distance" in "\n".join(s.details)


def test_mujoco_echo_of_cube_pos_as_bin_is_caught_and_fails():
    r = SubprocessRunner("", 60).run(code(CHEAT_PICK_PLACE_ECHOES_CUBE_POS), seed=4)
    assert r.ok
    assert r.final_state.get("_sim_steps", 0) >= 20
    assert r.final_state["cube_pos"] != r.final_state["bin_center"]  # the lie was overwritten
    s = verify("pick_place", [r])
    assert not s.ok


def test_script_that_never_calls_mj_step_enough_fails_at_runtime_too():
    # Belt-and-suspenders: even if a script somehow got past pre-flight with a trivial
    # mj_step call, the contract's own step count must still gate it.
    script = '''import mujoco
from simsmith_contract import get_task, write_result
m = mujoco.MjModel.from_xml_string("""
<mujoco><option gravity="0 0 0"/><worldbody>
<body name="hand"><freejoint/><geom type="sphere" size="0.01"/></body>
</worldbody></mujoco>
""")
d = mujoco.MjData(m)
mujoco.mj_step(m, d)  # exactly 1 step: far below MIN_MJ_STEPS
target = get_task("reach")["target_pos"]
write_result({"ee_pos": target, "target_pos": target})
'''
    r = SubprocessRunner("", 20).run(script, seed=1)
    assert r.ok
    assert r.final_state.get("_sim_steps") == 1
    s = verify("reach", [r])
    assert not s.ok and s.crashes == 1
    assert "need >=" in s.failure_text


def test_nan_and_garbage_rejected():
    assert not TEMPLATES["reach"].check({"ee_pos": [float("nan"), 0, 0], "target_pos": [0, 0, 0]})[0]
    assert not TEMPLATES["pick_place"].check(_pick_place_state(cube_pos="x"))[0]


def test_verify_counts_crashes_and_failures():
    t0 = simsmith_contract.task_for("reach", 0, {})["target_pos"]
    t1 = simsmith_contract.task_for("reach", 1, {})["target_pos"]
    good = RunResult(True, {"ee_pos": t0, "target_pos": t0}, seed=0)
    far = RunResult(True, {"ee_pos": [t1[0] + 1, t1[1], t1[2]], "target_pos": t1}, seed=1)
    crash = RunResult(False, error="boom", seed=2)
    s = verify("reach", [good, good, good])
    assert s.ok and s.success_rate == 1.0
    s = verify("reach", [good, far, good])
    assert not s.ok and abs(s.success_rate - 2 / 3) < 1e-9
    s = verify("reach", [good, good, crash])
    assert not s.ok and s.crashes == 1 and "crashed" in s.failure_text


def test_verify_rejects_self_reported_success_flag():
    r = RunResult(True, {"success": True}, seed=0)
    s = verify("reach", [r])
    assert not s.ok and s.crashes == 1


def test_closest_template_keywords():
    assert closest_template("put the cube in the bin") == "pick_place"
    assert closest_template("drive to the goal") == "reach"   # navigate was removed; falls back to reach
    assert closest_template("something odd") == "reach"


# ---- preflight --------------------------------------------------------------
def test_preflight_accepts_good_script():
    assert check(code(GOOD)).ok


def test_preflight_rejections():
    assert any("banned import" in p for p in check(code(BANNED)).problems)
    assert any("write_result" in p for p in check("x = 1\n").problems)
    assert any("syntax" in p for p in check("def (:\n").problems)
    assert any("os.system" in p for p in check("import os\nos.system('ls')\nfrom simsmith_contract import write_result\nwrite_result({})\n").problems)
    assert any("eval" in p for p in check("eval('1')\nwrite_result({})\n").problems)


def test_preflight_allowlist_catches_unknown_isaac_module():
    src = ("from isaacsim.fake.module import thing\nimport mujoco\n"
          "from simsmith_contract import write_result\nmujoco.mj_step(1, 2)\nwrite_result({})\n")
    assert check(src, {"isaacsim.core"}).ok is False
    assert check(src, set()).ok is True        # empty allow-list = check disabled


def test_preflight_requires_mujoco_simulation():
    # Calls write_result honestly, but never touches mujoco at all -- the simplest cheat.
    src = "from simsmith_contract import write_result\nwrite_result({'ee_pos': [0, 0, 0]})\n"
    result = check(src)
    assert not result.ok
    assert any("must import mujoco and call mj_step" in p for p in result.problems)


def test_preflight_requires_mj_step_not_just_the_import():
    # Imports mujoco (e.g. to build a model) but never advances it in time.
    src = ("import mujoco\nfrom simsmith_contract import write_result\n"
          "m = mujoco.MjModel.from_xml_string('<mujoco/>')\nwrite_result({'ee_pos': [0, 0, 0]})\n")
    result = check(src)
    assert not result.ok
    assert any("must import mujoco and call mj_step" in p for p in result.problems)


def test_preflight_accepts_bare_mj_step_import():
    # `from mujoco import mj_step` + a bare call must count too, not just `mujoco.mj_step(...)`.
    src = ("import mujoco\nfrom mujoco import mj_step\nfrom simsmith_contract import write_result\n"
          "mj_step(1, 2)\nwrite_result({'ee_pos': [0, 0, 0]})\n")
    assert check(src).ok


# ---- preflight: mujoco attribute resolution against the real installed module -------
def test_preflight_rejects_invented_mujoco_function():
    src = ("import mujoco\nfrom simsmith_contract import write_result\n"
          "mujoco.mj_setp(1, 2)\n"          # typo'd/invented, not a real mujoco function
          "mujoco.mj_step(1, 2)\nwrite_result({'ee_pos': [0, 0, 0]})\n")
    result = check(src)
    assert not result.ok
    problem = next(p for p in result.problems if "mj_setp" in p)
    assert "does not exist" in problem
    assert "mj_step" in problem   # difflib should surface the real function as a close match


def test_preflight_rejects_invented_chained_attribute():
    # mujoco.mjtObj is real; mjOBJ_NOT_A_THING is not a member of it.
    src = ("import mujoco\nfrom simsmith_contract import write_result\n"
          "x = mujoco.mjtObj.mjOBJ_NOT_A_THING\nmujoco.mj_step(1, 2)\n"
          "write_result({'ee_pos': [0, 0, 0]})\n")
    result = check(src)
    assert not result.ok
    problem = next(p for p in result.problems if "mjOBJ_NOT_A_THING" in p)
    assert "mujoco.mjtObj" in problem


def test_preflight_accepts_real_mujoco_attrs_including_chained():
    src = ("import mujoco\nfrom simsmith_contract import write_result\n"
          "x = mujoco.mjtObj.mjOBJ_BODY\n"
          "m = mujoco.MjModel.from_xml_string('<mujoco/>')\n"
          "mujoco.mj_step(1, 2)\nwrite_result({'ee_pos': [0, 0, 0]})\n")
    assert check(src).ok


def test_preflight_mujoco_attr_check_respects_import_alias():
    src = ("import mujoco as mj\nfrom simsmith_contract import write_result\n"
          "mj.mj_fake_hallucinated_call(1)\nmj.mj_step(1, 2)\n"
          "write_result({'ee_pos': [0, 0, 0]})\n")
    result = check(src)
    assert not result.ok
    assert any("mj_fake_hallucinated_call" in p for p in result.problems)


# ---- real examples still pass preflight + the full runtime/verifier pipeline --------
def test_real_reach_mujoco_example_passes_preflight_and_verify():
    with open(os.path.join(ROOT, "examples", "reach_mujoco.py"), encoding="utf-8") as f:
        script = f.read()
    assert check(script).ok
    r = SubprocessRunner("", 60).run(script, seed=2)
    assert r.ok
    assert r.final_state.get("_sim_steps", 0) > 0
    assert verify("reach", [r]).ok


def test_real_pick_place_mujoco_example_passes_preflight_and_verify():
    with open(os.path.join(ROOT, "examples", "pick_place_mujoco.py"), encoding="utf-8") as f:
        script = f.read()
    assert check(script).ok
    r = SubprocessRunner("", 60).run(script, seed=2)
    assert r.ok
    assert r.final_state.get("_sim_steps", 0) > 0
    assert verify("pick_place", [r]).ok


# ---- runner ---------------------------------------------------------------
def test_runner_executes_script_and_reads_result():
    r = SubprocessRunner("", 20).run(code(GOOD), seed=3)
    assert r.ok and "ee_pos" in r.final_state


def test_runner_seed_changes_scene():
    a = SubprocessRunner("", 20).run(code(GOOD), seed=1).final_state["target_pos"]
    b = SubprocessRunner("", 20).run(code(GOOD), seed=2).final_state["target_pos"]
    assert a != b


def test_runner_reports_crash_and_timeout():
    r = SubprocessRunner("", 20).run(code(CRASH), seed=0)
    assert not r.ok and "hallucinated_fn" in r.error
    r = SubprocessRunner("", 1).run("import time\ntime.sleep(5)\n", seed=0)
    assert not r.ok and "timeout" in r.error


def test_runner_no_result_file():
    r = SubprocessRunner("", 20).run("x = 1\n", seed=0)
    assert not r.ok and "no result file" in r.error


# ---- library / docs -------------------------------------------------------
def test_library_roundtrip_and_template_filter(tmp_path):
    lib = Library(str(tmp_path / "lib.jsonl"))
    lib.add("reach the red target", "reach", "print(1)", 1.0)
    lib.add("put cube in bin", "pick_place", "print(2)", 1.0)
    assert len(Library(str(tmp_path / "lib.jsonl"))) == 2
    got = lib.retrieve("reach target", "reach")
    assert len(got) == 1 and got[0]["template"] == "reach"


def test_docindex_retrieval(tmp_path):
    (tmp_path / "a.md").write_text("articulation controller joint positions franka")
    (tmp_path / "b.md").write_text("camera annotator replicator rendering")
    idx = DocIndex(str(tmp_path))
    top = idx.retrieve("franka joint controller", k=1)
    assert top and "a.md" in top[0]


# ---- pipeline -------------------------------------------------------------
def make_pipe(tmp_path, llm, seeds=(0, 1, 2)):
    cfg = Config(seeds=seeds, run_timeout_s=20)
    return Pipeline(llm, SubprocessRunner("", 20), Library(str(tmp_path / "lib.jsonl")),
                    DocIndex(str(tmp_path / "none")), cfg)


def test_pipeline_selects_by_execution_not_by_order(tmp_path):
    # generate() dispatches candidates concurrently (ThreadPoolExecutor), so which mock
    # script lands on which candidate index is not guaranteed -- assert by outcome, not
    # position, which is also a more faithful test of "selects by execution not by order."
    llm = MockLLM([WRONG, BANNED, CRASH, SELF_GRADE, GOOD])
    pipe = make_pipe(tmp_path, llm)
    res = pipe.solve("reach the target", n_candidates=5, do_repair=False)
    assert res.status == "VERIFIED"
    assert res.winner.score.ok   # GOOD wins regardless of which slot it landed in
    # BANNED (subprocess import) and CRASH (never calls write_result) are rejected before any run
    assert sum(1 for c in res.candidates if not c.preflight_ok) == 2
    # WRONG (bad pose) did execute and failed verification without missing-key crashes
    assert sum(1 for c in res.candidates if c.score and not c.score.ok and c.score.crashes == 0) == 1
    # SELF_GRADE (success flag, no poses) executed but is missing the required keys every seed
    assert sum(1 for c in res.candidates if c.score and c.score.crashes == 3) == 1
    assert len(pipe.library) == 1


def test_pipeline_repair_rescues_failing_candidates(tmp_path):
    llm = MockLLM([WRONG, WRONG], repair_outputs=[GOOD, GOOD])
    pipe = make_pipe(tmp_path, llm)
    res = pipe.solve("reach the target", n_candidates=2, do_repair=True)
    assert res.status == "VERIFIED" and res.winner.stage == "repaired"
    assert llm.calls["repair"] == 2


def test_pipeline_repair_is_capped(tmp_path):
    llm = MockLLM([WRONG] * 5, repair_outputs=[WRONG] * 5)
    res = make_pipe(tmp_path, llm).solve("reach", n_candidates=5, do_repair=True, max_repairs=2)
    assert llm.calls["repair"] == 2
    assert res.status in ("FAILED", "PARTIAL")


def test_pipeline_never_fake_success_and_no_library_pollution(tmp_path):
    llm = MockLLM([WRONG] * 3)
    pipe = make_pipe(tmp_path, llm)
    res = pipe.solve("reach", n_candidates=3, do_repair=False)
    assert res.status != "VERIFIED" and len(pipe.library) == 0


def test_pipeline_save_to_library_flag(tmp_path):
    pipe = make_pipe(tmp_path, MockLLM([GOOD]))
    pipe.solve("reach", n_candidates=1, do_repair=False, save_to_library=False)
    assert len(pipe.library) == 0


def test_intake_out_of_scope_reports_gap(tmp_path):
    llm = MockLLM([GOOD], intake={"template": "reach", "exact_match": False,
                                  "gap": "no welding robot", "goal": "reach seam"})
    res = make_pipe(tmp_path, llm).solve("design a welding robot", n_candidates=1, do_repair=False)
    assert not res.exact_match and any("Gap" in n for n in res.notes)


def test_intake_falls_back_when_llm_returns_garbage(tmp_path):
    class Garbage(MockLLM):
        def complete(self, system, *a, **k):
            from simsmith import prompts
            if system == prompts.INTAKE_SYSTEM:
                return "not json at all"
            return super().complete(system, *a, **k)
    res = make_pipe(tmp_path, Garbage([GOOD])).solve("put the cube in the bin", n_candidates=1, do_repair=False)
    assert res.template == "pick_place" and not res.exact_match


def test_extract_code_handles_unfenced_and_fenced():
    assert extract_code("```python\nx=1\n```").strip() == "x=1"
    assert extract_code("x=1").strip() == "x=1"
