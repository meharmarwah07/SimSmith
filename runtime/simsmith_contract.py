"""Contract between generated sim scripts and the SimSmith runner/verifier.

Generated scripts import this module. It has no hard dependencies (MuJoCo is
imported lazily and only if present), so it degrades gracefully outside a
MuJoCo environment too.

Rules for generated scripts:
  * Get this run's task parameters (target position, cube start position, bin
    position, ...) from get_task(template). NEVER invent or compute these
    yourself and never report back a value you derived from your own
    simulation (e.g. target_pos = ee_pos) -- the verifier recomputes the same
    parameters independently from SIMSMITH_SEED and rejects a report that
    doesn't match, so self-reporting fake parameters is caught, not trusted.
  * Use get_rng() only for things get_task() doesn't cover.
  * Honour get_perturb() values where relevant (used by robustness testing);
    get_task() already applies them to the parameters it returns.
  * At the end, call write_result(final_state) with RAW poses only. Never
    report a success flag: the verifier computes success itself, so a script
    cannot grade its own work.
  * ee_pos, cube_pos, cube_velocity and gripper_opening are OUTCOME fields,
    not task parameters -- get_task() doesn't cover them because they depend
    on what your script's simulation actually did. Importing this module
    hooks mujoco.MjData/mj_step so write_result() can independently
    re-measure these four fields itself, by body/joint name, from whichever
    MjData instance actually got stepped the most. Whatever you pass for
    them is informational only; the measured value always wins, and it is
    the measured value that fails the preflight simulation check if you
    never really call mj_step.
"""
import json
import os
import random

# Templates that have harness-owned task parameters, and the keys get_task() returns
# for each. This is the single place the sampling formula lives: get_task() (called by
# scripts, via env-derived seed/perturb) and the verifier (recomputing ground truth
# independently from a run's recorded seed/perturb) both call task_for() below, so
# there is no way for the two to drift apart.
TASK_KEYS = {
    "reach": ("target_pos",),
    "pick_place": ("cube_initial_pos", "bin_center"),
}


def get_seed() -> int:
    return int(os.environ.get("SIMSMITH_SEED", "0"))


def get_rng() -> random.Random:
    return random.Random(get_seed())


def get_perturb() -> dict:
    raw = os.environ.get("SIMSMITH_PERTURB", "") or "{}"
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def task_for(template: str, seed: int, perturb: dict | None = None) -> dict:
    """Pure function: the harness-owned task parameters for (template, seed, perturb).

    No env vars, no side effects -- this is what lets the verifier recompute the
    same ground truth a script saw, from a recorded seed and perturb, without
    running in the same process/environment as the script did.
    """
    if template not in TASK_KEYS:
        raise ValueError(f"no task parameters defined for template {template!r}")
    perturb = perturb or {}
    offset = perturb.get("target_offset", [0, 0, 0])
    rng = random.Random(seed)
    if template == "reach":
        target = [rng.uniform(0.3, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.25, 0.55)]
        return {"target_pos": [target[i] + offset[i] for i in range(3)]}
    if template == "pick_place":
        cube = [rng.uniform(0.45, 0.55), rng.uniform(-0.1, 0.1), 0.02]
        cube = [cube[i] + offset[i] for i in range(3)]
        bin_center = [rng.uniform(0.3, 0.4), rng.uniform(0.15, 0.25), 0.02]  # not perturbed
        return {"cube_initial_pos": cube, "bin_center": bin_center}
    raise AssertionError("unreachable")  # TASK_KEYS and the branches above must stay in sync


def get_task(template: str) -> dict:
    """This run's harness-owned task parameters for `template`, deterministic from
    SIMSMITH_SEED (and SIMSMITH_PERTURB). Scripts must read target/cube/bin positions
    from here, not compute or report them independently."""
    return task_for(template, get_seed(), get_perturb())


# --- MuJoCo outcome measurement -------------------------------------------------
# A script cannot be trusted to report where its own simulation ended up (it could
# just echo the target back). So we hook mujoco.MjData/mj_step at import time -- before
# the script creates anything -- track how many times each MjData instance actually got
# stepped, and at write_result() time measure ee_pos/cube_pos/cube_velocity/
# gripper_opening ourselves, by name, from whichever instance was stepped the most
# (a script's throwaway MjData copies used for IK sub-solves never call mj_step, so
# they're never "the busiest" and are naturally ignored).

_MJ_TRACKED: dict = {}   # id(MjData instance) -> {"data": data, "model": model, "steps": int}
_MJ_HOOKED = False


def _hook_mujoco() -> None:
    global _MJ_HOOKED
    if _MJ_HOOKED:
        return
    try:
        import mujoco
    except ImportError:
        return
    _MJ_HOOKED = True

    _orig_mjdata = mujoco.MjData
    _orig_mj_step = mujoco.mj_step

    def _tracked_mjdata(model, *args, **kwargs):
        data = _orig_mjdata(model, *args, **kwargs)
        _MJ_TRACKED[id(data)] = {"data": data, "model": model, "steps": 0}
        return data

    def _tracked_mj_step(model, data, nstep: int = 1):
        entry = _MJ_TRACKED.get(id(data))
        if entry is None:
            entry = _MJ_TRACKED[id(data)] = {"data": data, "model": model, "steps": 0}
        entry["steps"] += nstep
        return _orig_mj_step(model, data, nstep)

    mujoco.MjData = _tracked_mjdata
    mujoco.mj_step = _tracked_mj_step


_hook_mujoco()  # must run before the script's own `import mujoco` for the patch to apply


def _busiest_mj_instance() -> dict | None:
    if not _MJ_TRACKED:
        return None
    return max(_MJ_TRACKED.values(), key=lambda e: e["steps"])


def _measure_from_mujoco(final_state: dict) -> None:
    """Overwrite ee_pos/cube_pos/cube_velocity/gripper_opening in place, from
    the busiest tracked MjData instance -- never from what the script reported. No-ops
    (and adds no "_sim_steps" key) if the script never actually touched mujoco, which
    is fine: that case is caught separately, statically, by the pre-flight check."""
    busiest = _busiest_mj_instance()
    if busiest is None:
        return
    import mujoco  # already imported if we have a tracked instance; safe to re-import

    final_state["_sim_steps"] = busiest["steps"]
    data, model = busiest["data"], busiest["model"]

    def body_pos(name: str):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return [float("nan")] * 3
        return [float(x) for x in data.xpos[bid]]

    def joint_qvel3(name: str):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return [float("nan")] * 3
        adr = model.jnt_dofadr[jid]
        return [float(x) for x in data.qvel[adr:adr + 3]]

    def finger_opening_sum(names) -> float:
        total = 0.0
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                return float("nan")
            total += float(data.qpos[model.jnt_qposadr[jid]])
        return total

    if "ee_pos" in final_state:
        final_state["ee_pos"] = body_pos("hand")
    if "cube_pos" in final_state:
        final_state["cube_pos"] = body_pos("cube")
    if "cube_velocity" in final_state:
        final_state["cube_velocity"] = joint_qvel3("cube_free")
    if "gripper_opening" in final_state:
        final_state["gripper_opening"] = finger_opening_sum(("finger_joint1", "finger_joint2"))


def write_result(final_state: dict, extra: dict | None = None) -> None:
    path = os.environ.get("SIMSMITH_RESULT")
    if not path:
        raise RuntimeError("SIMSMITH_RESULT is not set; run via the SimSmith runner")
    _measure_from_mujoco(final_state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"final_state": final_state, "extra": extra or {}}, f)
