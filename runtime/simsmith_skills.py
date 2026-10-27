"""Trusted helper functions for generated MuJoCo scripts (the "skills" ablation).

Import style matches simsmith_contract: `from simsmith_skills import move_to, ...`.
This module is on PYTHONPATH the same way simsmith_contract is (see
simsmith/runner.py's CONTRACT_DIR), so no extra wiring is needed to use it.

These are low-level, reusable PRIMITIVES on a (model, data) pair the calling script
already created -- they do not pick a plan. There is no pick_place sequence, no
grasp height, no waypoint, no settle-duration, and no "how many times to re-aim"
baked in anywhere here: a script decides all of that itself, the same way it would
without these helpers. All five functions only ever touch the arm's 7 joint dofs
(named `joint1`..`joint7`) and the gripper actuator (ctrl[7]) -- never a free-joint
body like the pick_place scene's `cube`, so using them cannot accidentally move or
reset anything a script didn't ask them to.

Units: positions in metres, world frame; joint angles in radians; gripper opening on
the actuator's own ctrlrange [0, 255] (255 = fully open), same convention as
docs_index/scene_pick_place.md's Actuators table.

Trust/harness notes (why these are safe to hand to a generated script as-is):
  * Every mj_step call below is `mujoco.mj_step(model, data)` -- an attribute lookup
    on the `mujoco` module at call time, never a cached/imported function reference --
    so it always goes through whatever simsmith_contract's import-time hook currently
    has installed there (model-identity check, per-step snapshot for write_result()'s
    independent re-measurement). These helpers add no code path around that hook.
  * `data` is always the SAME MjData object the calling script created (and that
    simsmith_contract is therefore already tracking by id()); these helpers never
    create their own throwaway MjData to step, so every step they take counts toward
    the harness's MIN_MJ_STEPS floor and the last-step snapshot exactly as if the
    script had called mj_step directly itself.
  * No helper here imports or touches simsmith_contract at all -- they only need
    `model`/`data`, which the calling script already has from its own get_task()/
    MjModel.from_xml_path() calls, so there is no interaction with the preflight
    restriction on simsmith_contract attribute access.
"""
from __future__ import annotations

import mujoco
import numpy as np

_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
_ARM_CTRL_SLICE = slice(0, 7)   # actuator1..7 drive joint1..7, in that order (see scene cards)
_GRIPPER_CTRL_IDX = 7           # actuator8, tendon-coupled fingers, ctrlrange [0, 255]
_DOWN = np.array([0.0, 0.0, -1.0])


def _arm_joint_ids(model) -> list:
    ids = []
    for name in _ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"model has no joint named {name!r} -- not an official "
                             "reach/pick_place scene?")
        ids.append(jid)
    return ids


def body_pos(model, data, name: str) -> np.ndarray:
    """World position [x, y, z] of the body named `name`, read directly from the
    CURRENT `data.xpos` -- no recompute. If you just wrote a body's free-joint qpos
    (e.g. the pick_place `cube`) yourself, call mujoco.mj_forward(model, data) (or
    mj_step) first, or this will still reflect the pose before that write.

    Returns a numpy array (a copy, safe to mutate), same as `data.xpos[bid]` itself
    and the same convention as arm_jacobian() -- call `.tolist()` on it yourself if
    write_result() or some other call needs a plain list.

    Raises ValueError if no body has this name.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"model has no body named {name!r}")
    return data.xpos[bid].copy()


def arm_jacobian(model, data, body: str) -> np.ndarray:
    """3x7 translational Jacobian of `body`'s origin with respect to joint1..joint7
    ONLY, in that order -- never the gripper fingers or any other dof the model has
    (e.g. the pick_place scene's cube_free, which a script must never be able to
    perturb through a Jacobian/IK call). Column i is d(body_pos)/d(qvel[joint_i]).

    Calls mujoco.mj_kinematics(model, data) then mj_comPos(model, data) internally
    first, so the result is always consistent with whatever `data.qpos` currently is,
    even if you haven't called mj_forward/mj_step yourself since your last qpos
    write (skipping that pair before mj_jac is a documented way to silently get a
    wrong Jacobian -- see docs_index/mujoco_notes.md). Neither call advances
    simulation time or counts as an mj_step.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"model has no body named {body!r}")
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    dof_idx = [model.jnt_dofadr[jid] for jid in _arm_joint_ids(model)]
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, data.xpos[bid], bid)
    return jacp[:, dof_idx]


def ik_step(model, data, body: str, target_pos, keep_gripper_down: bool = True,
            damping: float = 1e-3, max_delta: float = 0.2) -> list:
    """One damped-least-squares IK step for `body`'s origin toward `target_pos`
    ([x, y, z], world frame, metres), returning a 7-vector of joint position TARGETS
    (radians, one per joint1..joint7, in that order) -- NOT a ctrl value by itself and
    NOT applied to `data`/`model` in any way. Assign the result into ctrl[0:7]
    yourself (position actuators), or use move_to() which does that plus the
    stepping/ramping.

    This is ONE linearized step from `data`'s CURRENT pose, not an iterate-to-
    convergence solve: for a target far from the arm's current position, one call
    will under-shoot. Call it again (or call move_to() again toward the same
    target_pos) from the arm's new, closer pose to refine further -- how many times
    to re-aim, and when, is your script's decision, not this function's.

    If `keep_gripper_down` (default True), also minimizes rotational error against a
    fixed goal: `body`'s local +Z axis pointing along world -Z (straight down) --
    the usual top-down grasp/place approach for this gripper. Pass False for a task
    where approach angle doesn't matter.

    `damping` is the Levenberg-Marquardt regularizer on J @ J.T (larger = more
    stable/slower near a singularity, smaller = more responsive). `max_delta` clips
    the raw per-joint step (radians) before adding it to the current joint angle, so
    a large one-shot error (e.g. the first call from the arm's resting pose) can't
    request an unreasonably large jump; the result is then also clipped to each
    joint's own range (model.jnt_range), the same limits listed in the scene cards.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"model has no body named {body!r}")
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    joint_ids = _arm_joint_ids(model)
    dof_idx = [model.jnt_dofadr[jid] for jid in joint_ids]
    qpos_idx = [model.jnt_qposadr[jid] for jid in joint_ids]

    point = np.array(data.xpos[bid], dtype=float)
    err_p = np.asarray(target_pos, dtype=float) - point
    jacp = np.zeros((3, model.nv))

    if keep_gripper_down:
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, point, bid)
        z_world = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        err_r = np.cross(z_world, _DOWN)
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        err = np.concatenate([err_p, err_r])
    else:
        mujoco.mj_jac(model, data, jacp, None, point, bid)
        J = jacp[:, dof_idx]
        err = err_p

    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), err)
    dq = np.clip(dq, -max_delta, max_delta)

    current = np.array([data.qpos[i] for i in qpos_idx], dtype=float)
    target_q = current + dq
    for k, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            target_q[k] = min(max(target_q[k], lo), hi)
    return [float(x) for x in target_q]


def move_to(model, data, target_pos, steps: int, body: str = "hand",
           keep_gripper_down: bool = True, damping: float = 1e-3, ramp_frac: float = 0.6) -> list:
    """Drives `body` toward `target_pos` ([x, y, z], world frame, metres) over
    exactly `steps` calls to mujoco.mj_step(model, data) -- one per loop iteration, so
    the harness's snapshot hook sees every one of them, same as if your script called
    mj_step itself. Computes a single ik_step() target at the START of this call (from
    `data`'s pose at that moment) and ramps ctrl[0:7] from its current value to that
    target over the first `ramp_frac` fraction of `steps` (default 60%), holding at
    the target for the remainder so the arm actually settles before this call
    returns -- snapping ctrl straight to a new target can whip the arm fast enough to
    shake a held object loose.

    Does NOT touch ctrl[7] (the gripper): whatever it was set to before this call
    (e.g. via set_gripper()) stays exactly as is for the whole move. Does NOT loop
    ik_step internally to converge closer over the course of the move -- call move_to
    again (same or adjusted target_pos) afterward if the first call undershot; your
    script decides how many calls, at what heights/waypoints, in what order.

    Returns the 7-vector IK target this call ramped toward (radians, joint1..joint7),
    in case you want to compare it against the arm's qpos afterward.
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    target_q = np.array(ik_step(model, data, body, target_pos,
                                keep_gripper_down=keep_gripper_down, damping=damping))
    start_q = np.array(data.ctrl[_ARM_CTRL_SLICE], dtype=float)
    n_ramp = max(int(steps * ramp_frac), 1)
    for i in range(steps):
        alpha = min(1.0, i / n_ramp)
        data.ctrl[_ARM_CTRL_SLICE] = start_q + alpha * (target_q - start_q)
        mujoco.mj_step(model, data)
    return [float(x) for x in target_q]


def set_gripper(model, data, opening: float, steps: int) -> None:
    """Sets the gripper actuator (ctrl[7]) to `opening` (ctrlrange [0, 255]; 255 =
    fully open, 0 = fully closed -- see docs_index/scene_pick_place.md's Actuators
    table) and calls mujoco.mj_step(model, data) `steps` times so the fingers
    physically move/settle before your script continues. Leaves ctrl[0:7] (the arm)
    completely untouched, whatever it currently is -- call this between move_to()
    calls, not interleaved with one.
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    data.ctrl[_GRIPPER_CTRL_IDX] = float(np.clip(opening, 0.0, 255.0))
    for _ in range(steps):
        mujoco.mj_step(model, data)
