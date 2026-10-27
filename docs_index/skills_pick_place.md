# pick_place skills (`runtime/simsmith_skills.py`)

Trusted helper functions available for the `pick_place` template only. Import them the
same way as `simsmith_contract`:

```python
from simsmith_skills import body_pos, arm_jacobian, ik_step, move_to, set_gripper
```

These are low-level primitives on a `(model, data)` pair your script already created
(via `mujoco.MjModel.from_xml_path` / `mujoco.MjData`, after importing
`simsmith_contract`) -- not a plan. Nothing here decides a grasp height, a waypoint, a
settle duration, or how many times to re-aim; your script still writes all of that,
exactly as it would without these helpers. All five only ever touch the arm's 7 joint
dofs (`joint1`..`joint7`) and the gripper actuator (`ctrl[7]`); none of them can read or
write the cube's free-joint dofs.

Units: positions are `[x, y, z]` in metres, world frame. Joint angles are radians.
Gripper opening is on the actuator's own ctrlrange `[0, 255]` (255 = fully open, 0 =
fully closed), the same convention as the Actuators table in
`docs_index/scene_pick_place.md`.

## `body_pos(model, data, name) -> np.ndarray, shape (3,)`

World position `[x, y, z]` of the body named `name`, read directly from the current
`data.xpos` -- a numpy array (a copy, safe to mutate), the same convention as
`arm_jacobian()` below. Call `.tolist()` on it yourself if you need a plain list (e.g.
for `write_result()`). No recompute happens inside this call. If you just wrote a
body's free-joint `qpos` yourself (e.g. the cube's, before your first `mj_step`), call
`mujoco.mj_forward(model, data)` first -- otherwise this still reflects the pose from
before that write.

Raises `ValueError` if no body has this name.

## `arm_jacobian(model, data, body) -> np.ndarray, shape (3, 7)`

Translational Jacobian of `body`'s origin with respect to `joint1`..`joint7`, in that
order (column `i` is `d(body_pos)/d(qvel[joint_i])`). Internally calls
`mujoco.mj_kinematics` then `mujoco.mj_comPos` first, so the result is always
consistent with whatever `data.qpos` currently is -- including right after you write a
qpos value directly, without needing `mj_forward` first. Neither call advances
simulation time or counts as an `mj_step`.

Returns a numpy array (for your own further linear algebra), same convention as
`body_pos()` above -- unlike `ik_step()`/`move_to()` below, which return a plain list
of floats (joint targets, not a position/vector you'd typically do more linear algebra
on).

## `ik_step(model, data, body, target_pos, keep_gripper_down=True, damping=1e-3, max_delta=0.2) -> [q1..q7]`

One damped-least-squares IK step for `body`'s origin toward `target_pos`, returning a
7-vector of joint position *targets* (radians) -- not a ctrl value, and nothing in
`model`/`data` is modified by calling this. Assign the result into `ctrl[0:7]` yourself,
or use `move_to()` below, which does that plus the stepping/ramping.

**This is one linearized step from the arm's current pose, not an iterate-to-
convergence solve.** For a target far from where the arm is now, one call under-shoots.
Call it again (or call `move_to()` again) from the arm's new, closer pose to refine
further. How many times to re-aim, and when, is your script's decision.

`keep_gripper_down=True` (the default) adds a rotational term that keeps `body`'s local
+Z axis pointing along world -Z (straight down) -- the usual top-down grasp/place
approach angle for this gripper. Pass `False` if approach angle doesn't matter for your
task.

`damping` is the Levenberg-Marquardt regularizer on `J @ J.T`. `max_delta` clips the raw
per-joint step (radians) before it's added to the current joint angle, so a large
one-shot error can't request an unreasonable jump; the result is then also clipped to
each joint's own range (the same limits in the scene cards' Joints tables).

Raises `ValueError` if `body` doesn't exist.

## `move_to(model, data, target_pos, steps, body="hand", keep_gripper_down=True, damping=1e-3, ramp_frac=0.6) -> [q1..q7]`

Drives `body` toward `target_pos` over exactly `steps` calls to `mujoco.mj_step(model,
data)` -- one per iteration, so every one of them is seen by the harness the same as if
your script had called `mj_step` directly. Computes a single `ik_step()` target at the
*start* of the call (from `data`'s pose at that moment) and ramps `ctrl[0:7]` smoothly
from its current value to that target over the first `ramp_frac` fraction of `steps`
(default 60%), holding at the target for the rest so the arm actually settles before
the call returns. Snapping `ctrl` straight to a new target in one step can whip the arm
fast enough to shake a held object loose.

Does **not** touch `ctrl[7]` (the gripper) at all -- whatever it was set to before this
call stays exactly as is for the whole move. Does **not** loop `ik_step` internally to
converge closer over the course of the move; call `move_to` again afterward (same or
adjusted `target_pos`) if the first call under-shot.

Returns the 7-vector IK target this call ramped toward, in case you want to compare it
against the arm's `qpos` afterward.

Every `move_to`/`set_gripper` call's `steps` directly becomes that many real
`mj_step` calls; the harness requires at least 100 total across your whole script, and
the hand-written reference example on this task uses several thousand -- across all
your calls combined, not any single one.

## `set_gripper(model, data, opening, steps) -> None`

Sets the gripper actuator (`ctrl[7]`) to `opening` and calls `mj_step` `steps` times so
the fingers physically move/settle before your script continues. Leaves `ctrl[0:7]`
(the arm) completely untouched -- call this between `move_to()` calls, not interleaved
with one.

## Gotchas

- These helpers only recognize joints literally named `joint1`..`joint7`. They raise
  `ValueError` rather than silently operating on the wrong dofs if that assumption ever
  doesn't hold (it does, for the official `scene_pick_place.xml`).
- `ik_step`/`move_to` solve for `body`'s own origin (`xpos`), not an offset grasp point
  like a fingertip pad -- if your grasp geometry needs an offset, add it to
  `target_pos` yourself before calling (see `docs_index/mujoco_notes.md`'s note on
  `mj_jac` taking an arbitrary world point, for the general technique, which these
  helpers don't expose directly).
- `move_to` reads `data.ctrl[0:7]` as the ramp's starting point. If you've never set
  `ctrl[0:7]` before (a fresh `MjData`, no keyframe reset), that start is `0.0` for
  each -- not necessarily the arm's actual current pose -- so the first ramp may move
  further than the IK solve alone implies. Resetting to the `home` keyframe first (see
  the scene card) avoids this.
- None of these helpers call `get_task`, `get_rng`, `get_perturb`, or `write_result` --
  you still need `from simsmith_contract import ...` for those, exactly as without the
  skills helpers.
