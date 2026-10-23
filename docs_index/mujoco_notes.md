# MuJoCo + Franka Panda notes (pinned: mujoco 3.16)
Load: mujoco.MjModel.from_xml_path(os.path.join(os.environ['SIMSMITH_ROOT'],'assets','franka_emika_panda','scene.xml')).
Bodies: link0-link7, hand, left_finger, right_finger. No end-effector site: use body 'hand' (d.xpos[id]).
Actuators: 7 position-controlled arm joints (ctrl[:7] = target joint angles), ctrl[7] gripper (0-255, 255=open).
Keyframe 0 is 'home': mujoco.mj_resetDataKeyframe(m, d, 0).
IK: mujoco.mj_jacBody(m, dd, jacp, None, body_id), damped least squares on first 7 dofs.
Position actuators sag under gravity: re-aim goal by the residual error 2-4 times.
Step: mujoco.mj_step(m, d); timestep m.opt.timestep. Run headless, no viewer.

## Gotchas found the hard way

- **mj_jac/mj_jacBody need mj_comPos first.** Calling only mj_kinematics(m, dd) before
  mj_jac gives a silently wrong Jacobian (IK barely moves, or diverges). Always do
  mj_kinematics(m, dd) then mj_comPos(m, dd) before mj_jac/mj_jacBody in a from-scratch
  MjData copy used for an IK sub-solve.
- **mj_jac(m, dd, jacp, jacr, point, body_id) takes an arbitrary world-frame POINT**,
  not just the body origin. Use it (recomputing `point` each iteration from the body's
  current xpos/xmat) to target an offset like a fingertip grasp point that isn't the
  body's own frame origin, e.g. `point = xpos[hand] + xmat[hand].reshape(3,3) @ local_offset`.
- **Position-only IK (3 DOF) is not enough for grasping.** The damped-least-squares
  pseudo-inverse will happily converge to the target position with the gripper
  approaching from an arbitrary angle, which often lets the palm/wrist clip the object
  before the fingers get low enough to straddle it. Add an axis-alignment term so the
  gripper's approach axis points straight down: err_rot = cross(z_world, [0,0,-1]),
  stack with the position error, and use both jacp and jacr (6xN Jacobian, 6x6 damping).
- **Don't snap ctrl straight to a new IK solution when holding something.** A step input
  to these stiff position actuators (gainprm ~2000-4500) whips the arm fast enough to
  shake a grasped object loose from friction-only contact. Ramp ctrl from the current
  value to the new target over most of the settle duration instead of assigning it in
  one shot.
- **Free-joint qvel, not xpos deltas, for reported velocity.** For a body with a
  `<freejoint>`, `d.qvel[m.jnt_dofadr[joint_id] : +3]` is the body's linear velocity in
  the WORLD frame directly (no finite-differencing needed); the next 3 are angular
  velocity in the body's local frame.
- **Franka gripper "opening" convention**: ctrl[7] (actuator8, tendon-coupled fingers)
  in [0, 255], 255 = fully open (~0.04 m per finger), 0 = fully closed. A good scalar
  "gripper_opening" to report is `d.qpos[finger1] + d.qpos[finger2]` (0 to 0.08 m);
  a 4cm grasped cube reads ~0.04, so use a threshold above that (e.g. 0.06) to tell
  "open/released" from "still holding something."
