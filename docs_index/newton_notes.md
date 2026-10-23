# Newton physics notes (pinned: newton 1.7.0)
Load a robot: `builder = newton.ModelBuilder(); builder.add_urdf(path, floating=False, ...); model = builder.finalize()`.
Kinematic IK: `newton.ik.IKSolver(model, n_problems, objectives, jacobian_mode=...)`, objectives include
`IKObjectivePosition`, `IKObjectiveRotation`, `IKObjectiveJointLimit`. `solver.step(joint_q, joint_q, iterations=...)`
mutates `joint_q` in place; read body poses via `eval_fk_batched` (batched) or `newton.eval_fk` (single state).
Dynamics: `newton.solvers.SolverXPBD(model, iterations=...)`, step with `solver.step(state_in, state_out, control, contacts, dt)`;
`contacts=None` skips rigid/particle-shape contact handling (fine for a single arm with no scene objects).
Actuation: set `builder.default_joint_cfg.target_ke/target_kd` BEFORE `add_urdf(..., joint_damping_mode="drive")`,
then drive position targets at runtime via `control.joint_target_q = wp.array(q, dtype=wp.float32)` (coord layout).

## Gotchas found the hard way

- **`add_urdf(..., collapse_fixed_joints=True)` is close to mandatory for dynamics.**
  Franka-style URDFs have massless "frame" links joined by FIXED joints (e.g. a dummy
  link8, the hand-mount frame, the tcp frame). Without collapsing them into their
  dynamic parent, `SolverXPBD` diverges to NaN velocity within a single step — even
  with the control target set equal to the current pose (zero tracking error). Kinematic
  IK alone (no solver stepping) doesn't need this, so it's easy to miss until you add
  an actual physics step.
- **A URDF joint's own `<dynamics damping=...>` overrides `default_joint_cfg.target_kd`.**
  `joint_damping_mode="drive"` maps the URDF's (usually tiny, e.g. 0.003) passive
  damping value into `joint_target_kd` directly; `default_joint_cfg.target_kd` only
  applies where the URDF specifies none. To force your own PD damping regardless,
  overwrite `builder.joint_target_kd` (and `joint_target_ke`) as a list after `add_urdf`,
  before `finalize()`.
- **`builder.body_q` (pre-finalize) is NOT the resolved world rest pose** — it's close to
  a placeholder (often identity) for URDF-imported bodies. To get a link's real rest-pose
  transform (e.g. to compute a fixed local offset like a tcp point relative to a parent
  link after collapsing), `finalize()` the model, call `newton.eval_fk(model, model.joint_q, model.joint_qd, state)`,
  and read `state.body_q` — not the builder's list.
- **Body/joint ordering after `collapse_fixed_joints=True`** drops the collapsed bodies
  entirely (body_count shrinks); re-resolve any body index you need (e.g. via
  `model.body_label.index(...)`) against the collapsed model, don't assume the index
  from an uncollapsed build still lines up.
- A no-op `solver.step(state, state2, control, None, 1e-9)` right after building the
  model is a cheap way to force warp's kernel JIT compile before timing the real solve
  (the first real call otherwise eats that compile time).
