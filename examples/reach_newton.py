"""Hand-written Newton 1.7.0 reach (Franka FR3, headless, IK + XPBD physics step).

Reports raw poses only. The final ee_pos comes from stepping an actual rigid-body
physics solver (SolverXPBD, gravity + PD joint actuation) to rest, not from the
kinematic IK solution directly.
"""
import os
import sys
import time

_t_proc_start = time.perf_counter()

import numpy as np
import warp as wp
import newton
import newton.ik as ik
import newton.utils
from newton.solvers import SolverXPBD
from simsmith_contract import get_task, write_result

_t_imports_done = time.perf_counter()

EE_LINK_LABEL = "fr3/fr3_link7"
EE_OFFSET = wp.vec3(0.0, 0.0, 0.21039998531341553)  # fr3_hand_tcp in fr3_link7's local frame (fixed joint)


def solve_ik(model, target):
    joint_q = wp.zeros((1, model.joint_coord_count), dtype=wp.float32)
    pos_obj = ik.IKObjectivePosition(
        link_index=EE_LINK, link_offset=EE_OFFSET,
        target_positions=wp.array([target.tolist()], dtype=wp.vec3),
    )
    joint_limit_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0,
    )
    solver = ik.IKSolver(model=model, n_problems=1, objectives=[pos_obj, joint_limit_obj],
                          lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    solver.step(joint_q, joint_q, iterations=100)
    return joint_q.numpy()[0].copy()


def ee_world_pos(state):
    tf = wp.transform(*state.body_q.numpy()[EE_LINK])
    return np.array(wp.transform_point(tf, EE_OFFSET))


with wp.ScopedDevice("cpu"):
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.target_ke = 2000.0
    builder.default_joint_cfg.target_kd = 50.0
    urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    builder.add_urdf(
        str(urdf_path), floating=False, joint_damping_mode="drive",
        enable_self_collisions=False,
        collapse_fixed_joints=True,  # merges the massless fixed-joint frames (hand, tcp, ...)
                                      # into their dynamic parent; without this the massless
                                      # dummy links make SolverXPBD diverge to NaN in one step.
    )
    builder.joint_target_kd = [50.0] * len(builder.joint_target_kd)  # URDF's own damping is ~0.003, too weak
    model = builder.finalize()
    EE_LINK = model.body_label.index(EE_LINK_LABEL)
    _t_model_built = time.perf_counter()

    state_a = model.state()
    state_b = model.state()
    control = model.control()
    phys_solver = SolverXPBD(model, iterations=10)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)
    phys_solver.step(state_a, state_b, control, None, 1e-9)  # trigger kernel JIT compile
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)  # reset after the warmup no-op step
    _t_kernels_loaded = time.perf_counter()

    # target_pos is harness-owned: the verifier recomputes it independently from the seed
    # and rejects a report that doesn't match, so it is never ours to invent.
    target = np.array(get_task("reach")["target_pos"])

    sim_dt = 1.0 / 1000.0
    settle_steps = 1500  # 1.5s simulated settle time, matches reach_mujoco.py
    aim = target.copy()
    for correction in range(3):  # re-aim to cancel steady-state gravity sag, like reach_mujoco.py
        joint_target = solve_ik(model, aim)
        control.joint_target_q = wp.array(joint_target, dtype=wp.float32)
        s_in, s_out = state_a, state_b
        for _ in range(settle_steps if correction == 0 else 500):
            phys_solver.step(s_in, s_out, control, None, sim_dt)
            s_in, s_out = s_out, s_in
        state_a, state_b = s_in, s_out
        ee_pos = ee_world_pos(state_a)
        err = target - ee_pos
        if np.linalg.norm(err) < 0.005:
            break
        aim = aim + err
    _t_solved = time.perf_counter()

write_result({"ee_pos": ee_pos.tolist(), "target_pos": target.tolist()})

print(
    "TIMING "
    f"import={_t_imports_done - _t_proc_start:.3f}s "
    f"model_build={_t_model_built - _t_imports_done:.3f}s "
    f"kernel_load={_t_kernels_loaded - _t_model_built:.3f}s "
    f"solve={_t_solved - _t_kernels_loaded:.3f}s "
    f"total={_t_solved - _t_proc_start:.3f}s",
    file=sys.stderr,
)
