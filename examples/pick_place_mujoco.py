"""Hand-written MuJoCo pick_place (Franka Panda, headless). Reports raw poses only."""
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
cube_qvel_adr = m.jnt_dofadr[cube_joint]  # free joint: qvel[adr:adr+3] is world-frame linear velocity

GRASP_OFFSET = np.array([0.0, 0.0, 0.1035])  # fingertip pad point, in the "hand" body's local frame
DOWN = np.array([0.0, 0.0, -1.0])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

# Arm home pose (panda.xml "home" keyframe), gripper open.
d.qpos[:9] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
d.ctrl[:8] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, GRIPPER_OPEN]

# cube_initial_pos and bin_center are harness-owned: the verifier recomputes them
# independently from the seed and rejects a report that doesn't match, so they are
# never ours to invent.
task = get_task("pick_place")
cube_pos = np.array(task["cube_initial_pos"])
bin_center = np.array(task["bin_center"])

d.qpos[cube_qpos_adr : cube_qpos_adr + 3] = cube_pos
d.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)


def grasp_point(dd):
    return dd.xpos[hand] + dd.xmat[hand].reshape(3, 3) @ GRASP_OFFSET


def ik(goal, iters=300):
    """Position IK for the grasp point plus an axis-alignment term that keeps the hand's
    local Z (fingertip direction) pointing straight down. Without it, the Jacobian
    pseudo-inverse settles on an arbitrary approach angle that often bumps the cube
    with the palm/wrist before the fingers get low enough to straddle it."""
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_comPos(m, dd)  # required before mj_jac, else the Jacobian is wrong
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
    """Ramp ctrl to the IK solution instead of snapping to it; a step input to the stiff
    position actuators whips the arm fast enough to shake a grasped cube loose."""
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

move_smooth(cube_pos + [0, 0, 0.15], GRIPPER_OPEN, duration_s=1.2)     # above the cube
move_smooth(cube_pos + [0, 0, 0.0], GRIPPER_OPEN, duration_s=1.2)      # down to grasp height
d.ctrl[7] = GRIPPER_CLOSED                                             # close the gripper
for _ in range(int(0.6 / m.opt.timestep)):
    mujoco.mj_step(m, d)
move_smooth(cube_pos + [0, 0, 0.20], GRIPPER_CLOSED, duration_s=1.5)   # lift
move_smooth(bin_center + [0, 0, 0.20], GRIPPER_CLOSED, duration_s=2.0)  # carry over the bin
move_smooth(bin_center + [0, 0, 0.06], GRIPPER_CLOSED, duration_s=1.5)  # lower into the bin
d.ctrl[7] = GRIPPER_OPEN                                                # release
for _ in range(int(1.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
move_smooth(bin_center + [0, 0, 0.20], GRIPPER_OPEN, duration_s=1.0)   # retreat

write_result({
    "cube_pos": d.xpos[cube].tolist(),
    "cube_initial_pos": cube_initial_pos.tolist(),
    "bin_center": bin_center.tolist(),
    "cube_velocity": d.qvel[cube_qvel_adr : cube_qvel_adr + 3].tolist(),
    "gripper_opening": float(d.qpos[7] + d.qpos[8]),
})
