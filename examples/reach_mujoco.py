"""Hand-written MuJoCo reach (Franka Panda, headless). Reports raw poses only."""
import os
import numpy as np
import mujoco
from simsmith_contract import get_task, write_result

ROOT = os.environ.get("SIMSMITH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", "franka_emika_panda", "scene.xml"))
d = mujoco.MjData(m)
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

# target_pos is harness-owned: the verifier recomputes it independently from the seed
# and rejects a report that doesn't match, so it is never ours to invent.
target = np.array(get_task("reach")["target_pos"])


def ik(goal, iters=200):
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    jp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_comPos(m, dd)
        err = goal - dd.xpos[hand]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacBody(m, dd, jp, None, hand)
        J = jp[:, :7]
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err), -0.1, 0.1)
        dd.qpos[:7] = np.clip(dd.qpos[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
    return dd.qpos[:7].copy()


goal = target.copy()
for _ in range(4):  # re-aim to cancel steady-state gravity sag of the position actuators
    d.ctrl[:7] = ik(goal)
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    err = target - d.xpos[hand]
    if np.linalg.norm(err) < 0.005:
        break
    goal = goal + err

write_result({"ee_pos": d.xpos[hand].tolist(), "target_pos": target.tolist()})
