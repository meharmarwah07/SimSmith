"""Generate docs_index/scene_reach.md and docs_index/scene_pick_place.md: facts about
the OFFICIAL scene XML for each template, extracted by introspection from the
actually-loaded mujoco.MjModel -- never hand-typed, so it can't drift from the real
assets/franka_emika_panda/scene*.xml. Re-run this whenever those XML files change.

This is a REFERENCE CARD, not a worked example: body/joint/actuator/keyframe facts
and where get_task() values apply, no IK strategy, no control loop, no solution code.
runtime/simsmith_contract.py structurally rejects any script that doesn't load the
exact official scene (same body names/nq/nu/ngeom), so these facts are a hard contract,
not just convention.

Usage:
  python scripts/gen_scene_cards.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "franka_emika_panda")
OUT_DIR = os.path.join(ROOT, "docs_index")

_JNT_TYPE_NAME = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}
_JNT_QPOS_DIM = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
_JNT_DOF_DIM = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}

SCENES = {
    "reach": {
        "xml": "scene.xml",
        "title": "reach scene",
        "intro": (
            "Official scene for the `reach` template. No cube, no bin: the only task "
            "object is the target point itself, which is NOT a body in this scene at "
            "all (see \"Where get_task() values apply\" below)."
        ),
    },
    "pick_place": {
        "xml": "scene_pick_place.xml",
        "title": "pick_place scene",
        "intro": (
            "Official scene for the `pick_place` template. Same robot as `reach`, plus "
            "one free body named \"cube\". There is NO body or geom for a bin anywhere "
            "in this scene (see \"Where get_task() values apply\" below)."
        ),
    },
}


def fmt_range(r) -> str:
    return f"[{r[0]:.4g}, {r[1]:.4g}]"


def body_table(model) -> list:
    rows = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"(unnamed body {i})"
        parent = model.body_parentid[i]
        parent_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)
                      if i > 0 else None)
        pos = model.body_pos[i]
        rows.append((name, parent_name, pos))
    return rows


def joint_table(model) -> list:
    rows = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"(unnamed joint {j})"
        jtype = _JNT_TYPE_NAME[int(model.jnt_type[j])]
        qadr = int(model.jnt_qposadr[j])
        vadr = int(model.jnt_dofadr[j])
        qdim = _JNT_QPOS_DIM[jtype]
        vdim = _JNT_DOF_DIM[jtype]
        limited = bool(model.jnt_limited[j])
        rng = model.jnt_range[j]
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j])
        rows.append({
            "name": name, "type": jtype, "body": body,
            "qpos_adr": qadr, "qpos_dim": qdim, "qpos_slice": (qadr, qadr + qdim),
            "dof_adr": vadr, "dof_dim": vdim, "dof_slice": (vadr, vadr + vdim),
            "limited": limited, "range": rng,
        })
    return rows


def actuator_table(model) -> list:
    rows = []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"(unnamed actuator {a})"
        trntype = model.actuator_trntype[a]
        target_id = model.actuator_trnid[a][0]
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            target = f"joint {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_id)}"
        elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
            target = f"tendon {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, target_id)}"
        else:
            target = f"(transmission type {int(trntype)})"
        rows.append({
            "name": name, "target": target,
            "ctrlrange": model.actuator_ctrlrange[a], "ctrllimited": bool(model.actuator_ctrllimited[a]),
        })
    return rows


def keyframe_table(model) -> list:
    rows = []
    for k in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"(unnamed key {k})"
        rows.append({
            "name": name,
            "qpos": [float(x) for x in model.key_qpos[k]],
            "ctrl": [float(x) for x in model.key_ctrl[k]],
        })
    return rows


def card_for(template: str, spec: dict) -> str:
    path = os.path.join(ASSETS, spec["xml"])
    model = mujoco.MjModel.from_xml_path(path)

    bodies = body_table(model)
    joints = joint_table(model)
    actuators = actuator_table(model)
    keyframes = keyframe_table(model)

    lines = [
        f"# {spec['title']} (template: `{template}`)",
        "",
        f"Generated by `scripts/gen_scene_cards.py` via introspection of "
        f"`assets/franka_emika_panda/{spec['xml']}` with mujoco {mujoco.__version__} -- "
        f"facts about the real scene, not a worked solution. "
        f"`runtime/simsmith_contract.py` rejects any script whose loaded model's body "
        f"names/nq/nu/ngeom don't match this exact file, so these facts are a hard "
        f"contract, not just convention.",
        "",
        spec["intro"],
        "",
        f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  "
        f"ngeom={model.ngeom}  timestep={model.opt.timestep:g}s  "
        f"gravity={[float(g) for g in model.opt.gravity]}",
        "",
        "## World-frame conventions",
        "",
        "- World frame is the scene's root frame (body `world`, id 0); all positions "
        "below are in this frame, metres, right-handed, **z-up** (gravity is "
        f"{[float(g) for g in model.opt.gravity]} m/s^2, i.e. -z).",
        "- The robot base (body `link0`) sits at the world origin "
        f"(pos={[float(x) for x in model.body_pos[1]]}, no rotation) -- "
        "world coordinates ARE base coordinates for this robot.",
        "- The ground/table surface is the `floor` plane geom at world z=0 "
        "(`<geom name=\"floor\" type=\"plane\" .../>` in the included scene file). "
        "There is no separate table body/geom; the floor plane is the support surface "
        + ("the cube and the robot rest on." if template == "pick_place" else "the robot rests on."),
        "",
        "## Bodies",
        "",
        "| name | parent | pos (relative to parent) |",
        "|---|---|---|",
    ]
    for name, parent, pos in bodies:
        parent_s = parent if parent is not None else "(root)"
        pos_s = "[" + ", ".join(f"{float(x):.4g}" for x in pos) + "]"
        lines.append(f"| `{name}` | `{parent_s}` | {pos_s} |")
    lines += [
        "",
        "`hand` is the end-effector proxy body: `write_result()` in "
        "`runtime/simsmith_contract.py` measures `ee_pos` as this body's world position "
        "(`xpos`), from the snapshot at the last real `mj_step` call -- not from "
        "whatever you report.",
    ]
    if template == "pick_place":
        lines.append(
            "`cube` is the manipulated object; its world position is measured the same "
            "way for `cube_pos`."
        )

    def group_for(jname: str) -> str:
        if jname.startswith("finger_joint"):
            return "fingers"
        if jname == "cube_free":
            return "cube"
        return "arm"

    groups = {"arm": [], "fingers": [], "cube": []}
    for j in joints:
        groups[group_for(j["name"])].append(j)
    lines += ["", "## DOF layout", "", f"Total: nq={model.nq}  nv={model.nv} (= qpos size, qvel/dof size)."]
    for gname, label in (("arm", "arm (7 hinge joints)"), ("fingers", "fingers (2 slide joints)"),
                        ("cube", "cube (1 free joint)")):
        gj = groups[gname]
        if not gj:
            continue
        qdim = sum(j["qpos_dim"] for j in gj)
        vdim = sum(j["dof_dim"] for j in gj)
        names = ", ".join(f"`{j['name']}`" for j in gj)
        lines.append(f"- **{label}**: {qdim} qpos dof, {vdim} qvel dof -- {names}")
    lines.append("")
    lines += ["## Joints", "", "| name | type | body | qpos slice | qvel/dof slice | range (if limited) |", "|---|---|---|---|---|---|"]
    for j in joints:
        rng_s = fmt_range(j["range"]) if j["limited"] else "(unlimited)"
        lines.append(
            f"| `{j['name']}` | {j['type']} | `{j['body']}` | "
            f"qpos[{j['qpos_slice'][0]}:{j['qpos_slice'][1]}] ({j['qpos_dim']} dim) | "
            f"qvel[{j['dof_slice'][0]}:{j['dof_slice'][1]}] ({j['dof_dim']} dim) | {rng_s} |"
        )
    lines.append("")
    if template == "pick_place":
        cube_j = next(j for j in joints if j["name"] == "cube_free")
        qa, qb = cube_j["qpos_slice"]
        va, vb = cube_j["dof_slice"]
        lines += [
            "### `cube_free` layout (free joint)",
            "",
            f"- `qpos[{qa}:{qb}]` = `[x, y, z, qw, qx, qy, qz]` -- **position (3) then "
            "quaternion (4, scalar-first `wxyz`)**. This is the cube body's pose "
            "relative to its parent (`cube` body's own `pos` is `0 0 0` in its XML, so "
            "this IS the cube's world pose).",
            f"- `qvel[{va}:{vb}]` = `[vx, vy, vz, wx, wy, wz]` -- linear velocity (3) "
            "then angular velocity (3), both in the world frame for a free joint.",
            "- `cube_initial_pos` from `get_task(\"pick_place\")` is where a script must "
            "WRITE this joint's position qpos entries (quaternion left as the scene's "
            "default, identity `1 0 0 0`, unless the task perturbs orientation) before "
            "stepping physics forward; nothing sets this automatically.",
            "- `write_result()` measures `cube_velocity` as this joint's 3 linear qvel "
            "entries (not angular) from the last-step snapshot.",
            "",
        ]
    lines += ["## Actuators", "", "| name | drives | ctrlrange |", "|---|---|---|"]
    for a in actuators:
        rng_s = fmt_range(a["ctrlrange"]) if a["ctrllimited"] else "(unlimited)"
        lines.append(f"| `{a['name']}` | {a['target']} | {rng_s} |")
    gripper = actuators[-1]
    lines += [
        "",
        f"`{gripper['name']}` (ctrlrange {fmt_range(gripper['ctrlrange'])}) is the "
        "gripper: it drives the `split` tendon coupling `finger_joint1`+`finger_joint2` "
        "(each a slide joint, range [0, 0.04] m individually), remapped so **255 = fully "
        "open, 0 = fully closed**. `write_result()` measures `gripper_opening` as the "
        "SUM of the two finger joints' raw qpos (metres), not the ctrl value -- a closed "
        "grasp on a 4 cm cube reads roughly 0.04, versus ~0.08 fully open.",
        "",
        "## Keyframes",
        "",
    ]
    if keyframes:
        for k in keyframes:
            lines.append(f"- `{k['name']}`: `qpos={k['qpos']}`  `ctrl={k['ctrl']}`")
            if template == "pick_place":
                lines.append(
                    "  (the trailing 7 qpos values are `cube_free`'s keyframe pose -- "
                    "the cube's position as placed in the XML, `0.5 0 0.02`, NOT "
                    "necessarily this run's `cube_initial_pos`, which `get_task()` "
                    "samples per-seed)."
                )
    else:
        lines.append("(none)")
    lines += [
        "",
        "## Where get_task() values apply",
        "",
    ]
    if template == "reach":
        lines += [
            "- `get_task(\"reach\")[\"target_pos\"]` is a free-standing world-frame "
            "point -- **there is no body, geom, site, or any other scene element for "
            "it**. It exists only as a number the script reads and the verifier "
            "independently recomputes from the seed; nothing in the XML represents it "
            "visually or physically.",
            "- Success compares this point only to the measured `ee_pos` (the `hand` "
            "body's position); the target is never something a script spawns in the "
            "scene.",
        ]
    else:
        lines += [
            "- `get_task(\"pick_place\")[\"cube_initial_pos\"]` is where the script must "
            "set the EXISTING `cube` body's free-joint position (see `cube_free` layout "
            "above) before stepping physics -- the cube is already in the scene XML at "
            "a fixed placeholder pose (`0.5 0 0.02`); a script must override that with "
            "this run's sampled value, not add a new body.",
            "- `get_task(\"pick_place\")[\"bin_center\"]` is a free-standing world-frame "
            "point with **no physical representation at all** -- no body, no geom, no "
            "collision volume, nothing the cube can rest in or collide with. The success "
            "check (`simsmith/templates.py`, not reproduced here) compares final "
            "`cube_pos` against this point and against `cube_initial_pos`/`cube_velocity`/"
            "`gripper_opening`; a script does not need to, and cannot, place the cube "
            "\"inside\" anything physical.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for template, spec in SCENES.items():
        content = card_for(template, spec)
        out_path = os.path.join(OUT_DIR, f"scene_{template}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"wrote {out_path} ({len(content)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
