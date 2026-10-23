"""Task templates and their deterministic success checks.

The success check is fixed by the template, not by the generated script, so a
script cannot grade itself. A check receives the RAW final state written by
the script and returns (ok, detail).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def _dist(a, b, dims=3) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(dims)))


def _vec_ok(v, n=3) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) >= n
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v[:n]))


def _num_ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


PICK_PLACE_GRIPPER_OPEN_MIN = 0.06  # metres of finger separation; a grasped 4cm cube reads ~0.04


@dataclass(frozen=True)
class Template:
    name: str
    robot: str
    description: str
    required_keys: tuple
    check: Callable[[dict], tuple]
    state_help: str


def _check_reach(s: dict):
    if not (_vec_ok(s.get("ee_pos")) and _vec_ok(s.get("target_pos"))):
        return False, "bad ee_pos/target_pos"
    d = _dist(s["ee_pos"], s["target_pos"])
    return d <= 0.02, f"ee-target distance {d:.4f} m (need <= 0.02)"


def _check_pick_place(s: dict):
    for k in ("cube_pos", "cube_initial_pos", "bin_center", "cube_velocity"):
        if not _vec_ok(s.get(k)):
            return False, f"bad {k}"
    if not _num_ok(s.get("gripper_opening")):
        return False, "bad gripper_opening"
    moved = _dist(s["cube_pos"], s["cube_initial_pos"])
    d_xy = _dist(s["cube_pos"], s["bin_center"], dims=2)
    # cube_initial_pos is measured at rest on the floor before pickup, so its z doubles
    # as "the resting height" without the template having to hardcode table geometry.
    resting_z = float(s["cube_initial_pos"][2])
    dz = abs(float(s["cube_pos"][2]) - resting_z)
    speed = math.sqrt(sum(float(v) ** 2 for v in s["cube_velocity"][:3]))
    gripper_opening = float(s["gripper_opening"])
    gripper_open = gripper_opening >= PICK_PLACE_GRIPPER_OPEN_MIN
    ok = moved >= 0.10 and d_xy <= 0.05 and dz <= 0.01 and speed < 0.05 and gripper_open
    detail = (
        f"cube moved {moved:.3f} m (need >= 0.10), xy distance to bin {d_xy:.3f} m (need <= 0.05), "
        f"z vs resting height {dz:.3f} m (need <= 0.01), speed {speed:.3f} m/s (need < 0.05), "
        f"gripper_opening {gripper_opening:.3f} m (need >= {PICK_PLACE_GRIPPER_OPEN_MIN})"
    )
    return ok, detail


TEMPLATES = {
    "reach": Template(
        "reach", "franka_panda",
        "Move the arm end effector to a target position.",
        ("ee_pos", "target_pos"), _check_reach,
        "final_state = {'ee_pos': [x,y,z], 'target_pos': [x,y,z]}  (world frame, metres)"),
    "pick_place": Template(
        "pick_place", "franka_panda",
        "Pick up a cube and place it in a bin.",
        ("cube_pos", "cube_initial_pos", "bin_center", "cube_velocity", "gripper_opening"), _check_pick_place,
        "final_state = {'cube_pos': [x,y,z], 'cube_initial_pos': [x,y,z], 'bin_center': [x,y,z], "
        "'cube_velocity': [vx,vy,vz] (m/s), 'gripper_opening': <metres of finger separation>}"),
}


def closest_template(text: str) -> str:
    """Keyword fallback used when the LLM intake fails."""
    t = text.lower()
    if any(w in t for w in ("pick", "place", "bin", "grasp", "put", "cube", "move the")):
        return "pick_place"
    return "reach"
