"""Prompt text. Keep simulator-version-specific guidance in docs_index/, not here."""

# Bump this whenever INTAKE/PLAN/GENERATE/REPAIR text OR the generation config
# (max_tokens, extra_payload) changes, and record it alongside any eval/baseline
# results -- without it a pass-rate number can't be tied to what produced it.
# v2 vs v1: GENERATE max_tokens 8000 -> 24000 (v1 measured pick_place truncating
# 100% of the time at 8000), and callers may pass extra_payload=
# {"chat_template_kwargs": {"enable_thinking": False}} to suppress Nemotron's
# "reasoning" field -- probed to cut completion tokens/cost/latency roughly 3-6x
# and eliminate truncation entirely, at the cost of a higher hallucinated-API rate
# on pick_place specifically in a small (n=5) probe. Prompt text itself is unchanged.
PROMPT_VERSION = "mujoco-v5"
# v3 vs v2: generate()/repair() now always append the introspected scene card
# (docs_index/scene_<template>.md, see scripts/gen_scene_cards.py and
# DocIndex.scene_card()) for the active template, independent of the keyword
# ranker's output -- so every candidate/repair sees ground-truth body/joint/
# actuator/keyframe facts for the scene it must load, not just whatever the
# ranker happened to retrieve. Prompt text itself is otherwise unchanged.
# v4 vs v3: GENERATE_SYSTEM's SIMSMITH_ROOT/scene-loading step used to describe the
# fallback in prose ("read it with os.environ, with os.path.dirname(...) as a local
# fallback"), which measurably caused the model to invent-then-misrecall a garbled
# variable name a line or two later (e.g. defines simsmith_root, references
# smithsonian_root) -- docs/diagnosis_2026-10-24.md Part 1 found this was 50% of all
# pick_place crashes in one run. Replaced with a literal, exact snippet the model is
# told to copy verbatim, removing the need to invent-and-recall a name at all. Also
# lists the newly-banned preflight patterns (getattr/setattr/importlib/builtins/
# sys.modules/dunder attrs/non-public simsmith_contract attrs) so a candidate doesn't
# waste a generation on something that will be statically rejected anyway.
# v5 vs v4: the v4 "copy this preamble verbatim" fix removed the SIMSMITH_ROOT-name-
# corruption crashes entirely (docs/diagnosis_2026-10-25.md) but introduced a new one --
# models treated the preamble as closed and ~31-55% of reach candidates then never
# `import numpy` despite needing `np.` throughout, a NameError (the single largest
# crash cause in a v4 regression run). Fixed at the source: numpy is now part of the
# literal preamble. Also made explicit (previously only enforced by preflight, not
# stated): load the scene ONLY via mujoco.MjModel.from_xml_path, never open()/vars(),
# and list every import at the top -- the last of these is also now a precondition for
# preflight's new undefined-name check (see simsmith/preflight.py), which rejects a
# script using a name no import/assignment/builtin accounts for, before any sim time.

INTAKE_SYSTEM = """You map a user's robot request onto ONE supported task template.
Supported templates:
- reach: move a Franka arm end effector to a target position
- pick_place: Franka arm picks a cube and places it in a bin
Reply with ONLY a JSON object:
{"template": "reach|pick_place", "exact_match": true|false,
 "gap": "<if not an exact match, one sentence on what differs, else empty>",
 "goal": "<one-sentence restatement of the task as supported>"}
Never refuse. If the request is outside scope, choose the closest template and describe the gap."""

PLAN_SYSTEM = """You are a robotics simulation engineer planning a MuJoCo script for a
Franka Panda arm. Write a short numbered plan: how you load the scene, how you command
the arm/gripper (IK + position actuators, stepped forward in time with mj_step -- not
just computed kinematically), motion phases, and the exact raw state the script must
report at the end. The success check is fixed by the system and the script must NOT
compute or report success itself."""

GENERATE_SYSTEM = """You write complete, runnable MuJoCo Python scripts (headless, no viewer)
for a Franka Panda arm.
Hard requirements:
1. Single file. Return ONLY one ```python code block.
2. Start with EXACTLY this preamble, copied verbatim (same variable names throughout --
   do not rename them or re-derive them differently later in the script), then list
   every other import you need, also at the top, before any other code:
   ```python
   import os
   import mujoco
   import numpy as np
   from simsmith_contract import get_task, get_rng, get_perturb, write_result
   SIMSMITH_ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   ```
   Every name you use later must come from an import at the top, a builtin, or something
   you assign/define yourself -- a name you never imported (e.g. using `np.` without
   this import) is rejected before any sim time is spent.
3. Load the scene using that exact `SIMSMITH_ROOT` variable, nothing else, and ONLY via
   `mujoco.MjModel.from_xml_path` (never `from_xml_string`, never read/parse the XML
   file yourself):
   ```python
   model = mujoco.MjModel.from_xml_path(os.path.join(SIMSMITH_ROOT, "assets", "franka_emika_panda", "scene.xml"))
   ```
   Use `scene.xml` for a reach task, `scene_pick_place.xml` for a pick_place task (it
   adds a free body named "cube"). `model`/`data` variable names otherwise your choice.
4. You MUST actually simulate: create mujoco.MjData, then call mujoco.mj_step(model, data)
   enough times to let the arm/gripper actually get where it's going (hundreds of steps
   at the model's own timestep, not one or two). A script that never steps physics
   forward in time is rejected before it runs (pre-flight requires an mj_step call) and
   its reported end-effector/cube state is independently re-measured from the simulated
   MjData and WILL NOT match a faked value -- there is no shortcut around actually
   simulating.
5. Known names in these scenes: body "hand" (end effector proxy), joints
   "joint1".."joint7" (arm, actuators ctrl[0:7]), "finger_joint1"/"finger_joint2" and
   actuator ctrl[7] in [0,255] (255=open) for the gripper (pick_place only), body "cube"
   and freejoint "cube_free" (pick_place only, already in the scene -- just read/set its
   qpos/qvel, don't add a body for it). Position actuators sag under gravity: re-aim the
   IK goal by the residual error 2-4 times rather than solving once and trusting it.
6. Get this run's target/cube/bin position(s) from get_task("<template>") (e.g.
   get_task("reach")["target_pos"], get_task("pick_place")["cube_initial_pos"] and
   ["bin_center"]). NEVER compute these yourself and NEVER report a value derived from
   your own simulation as if it were the task parameter (e.g. target_pos = ee_pos) --
   the verifier recomputes the real parameters independently from the seed and rejects
   a report that doesn't match exactly. get_task() already applies any perturbation.
7. Use rng = get_rng() only to randomise things get_task() doesn't cover. Call
   get_perturb() yourself only for perturbation keys that affect your script's behavior
   but aren't positions (friction_scale, mass_scale, action_delay_steps,
   pose_noise_std); missing keys mean no perturbation.
8. At the end call write_result(final_state) with RAW positions/velocities only, built
   from get_task()'s values and your own measured final state (ee_pos for reach;
   cube_pos, cube_initial_pos, bin_center, cube_velocity, gripper_opening for
   pick_place). Never a success flag.
9. Use only APIs from the reference material (MuJoCo API notes and gotchas are
   retrieved for you below) or ones you are certain exist. Do not invent function names.
10. No subprocess, sockets, network access, eval/exec/compile/open/input/getattr/
    setattr/globals/vars/importlib/builtins/sys.modules, and no attribute access of any
    kind on `simsmith_contract` other than calling get_task/get_rng/get_perturb/
    write_result (the normal `from simsmith_contract import ...` form is fine and
    expected; `import simsmith_contract` to poke at anything else on it is not). In
    particular: never call `open()` (not even to read the scene XML or write a debug
    log -- print() instead) and never call `vars()` (print the specific values you
    want instead). These are all statically rejected before any sim time is spent."""

REPAIR_SYSTEM = GENERATE_SYSTEM + """
You are repairing a script that failed. You are given the script and exact failure output.
Fix the root cause; do not add success flags. Return the full corrected script."""

FRAMINGS = [
    "Prefer the simplest, most direct implementation.",
    "Prefer a structured implementation with small helper functions and explicit phases.",
    "Be defensive: add settling steps and check intermediate states before moving on.",
    "Prefer the approach used in the reference examples as closely as possible.",
    "Think about what commonly goes wrong in this kind of task and guard against it.",
]
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]


# =================================================================================
# Skills ablation (pick_place only). Appended by Pipeline.generate()/repair() to the
# USER message, never baked into GENERATE_SYSTEM/REPAIR_SYSTEM above -- those stay
# byte-identical to PROMPT_VERSION "mujoco-v5" for BOTH templates, so MockLLM's exact
# system-string dispatch (tests/mocks.py) and every existing reach measurement under
# that version stay valid. reach never sees this block. Label results produced with
# it as PICK_PLACE_SKILLS_PROMPT_VERSION, not the global PROMPT_VERSION above, since
# this is an additive, pick_place-only change to the user content, not a new version
# of the shared system prompt text.
PICK_PLACE_SKILLS_PROMPT_VERSION = "mujoco-v6-skills"

PICK_PLACE_SKILLS_BLOCK = """For this pick_place task, trusted helper functions are also available via
`from simsmith_skills import ...` (same import style as simsmith_contract):
body_pos, arm_jacobian, ik_step, move_to, set_gripper. Their exact signatures, units,
and gotchas are in the reference material below -- read it before using them. They
are low-level primitives, not a plan: no helper picks a grasp height, a waypoint, a
settle duration, or how many times to re-aim -- that sequencing is still yours to
write, exactly as it would be without them. Using them is optional; you may write
your own arm control with raw mujoco calls instead if you prefer."""
# =================================================================================
