# SimSmith

Describe a robot task in plain language. SimSmith writes several NVIDIA Isaac Sim
scripts with Nemotron (Nebius Token Factory), runs them, **verifies the outcome
by execution**, repairs failures, and keeps verified scripts in a library that
improves later runs.

> Simulator: MuJoCo (Franka Panda from mujoco_menagerie, Apache-2.0, in `assets/`) is the working baseline; Isaac Sim is an optional runner if GPU access is secured. `examples/reach_mujoco.py` passes the `reach` check on 8/8 seeds, and `examples/pick_place_mujoco.py` — the real demo, IK + a physical grasp (contact friction, no teleporting) — passes `pick_place` on 8/8 seeds, ~0.7s/run (executed for real, headless).
>
> Newton 1.7.0 is a second, validated runner: `examples/reach_newton.py` passes `reach` on 8/8 seeds using real rigid-body dynamics (SolverXPBD, gravity + PD joint actuation settled to rest, not just the kinematic IK pose), ~13s/run including first-run kernel compile. Getting past kinematic-IK-only required `collapse_fixed_joints=True` on `add_urdf` — without it the massless fixed-joint frames in the Franka URDF (hand/tcp mounts) make SolverXPBD diverge to NaN in one step.

> Status: early base. The pipeline, verifier, eval harness and tests are in
> place and tested **with mocks**. It has **not yet been run against real Isaac
> Sim or the live Token Factory API**. Do that first (see "Day-1 gate").

## Why execution-verified

An LLM can write Isaac Sim code that runs without error but never completes the
task. SimSmith fixes the success check per task template (`simsmith/templates.py`)
and computes success from the **raw final poses** a script reports. A script
cannot report its own success, and the LLM never picks the winner.

## Pipeline

1. **Intake** (Nemotron fast model): map the request to a supported template
   (`reach`, `pick_place`). Out-of-scope requests get the closest
   template plus an explicit gap statement, never a silent substitution.
2. **Plan** (Nemotron plan model).
3. **Retrieve**: pinned-version Isaac docs (`docs_index/`) and verified library scripts.
4. **Generate** N candidates, each with different retrieved context, framing and temperature.
5. **Pre-flight (no GPU)**: syntax, banned imports/calls, optional API allow-list.
6. **Execute** each candidate per seed in a separate process (Isaac headless on Nebius).
7. **Verify**: deterministic success check over several seeds.
8. **Repair**: capped number of repair rounds using the real failure output.
9. **Select and learn**: best verified candidate wins and is saved to the library.

Not yet implemented: Tavily-grounded failure-mode research and perturbation
robustness testing (the runner already passes `SIMSMITH_PERTURB` to scripts, so
these can be added without changing the script contract).

## Setup

```bash
cp .env.example .env          # fill in Token Factory URL/key, model IDs, SIM_PYTHON
python -m pytest -q           # 27 tests, no GPU or network needed
```

Put text/markdown excerpts of the docs for your **pinned** Isaac Sim version in
`docs_index/` (mixing versions is the main source of hallucinated APIs).
Optionally list valid module paths in `docs_index/isaac_api_allowlist.txt`.

## Day-1 gate (run on the Nebius GPU instance)

```bash
python scripts/gate_day1.py --handwritten examples/reach_isaac.py
```

You must write `examples/reach_isaac.py` yourself for your pinned version
(only a pure-Python mock is included). The gate prints seconds per run, which
decides how large the final evaluation can be within your credit budget.

## Evaluation

```bash
python -m eval.run_eval --max-runs 300
```

Conditions: `single_shot`, `best_of_5` (+repair), `best_of_5_library`. The
library is built from **dev** tasks only and frozen; held-out runs never write to
it. A hard cap on sim executions protects the budget and the report says when
it truncated. Report every condition, including losses.

## Script contract

Generated scripts import `simsmith_contract` (see `runtime/`). Target/cube/bin
positions are harness-owned: scripts read them from `get_task(template)`, never
compute or self-report them — the verifier recomputes the same parameters from
the seed independently and fails any run whose reported value doesn't match, so
a script claiming e.g. `target_pos = ee_pos` is caught, not trusted. Scripts
randomise anything else with `get_rng()`, honour `get_perturb()`, and end with
`write_result(final_state)` containing raw positions only.

## Security

**Threat model: accidental gaming, not adversarial code.** The verifier's design (fixed
success checks, independent re-measurement from MuJoCo state, a never-trusted
self-report) and the pre-flight checks (`simsmith/preflight.py`) exist to catch an LLM
*accidentally or lazily* gaming its own grade — echoing a target back as the answer,
self-reporting success, drifting into an API it doesn't actually have, etc. They are
**not** a security boundary against a script an attacker deliberately crafted to
defeat the harness. `preflight.py`'s checks are AST-literal pattern matches: they close
every bypass found during a specific audit (`docs/diagnosis_2026-10-24.md`) --
`importlib`-based reflection, `getattr`/`setattr` indirection, dunder-attribute escapes,
reaching into `simsmith_contract`'s private hooking state, shadowing `write_result` --
but a sufficiently motivated attacker can always find one more level of indirection an
AST pattern doesn't cover. `simsmith/runner.py` strips the child process's environment
to a small allow-list (no API keys reach the script) and best-effort isolates its
network namespace (`unshare --net --map-root-user`, confirmed working in this
environment; plain `unshare --net` is not permitted here) as real, if partial,
mitigations — but the script still runs as a plain subprocess of the harness, with
full filesystem read access and no CPU/memory limits. If this project is ever used
against untrusted or adversarial model output (not just "an LLM trying its honest
best"), running candidate scripts in a real sandbox (container/VM, unprivileged user,
read-only filesystem) is a prerequisite, not an enhancement.

## Known limitations

- Two task templates, one robot; one pinned Isaac version.
- Success checks are simple geometric thresholds; they do not judge motion quality or safety.
- Robustness inside a simulator does not establish real-world transfer.
- Retrieval is a small keyword scorer, not embeddings.
- Isaac Sim needs an RTX-class (ray-tracing) GPU; confirm availability on Nebius.

## Models and services (fill in as used)

- Nebius Token Factory: Nemotron (plan/generate/fast) via the OpenAI-compatible API.
- Nebius AI Cloud: GPU instance or Serverless Jobs for headless Isaac Sim runs.

## Feedback notes for the submission (keep adding as you go)

- Token Factory: _..._
- AI Cloud / Serverless Jobs: _..._
- NVIDIA models/tools: _..._

## License

MIT, see `LICENSE`.
