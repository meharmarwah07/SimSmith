# SimSmith

*Nebius × NVIDIA Hackathon — Physical AI track*

Describe a robot task in plain language. SimSmith has Nemotron write several
MuJoCo scripts, runs every one of them for real, **verifies the outcome by
execution** (never by asking the model), repairs the ones that fail, and keeps
whatever actually worked in a library that improves later runs.

## The problem

An LLM can write simulation code that runs without a single error and still
never accomplishes the task — a plausible-looking script is not evidence that
the robot did anything. Worse, a model asked to grade its own work will
sometimes just say it worked: echo the target back as the answer, report a
success flag, or quietly build a toy scene instead of the real one. For a
robot-sim pipeline, "the code ran" and "the task succeeded" are different
claims, and only one of them is safe to take on the model's word. SimSmith's
only job is to never confuse the two.

## The loop

1. **Intake** (Nemotron, fast model) maps the request to a supported task
   template (`reach`, `pick_place`). An out-of-scope request gets the closest
   template plus an explicit gap statement — never a silent substitution.
2. **Plan** (Nemotron, plan model) sketches an approach for the matched
   template.
3. **Retrieve** pinned-version MuJoCo API notes (`docs_index/`) and any
   verified scripts already in the library.
4. **Generate 5 candidates** (Nemotron, generate model, Nebius Token Factory)
   — each with a different framing, temperature, and slice of the retrieved
   context, so the five are genuinely different attempts, not five rolls of
   the same prompt.
5. **Pre-flight** (no GPU): a static check rejects a candidate before any
   simulator time is spent on it — banned imports/calls, must actually import
   MuJoCo and step physics forward, must call the contract's `write_result`,
   and (if configured) an API allow-list.
6. **Execute** each surviving candidate, per seed, **headless in MuJoCo**, in
   its own subprocess.
7. **Verify**: a deterministic, harness-owned check per template decides
   pass/fail from the raw final simulator state — the script never grades
   itself.
8. **Repair**: up to 2 rounds, feeding the real failure output (stderr,
   traceback, or the verifier's own explanation) back to the model for the
   candidates closest to passing.
9. **Select and learn**: the best candidate that actually passed wins, by
   execution, not by a vote or a model's opinion — and only then is it saved
   to the verified library.

Stage 7a (Tavily-grounded real-world failure-mode hints into the plan step)
exists and is wired in, off by default, and excluded from every official eval
condition — see [Tavily](#tavily-optional-off-by-default) below.

## Never a dead end, never a fake success

Two rules hold for every single task-run, win or loss:

- **Never a dead end.** Every task-run produces a result — `VERIFIED`,
  `PARTIAL`, or `FAILED` — and that result (including the losing candidates'
  code, preflight problems, and per-seed stdout/stderr) is written to disk.
  Nothing is silently dropped, and no report omits a condition because it
  lost.
- **Never a fake success.** A script cannot report its own success — only
  raw final poses. Three things make that stick:
  - **Harness-owned task parameters.** Target/cube/bin positions come from
    `get_task(template)`, derived purely from the run's seed. The verifier
    recomputes the same parameters independently and fails any run whose
    reported value disagrees — a script that does `target_pos = ee_pos` is
    caught, not trusted.
  - **Independent, last-step measurement.** Outcome fields (`ee_pos`,
    `cube_pos`, `cube_velocity`, `gripper_opening`) are re-measured by the
    harness itself from a snapshot taken at the script's *last real*
    `mj_step` call — not from whatever the script passes to `write_result`,
    and not from live simulator state when that call happens. Writing to
    `qpos` after the last real step, or calling an un-hooked `mj_forward`,
    cannot move the reported pose.
  - **A simulation floor.** A script that imports MuJoCo and technically
    calls `mj_step` but barely moves the clock forward (e.g. to tick a
    pre-flight box) is rejected — too few real steps means it never actually
    simulated.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for exactly how the contract,
pre-flight, and verifier enforce this, and for the audit trail
(`docs/diagnosis_2026-10-24.md`) of the adversarial scripts tried against it.

## Results

{{RESULTS_TABLE}}

**Reach is the headline template** — it is small (one target point, 7 DOFs),
and is where the pipeline is expected to look strongest.

**`pick_place` is experimental and currently failing in dev-task regeneration**
(0/24 verified with the `pick_place` skills helpers enabled, 0/24 verified in
an earlier run without them) — it needs a 15-DOF scene (arm + gripper + the
cube's free joint), a grasp, and a release, and the generated candidates have
not yet cleared that bar. This is reported as a negative result, not hidden.

**Newton is a second, hand-written backend, not an LLM-generated one.**
`examples/reach_newton.py` is a hand-written reference script against Newton
1.7.0 (real rigid-body dynamics via `SolverXPBD`), used to validate that the
contract and verifier concepts aren't MuJoCo-specific — Nemotron does not
generate Newton code, and Newton is not part of the eval conditions below.

**Tavily is optional and off by default** (see
[below](#tavily-optional-off-by-default)) and is excluded from all four
official held-out conditions regardless of environment configuration.

**Held-out sample size**: `eval/tasks.json`'s held-out set has 6 tasks, 3 per
template (`reach`: h1–h3, `pick_place`: h4–h6). The default evaluation repeats
each held-out task 5 times per condition — an effective sample of 3 tasks ×
5 repeats per template per condition. This is small; treat differences below
roughly 2 tasks' worth of outcomes as noise, and read every number in
{{RESULTS_TABLE}} with that in mind.

## How to run it

```bash
git clone <this repo> && cd simsmith
pip install -r requirements.txt       # stdlib runtime; pytest is the only dev dependency
cp .env.example .env                   # fill in Token Factory URL/key, model IDs, SIM_PYTHON
python -m pytest -q                    # runs fully offline, no GPU or network needed
```

`.env.example` lists every variable name the project reads (Token Factory
base URL/key, the three Nemotron model IDs, the sim Python launcher, per-run
limits, per-tier pricing for cost tracking, and the optional Tavily toggle) —
copy it to `.env` and fill in your own values; `.env` is git-ignored and
never committed.

Put plain-text/markdown excerpts of the docs for your **pinned** MuJoCo
version in `docs_index/` (mixing versions is the main source of hallucinated
APIs) — `python scripts/gen_mujoco_api_doc.py` regenerates the introspected
API reference from whatever `mujoco` is actually installed, so it can't drift
from the pinned version.

### Day-1 gate

Before spending real budget on an eval, confirm the whole chain actually
works end to end against the real Token Factory and a real simulator:

```bash
python scripts/gate_day1.py
```

It checks Token Factory connectivity for all three model roles, that the sim
Python launcher can import the contract, that a hand-written reference script
(`examples/reach_mujoco.py` by default) runs and verifies, and that a
Nemotron-generated script for the same task at least runs — then prints
seconds/run, which is what determines how large an eval you can afford.

### Demo UI (optional)

```bash
python -m app.server --mock --port 8000    # scripted LLM + real installed mujoco, no API calls
```

Every run the UI shows carries a `source` field (`mock` / `live` /
`recorded_live`) and an unmissable badge saying which — a mocked run is never
presented as if it used the real model.

## Reproduce

The held-out evaluation lives in `eval/run_eval.py`, driven by the single
frozen reference config in `config/frozen.json`:

```bash
python -m eval.run_eval --dry-run              # prints the task-run/cost estimate, makes no network call
python -m eval.run_eval --repeats 5 --workers 4 --yes
```

`config/frozen.json` pins every value that must be identical across the four
conditions below for the comparison to mean anything — model, reasoning mode,
max tokens, the temperature schedule, seeds, worker count, and prompt
version. `eval/run_eval.py` checks the live environment against it before
making a single call and **refuses to run** on any drift outside a
condition's own intended variable (`n_candidates`, `max_repairs`); it stamps
a hash of the frozen config into every result row so a report can always be
tied back to exactly the config that produced it.

The four official held-out conditions:

| Condition | Candidates | Repair | Library |
|---|---|---|---|
| `single_shot` | 1 | no | no |
| `best_of_5` | 5 | no | no |
| `best_of_5_repair` | 5 | up to 2 rounds | no |
| `best_of_5_repair_library` | 5 | up to 2 rounds | yes (built from **dev** tasks only, then frozen) |

Held-out task-runs never write to the library — it is built once from dev
tasks before any held-out measurement starts. A hard cap on total simulator
executions (`--max-runs`) and, optionally, total spend (`--max-usd`) protect
the budget; if either is hit, the report says so explicitly and names which
cap, rather than silently presenting partial numbers as complete. The run is
resumable: `--progress-file` checkpoints after every task-run, so a crash or
Ctrl-C can be resumed by rerunning the same command without re-paying for any
task-run already completed. All results — wins and losses — are written to
`eval/results/`.

### Tavily (optional, off by default)

`simsmith/tavily.py` can fetch sanitized, source-cited real-world
failure-mode hints into the plan-step prompt. It is opt-in
(`SIMSMITH_TAVILY=1` plus `TAVILY_API_KEY`), feeds only `scripts/baseline_dev.py`
and `scripts/baseline_best_of_5.py`, and is **never** used by
`eval/run_eval.py`'s four official conditions regardless of how the
environment is configured — only a separately requested `tavily_on` condition
would ever call it there. Treat any number produced with Tavily enabled as a
distinct, clearly-labeled condition, not a variant of the official ones.

## Security and integrity

**Threat model: accidental gaming by an honest-but-sloppy model, not a
deliberately adversarial script author.** Every defense below exists to stop
an LLM from *accidentally or lazily* gaming its own grade — echoing a target
back as the answer, self-reporting success, drifting into an API it doesn't
actually have — not to withstand a human attacker deliberately crafting a
script to defeat the harness.

- **Harness-owned parameters, independent re-measurement.** As described
  above: task parameters and outcome fields are never trusted from the
  script's own report.
- **Static pre-flight checks** (`simsmith/preflight.py`) are AST-literal
  pattern matches — a cheap pre-filter, not a sandbox. They ban a fixed list
  of modules/calls/attribute accesses, reject any access to
  `simsmith_contract`'s private state (only `get_task`/`get_rng`/
  `get_perturb`/`write_result` are allowed), and reject a script that
  redefines `write_result` itself. An audit of exactly which bypasses this
  closes — and which remain open regardless, because they require real
  process isolation — is in `docs/diagnosis_2026-10-24.md`.
- **Process-level mitigations, not a sandbox.** `simsmith/runner.py` strips
  the child process's environment to a small allow-list (`PATH`, `HOME`,
  `LANG`, `LC_ALL`, and `SIMSMITH_*`) so no API key reaches a generated
  script, and best-effort isolates its network namespace
  (`unshare --net --map-root-user`) — but the script still runs as a plain
  subprocess with full filesystem read access and no CPU/memory limits.
- **If this is ever pointed at untrusted or adversarial model output** (not
  just "an LLM trying its honest best"), running candidate scripts in a real
  sandbox — container, VM, unprivileged user, read-only filesystem — is a
  prerequisite, not an enhancement. That work is not done here.

## Known limitations

- Two task templates (`reach`, `pick_place`), one robot (Franka Panda), one
  pinned MuJoCo version.
- `pick_place` does not currently pass verification in dev-task regeneration
  (see [Results](#results)) — it is included for completeness and honesty
  about the harness's current limits, not as a working capability.
- Success checks are simple geometric/kinematic thresholds (distance, height,
  speed, gripper opening) — they judge task completion, not motion quality,
  safety, or elegance.
- Passing inside a simulator establishes nothing about real-world transfer.
- Retrieval over `docs_index/` is a small keyword scorer, not embeddings.
- No eval run is bit-for-bit reproducible even from an identical config — the
  harness side (seeds, framing/temperature schedule, which docs/library
  entries) is deterministic, but the LLM completion itself is sampled and not
  server-side-seeded.
- Scaffolding for an Isaac Sim backend exists (`simsmith/preflight.py`'s
  Isaac API allow-list path, `eval/run_eval.py --isaac-version`) but is inert
  and untested here — MuJoCo is the only backend Nemotron actually generates
  code for in this submission.
- Pre-flight's AST checks are a pre-filter against sloppy generation, not a
  security boundary — see [Security and integrity](#security-and-integrity).

## Models and services

- **Nebius Token Factory**: Nemotron (plan / generate / fast roles) via the
  OpenAI-compatible chat-completions API.
- **Nebius AI Cloud**: the GPU/CPU instance that actually runs the headless
  MuJoCo subprocesses (`SIMSMITH_SIM_PYTHON`).

## License

MIT, see [LICENSE](LICENSE).
