# Architecture

This is the technical companion to [README.md](../README.md): how each piece
of SimSmith actually enforces "never a dead end, never a fake success," not
just the summary of what it does. Everything here is read from the code as
it stands; where a claim needed checking against an adversarial script, that
audit is `diagnosis_2026-10-24.md` in this same directory — this document
cites it rather than repeating it.

## File map

```
simsmith/            the harness (runs in the parent process, trusted)
  pipeline.py         intake -> plan -> retrieve -> generate -> preflight ->
                      execute -> verify -> repair -> select -> learn
  templates.py        task templates + their fixed, deterministic success checks
  preflight.py        static AST checks, before any simulator time is spent
  runner.py           subprocess execution, env stripping, network isolation
  verifier.py         turns per-seed RunResults into a pass/fail Score
  library.py          JSONL store of verified scripts, used as few-shot examples
  docs.py             keyword retrieval over docs_index/
  llm.py              Token Factory client (cost tracking, retries, budget cap)
  tavily.py           optional, sanitized real-world failure-mode search
  frozen_config.py    enforces config/frozen.json against the live environment
  run_meta.py         per-row provenance stamped onto every result
  config.py           reads .env / environment into a Config

runtime/             the contract (imported INSIDE the generated script's own
                     process; this is the only harness code a candidate script
                     ever runs with)
  simsmith_contract.py  get_task/get_rng/get_perturb/write_result + the
                        mj_step-hooking independent measurement
  simsmith_skills.py    trusted move_to/set_gripper/... primitives for
                        pick_place (skills ablation)

eval/run_eval.py      the four-condition held-out evaluator
config/frozen.json    the single reference config eval/run_eval.py runs under
docs_index/           retrieved reference material (generated + hand-written)
examples/             hand-written reference scripts (never fed to the model)
app/                  local demo UI (stdlib only, mock/live/recorded_live)
docs/diagnosis_2026-10-24.md   a time-boxed adversarial audit of this design
```

## Pipeline walkthrough (`simsmith/pipeline.py`)

`Pipeline.solve(task)` runs, in order:

1. **`intake(task)`** — Nemotron's fast model maps free text to one of
   `TEMPLATES` (`simsmith/templates.py`). On any LLM/parse failure it falls
   back to a keyword matcher (`closest_template`), so intake degrading never
   stops the pipeline — it just loses the "exact match" guarantee and reports
   a gap instead.
2. **`plan(goal, template)`** — a planning completion, optionally extended
   with a block of sanitized, source-cited Tavily hints
   (`_tavily_reference_block`) if a `tavily_client` was explicitly passed in.
   That block is always labeled `UNTRUSTED reference data` in the prompt
   itself — the plan model is told never to let it override the task.
3. **`generate(goal, template, plan, n, use_library)`** — builds `n`
   candidates concurrently (thread pool; this is network-bound, not
   CPU-bound). Each candidate gets a different entry from `FRAMINGS` and
   `TEMPERATURES` (cycled by index, not randomized — the schedule itself is
   deterministic even though the completions aren't), a different windowed
   slice of the retrieved docs, the **unwindowed** scene card for the active
   template (every candidate sees it, regardless of what the keyword ranker
   picked), and — for `pick_place` only — the skills-helpers prompt block and
   reference doc.
4. **`evaluate(c, template)`** (preflight + execute + verify) — see the
   dedicated sections below. Runs once per candidate, across every seed in
   `cfg.seeds`.
5. **Repair** — candidates that didn't pass are sorted by success rate
   descending (closest-to-working first) and up to `max_repairs` of them are
   repaired: the real failure text (preflight problems, or the verifier's
   per-seed failure lines) is fed back to the model, a new candidate is
   produced and evaluated the same way, and appended to the candidate pool
   rather than replacing the original.
6. **Select** — all scored candidates are sorted by `(ok, success_rate,
   -mean_seconds)`; the winner is whichever genuinely scores highest, never a
   vote and never the model's own opinion. `status` is `VERIFIED` only if the
   winner's score passed the bar; `PARTIAL` if the best score is a nonzero
   partial; `FAILED` otherwise. A `VERIFIED` winner is the only thing ever
   written to the library, and only when the caller asked
   (`save_to_library=True`).

## The script contract (`runtime/simsmith_contract.py`)

This is the one piece of harness code that runs **inside** a generated
script's own process — everything else in `simsmith/` stays in the parent.
Because of that, its job is narrower and more defensive than the rest of the
harness: assume the script might be sloppy, not malicious, but never trust
anything it computes about its own outcome.

- **`get_task(template)`** returns this run's target/cube/bin positions,
  derived purely from `task_for(template, seed, perturb)` — a pure function
  with no env reads and no side effects. The verifier calls the *same*
  function independently, from the seed/perturb recorded on the `RunResult`,
  to recompute what the script *should* have seen — so there is exactly one
  formula, used twice, never two formulas that could drift apart.
- **Independent, last-step measurement.** Importing this module hooks
  `mujoco.MjModel.from_xml_path`/`from_xml_string`, `mujoco.MjData`, and
  `mujoco.mj_step` *before* the script creates anything:
  - Every `MjModel` built is checked against a fingerprint (body names, `nq`,
    `nu`, `ngeom`) of the real official scene for the active template — a
    script that builds its own toy world instead of the real one fails
    immediately, with a clear diff.
  - Every `mj_step` call snapshots `xpos`/`qvel`/`qpos` at that exact moment,
    overwriting any earlier snapshot for that `MjData` instance. `write_result`
    measures `ee_pos`/`cube_pos`/`cube_velocity`/`gripper_opening` from the
    snapshot of whichever tracked instance was stepped the most (a throwaway
    IK-scratch `MjData` that's never stepped is naturally ignored) — never
    from live state. Writing to `qpos` directly, or calling an un-hooked
    `mj_forward`, after the last real step changes nothing the harness
    reports.
- **`write_result(final_state)`** overwrites those four outcome fields in
  place with the measured values before writing the result file — whatever
  the script passed for them is informational only and is discarded.

### The skills ablation (`runtime/simsmith_skills.py`)

For `pick_place` only, a script may import `body_pos`/`arm_jacobian`/
`ik_step`/`move_to`/`set_gripper` instead of writing its own IK/stepping
loop. These are primitives, not a plan — nothing here picks a grasp height,
a waypoint, or how many times to re-aim. They matter to the contract in one
specific way: every `mj_step` they take is a real, attribute-looked-up call
on the `mujoco` module (never a cached reference), so it always goes through
whatever hook `simsmith_contract` currently has installed — using the skills
helpers is exactly as "real" to the measurement system as a script stepping
MuJoCo by hand.

## Pre-flight (`simsmith/preflight.py`)

Runs before any simulator time is spent. It is an **AST-literal pattern
check** — explicitly a cheap pre-filter against accidental/careless gaming,
not a sandbox and not a defense against a script an attacker deliberately
wrote to defeat it (see
[Security and integrity](../README.md#security-and-integrity)). What it
checks:

- **Banned modules/calls** (`subprocess`, `socket`, `requests`, `importlib`,
  `eval`, `exec`, `open`, `getattr`/`setattr` outside a narrow safe shape,
  …) and a fixed list of banned attribute calls (`os.system`, `os.popen`, …).
- **No reflection/introspection escapes**: any `obj.__dunder__` attribute
  access is rejected outright, regardless of `obj`.
- **`simsmith_contract`'s private state is off-limits.** A script that does
  `import simsmith_contract` may only touch `get_task`/`get_rng`/
  `get_perturb`/`write_result` on it — any other attribute access (the
  measurement internals, the step-tracking dict) is rejected. The ordinary
  `from simsmith_contract import get_task, ...` form never creates an
  `Attribute` node at all, so normal usage is unaffected.
- **`write_result` cannot be redefined or shadowed** — a script defining its
  own `write_result` (function or plain assignment) is rejected, so it can't
  satisfy "calls write_result by name" while never running the real one.
- **Must actually simulate**: the script must import `mujoco` and call
  `mj_step` (directly, or via the skills helpers' stepping calls) — reporting
  a position without ever stepping physics forward is rejected statically,
  and `verifier.py`'s `MIN_MJ_STEPS` floor catches the case where it stepped
  only a token amount.
- **MuJoCo API hallucination check**: every `mujoco.<attr>` chain the script
  references is resolved against the *actually installed* `mujoco` module by
  walking `getattr`, with `difflib` suggesting the likely intended name on a
  miss — this can't go stale against the pinned version because it isn't a
  hand-maintained symbol list.
- **Undefined-name check**: any `Name` read that's never imported, assigned,
  or a builtin is flagged — deliberately not scope-correct (a name bound
  anywhere in the file counts as defined everywhere), because the one thing
  this check must never do is false-positive on an honest script.
- **Isaac allow-list** (inert unless a caller passes one): if an allow-list
  of module paths is configured, Isaac-namespace imports (`omni`,
  `isaacsim`, `pxr`, `isaaclab`) outside it are rejected. Unused in this
  submission's eval — see README's Known Limitations.

## Verification (`simsmith/verifier.py` + `simsmith/templates.py`)

`verify(template, results)` turns the per-seed `RunResult` list into a
`Score`. For each seed, in order:

1. A crashed run (`ok=False`) counts as a failure with the real error
   attached for repair.
2. Missing required keys in the reported `final_state` → failure.
3. **`_simulation_too_short`**: fewer than `MIN_MJ_STEPS` (100) real steps,
   read from the contract's own step count (never the script's) → failure.
4. **`_task_mismatch`**: recomputes the expected task parameters from
   `(seed, perturb)` via `simsmith_contract.task_for` and compares them
   against what the script reported — any disagreement fails the seed,
   labeled explicitly as "self-reported task parameters are never trusted."
5. The template's own fixed check (`templates.py`) runs against the raw
   final state:
   - **`reach`**: end-effector-to-target distance ≤ 0.02 m.
   - **`pick_place`**: cube moved ≥ 0.10 m from its resting position, within
     0.05 m (xy) of the bin center, within 0.01 m of resting height, moving
     slower than 0.05 m/s, with the gripper open ≥ 0.06 m (i.e. actually
     released, not still clamping the cube up at the drop point).

A `Score.ok` requires zero crashes across all seeds *and* a success rate at
or above `pass_threshold` (0.8). `failure_text` collects the real per-seed
reasons and is exactly what feeds the repair prompt.

## Execution isolation (`simsmith/runner.py`)

Each candidate run is a fresh subprocess, one per (candidate, seed):

- **Environment is stripped to an allow-list** (`PATH`, `HOME`, `LANG`,
  `LC_ALL`, plus the `SIMSMITH_*` variables the contract needs) — not the
  parent's full environment, so no Token Factory or Tavily API key is ever
  visible to a generated script.
- **Best-effort network namespace isolation**: `unshare --net
  --map-root-user` is probed once per process and, where it succeeds, every
  candidate subprocess runs with a loopback-only network interface. Plain
  `unshare --net` needs a privilege an unprivileged process doesn't have, so
  the `--map-root-user` form (enter an unprivileged *user* namespace first,
  then create a network namespace from inside it) is tried first.
  `SubprocessRunner.network_isolated` reports which case actually applied,
  rather than assuming isolation happened.
- **Timeouts never count as a pass**: a `TimeoutExpired` returns `ok=False`
  before any result file is even checked for.
- Full stdout/stderr/exit code are always captured (capped at 20k chars per
  field when persisted via `candidate_log()`), so a crash can be diagnosed
  after the fact without rerunning it.

This is a real, partial mitigation — not a sandbox. The script still runs as
a plain subprocess with full filesystem read access and no CPU/memory
limits. See [Security and integrity](../README.md#security-and-integrity).

## The held-out evaluator (`eval/run_eval.py`)

Four conditions (`CONDITIONS` dict), varying only `n_candidates`/
`use_library`/`do_repair` — everything else about how a task-run is solved
is identical, enforced by `simsmith/frozen_config.py`:

- **`config/frozen.json`** is the single source of truth for model, reasoning
  mode, max-tokens-per-call-kind, the temperature schedule, seeds, worker
  count, and prompt version. `require_config_matches_frozen` is called
  before any network call and raises `ConfigDrift` (exit code 2, not just a
  warning) if the live environment disagrees in any field outside a
  condition's own intended variable.
- A hash of the frozen config (`config_hash`) is stamped into every result
  row via `run_meta.build_run_meta`, alongside the model, prompt version,
  classifier version, seeds, worker count, and git commit — a row can always
  be tied back to exactly the config and code that produced it, even across
  a resumed, multi-invocation run.
- **Library integrity**: the library used by `best_of_5_repair_library` is
  built *once*, from `tasks["dev"]` only, before any held-out task-run
  starts; held-out solves always pass `save_to_library=False`, and the other
  three conditions get a Library instance pointed at an unused, never-read,
  never-shared path — enforced at the data level, not just by a flag.
- **Budget protection**: `--max-runs` (simulator executions) and `--max-usd`
  (spend, via `TokenFactoryLLM.spent_usd`/`budget_usd`) are both checked
  before each new job is submitted, inside a bounded sliding window of at
  most `--workers` in-flight jobs — so a cap can be overshot by at most one
  window's worth of already-running work, never by the whole remaining job
  list. A truncated run says so explicitly in the report, including which
  cap triggered it.
- **Resumability**: `--progress-file` checkpoints after every task-run
  (`(kind, condition, task_id, repeat_idx)`-keyed); rerunning the same
  command after a crash or Ctrl-C loads completed rows from disk and only
  submits what's left, including the dev-task library build.
- **Cost estimation vs. measurement**: `print_cost_estimate` is a
  pre-run estimate only, from this project's own historical measured
  per-task-run costs (or a documented fallback/scaling when a
  condition/template combination hasn't been measured yet) — it never
  substitutes for the real, measured `cost_usd` that ends up in the report.

## Honest gaps (beyond README's Known Limitations)

- Pre-flight and the contract close every *in-process* self-grading path
  found during the adversarial audit in `diagnosis_2026-10-24.md` (Part 4):
  monkeypatching the measurement function, injecting a fake tracked-step
  entry, and shadowing `write_result` are all now rejected statically, by
  banning any non-public attribute access on `simsmith_contract` and any
  redefinition of `write_result`. None of this is process isolation — it is
  pattern-matching against the specific exploits that were actually tried.
  A deliberately adversarial script author who finds one more level of
  indirection is not covered; see the Security section in README.md.
- The eval harness's own cost estimator is explicitly labeled as historical
  means, not a guarantee — real per-run cost varies with how many repairs a
  task-run actually needs, and this is surfaced to the user rather than
  smoothed over.
- LLM sampling is not server-side seeded, so no eval run is bit-for-bit
  reproducible even from an identical `config/frozen.json` — only the
  harness-side schedule (seeds, framing/temperature cycling, which docs/
  library entries get retrieved) is deterministic.
