# SimSmith diagnosis — 2026-10-24

Scope: time-boxed (~40 min target), budget cap $1 for any new LLM/sim spend. Held-out
tasks were not read or run. No fixes were applied except Part 0 (logging), as instructed.

**Note on repo state:** the codebase was being actively edited (by the user, in parallel
with this review) while this diagnosis was written — `simsmith/docs.py`, `simsmith/pipeline.py`,
`simsmith/prompts.py`, and `docs_index/scene_{reach,pick_place}.md` all changed on disk
mid-review (a new "scene card" feature, `PROMPT_VERSION` bumped `mujoco-v2` → `mujoco-v3`).
Findings below reflect the code as it stood after those changes. Re-run if the repo has
moved further since.

---

## Part 0 — Logging (implemented)

Previously, `RunResult` only kept a truncated tail of stderr on failure, and the
`baseline_dev.py` / `baseline_best_of_5.py` per-candidate rows recorded only a
classification label, not the candidate's code or what actually happened per seed. A
crash could not be diagnosed after the fact without rerunning it.

Changes:
- `simsmith/runner.py`: `RunResult` now always carries `exit_code`, full `stdout`, and
  full `stderr` (not just a 1500-char tail on failure), for every outcome including
  timeouts.
- `simsmith/pipeline.py`: `Candidate.runs` now holds the raw per-seed `RunResult` list
  (previously discarded after `verify()` reduced it to a `Score`). Added
  `candidate_log(c)`, returning `{script, stage, preflight_ok, preflight_problems, runs:
  [{seed, ok, exit_code, seconds, error, stdout, stderr}, ...]}`.
- `scripts/baseline_dev.py`, `scripts/baseline_best_of_5.py`: each candidate's row now
  includes `candidate_log(c)` (stdout/stderr capped at 20k chars each to keep result
  files sane).
- Test added: `tests/test_core.py::test_runner_captures_exit_code_and_full_stdout_stderr`
  and `::test_candidate_log_persists_full_code_preflight_and_per_seed_runs`.

Full suite: **77 passed** (`python3 -m pytest tests/ -q -p no:anyio`) after the change.

**Not yet covered:** `eval/run_eval.py` (the three-condition held-out evaluator) still
only records the winner's score, not per-candidate logs. Scoped out here since the
instruction was logging in "the results JSON" for the Part 1 regeneration path
specifically; if `run_eval.py` is going to be used for the real held-out run, it needs
the same treatment before that run, not after.

---

## Part 1 — pick_place root-cause breakdown

Regenerated 12 dev pick_place task-runs (`d4`/`d5`/`d6`, 4 runs each) with
`best_of_5` + up to 2 repairs, `nvidia/nemotron-3-super-120b-a12b`, reasoning off
(`enable_thinking: false`), using the new Part 0 logging. Full report:
`eval/results/best_of_5_1792853506.json` (tag: `diagnosis-2026-10-24: part1 12
best_of_5 pick_place dev, super, reasoning_off, new logging`). Cost: **$0.089** total
(mean $0.0074/task-run).

**Result: 0/12 VERIFIED.** Across all 12 task-runs' 84 candidates (generated + repaired):
`crash` 40, `wrong_logic` 27, `hallucinated_api` 14 *(see caveat below — this number is
not trustworthy)*, `task_param_mismatch` 1, `preflight_other` 1, `truncation` 1.

### Crashes (40), by actual root cause (read from the new per-seed stdout/stderr, not the `classify()` label)

| Count | Root cause | Representative traceback |
|---|---|---|
| 20 (50%) | **Variable-name corruption**: the model writes the correct fallback assignment `simsmith_root = os.environ.get('SIMSMITH_ROOT') or os.path.dirname(...)`, then references it two lines later under a *different, similar-sounding* name it never defined (`smithsonian_root`, `smithson_root`, `smithsmith_root`, `smithith_root`, `smithth_root`, `simsmth_root`...) — a different garbled spelling nearly every time. Happens in both `generated` and `repaired` candidates (repair regenerates from the same model and reproduces the same failure mode), across all 3 dev pick_place tasks. | `xml_path = os.path.join(smithsonian_root, 'assets', ...)` → `NameError: name 'smithsonian_root' is not defined` |
| 18 (45%) | **IK Jacobian/qpos dimension mismatch, specific to pick_place**: `model.nv` for `scene_pick_place.xml` is **15** (7 arm + 2 finger + 6 from the cube's free joint), not 7. The model repeatedly allocates `jacp = np.zeros((3, model.nv))`, solves the full 15-column Jacobian without slicing to the arm's first 7 columns, and gets a 15-dim `dq` it then tries to add to `data.qpos[:7]` or `data.ctrl[:7]`. This can't happen on `reach` (no cube, `nv=7`), which is presumably why it wasn't caught before. | `q_new = data.qpos[:7] + dq * 0.1` → `ValueError: operands could not be broadcast together with shapes (7,) (15,)` |
| 1 | Wrong API usage: `mujoco.mjtObj[type_str.upper()]` (subscripting the enum type instead of attribute access, e.g. `mujoco.mjtObj.mjOBJ_BODY`) | `TypeError: 'pybind11_type' object is not subscriptable` |
| 1 | Gripper control: off-by-one actuator index (`data.ctrl[finger_joint2_id]` where `finger_joint2_id` is a *joint* id, not the gripper's single combined actuator index 7) | `IndexError: index 8 is out of bounds for axis 0 with size 8` |

**Not observed this run**: wrong body/joint names (the new scene cards + `mujoco_api.md`
appear to be doing their job here — no `mj_name2id` returning `-1` or
`KeyError`/unknown-name crashes), gripper-missing-release, or bare timeouts.

**Actionable fix, highest leverage**: the #1 cause (50% of all crashes) is entirely a
prompt-format issue, not a reasoning failure — `GENERATE_SYSTEM` currently *describes*
the `SIMSMITH_ROOT` fallback in prose ("read it with os.environ, with
`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` as a local fallback")
instead of giving a literal line to copy. Replacing that with an exact snippet, e.g.
```python
SIMSMITH_ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
removes the model's need to invent-and-then-recall a variable name across lines, which
is exactly where it's drifting. This alone would plausibly fix half of this run's crashes
for near-zero cost.

**Second fix**: add an explicit Jacobian-slicing gotcha to `docs_index/mujoco_notes.md`
(which already has related IK gotchas) and/or `docs_index/scene_pick_place.md` (which
already prints `nv=15` in its header but doesn't explain why that matters for IK):
*"`mj_jacBody`/`mj_jac` fill `jacp`/`jacr` with `model.nv` columns — for `pick_place`
that's 15, not 7, because it includes the cube's free-joint DOFs. Always slice to
`jacp[:, :7]` (and `jacr[:, :7]`) before solving for the arm's 7 joint velocities, or
the solve's output can't be added to `qpos[:7]`/`ctrl[:7]`."* This is exactly the
pattern the project's own `examples/pick_place_mujoco.py` already does correctly
(`J = np.vstack([jacp[:, :7], jacr[:, :7]])`) — the generated scripts just aren't
reliably doing it, and nothing currently tells them to.

**Caveat — the `hallucinated_api` classification in `classify()` (`scripts/baseline_dev.py`)
is unreliable.** It matches `"attributeerror" in text and "module" in text`. Checked
all 14 candidates this run labeled `hallucinated_api`: **14/14 are false positives** —
every Python traceback's stack frame is literally rendered as `in <module>`, so the
substring match fires on *any* `AttributeError` crash, regardless of whether a module
attribute was involved. E.g. one of the 14 is simply `AttributeError: 'list' object has
no attribute 'tolist'` (calling `.tolist()` on a plain list) — nothing to do with a
hallucinated MuJoCo API. **This means every historical baseline/best_of_5 report's
`hallucinated_api` count (including the pre-existing files in `eval/results/`) is
likely substantially overstated**, and the real hallucinated-API rate for this run is
closer to **0%**, not the reported ~17% (14/84). This should be fixed and the historical
numbers re-read with that in mind before anyone draws conclusions from them.

---

## Part 2 — Security review of generated-script execution

**Summary: generated scripts run with no sandboxing at all.** They get the full parent
environment (including API keys), full filesystem read access, and — because the
pre-flight checker only pattern-matches literal `import X` / `X.Y(...)` syntax — the
module/call bans are trivially defeated by one level of indirection. All items below were
verified by actually running the PoC script through `simsmith.preflight.check` +
`simsmith.runner.SubprocessRunner.run` (not just read from source).

| # | Question | Finding |
|---|---|---|
| 1 | Does the runner pass the parent env to scripts? | Yes. `runner.py`: `env = dict(os.environ)` — the full parent process environment, unfiltered, including `TOKEN_FACTORY_API_KEY`/`TOKEN_FACTORY_BASE_URL` (and `TAVILY_API_KEY` or anything else present in whatever shell/service runs the pipeline). PoC `env_leak_api_key` confirmed `os.environ` is fully visible inside the subprocess. |
| 2 | Can a script read `.env`? | Yes. `open` is not banned (only the call named `open_url` is, which doesn't exist in Python — looks like a typo for `open`). PoC `read_dotenv_file` read `.env` directly and found `TOKEN_FACTORY_API_KEY` in it. |
| 3 | Can a script read `~/`? | Yes, same `open`/`os.listdir` gap. PoC `read_home_ssh` listed the real `$HOME` (`.ssh`, `.gitconfig`, `.bash_history`, etc. all enumerable/readable). |
| 4 | What do the preflight bans cover? | `BANNED_MODULES` = `subprocess, socket, requests, urllib, http, ftplib, shutil, ctypes, multiprocessing, pty` (direct `import`/`from X import` only). `BANNED_CALLS` = `eval, exec, compile, __import__, open_url, input` (direct bare-name calls only). `BANNED_ATTR_CALLS` = a fixed list of `os.system`/`os.popen`/`os.remove`/`os.rmdir`/`os.unlink`/`os.kill`, matched only as a literal `os.<name>(...)` AST attribute call. `importlib` and `os` themselves are **not** banned. |
| 5 | Does `importlib` bypass the module ban? | Yes. PoC `importlib_bypass_socket` did `importlib.import_module("socket")` then opened a real TCP connection to `1.1.1.1:80` — preflight passed, the connection succeeded. PoC `importlib_bypass_urllib` did the same via `importlib.import_module("urllib.request")` and fetched `http://example.com` for real. Any banned module is reachable this way. |
| 6 | Does a getattr trick bypass `BANNED_ATTR_CALLS`? | Yes. PoC `os_system_via_getattr` ran `getattr(os, "sys"+"tem")("id > ...")` — not a literal `os.system(...)` AST node, so it's invisible to the check. Arbitrary shell command execution confirmed (exit 0, file written). The same trick defeats the `eval`/`exec` ban (`getattr(builtins, "ev"+"al")(...)`). |
| 7 | Can internal harness state be reached and mutated? | Yes, and this is the most serious issue — see Part 4. `import simsmith_contract` inside the generated script gets a live, mutable reference to the exact module the runtime hook and the measurement logic live in. Nothing stops a script from monkeypatching `simsmith_contract._measure_from_mujoco`, writing directly into `simsmith_contract._MJ_TRACKED`, or shadowing `write_result` — all three fully defeat the "independent re-measurement" design (Part 4, findings #1-3). |

### Recommended fixes, ranked by severity

1. **Sandbox script execution** (highest). Run candidate scripts as an unprivileged
   user/container/VM with no network namespace and a scratch-only filesystem mount,
   not as a plain subprocess of the pipeline process. This is the only fix that closes
   #1, #3, #5, #6, and the `simsmith_contract` mutation issue (#7) at once — pattern-matching
   the AST can never be complete (see #5/#6), so the preflight checker should be treated
   as a cheap pre-filter, never the security boundary.
2. **Strip the environment passed to the child** (cheap, do regardless of sandboxing).
   `runner.py` should pass an explicit allow-list (e.g. just `PATH`, `PYTHONPATH`,
   `SIMSMITH_*`), not `dict(os.environ)`. Closes #1 immediately.
3. **Fix `open_url` → `open`** in `BANNED_CALLS`, and ban `getattr`/`setattr` entirely
   (or at minimum flag any `getattr(module_obj, non_literal_string)` pattern) — closes
   #2/#3 partially and #6, though it's still an incomplete AST-level patch, not a
   real fix (see #1 above for why).
4. **Make `importlib` itself banned**, and recursively re-check any string literal
   passed to `importlib.import_module(...)` against `BANNED_MODULES` — closes #5. Still
   bypassable by `importlib.import_module("imp"+"ortlib")`-style string-building, so
   this is a stopgap, not a fix.
5. **Protect `simsmith_contract`'s own integrity in-process** is not realistically
   fixable while scripts run in the same OS process tree with a live Python import of
   it — Python does not enforce module-private state against a cooperating importer.
   This is really the same finding as #1 (sandboxing): the fix is process isolation, not
   a smarter module. See Part 4 for the concrete exploits.

---

## Part 3 — Eval integrity

Checked every path by which held-out task info or a hand-written solution could reach
the generator:

- **`docs_index/`**: `mujoco_api.md` (introspected function signatures/docstrings) and
  the new `scene_reach.md` / `scene_pick_place.md` (introspected body/joint/actuator/
  keyframe tables) are both generated by introspecting the installed `mujoco` module /
  real scene XML, not hand-typed, and both explicitly avoid including IK/control code or
  the numeric success thresholds (those stay in `simsmith/templates.py`, never retrieved
  into a prompt). `mujoco_notes.md`/`newton_notes.md` are hand-written API gotchas, no
  worked solutions. None of this is task-specific — it's the same for every task on a
  given template, dev or held-out, so it isn't a leak of held-out-specific information;
  it's reference material, by design. **Minor accuracy issue**: `mujoco_notes.md` says
  "pinned: mujoco 3.16" but the installed version in this environment is actually
  **3.15.0** — if that's also true in the real eval environment, the model is being told
  the wrong pinned version, which is exactly the inconsistency that file's own job is to
  prevent.
- **Scene cards** (new, `scripts/gen_scene_cards.py` → `DocIndex.scene_card()`, injected
  unconditionally into every generate/repair prompt as of `PROMPT_VERSION = mujoco-v3`):
  reviewed the generator and its output. It's facts about the scene XML (names, qpos/dof
  layout, ctrlrange, keyframe values) plus "where `get_task()` values apply" — no IK
  code, no thresholds. Clean. Note it bypasses `docs.retrieve()`'s keyword ranking
  entirely (always included, template-keyed, not task-keyed) — correct for what it's
  for, but worth knowing it's not subject to the same "windowed per candidate" treatment
  as the rest of the retrieved docs.
- **Library** (`simsmith/library.py`): only ever populated by `Library.add()`, called
  from `Pipeline.solve()` only when `save_to_library=True`. `eval/run_eval.py` sets
  `save_to_library=True` only for dev-task solves building the frozen library, and
  `save_to_library=False` for every held-out solve — confirmed by reading `run_eval.py`
  directly (not just the docstring's claim). No committed/pre-seeded `library.jsonl`
  exists in the repo. Clean.
- **`examples/`** (hand-written `reach_mujoco.py`, `pick_place_mujoco.py`, etc.): grepped
  every reference across `simsmith/`, `eval/`, `scripts/` — the only consumers are
  `tests/mocks.py` and `tests/test_core.py`. `DocIndex` only reads from `docs_index/`,
  `Library` only reads from its own `.jsonl`. These full worked solutions never reach
  the generator. Clean.
- **Prompts** (`simsmith/prompts.py`): `GENERATE_SYSTEM`/`REPAIR_SYSTEM` describe the API
  contract and known body/joint names generically; no task-specific numbers. Clean.

**Prompt version / model recorded per result:**
- `scripts/baseline_dev.py` and `scripts/baseline_best_of_5.py` both stamp
  `prompt_version`, `model`, `sim_python`, `sim_seeds`, `tag`, `extra_payload` at the
  **top level** of their output report. Gap: this is per-run-invocation, not per-row —
  both scripts support resuming from a `--progress-file` across separate invocations
  (e.g. after an interruption), and if a resumed run uses a different `--model-generate`
  or prompt version than the original, the rows silently merge under one report with no
  per-row marker of which config actually produced which row.
- `eval/run_eval.py`'s report (the actual three-condition held-out evaluator) records
  **none** of this — no `prompt_version`, no `model`, no `sim_python`. An `eval_<ts>.json`
  file cannot be tied to the prompt/model config that produced it at all without cross-
  referencing shell history or `usage.jsonl` timestamps by hand. This is the eval script
  most likely to produce the numbers that end up in a final report, and it's the one
  missing this.

**Condition parity** (`eval/run_eval.py` `CONDITIONS` dict): `single_shot`
(`n_candidates=1, use_library=False, do_repair=False`), `best_of_5`
(`n_candidates=5, use_library=False, do_repair=True`), `best_of_5_library`
(`n_candidates=5, use_library=True, do_repair=True`) — all three go through the same
`Pipeline`/model/temperature-and-framing schedule, varying only the documented
intended variable. One minor wrinkle: `Pipeline.generate()` fixes the single candidate's
temperature at `0.3` when `n<=1`, whereas `best_of_5`'s first candidate (same framing,
index 0) runs at `0.2` (`TEMPERATURES[0]`) — a small, probably-intentional but
undocumented difference that means `single_shot` isn't quite "best_of_5's first
candidate in isolation" if anyone goes on to make that comparison.

---

## Part 4 — Verifier robustness

Five new adversarial scripts (not already in `tests/mocks.py`) run through the real
`preflight.check` → `SubprocessRunner.run` → `verifier.verify` pipeline, against the
official `reach` scene:

| # | Script | What it does | Result |
|---|---|---|---|
| 1 | `monkeypatch_measurement` | `import simsmith_contract as sc; sc._measure_from_mujoco = lambda s: None`, then runs the real scene for 150 real steps without ever moving the arm, then calls the real `write_result()` with a lie (`ee_pos = target_pos`). | **PASSES.** `success_rate=1.0`. The re-measurement that's supposed to make self-reported poses untrustable never ran, because the script disabled it before calling `write_result`. |
| 2 | `fake_mj_tracked_entry` | Runs the real scene honestly (arm stays at rest), then injects a fabricated entry into `simsmith_contract._MJ_TRACKED` claiming 10,000,000 steps and a snapshot with `hand` already at the target. | **PASSES.** `_sim_steps` reported as `10000000`, `success_rate=1.0`. `_busiest_mj_instance()` picks the fabricated entry because nothing validates that a tracked entry actually came from a real hooked `mj_step` call. |
| 3 | `shadow_write_result` | Defines a local no-op `def write_result(...)`  (satisfies the static "calls write_result" pre-flight check by name only), does 150 real but irrelevant steps, then writes the result file **directly** with `json.dump(...)`, bypassing `simsmith_contract` entirely. | **PASSES.** `success_rate=1.0`. Confirms the static check only verifies a name is called, never which function actually binds to that name, and never that the real contract module's `write_result` is what runs. |
| 4 | `swallow_exception_fake_success` | Runs the real scene, hits a genuine crash (simulated here as a deliberate `AttributeError`), catches it, and calls the **real** `write_result()` with a fabricated success state in the `except` block. | **Correctly fails** (`success_rate=0.0`). Because the real `write_result()` ran, the independent re-measurement overwrote the lie with the arm's true (unmoved) position. This specific cheat is already closed, as long as the script calls the real `write_result`. |
| 5 (control) | `CHEAT_QPOS_AFTER_LAST_STEP` from `tests/mocks.py`, rerun through this same harness | Known-blocked exploit (fakes `qpos` + calls the un-hooked `mj_forward` after the last real `mj_step`). | **Correctly fails**, confirming the harness isn't just rubber-stamping everything — the last-step-snapshot design does work for the case it was built for. |

**3 of 5 novel adversarial scripts fully defeat the verifier**, reporting a 100% success
rate despite the arm never moving. All three share one root cause: a generated script
runs `import simsmith_contract` in the **same process** as the harness's own hooking
logic, so it has a live, writable reference to every "private" name (`_measure_from_mujoco`,
`_MJ_TRACKED`) and can shadow `write_result` outright. This is the same root cause as
Part 2 finding #7 — it is a process-isolation gap, not a logic bug in the measurement
code itself (the measurement code is in fact correct and hard to fool *if it actually
runs*, as shown by #4 and the control case).

Other items from the prompt, checked and found **not** currently exploitable:
- **Timeouts counted as pass?** No. `runner.py` returns on `TimeoutExpired` before ever
  checking for a result file; a timed-out run is always `ok=False`.
- **Perturbation hook misuse?** Not currently live — `pipeline.py`'s own header says
  stage 7b (perturbation robustness) isn't implemented yet, and no caller ever passes a
  non-empty `perturb` into `runner.run()`. `get_perturb()` is wired through and callable
  by a script, but there is no mechanism today for a script to detect "this is a
  robustness/adversarial run" and behave differently, because there's no such run yet.
  Worth re-checking this specific item once 7b ships.
- **Reading `get_task()` and hardcoding instead of using it live?** Not exploitable: the
  verifier recomputes expected values independently from `(seed, perturb)` via the same
  pure `task_for()` function, in the parent process, regardless of what the child did or
  didn't call.

### Recommended fix
Same as Part 2's top recommendation: run candidate scripts out-of-process (sandboxed),
so a script's `import simsmith_contract` binds to a *different* module instance than the
one whose state the harness actually trusts — or, short of full sandboxing, have
`write_result()`'s outcome measurement happen in a **separate process** from the
candidate script (e.g. the harness re-opens the result file's declared model path itself
and re-simulates/re-reads state, rather than trusting anything computed inside the
candidate's own interpreter). Patching individual exploits (e.g. "also check
`_MJ_TRACKED` entries have a real `data` object") is a losing game against a script that
can `import` and mutate the same module.

---

## Part 5 — Reproducibility

- **Seeds**: `Config.seeds = tuple(range(n_seeds))` (`SIMSMITH_SEEDS`, currently `5`) —
  always `0..n-1`, deterministic and identical across runs/conditions. Not randomly
  sampled per run. Good.
- **Simulation determinism**: given the same script + seed, MuJoCo stepping itself is
  deterministic (no hidden RNG inside `mj_step`); `get_rng()` is seeded from
  `SIMSMITH_SEED`. A specific candidate script's sim behavior is fully reproducible.
- **Temperature / candidate diversity**: deterministic *schedule* (`FRAMINGS`/
  `TEMPERATURES` cycle by candidate index `i % len(...)`), but the actual **LLM
  completion is not deterministic** — `TokenFactoryLLM.complete()`'s payload never sends
  a `seed` field to the chat-completions API, so re-running the identical prompt at
  temperature > 0 (every candidate except an edge case) gives a different script each
  time. This is inherent to using a sampled LLM, not a bug, but it does mean **no eval
  run is bit-for-bit reproducible even from an identical config** — only the harness
  side (which seeds, which framings, which temperatures, which docs/library entries) is.
- **Can a run be repeated from a config file?** No — there is no serialized "run spec."
  `baseline_dev.py`/`baseline_best_of_5.py` partially reconstruct their own config after
  the fact (model, sim_python, seeds, prompt_version, tag, extra_payload, all recorded
  at the top of their own output JSON — see Part 3), which is enough to *approximately*
  rerun one by hand, but there's no single `--config run.json` flag and no automatic
  capture of `SIMSMITH_*` env vars actually in effect (e.g. `SIMSMITH_RUN_TIMEOUT_S`,
  `SIMSMITH_SEEDS`, `SIMSMITH_MAX_TOKENS_*` aren't echoed anywhere in the output). As
  noted in Part 3, `eval/run_eval.py` doesn't even do the partial version of this.
- **Other nondeterminism sources**: thread-pool scheduling order for concurrent
  `generate()` calls doesn't affect *which* framing/temperature a given candidate index
  gets (that's `i`-keyed, not arrival-order-keyed) — just wall-clock timing and usage-log
  row order, both already handled by tag-based attribution.

---

## Ranked findings

| Severity | Finding | Evidence | Suggested fix | Est. effort |
|---|---|---|---|---|
| **Critical** | Generated scripts run unsandboxed with the full parent environment and filesystem access; `BANNED_MODULES`/`BANNED_CALLS`/`BANNED_ATTR_CALLS` are AST-literal pattern matches, trivially bypassed via `importlib.import_module(...)` or `getattr(os,"sys"+"tem")`. Confirmed: real network egress, `.env`/`~` reads, arbitrary shell exec. | Part 2, PoCs #1,2,3,5,6 | Sandbox execution (container/VM/unprivileged user, no network namespace); strip env to an allow-list in the meantime | Medium (env strip: <1hr; real sandboxing: days) |
| **Critical** | 3 of 5 novel adversarial scripts fully defeat the verifier (100% false "success") by importing `simsmith_contract` in-process and either monkeypatching `_measure_from_mujoco`, injecting a fake `_MJ_TRACKED` entry, or shadowing `write_result` to skip it entirely. | Part 4, exploits #1-3, reproduced live | Run candidate scripts in a separate process/sandbox from the measurement logic (same fix as above); don't trust in-process "private" state against a cooperating importer | Medium-High (shares the sandboxing fix above) |
| High | Cost/budget tracking (`SIMSMITH_BUDGET_USD`) is silently inert unless `SIMSMITH_PRICE_IN`/`SIMSMITH_PRICE_OUT` are also set — with neither in the current `.env`, every usage row computes `cost_usd=0.0` and `BudgetExceeded` can never trigger regardless of actual spend. Found while launching Part 1's regeneration (usage log showed `cost_usd=0.0` on real billed calls). | Observed directly in `eval/results/best_of_5_usage_1792782009.jsonl` during this session | Either hard-fail `TokenFactoryLLM.__init__` when a budget is set but prices are both zero/unset, or default prices from a per-model table instead of a flat env var | Low (a few lines + a models→price table) |
| Medium | `eval/run_eval.py`'s report records no `prompt_version`, `model`, or `sim_python` at all — an `eval_<ts>.json` can't be tied to the config that produced it without cross-referencing shell history. (`baseline_dev.py`/`baseline_best_of_5.py` already do this at the top level, so the fix is to match that pattern here.) | Part 3 / Part 5, read directly from `run_eval.py`'s `report` dict construction | Add the same top-level config fields `baseline_*.py` already writes | Low (~10 lines) |
| Medium | `baseline_dev.py`/`baseline_best_of_5.py` support resuming from a `--progress-file` across invocations, but rows have no per-row model/prompt-version stamp — a resumed run with a different `--model-generate` merges silently into one report. | Part 3, code reading of `_do_one`/`_load_progress` | Stamp `model`, `prompt_version`, `extra_payload` onto each row, not just the top-level report | Low |
| High | `classify()`'s `hallucinated_api` detection (`scripts/baseline_dev.py`, reused by `baseline_best_of_5.py`) matches on `"attributeerror" in text and "module" in text` — but every Python traceback frame is rendered as `in <module>`, so *any* `AttributeError` crash matches, not just real module-attribute hallucinations. Verified: 14/14 `hallucinated_api` labels in the Part 1 regeneration are false positives (e.g. `'list' object has no attribute 'tolist'`). Every historical baseline report's hallucinated-API rate is likely overstated. | Part 1, checked directly against all 14 of this run's labels | Require a real pattern, e.g. `re.search(r"module '[\w.]+' has no attribute", text)` or `ModuleNotFoundError`/`ImportError` only | Trivial (~5 min) |
| Medium | 50% of this run's crashes (20/40) are the model corrupting the `SIMSMITH_ROOT` fallback variable name between definition and use (e.g. defines `simsmith_root`, then references `smithsonian_root`) — caused by `GENERATE_SYSTEM` describing the fallback in prose instead of giving a literal line. 45% (18/40) are pick_place-specific IK code that doesn't slice the Jacobian to the arm's 7 DOFs out of `model.nv=15` (which includes the cube's free joint) before solving. | Part 1, read directly from regenerated candidates' code + tracebacks | Give `GENERATE_SYSTEM` a literal copy-paste `SIMSMITH_ROOT` line instead of prose; add the Jacobian-slicing gotcha to `mujoco_notes.md`/`scene_pick_place.md` | Trivial (prompt text only) |
| Medium | `docs_index/mujoco_notes.md` claims "pinned: mujoco 3.16"; the installed version in this environment is 3.15.0. If also true elsewhere, the model is being told the wrong pinned version in exactly the file whose purpose is to prevent that. | Part 3; `python3 -c "import mujoco; print(mujoco.__version__)"` → `3.15.0` | Regenerate/correct the version line (or auto-stamp it from the installed version like `mujoco_api.md`/scene cards already do) | Trivial |
| Low | `BANNED_CALLS` includes `open_url` (not a real Python builtin) where `open` was almost certainly intended. | Part 2 #2 | One-line fix: `open_url` → `open` (plus the broader AST-bypass caveat above still applies) | Trivial |
| Low | `single_shot` candidates generate at a fixed temperature of 0.3, while `best_of_5`'s index-0 candidate (same framing) generates at 0.2 — a small, undocumented asymmetry if `single_shot` is ever treated as "best_of_5's first candidate in isolation." | Part 3, `Pipeline.generate()` | Document the intentional difference, or align them | Trivial |
| Info | No eval run is reproducible bit-for-bit even from an identical config, because LLM sampling isn't seeded server-side and no config file/env-var snapshot is captured per run. Harness-side determinism (seeds, framing/temperature schedule) is solid. | Part 5 | Not really "fixable" for the LLM side; capture a full config snapshot (including all `SIMSMITH_*` env vars in effect) per run for partial reproducibility | Low-Medium |

---

## Budget / scope accounting

- Part 1 regeneration: 12 pick_place dev task-runs, `nvidia/nemotron-3-super-120b-a12b`,
  reasoning off, best_of_5 + up to 2 repairs. A prior run with this same configuration
  (tag `part3: best_of_5+2repair, super, reasoning_off`) had cost **$0.064** total; used
  as a calibration point before spending further. This run's actual cost: **$0.089**
  total (`eval/results/best_of_5_1792853506.json`), well inside the $1 cap.
- No held-out task (`eval/tasks.json`'s `"heldout"` list) was read, retrieved, or run as
  part of this diagnosis.
- All Part 2/4 PoC scripts ran locally (`sim_python=""`, local `python3`/mujoco), no LLM
  calls, $0 cost.
