"""Local demo UI server for SimSmith.

Usage:
  python -m app.server [--mock] [--replay PATH] [--port 8000]
                        [--results-dir DIR] [--media-dir DIR]

Modes (pick one; --replay wins if both are given):
  (default)  builds the real pipeline from Config.from_env(), same as
             scripts/gate_day1.py -- needs TOKEN_FACTORY_* env vars and a
             sim Python with mujoco installed.
  --mock     builds the pipeline from tests/mocks.py (MockLLM + a plain
             SubprocessRunner) so the whole demo works with no network calls
             and no API key, using the mujoco already installed here.
  --replay   serves one previously saved run (see serialize_result() below)
             with no LLM or sim calls at all. Read-only: /api/run returns 409.

Every serialized run carries a "source" field ("mock" | "live" |
"recorded_live") so the UI can show an unmissable integrity badge instead of
letting a recorded or scripted run pass as a live model call. A --replay
fixture that is missing this field is refused at startup rather than served
unlabeled.

--results-dir (default eval/results/) feeds the /results page, which reads
whatever eval result JSON files are sitting there and renders a model x
condition comparison table. --media-dir (default app/media/) is the only
directory /api/media will ever serve a clip from.

Binds to 127.0.0.1 only. Never put the API key, base_url, or any other env
var into a response -- only the small, explicit fields built by
serialize_result() cross that boundary.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.llm import LLMError, TokenFactoryLLM  # noqa: E402
from simsmith.pipeline import Candidate, Pipeline, SolveResult  # noqa: E402
from simsmith.prompts import PROMPT_VERSION  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.templates import TEMPLATES, closest_template  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
DEFAULT_RESULTS_DIR = os.path.join(ROOT, "eval", "results")
DEFAULT_MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
BACKEND_COMPARISON_FIXTURE = os.path.join(FIXTURES_DIR, "backend_comparison.json")
MAX_BODY_BYTES = 10 * 1024
MAX_TASK_CHARS = 2000

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/results": ("results.html", "text/html; charset=utf-8"),
    "/results.html": ("results.html", "text/html; charset=utf-8"),
    "/results.js": ("results.js", "application/javascript; charset=utf-8"),
}

# Only these extensions are ever served by /api/media -- anything else (even
# if it exists under --media-dir) is rejected, so the allow-list is by type
# as well as by directory.
MEDIA_CONTENT_TYPES = {
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".png": "image/png",
}

# Example tasks offered on the page. Only the exact-match ones map to the
# template they claim; anything else (including free text) falls back to the
# same keyword heuristic the real pipeline uses when intake fails, so the
# "closest template + gap" state is always reachable, not just hard-coded.
EXAMPLE_TASKS = [
    {"task": "Move the arm's gripper to the target point.",
     "template": "reach", "exact_match": True, "gap": ""},
    {"task": "Pick up the cube and place it in the bin.",
     "template": "pick_place", "exact_match": True, "gap": ""},
    {"task": "Stack two cubes on top of each other.",
     "template": "pick_place", "exact_match": False,
     "gap": "SimSmith has no stacking check yet; mapped to pick_place "
            "(cube to bin) as the closest supported template."},
]

_SEED_LINE = re.compile(r"^seed (\d+): (PASS|FAIL|CRASH)\b\s*(.*)$")
_SEED_PREFIX = re.compile(r"^seed (\d+): (.*)$")


def _mock_intake_for(task: str) -> dict:
    for ex in EXAMPLE_TASKS:
        if ex["task"] == task:
            return {"template": ex["template"], "exact_match": ex["exact_match"],
                     "gap": ex["gap"], "goal": task}
    tpl = closest_template(task)
    return {"template": tpl, "exact_match": False,
            "gap": f"No exact example matched; mapped to the closest supported "
                   f"template '{tpl}' by keyword heuristic.", "goal": task}


def build_mock_pipeline(task: str, cfg: Config, library: Library, docs: DocIndex) -> Pipeline:
    """A fresh MockLLM per request, seeded with a mix of outcomes (verified,
    needs repair, crashes, pre-flight rejected, cheats caught by the
    verifier) so one demo run exercises the whole UI."""
    from tests.mocks import BANNED, CRASH, GOOD, SELF_GRADE, WRONG, MockLLM

    llm = MockLLM(
        generate_outputs=[GOOD, WRONG, CRASH, BANNED, SELF_GRADE],
        repair_outputs=[GOOD, GOOD],
        intake=_mock_intake_for(task),
    )
    runner = SubprocessRunner("", cfg.run_timeout_s)
    return Pipeline(llm, runner, library, docs, cfg)


def build_live_pipeline(cfg: Config, library: Library, docs: DocIndex) -> Pipeline:
    llm = TokenFactoryLLM(cfg.base_url, cfg.api_key,
                          usage_log_path=os.path.join(ROOT, "app", "usage.jsonl"))
    runner = SubprocessRunner(cfg.sim_python, cfg.run_timeout_s)
    return Pipeline(llm, runner, library, docs, cfg)


def classify_seed_lines(details: list) -> list:
    """Score.details is a list of one human-readable line per seed, in seed
    order (verify() iterates results in that order). Lines that explicitly
    say PASS/FAIL/CRASH are classified directly; everything else (missing
    keys, task-mismatch, simulation-too-short) is a run that did not crash at
    the process level but failed an integrity check before ever reaching the
    template's own pass/fail test, so it is shown as a crash too -- the
    script did not produce a trustworthy result."""
    out = []
    for i, line in enumerate(details):
        m = _SEED_LINE.match(line)
        if m:
            out.append({"seed": int(m.group(1)), "status": m.group(2).lower(), "detail": m.group(3)})
            continue
        m = _SEED_PREFIX.match(line)
        if m:
            out.append({"seed": int(m.group(1)), "status": "crash", "detail": m.group(2)})
        else:
            out.append({"seed": i, "status": "crash", "detail": line})
    return out


def serialize_candidate(c: Candidate) -> dict:
    score = None
    if c.score is not None:
        score = {
            "ok": c.score.ok,
            "success_rate": c.score.success_rate,
            "crashes": c.score.crashes,
            "mean_seconds": c.score.mean_seconds,
            "failure_text": c.score.failure_text,
            "seeds": classify_seed_lines(c.score.details),
        }
    if not c.preflight_ok:
        status = "rejected_preflight"
    elif score is not None and score["ok"]:
        status = "verified"
    else:
        status = "failed_verification"
    return {
        "idx": c.idx,
        "stage": c.stage,
        "framing": c.framing,
        "temperature": c.temperature,
        "script": c.script,
        "preflight_ok": c.preflight_ok,
        "preflight_problems": list(c.preflight_problems),
        "status": status,
        "score": score,
        # Not a field on Candidate today -- read defensively via getattr so a
        # future pipeline change that starts recording clips (gif/mp4/png)
        # shows up here with no change needed on this side, and anything
        # without one just renders "no recording".
        "clip_path": getattr(c, "clip_path", None),
    }


def serialize_result(res: SolveResult, *, library_size: int, elapsed_s: float,
                      source: str, tokens: int = 0, cost_usd: float = 0.0, sim_runs: int = 0,
                      model: str | None = None, prompt_version: str | None = None,
                      date: str | None = None) -> dict:
    """source must be one of "mock" | "live" | "recorded_live" -- it is what
    the UI's integrity badge is driven from, so every caller must pass one
    explicitly rather than relying on a default."""
    t = TEMPLATES.get(res.template)
    groups: dict = {}
    order: dict = {}
    winner_group, winner_attempt = None, None
    for pos, c in enumerate(res.candidates):
        groups.setdefault(c.idx, []).append(c)
        order.setdefault(c.idx, pos)
        if res.winner is c:
            winner_group = c.idx
            winner_attempt = len(groups[c.idx]) - 1

    candidates_out = []
    repair_rounds = []
    for idx in sorted(groups, key=lambda i: order[i]):
        attempts = [serialize_candidate(c) for c in groups[idx]]
        candidates_out.append({"idx": idx, "attempts": attempts})
        for n, c in enumerate(groups[idx][1:], start=1):
            repair_rounds.append({
                "candidate_idx": idx,
                "attempt": n,
                "before_status": attempts[n - 1]["status"],
                "after_status": attempts[n]["status"],
            })

    return {
        "task": res.task,
        "template": {"name": res.template, "robot": t.robot if t else "",
                     "description": t.description if t else ""},
        "exact_match": res.exact_match,
        "gap": res.gap,
        "plan": res.plan,
        "candidates": candidates_out,
        "winner": {"idx": winner_group, "attempt": winner_attempt} if winner_group is not None else None,
        "status": res.status,
        "notes": list(res.notes),
        "repair_rounds": repair_rounds,
        "library_size": library_size,
        "elapsed_s": elapsed_s,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "sim_runs": sim_runs,
        "source": source,
        "model": model,
        "prompt_version": prompt_version,
        "date": date,
    }


def _count_new_usage(usage_log_path: str, start_offset: int) -> tuple:
    """Reads only the bytes appended to usage_log_path since start_offset and
    sums tokens/cost for this run. Returns (tokens, cost_usd, new_offset)."""
    if not usage_log_path or not os.path.exists(usage_log_path):
        return 0, 0.0, start_offset
    tokens, cost = 0, 0.0
    with open(usage_log_path, encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            tokens += int(rec.get("prompt_tokens", 0)) + int(rec.get("completion_tokens", 0))
            cost += float(rec.get("cost_usd", 0.0))
        new_offset = f.tell()
    return tokens, cost, new_offset


class ServerState:
    def __init__(self, mode: str, pipeline: Pipeline | None = None,
                 library: Library | None = None, replay: dict | None = None,
                 results_dir: str = DEFAULT_RESULTS_DIR, media_dir: str = DEFAULT_MEDIA_DIR):
        self.mode = mode  # "mock" | "live" | "replay"
        self.pipeline = pipeline
        self.library = library
        self.replay = replay
        self.results_dir = results_dir
        self.media_dir = media_dir
        self.lock = threading.Lock()
        self.busy = False
        self.run_state = "done" if replay is not None else "idle"
        self.task = None
        self.result = replay
        self.error = None
        self.started_at = None


def run_task(state: ServerState, task: str, mock: bool, cfg: Config, docs: DocIndex | None) -> None:
    """Runs one solve() call to completion in a background thread; /api/status
    only reports idle/running/done/error, not sub-stage progress. Pipeline.solve()
    is a single blocking call with no progress hook, and simsmith/pipeline.py is
    being actively edited in another session right now -- re-implementing its
    stage-by-stage orchestration here (intake/plan/generate/evaluate/repair) to
    get finer progress would silently drift the moment that method changes
    elsewhere. See app/README.md "What to add after merge" for the follow-up."""
    t0 = time.time()
    pipe = state.pipeline
    sim_runs_before = pipe.runner.runs
    usage_path = getattr(pipe.llm, "usage_log_path", None)
    usage_offset_before = 0
    if usage_path and os.path.exists(usage_path):
        usage_offset_before = os.path.getsize(usage_path)
    try:
        if mock:
            pipe = build_mock_pipeline(task, cfg, state.library, docs)
            state.pipeline = pipe
            sim_runs_before = 0
        res = pipe.solve(task)
        tokens, cost_usd, _ = _count_new_usage(usage_path, usage_offset_before)
        source = "mock" if mock else "live"
        model = None if mock else (cfg.model_generate or "unknown-model")
        payload = serialize_result(
            res, library_size=len(state.library), elapsed_s=time.time() - t0,
            tokens=tokens, cost_usd=cost_usd, sim_runs=pipe.runner.runs - sim_runs_before,
            source=source, model=model, prompt_version=PROMPT_VERSION,
            date=datetime.date.today().isoformat(),
        )
        with state.lock:
            state.result = payload
            state.run_state = "done"
            state.busy = False
    except Exception as e:  # noqa: BLE001 -- a run must never crash the server
        with state.lock:
            state.error = str(e)
            state.run_state = "error"
            state.busy = False


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Same formula used by scripts/baseline_dev.py, reimplemented here so
    app/ has no import dependency on scripts/."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _iso_date(ts) -> str:
    try:
        return datetime.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "unknown"


def _classifier_warning(data: dict) -> bool:
    """True if this result file's failure-mode labels (hallucinated_api etc.)
    come from a classifier version older than 2, or don't say -- a file with
    no classifier_version at all is treated the same as "old", since there's
    no basis to trust its labels either way. This only ever gates the
    label/breakdown warning; it must never be used to adjust a pass rate."""
    v = data.get("classifier_version")
    try:
        return not (v is not None and float(v) >= 2)
    except (TypeError, ValueError):
        return True


def _row_n_candidates(data: dict) -> float | None:
    """The eval scripts record n_candidates per task row, not at the file
    level. Every row in one result file was generated under the same
    condition, so the first row with the field is enough to read it off."""
    rows = data.get("rows")
    if not isinstance(rows, dict):
        return None
    for row_list in rows.values():
        if isinstance(row_list, list) and row_list and isinstance(row_list[0], dict):
            nc = row_list[0].get("n_candidates")
            if isinstance(nc, (int, float)):
                return nc
    return None


# The canonical condition vocabulary the main session's eval code actually
# writes -- kept in sync by hand with simsmith/run_meta.py's VALID_CONDITIONS
# (and eval/run_eval.py's CONDITIONS dict, which uses the same five names).
# detect_condition() trusts an explicit "condition" field whose value is one
# of these literally; it never tries to re-derive or "correct" it from
# max_repairs/n_candidates the way the old, pre-vocabulary heuristics did --
# those heuristics now only run for files that have no "condition" field at
# all (see detect_condition's docstring).
VALID_CONDITIONS = frozenset({
    "single_shot", "best_of_5", "best_of_5_repair",
    "best_of_5_repair_library", "best_of_5_repair_skills",
})

# Tavily (web search) is an orthogonal flag, not a sixth condition -- a run
# can be e.g. best_of_5_repair_skills with or without it. The main session
# hasn't written any file with this yet, so both forms it described are
# supported defensively: a boolean "tavily_on" field, or a "_tavily"/
# "_tavily_on" suffix tacked onto the condition string itself.
_TAVILY_SUFFIXES = ("_tavily_on", "_tavily")


def _strip_tavily_suffix(cond: str) -> tuple:
    for suffix in _TAVILY_SUFFIXES:
        if cond.endswith(suffix):
            return cond[: -len(suffix)], True
    return cond, False


def detect_tavily_on(data: dict) -> bool:
    if data.get("tavily_on") is True:
        return True
    cond = data.get("condition")
    if isinstance(cond, str) and cond:
        _, suffixed = _strip_tavily_suffix(cond)
        if suffixed:
            return True
    return False


def detect_condition(data: dict, filename: str) -> str:
    """One of the VALID_CONDITIONS names above, or "unknown".

    An explicit top-level "condition" field (after stripping any tavily
    suffix -- see detect_tavily_on) is trusted verbatim when it's one of the
    five canonical names; an explicit field with any other value is
    "unknown", never guessed at.

    Only a file with NO "condition" field at all (older files that predate
    simsmith/run_meta.py's VALID_CONDITIONS) falls back to the old
    heuristics: a per-row "n_candidates" signal (1 -> single_shot, >1 ->
    best_of_5 or best_of_5_repair depending on max_repairs), then this
    repo's own filename prefix convention. A file with no evidence for any
    of this is "unknown", never silently folded into single_shot -- that
    silent default was the original bug this function was written to fix,
    since a best_of_5 run misreported as single_shot would be compared
    against the wrong baseline on the Results page."""
    cond = data.get("condition")
    if isinstance(cond, str) and cond:
        base, _ = _strip_tavily_suffix(cond)
        return base if base in VALID_CONDITIONS else "unknown"

    n_candidates = _row_n_candidates(data)
    if n_candidates is not None:
        if n_candidates <= 1:
            return "single_shot"
        max_repairs = data.get("max_repairs")
        if isinstance(max_repairs, (int, float)) and max_repairs > 0:
            return "best_of_5_repair"
        return "best_of_5"

    base = os.path.basename(filename or "")
    if base.startswith("baseline_"):
        return "single_shot"
    if base.startswith("best_of_5_"):
        return "best_of_5"
    return "unknown"


# Two real eval result shapes exist in this codebase today, and the task that
# asked for this page explicitly expects more to show up over time:
#   "baseline" shape (scripts/baseline_dev.py and the best_of_5 eval scripts
#     under scripts/): keyed by template under "summaries", already carries
#     model/prompt_version/timestamp + a full failure_breakdown per
#     template. The actual files sitting in eval/results/ today are all this
#     shape -- both the single_shot baseline runs and the best_of_5 runs,
#     distinguished only by the top-level "condition"/"max_repairs" fields
#     detect_condition() reads (see its docstring). A single file commonly
#     covers more than one template (e.g. both "reach" and "pick_place" in
#     one best_of_5 run), so this emits one row per template, not one
#     aggregated row per file -- the Results page groups by template, and an
#     aggregate would hide that two templates can have very different pass
#     rates in the same run.
#   "eval" shape (eval/run_eval.py, hypothetical/forward-looking -- no file
#     in this repo uses it today): keyed by condition name -- the actual
#     keys eval/run_eval.py's own CONDITIONS dict uses are canonical
#     (single_shot / best_of_5_repair / best_of_5_repair_library), matched
#     against VALID_CONDITIONS the same as detect_condition() does for an
#     explicit "condition" field -- with a template on each individual task
#     row rather than at the file level. Also split into one row per
#     (condition, template).
# _normalize_eval_shape tolerates both and returns [] (not an error) for
# anything that matches neither, so a results-dir with mixed/future files
# never crashes the page -- see load_eval_results for how a genuinely
# unparseable file is still reported instead of silently dropped.
def _normalize_eval_shape(data: dict, file_mtime: float, name: str = "") -> list:
    if not isinstance(data, dict):
        return []

    if "summaries" in data and isinstance(data.get("summaries"), dict):
        model = data.get("model") or "unknown"
        prompt_version = data.get("prompt_version") or "unknown"
        condition = detect_condition(data, name)
        tavily_on = detect_tavily_on(data)
        ts = data.get("timestamp")
        date = _iso_date(ts) if ts else _iso_date(file_mtime)
        classifier_version = data.get("classifier_version")
        low_confidence = _classifier_warning(data)

        rows_out = []
        for template, summary in data["summaries"].items():
            if not isinstance(summary, dict):
                continue
            n = summary.get("n", 0) or 0
            if n == 0:
                continue
            passes = summary.get("passes", 0) or 0
            lo, hi = wilson_interval(passes, n)
            rows_out.append({
                "model": model,
                "template": template,
                "condition": condition,
                "tavily_on": tavily_on,
                "prompt_version": prompt_version,
                "date": date,
                "n": n,
                "pass_rate": passes / n,
                "wilson_lo": lo, "wilson_hi": hi,
                "mean_seconds": summary.get("mean_wall_seconds"),
                "mean_cost_usd": summary.get("mean_cost_usd"),
                "failure_breakdown": dict(summary.get("failure_breakdown") or {}),
                "classifier_version": classifier_version,
                "low_confidence_labels": low_confidence,
                "shape": "baseline",
            })
        return rows_out

    if "conditions" in data and isinstance(data.get("conditions"), dict):
        model = data.get("model") or "unknown"
        prompt_version = data.get("prompt_version") or "unknown"
        tavily_on = detect_tavily_on(data)
        started = data.get("started")
        date = _iso_date(started) if started else _iso_date(file_mtime)
        classifier_version = data.get("classifier_version")
        low_confidence = _classifier_warning(data)

        rows_out = []
        for cond_name, cond in data["conditions"].items():
            if not isinstance(cond, dict):
                continue
            by_template: dict = {}
            for r in (cond.get("rows") or []):
                if not isinstance(r, dict):
                    continue
                by_template.setdefault(r.get("template") or "unknown", []).append(r)
            for template, trows in by_template.items():
                n = len(trows)
                if n == 0:
                    continue
                verified = sum(1 for r in trows if r.get("status") == "VERIFIED")
                lo, hi = wilson_interval(verified, n)
                seconds_vals = [r["elapsed_s"] for r in trows if isinstance(r.get("elapsed_s"), (int, float))]
                cost_vals = [r["cost_usd"] for r in trows if isinstance(r.get("cost_usd"), (int, float))]
                breakdown: dict = {}
                for r in trows:
                    status = r.get("status") or "unknown"
                    if status != "VERIFIED":
                        breakdown[status.lower()] = breakdown.get(status.lower(), 0) + 1
                cond_base, cond_tavily = _strip_tavily_suffix(cond_name)
                rows_out.append({
                    "model": model,
                    "template": template,
                    "condition": cond_base if cond_base in VALID_CONDITIONS else "unknown",
                    "tavily_on": tavily_on or cond_tavily,
                    "prompt_version": prompt_version,
                    "date": date,
                    "n": n,
                    "pass_rate": verified / n,
                    "wilson_lo": lo, "wilson_hi": hi,
                    "mean_seconds": (sum(seconds_vals) / len(seconds_vals)) if seconds_vals else None,
                    "mean_cost_usd": (sum(cost_vals) / len(cost_vals)) if cost_vals else None,
                    "failure_breakdown": breakdown,
                    "classifier_version": classifier_version,
                    "low_confidence_labels": low_confidence,
                    "shape": "eval",
                })
        return rows_out

    return []


def load_eval_results(results_dir: str) -> tuple:
    """Returns (rows, errors). A file that fails to parse or doesn't match a
    known shape is reported in errors, never raised -- one bad file in
    --results-dir must not take down the whole page."""
    rows: list = []
    errors: list = []
    if not results_dir or not os.path.isdir(results_dir):
        return rows, [f"results directory not found: {results_dir}"]
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            parsed = _normalize_eval_shape(data, os.path.getmtime(path), name)
            if not parsed:
                errors.append(f"{name}: unrecognized or empty result shape, skipped")
                continue
            for r in parsed:
                r["file"] = name
            rows.extend(parsed)
        except (ValueError, OSError, TypeError, KeyError) as e:
            errors.append(f"{name}: {e}")
    return rows, errors


def _safe_media_path(media_dir: str, rel_path: str) -> str | None:
    """Resolves rel_path under media_dir and refuses anything that would
    escape it: absolute paths, '..' segments, and symlinks that point
    outside are all rejected by checking the *resolved* path's prefix, not
    the string itself."""
    if not rel_path or not media_dir:
        return None
    if rel_path.startswith("/") or "\\" in rel_path or "\x00" in rel_path:
        return None
    base = os.path.realpath(media_dir)
    candidate = os.path.realpath(os.path.join(base, rel_path))
    if candidate != base and not candidate.startswith(base + os.sep):
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


def make_handler_class(state: ServerState, cfg: Config, docs: DocIndex | None):
    mock = state.mode == "mock"

    class Handler(BaseHTTPRequestHandler):
        server_version = "SimSmithUI/1"

        def log_message(self, fmt, *args):  # quiet by default; keep stdout for the demo output
            pass

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, name: str, content_type: str) -> None:
            path = os.path.join(STATIC_DIR, name)
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except OSError:
                self._send_json(404, {"error": "not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            parsed = urlparse(self.path)
            path = parsed.path
            if path in STATIC_FILES:
                name, ctype = STATIC_FILES[path]
                self._send_file(name, ctype)
                return
            if path == "/api/status":
                with state.lock:
                    self._send_json(200, {
                        "mode": state.mode,
                        "state": state.run_state,
                        "busy": state.busy,
                        "task": state.task,
                        "result": state.result,
                        "error": state.error,
                        "library_size": len(state.library) if state.library is not None else None,
                    })
                return
            if path == "/api/replay":
                if state.replay is None:
                    self._send_json(404, {"error": "server was not started with --replay"})
                else:
                    self._send_json(200, {"result": state.replay})
                return
            if path == "/api/results":
                # One row per (model, template, condition, prompt_version) --
                # the Results page's grouped table and its Conditions chart
                # blocks both read this same endpoint; there is no separate
                # /api/conditions any more now that /api/results already has
                # template-level granularity (see _normalize_eval_shape).
                rows, errors = load_eval_results(state.results_dir)
                self._send_json(200, {"results_dir": state.results_dir, "rows": rows, "errors": errors})
                return
            if path == "/api/backend-comparison":
                try:
                    with open(BACKEND_COMPARISON_FIXTURE, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    data = {"note": "backend_comparison.json fixture missing or invalid", "backends": []}
                self._send_json(200, data)
                return
            if path == "/api/media":
                rel = parse_qs(parsed.query).get("path", [""])[0]
                ext = os.path.splitext(rel)[1].lower()
                ctype = MEDIA_CONTENT_TYPES.get(ext)
                if not ctype:
                    self._send_json(400, {"error": "unsupported media type"})
                    return
                full_path = _safe_media_path(state.media_dir, rel)
                if full_path is None:
                    self._send_json(404, {"error": "not found"})
                    return
                with open(full_path, "rb") as f:
                    body = f.read()
                self._send_bytes(body, ctype)
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if path != "/api/run":
                self._send_json(404, {"error": "not found"})
                return

            length_hdr = self.headers.get("Content-Length")
            try:
                length = int(length_hdr) if length_hdr is not None else None
            except ValueError:
                length = None
            if length is None or length > MAX_BODY_BYTES:
                # Drain up to the limit so the connection can be reused/closed cleanly,
                # but never read an attacker-declared huge length into memory.
                if length is not None:
                    self.rfile.read(min(length, MAX_BODY_BYTES))
                self.close_connection = True
                self._send_json(413, {"error": f"request body exceeds {MAX_BODY_BYTES} bytes"})
                return
            raw = self.rfile.read(length)

            if state.mode == "replay":
                self._send_json(409, {"error": "server is running in --replay mode (read-only); "
                                               "restart without --replay to run live tasks"})
                return

            with state.lock:
                if state.busy:
                    self._send_json(409, {"error": "a run is already in progress"})
                    return
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send_json(400, {"error": "body must be JSON"})
                    return
                task = data.get("task") if isinstance(data, dict) else None
                if not isinstance(task, str) or not task.strip():
                    self._send_json(400, {"error": "'task' must be a non-empty string"})
                    return
                task = task.strip()[:MAX_TASK_CHARS]

                state.busy = True
                state.run_state = "running"
                state.task = task
                state.result = None
                state.error = None
                state.started_at = time.time()

            thread = threading.Thread(target=run_task, args=(state, task, mock, cfg, docs), daemon=True)
            thread.start()
            self._send_json(202, {"status": "started", "task": task})

    return Handler


def build_state(args) -> tuple:
    """Returns (state, cfg, docs) ready for make_handler_class()."""
    results_dir = getattr(args, "results_dir", DEFAULT_RESULTS_DIR) or DEFAULT_RESULTS_DIR
    media_dir = getattr(args, "media_dir", DEFAULT_MEDIA_DIR) or DEFAULT_MEDIA_DIR

    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            data = json.load(f)
        replay = data["result"] if isinstance(data, dict) and "result" in data else data
        if not isinstance(replay, dict) or "source" not in replay:
            print(f"Cannot start in --replay mode: {args.replay} has no 'source' field "
                  f"(mock/live/recorded_live). A recording must say what it is before it can "
                  f"be shown, so this fixture is refused rather than served without a badge -- "
                  f"regenerate it with the current serialize_result().", file=sys.stderr)
            sys.exit(1)
        return ServerState(mode="replay", replay=replay, results_dir=results_dir,
                           media_dir=media_dir), Config(), None

    cfg = Config.from_env()
    docs = DocIndex(os.path.join(ROOT, "docs_index"))
    library = Library(os.path.join(ROOT, "app", "ui_library.jsonl"))
    if args.mock:
        pipe = build_mock_pipeline("warm-up", cfg, library, docs)
        return ServerState(mode="mock", pipeline=pipe, library=library, results_dir=results_dir,
                           media_dir=media_dir), cfg, docs

    try:
        pipe = build_live_pipeline(cfg, library, docs)
    except LLMError as e:
        print(f"Cannot start in live mode: {e}", file=sys.stderr)
        print("Set TOKEN_FACTORY_BASE_URL / TOKEN_FACTORY_API_KEY, or run with --mock.", file=sys.stderr)
        sys.exit(1)
    return ServerState(mode="live", pipeline=pipe, library=library, results_dir=results_dir,
                       media_dir=media_dir), cfg, docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="use the mock LLM + no network, fully offline")
    ap.add_argument("--replay", default="", help="serve a previously saved run JSON, read-only")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="directory of eval result JSON files for the /results page")
    ap.add_argument("--media-dir", default=DEFAULT_MEDIA_DIR,
                    help="only directory /api/media will ever serve a clip from")
    args = ap.parse_args()

    state, cfg, docs = build_state(args)
    handler_cls = make_handler_class(state, cfg, docs)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    print(f"SimSmith UI ({state.mode} mode) on http://127.0.0.1:{args.port}/")
    if state.mode == "replay":
        print(f"Serving recorded run from {args.replay}")
    print(f"Results page reading from {state.results_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
