"""SimSmith pipeline: intake -> plan -> retrieve -> generate -> preflight ->
execute -> verify -> repair -> select -> learn.

Stages 7a (Tavily failure-mode research) and 7b (perturbation robustness) are
NOT implemented yet; the runner already passes SIMSMITH_PERTURB through so
they can be added without touching the generated-script contract.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import prompts
from .config import Config
from .docs import DocIndex
from .library import Library
from .llm import LLM, LLMError
from .preflight import check as preflight_check
from .runner import SubprocessRunner
from .templates import TEMPLATES, closest_template
from .verifier import Score, verify

_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _CODE.search(text)
    return (m.group(1) if m else text).strip() + "\n"


_LOG_TEXT_CAP = 20000  # per stdout/stderr field, in a persisted candidate_log() record


def candidate_log(c: "Candidate") -> dict:
    """Full diagnostic record for one candidate: its code, preflight result, and
    every per-seed run's exit code + stdout/stderr -- not just the pass/fail summary
    a Score reduces it to. Callers that write candidates to a results JSON should
    include this (see scripts/baseline_dev.py, scripts/baseline_best_of_5.py) so a
    crash/failure can be diagnosed after the fact without rerunning anything."""
    return {
        "script": c.script,
        "stage": c.stage,
        "preflight_ok": c.preflight_ok,
        "preflight_problems": c.preflight_problems,
        "runs": [
            {"seed": r.seed, "ok": r.ok, "exit_code": r.exit_code, "seconds": r.seconds,
             "error": r.error, "stdout": r.stdout[-_LOG_TEXT_CAP:], "stderr": r.stderr[-_LOG_TEXT_CAP:]}
            for r in c.runs
        ],
    }


@dataclass
class Candidate:
    idx: int
    script: str
    framing: str
    temperature: float
    stage: str = "generated"            # generated | repaired
    preflight_ok: bool = True
    preflight_problems: list = field(default_factory=list)
    score: Score | None = None
    runs: list = field(default_factory=list)  # raw per-seed RunResult, for post-hoc diagnosis/logging


@dataclass
class SolveResult:
    task: str
    template: str
    exact_match: bool
    gap: str
    plan: str
    candidates: list
    winner: Candidate | None
    status: str                         # VERIFIED | PARTIAL | FAILED
    notes: list = field(default_factory=list)


class Pipeline:
    def __init__(self, llm: LLM, runner: SubprocessRunner, library: Library,
                 docs: DocIndex, cfg: Config, api_allowlist: set | None = None,
                 isaac_version: str = "unset"):
        self.llm, self.runner, self.library, self.docs = llm, runner, library, docs
        self.cfg, self.api_allowlist, self.isaac_version = cfg, api_allowlist, isaac_version

    # 0. intake
    def intake(self, task: str):
        try:
            raw = self.llm.complete(prompts.INTAKE_SYSTEM, task, self.cfg.model_fast, 0.0, kind="intake")
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0))
            tpl = data.get("template")
            if tpl not in TEMPLATES:
                raise ValueError("unknown template")
            return tpl, bool(data.get("exact_match", True)), data.get("gap", ""), data.get("goal", task)
        except (LLMError, ValueError, AttributeError, TypeError):
            tpl = closest_template(task)
            return tpl, False, "intake fallback: matched by keywords", task

    # 1. plan
    def plan(self, goal: str, template: str) -> str:
        t = TEMPLATES[template]
        user = (f"Task: {goal}\nTemplate: {t.name} ({t.robot}). {t.description}\n"
                f"Final raw state to report: {t.state_help}")
        try:
            return self.llm.complete(prompts.PLAN_SYSTEM, user, self.cfg.model_plan, 0.2, kind="plan")
        except LLMError as e:
            return f"(plan unavailable: {e})"

    # 2-3. retrieve + generate
    def generate(self, goal: str, template: str, plan: str, n: int, use_library: bool) -> list:
        t = TEMPLATES[template]
        scene_card = self.docs.scene_card(template)
        docs = self.docs.retrieve(f"{goal} {template}", k=max(n + 2, 6))
        examples = self.library.retrieve(goal, template, k=2) if use_library else []
        # A caller's usage-tracking tag (see TokenFactoryLLM.set_tag) lives in its own
        # thread-local and does not propagate to the worker threads below; carry it over
        # explicitly so a caller tagging a whole solve() call still gets all N candidates'
        # usage attributed to it, not silently dropped.
        parent_tag = getattr(self.llm, "get_tag", lambda: None)()

        def build(i: int) -> Candidate:
            if parent_tag is not None:
                self.llm.set_tag(parent_tag)
            framing = prompts.FRAMINGS[i % len(prompts.FRAMINGS)]
            temp = prompts.TEMPERATURES[i % len(prompts.TEMPERATURES)] if n > 1 else 0.3
            # each candidate sees a different slice of the retrieved docs, but the scene
            # card for the active template is never subject to that windowing -- every
            # candidate gets it regardless of what the keyword ranker picked.
            slice_ = docs[i:i + 3] if docs else []
            user = [f"Task: {goal}", f"Robot/template: {t.robot} / {t.name}",
                    f"Report at the end: {t.state_help}", f"Plan:\n{plan}", f"Style: {framing}"]
            if scene_card:
                user.append(f"Scene reference (ground truth, from the real XML):\n{scene_card}")
            if slice_:
                user.append("Reference material (MuJoCo API notes/gotchas):\n" + "\n---\n".join(slice_))
            for ex in examples:
                user.append(f"Verified example for a similar task ({ex['task']!r}):\n```python\n{ex['script']}```")
            try:
                out = self.llm.complete(prompts.GENERATE_SYSTEM, "\n\n".join(user),
                                        self.cfg.model_generate, temp, kind="generate")
            except LLMError as e:
                return Candidate(i, "", framing, temp, preflight_ok=False,
                                 preflight_problems=[f"llm error: {e}"])
            return Candidate(i, extract_code(out), framing, temp)

        # Independent API calls to the same remote model, not CPU-bound and not touching
        # the sim: a thread pool overlaps their network latency instead of paying it N times
        # serially. Sim execution (evaluate()) stays sequential -- that's local CPU/GPU work.
        if n <= 1:
            return [build(0)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(build, range(n)))

    # 4-6. preflight + execute + verify
    def evaluate(self, c: Candidate, template: str) -> None:
        pf = preflight_check(c.script, self.api_allowlist) if c.script else None
        if pf is None or not pf.ok:
            c.preflight_ok = False
            c.preflight_problems = pf.problems if pf else (c.preflight_problems or ["empty script"])
            c.score = None
            return
        results = [self.runner.run(c.script, seed, template=template) for seed in self.cfg.seeds]
        c.runs = results
        c.score = verify(template, results)

    # 7. repair
    def repair(self, c: Candidate, template: str, goal: str) -> Candidate | None:
        if c.preflight_ok and c.score is not None:
            failure = c.score.failure_text or "unknown failure"
        else:
            failure = "pre-flight problems:\n" + "\n".join(c.preflight_problems)
        scene_card = self.docs.scene_card(template)
        docs = self.docs.retrieve(f"{goal} {template} {failure}", k=6)
        user = [f"Task: {goal}", f"Script:\n```python\n{c.script}```", f"Failure output:\n{failure}"]
        if scene_card:
            user.append(f"Scene reference (ground truth, from the real XML):\n{scene_card}")
        if docs:
            user.append("Reference material (MuJoCo API notes/gotchas):\n" + "\n---\n".join(docs[:3]))
        user = "\n\n".join(user)
        try:
            out = self.llm.complete(prompts.REPAIR_SYSTEM, user, self.cfg.model_generate, 0.3, kind="repair")
        except LLMError:
            return None
        fixed = Candidate(c.idx, extract_code(out), c.framing, 0.3, stage="repaired")
        self.evaluate(fixed, template)
        return fixed

    # full run
    def solve(self, task: str, n_candidates: int = 5, use_library: bool = True,
              do_repair: bool = True, max_repairs: int = 2, save_to_library: bool = True) -> SolveResult:
        template, exact, gap, goal = self.intake(task)
        plan = self.plan(goal, template)
        cands = self.generate(goal, template, plan, n_candidates, use_library)
        for c in cands:
            self.evaluate(c, template)

        notes = []
        if do_repair:
            failing = [c for c in cands if not (c.score and c.score.ok)]
            # repair the ones closest to working first (highest success rate)
            failing.sort(key=lambda c: -(c.score.success_rate if c.score else -1))
            for c in failing[:max_repairs]:
                fixed = self.repair(c, template, goal)
                if fixed:
                    cands.append(fixed)

        scored = [c for c in cands if c.score is not None]
        scored.sort(key=lambda c: (c.score.ok, c.score.success_rate, -c.score.mean_seconds), reverse=True)
        winner = scored[0] if scored else None
        if winner and winner.score.ok:
            status = "VERIFIED"
            if save_to_library:
                self.library.add(goal, template, winner.script, winner.score.success_rate, self.isaac_version)
        elif winner and winner.score.success_rate > 0:
            status = "PARTIAL"
            notes.append("No candidate met the pass bar; returning the best partial result.")
        else:
            status = "FAILED"
            notes.append("No candidate achieved the task on any seed.")
        if not exact:
            notes.append(f"Request mapped to closest supported template '{template}'. Gap: {gap}")
        return SolveResult(task, template, exact, gap, plan, cands, winner, status, notes)
