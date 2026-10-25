"""Per-row metadata stamped into every result file, so a single row can be tied to
exactly what produced it (model, prompt version, reasoning mode, seeds, the exact
commit...) without cross-referencing shell history or usage-log timestamps by hand --
see docs/diagnosis_2026-10-24.md Part 3 and docs/diagnosis_2026-10-24.md's own
follow-up finding that a resumed run with a different config silently merges into one
report with no per-row marker of which config actually produced which row.

Deliberately per-ROW, not just per-report: a --progress-file resume across two
invocations with different --model-generate (or any other flag) must not leave that
drift invisible, which a single top-level report field cannot catch.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from . import prompts
from .config import Config
from .llm import max_tokens_config

# Bumped when classify_failure()'s hallucinated_api detection changed from a bare
# "attributeerror" + "module" substring match (false-positived on every AttributeError,
# since every Python traceback frame is literally rendered "in <module>") to the real
# `_MODULE_ATTR_ERROR` regex -- see docs/diagnosis_2026-10-24.md Part 1 (found: 14/14
# hallucinated_api labels in one run were false positives under version 1). Any report
# stamped version < 2 should have its hallucinated_api counts treated as unreliable.
CLASSIFIER_VERSION = 2

# The conditions this project currently runs. Keep in sync with eval/run_eval.py's
# CONDITIONS dict and scripts/baseline_best_of_5.py's condition naming.
VALID_CONDITIONS = frozenset({
    "single_shot", "best_of_5", "best_of_5_repair",
    "best_of_5_repair_library", "best_of_5_repair_skills",
})

_REPO_ROOT = None  # set lazily below to avoid a hard import-time filesystem dependency


def _repo_root() -> str:
    global _REPO_ROOT
    if _REPO_ROOT is None:
        import os
        _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return _REPO_ROOT


def git_commit() -> str | None:
    """The exact commit this run executed under, or None if it can't be determined
    (not a git checkout, git missing, a dirty rebase, ...) -- never raises."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=_repo_root())
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def reasoning_mode(extra_payload: dict | None) -> str:
    """"off" only when the call explicitly suppresses Nemotron's reasoning field
    (chat_template_kwargs.enable_thinking=False); "on" otherwise (the model's default
    is reasoning-enabled, so an absent/true value means reasoning stayed on)."""
    if not extra_payload:
        return "on"
    ctk = extra_payload.get("chat_template_kwargs") or {}
    return "off" if ctk.get("enable_thinking") is False else "on"


def build_run_meta(*, condition: str, cfg: Config, extra_payload: dict | None,
                   n_candidates: int, max_repairs: int, model: str = "",
                   workers: int | None = None) -> dict:
    """The fields every result row must carry (see module docstring). Callers merge
    this dict into their own per-row dict; collisions with a caller's own fields of
    the same name are a bug in the caller, not handled here on purpose."""
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; add it to VALID_CONDITIONS "
                         "if this is a genuinely new condition")
    return {
        "condition": condition,
        "model": model or cfg.model_generate,
        "reasoning": reasoning_mode(extra_payload),
        "prompt_version": prompts.PROMPT_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "n_candidates": n_candidates,
        "max_repairs": max_repairs,
        "max_tokens": max_tokens_config(),
        "temperature": list(prompts.TEMPERATURES) if n_candidates > 1 else [0.3],
        "seeds": list(cfg.seeds),
        "workers": workers,
        "git_commit": git_commit(),
        "date": datetime.now(timezone.utc).isoformat(),
    }
