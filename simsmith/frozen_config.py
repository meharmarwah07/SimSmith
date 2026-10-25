"""The single reference config (config/frozen.json) eval/run_eval.py must run under,
and the drift check that refuses a run when the live environment doesn't match it in
any field except a condition's own intended variable.

Why: a held-out eval run's numbers are only comparable across conditions (single_shot
vs best_of_5_repair vs best_of_5_repair_library) if everything EXCEPT what those
conditions are defined to vary (n_candidates, max_repairs) is identical -- same model,
same reasoning mode, same max_tokens, same temperature schedule, same seeds, same
worker count, same prompt version. Any of those drifting silently (a stale .env, an
unreviewed prompts.py edit, a forgotten --extra-payload) would make the comparison
meaningless without anyone noticing. config/frozen.json makes the intended values
explicit and visible in git diff; this module is the thing that actually enforces them
and stamps a hash of the frozen config into every result row so a report can be tied
back to exactly which frozen config produced it.
"""
from __future__ import annotations

import hashlib
import json
import os

from .config import Config
from .llm import max_tokens_config
from .run_meta import reasoning_mode
from . import prompts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FROZEN_PATH = os.path.join(ROOT, "config", "frozen.json")

# Every field in config/frozen.json except these two is a global constant a run must
# match exactly. n_candidates/max_repairs are deliberately excluded: they are each
# eval/run_eval.py CONDITIONS entry's own intended variable, not a shared constant.
STRICT_FIELDS = ("model", "reasoning", "max_tokens", "temperature", "seeds", "workers", "prompt_version")


class ConfigDrift(RuntimeError):
    """Raised (not just printed) so a caller in a test can assert on it directly;
    eval/run_eval.py's CLI catches it at the top level and exits non-zero."""


def load_frozen_config(path: str | None = None) -> dict:
    path = path or DEFAULT_FROZEN_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist -- eval/run_eval.py requires a frozen reference "
            "config to run against (see config/frozen.json's _comment). Create one "
            "deliberately; it is never auto-generated.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_hash(frozen: dict) -> str:
    """Stable across key order and across the "_comment" field (which is documentation,
    not config) -- changing the comment must not change the hash."""
    payload = {k: v for k, v in frozen.items() if k != "_comment"}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def current_config(cfg: Config, extra_payload: dict | None, workers: int) -> dict:
    """The actual-config dict, shaped exactly like frozen.json, for a run about to
    start -- built from the real environment/CLI, never from frozen.json itself."""
    return {
        "model": cfg.model_generate,
        "reasoning": reasoning_mode(extra_payload),
        "max_tokens": max_tokens_config(),
        "temperature": list(prompts.TEMPERATURES),
        "seeds": list(cfg.seeds),
        "workers": workers,
        "prompt_version": prompts.PROMPT_VERSION,
    }


def diff_against_frozen(actual: dict, frozen: dict) -> list:
    """[] if every STRICT_FIELDS value matches; else one message per mismatch."""
    problems = []
    for key in STRICT_FIELDS:
        if actual.get(key) != frozen.get(key):
            problems.append(f"{key}: frozen={frozen.get(key)!r}  actual={actual.get(key)!r}")
    return problems


def require_config_matches_frozen(cfg: Config, extra_payload: dict | None, workers: int,
                                  frozen: dict) -> None:
    """Raises ConfigDrift (does not just warn) if anything outside a condition's own
    intended variable has drifted from config/frozen.json."""
    problems = diff_against_frozen(current_config(cfg, extra_payload, workers), frozen)
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise ConfigDrift(
            "refusing to run: the live config has drifted from config/frozen.json in "
            f"field(s) other than a condition's own intended variable:\n{lines}\n"
            "Fix the environment/CLI to match, or edit config/frozen.json deliberately "
            "if this drift is intended (it will show up in git diff).")
