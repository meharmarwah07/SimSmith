import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simsmith.config import Config  # noqa: E402
from simsmith.frozen_config import (  # noqa: E402
    ConfigDrift, config_hash, current_config, diff_against_frozen,
    load_frozen_config, require_config_matches_frozen,
)

FROZEN = {
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "reasoning": "off",
    "max_tokens": {"generate": 24000, "repair": 8000, "plan": 3000, "intake": 3000},
    "temperature": [0.2, 0.4, 0.6, 0.8, 1.0],
    "seeds": [0, 1, 2, 3, 4],
    "n_candidates": 5,
    "max_repairs": 2,
    "workers": 8,
    "prompt_version": "mujoco-v5",
}


def _matching_cfg():
    return Config(model_generate=FROZEN["model"], seeds=tuple(FROZEN["seeds"]))


def _matching_extra_payload():
    return {"chat_template_kwargs": {"enable_thinking": False}}


def test_real_frozen_json_loads_and_matches_schema():
    frozen = load_frozen_config()  # the real checked-in config/frozen.json
    for key in ("model", "reasoning", "max_tokens", "temperature", "seeds",
               "n_candidates", "max_repairs", "workers", "prompt_version"):
        assert key in frozen, key


def test_load_frozen_config_missing_file_raises(tmp_path):
    try:
        load_frozen_config(str(tmp_path / "nope.json"))
        assert False, "should have raised"
    except FileNotFoundError:
        pass


def test_config_hash_stable_and_ignores_comment():
    h1 = config_hash(FROZEN)
    h2 = config_hash({**FROZEN, "_comment": "irrelevant documentation"})
    assert h1 == h2
    assert h1 == config_hash(dict(reversed(list(FROZEN.items()))))  # key order doesn't matter


def test_config_hash_changes_when_a_real_field_changes():
    h1 = config_hash(FROZEN)
    h2 = config_hash({**FROZEN, "workers": 4})
    assert h1 != h2


def test_diff_against_frozen_empty_when_matching():
    actual = current_config(_matching_cfg(), _matching_extra_payload(), workers=8)
    # prompt_version/temperature/max_tokens come from the real installed code, so
    # patch FROZEN's copies to match what's actually installed for this check.
    from simsmith import prompts
    from simsmith.llm import max_tokens_config
    frozen = {**FROZEN, "prompt_version": prompts.PROMPT_VERSION,
             "temperature": list(prompts.TEMPERATURES), "max_tokens": max_tokens_config()}
    assert diff_against_frozen(actual, frozen) == []


def test_diff_against_frozen_flags_model_drift():
    actual = current_config(Config(model_generate="some-other-model", seeds=tuple(FROZEN["seeds"])),
                            _matching_extra_payload(), workers=8)
    problems = diff_against_frozen(actual, FROZEN)
    assert any("model" in p for p in problems)


def test_diff_against_frozen_flags_reasoning_drift():
    actual = current_config(_matching_cfg(), {}, workers=8)  # reasoning defaults "on"
    problems = diff_against_frozen(actual, FROZEN)
    assert any("reasoning" in p for p in problems)


def test_diff_against_frozen_flags_workers_drift():
    actual = current_config(_matching_cfg(), _matching_extra_payload(), workers=1)
    problems = diff_against_frozen(actual, FROZEN)
    assert any("workers" in p for p in problems)


def test_diff_against_frozen_never_flags_n_candidates_or_max_repairs():
    # These are each condition's own intended variable -- not in STRICT_FIELDS at all,
    # so they must never appear in a diff even if wildly different from frozen's.
    actual = current_config(_matching_cfg(), _matching_extra_payload(), workers=8)
    actual["n_candidates"] = 1
    actual["max_repairs"] = 0
    problems = diff_against_frozen(actual, FROZEN)
    assert not any("n_candidates" in p or "max_repairs" in p for p in problems)


def test_require_config_matches_frozen_raises_configdrift_on_mismatch():
    try:
        require_config_matches_frozen(Config(model_generate="wrong-model"),
                                      _matching_extra_payload(), 8, FROZEN)
        assert False, "should have raised ConfigDrift"
    except ConfigDrift as e:
        assert "model" in str(e)
