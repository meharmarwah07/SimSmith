import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.run_meta import CLASSIFIER_VERSION, VALID_CONDITIONS, build_run_meta  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from tests.mocks import GOOD, MockLLM  # noqa: E402

import baseline_best_of_5  # noqa: E402
import baseline_dev  # noqa: E402

SCHEMA_KEYS = ("condition", "model", "reasoning", "prompt_version", "classifier_version",
              "n_candidates", "max_repairs", "max_tokens", "temperature", "seeds",
              "workers", "config_hash", "git_commit", "date")


def test_build_run_meta_has_every_schema_field():
    cfg = Config(model_generate="nvidia/nemotron-3-super-120b-a12b", seeds=(0, 1, 2))
    meta = build_run_meta(condition="best_of_5_repair", cfg=cfg, extra_payload=None,
                          n_candidates=5, max_repairs=2, workers=4)
    for key in SCHEMA_KEYS:
        assert key in meta, key
    assert meta["condition"] == "best_of_5_repair"
    assert meta["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert meta["reasoning"] == "on"
    assert meta["classifier_version"] == CLASSIFIER_VERSION
    assert meta["n_candidates"] == 5
    assert meta["max_repairs"] == 2
    assert meta["seeds"] == [0, 1, 2]
    assert meta["workers"] == 4
    assert set(meta["max_tokens"]) == {"generate", "repair", "plan", "intake"}
    assert json.dumps(meta)  # every value must be JSON-serializable


def test_build_run_meta_reasoning_off_only_when_explicitly_suppressed():
    cfg = Config(model_generate="m")
    off = build_run_meta(condition="single_shot", cfg=cfg,
                         extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
                         n_candidates=1, max_repairs=0)
    on_absent = build_run_meta(condition="single_shot", cfg=cfg, extra_payload=None,
                               n_candidates=1, max_repairs=0)
    on_true = build_run_meta(condition="single_shot", cfg=cfg,
                             extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
                             n_candidates=1, max_repairs=0)
    assert off["reasoning"] == "off"
    assert on_absent["reasoning"] == "on"
    assert on_true["reasoning"] == "on"


def test_build_run_meta_rejects_unknown_condition():
    cfg = Config(model_generate="m")
    try:
        build_run_meta(condition="not_a_real_condition", cfg=cfg, extra_payload=None,
                       n_candidates=1, max_repairs=0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_valid_conditions_matches_eval_run_eval_naming():
    # Keep this in sync with eval/run_eval.py's CONDITIONS and
    # scripts/baseline_best_of_5.py's condition naming -- a drift here means a row
    # from one writer uses a name the schema doesn't recognize.
    assert {"single_shot", "best_of_5", "best_of_5_repair",
           "best_of_5_repair_library", "best_of_5_repair_skills"} == set(VALID_CONDITIONS)


def test_baseline_dev_do_one_row_carries_run_meta(tmp_path):
    llm = MockLLM([GOOD] * 5, extra_payload={"chat_template_kwargs": {"enable_thinking": False}})
    cfg = Config(seeds=(0, 1), run_timeout_s=20)
    runner = SubprocessRunner("", 20)
    docs = DocIndex("/nonexistent")
    usage_path = str(tmp_path / "usage.jsonl")
    progress_path = str(tmp_path / "progress.jsonl")
    row = baseline_dev._do_one("reach", {"id": "d1", "task": "reach target a"}, 0, 1,
                               llm, runner, docs, cfg, usage_path, progress_path, workers=2)
    for key in SCHEMA_KEYS:
        assert key in row, key
    assert row["condition"] == "single_shot"
    assert row["reasoning"] == "off"
    assert row["n_candidates"] == 1
    assert row["max_repairs"] == 0
    assert row["workers"] == 2


def test_baseline_best_of_5_do_one_row_carries_run_meta(tmp_path):
    llm = MockLLM([GOOD] * 10)
    cfg = Config(seeds=(0, 1), run_timeout_s=20)
    runner = SubprocessRunner("", 20)
    docs = DocIndex("/nonexistent")
    usage_path = str(tmp_path / "usage.jsonl")
    progress_path = str(tmp_path / "progress.jsonl")
    row = baseline_best_of_5._do_one("reach", {"id": "d1", "task": "reach target a"}, 0, 1,
                                     llm, runner, docs, cfg, usage_path, progress_path,
                                     max_repairs=2, workers=3)
    for key in SCHEMA_KEYS:
        assert key in row, key
    assert row["condition"] == "best_of_5_repair"
    assert row["n_candidates"] == 5
    assert row["max_repairs"] == 2
    assert row["workers"] == 3
    assert "n_candidates_generated" in row

    row0 = baseline_best_of_5._do_one("reach", {"id": "d1", "task": "reach target a"}, 0, 1,
                                      MockLLM([GOOD] * 10), runner, docs, cfg, usage_path,
                                      progress_path, max_repairs=0)
    assert row0["condition"] == "best_of_5"
