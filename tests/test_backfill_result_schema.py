import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backfill_result_schema as backfill  # noqa: E402

SCHEMA_KEYS = ("condition", "model", "reasoning", "prompt_version", "classifier_version",
              "n_candidates", "max_repairs", "max_tokens", "temperature", "seeds",
              "workers", "git_commit", "date")


def _write_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)


def test_backfill_never_modifies_the_original_file(tmp_path, monkeypatch):
    src = {
        "prompt_version": "mujoco-v2", "model": "m", "sim_seeds": [0, 1], "timestamp": 1700000000,
        "extra_payload": {}, "rows": {"reach": [{"task_id": "d1", "i": 0, "classification": "pass"}]},
    }
    path = tmp_path / "baseline_123.json"
    _write_report(path, src)
    original_bytes = path.read_bytes()

    monkeypatch.setattr(backfill, "RESULTS", str(tmp_path))
    monkeypatch.setattr(backfill, "OUT_DIR", str(tmp_path / "backfilled"))
    backfill.backfill_file(str(path))

    assert path.read_bytes() == original_bytes


def test_backfill_single_shot_infers_condition_and_n_candidates(tmp_path, monkeypatch):
    src = {
        "prompt_version": "mujoco-v2", "model": "m", "sim_seeds": [0, 1, 2], "timestamp": 1700000000,
        "extra_payload": {"chat_template_kwargs": {"enable_thinking": False}},
        "rows": {"reach": [{"task_id": "d1", "i": 0, "classification": "pass"}]},
    }
    path = tmp_path / "baseline_123.json"
    _write_report(path, src)
    monkeypatch.setattr(backfill, "RESULTS", str(tmp_path))
    monkeypatch.setattr(backfill, "OUT_DIR", str(tmp_path / "backfilled"))

    result = backfill.backfill_file(str(path))
    out = json.load(open(result["out_path"]))
    row = out["rows"]["reach"][0]
    for key in SCHEMA_KEYS:
        assert key in row, key
    assert row["condition"] == "single_shot"
    assert row["n_candidates"] == 1
    assert row["max_repairs"] == 0
    assert row["reasoning"] == "off"
    assert row["seeds"] == [0, 1, 2]
    assert row["git_commit"] is None       # genuinely unknown -- must not be guessed
    assert row["max_tokens"] is None       # genuinely unknown -- must not be guessed
    assert out["condition"] == "single_shot"


def test_backfill_best_of_5_renames_old_n_candidates_and_derives_condition(tmp_path, monkeypatch):
    src = {
        "prompt_version": "mujoco-v3", "model": "m", "sim_seeds": [0, 1], "timestamp": 1700000000,
        "max_repairs": 2, "extra_payload": {},
        "rows": {"reach": [{"task_id": "d1", "run_idx": 0, "n_candidates": 5, "failure_classification": None}]},
    }
    path = tmp_path / "best_of_5_123.json"
    _write_report(path, src)
    monkeypatch.setattr(backfill, "RESULTS", str(tmp_path))
    monkeypatch.setattr(backfill, "OUT_DIR", str(tmp_path / "backfilled"))

    result = backfill.backfill_file(str(path))
    out = json.load(open(result["out_path"]))
    row = out["rows"]["reach"][0]
    assert row["n_candidates_generated"] == 5      # old meaning, preserved under its own name
    assert row["n_candidates"] == 5                # new meaning: the configured target
    assert row["condition"] == "best_of_5_repair"  # derived from max_repairs=2, not trusted from old field
    assert row["max_repairs"] == 2


def test_backfill_best_of_5_norepair_condition(tmp_path, monkeypatch):
    src = {
        "prompt_version": "mujoco-v4", "model": "m", "sim_seeds": [0], "timestamp": 1700000000,
        "max_repairs": 0, "extra_payload": {},
        "rows": {"reach": [{"task_id": "d1", "run_idx": 0, "n_candidates": 5}]},
    }
    path = tmp_path / "best_of_5_456.json"
    _write_report(path, src)
    monkeypatch.setattr(backfill, "RESULTS", str(tmp_path))
    monkeypatch.setattr(backfill, "OUT_DIR", str(tmp_path / "backfilled"))

    result = backfill.backfill_file(str(path))
    out = json.load(open(result["out_path"]))
    assert out["rows"]["reach"][0]["condition"] == "best_of_5"
