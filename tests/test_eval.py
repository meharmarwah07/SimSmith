import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.run_eval import CONDITIONS, run_eval  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from tests.mocks import GOOD, MockLLM  # noqa: E402

TASKS = {
    "dev": [{"id": "d1", "template": "reach", "task": "reach target a"}],
    "heldout": [{"id": "h1", "template": "reach", "task": "reach target b"},
                {"id": "h2", "template": "reach", "task": "reach target c"}],
}


def factory(llm, seeds=(0, 1)):
    cfg = Config(seeds=seeds, run_timeout_s=20)
    runner = SubprocessRunner("", 20)
    return lambda lib: Pipeline(llm, runner, lib, DocIndex("/nonexistent"), cfg)


def test_heldout_never_writes_to_library(tmp_path):
    llm = MockLLM([GOOD] * 200)
    lib_path = str(tmp_path / "lib.jsonl")
    rep = run_eval(factory(llm), TASKS, ["best_of_5_repair_library"], 500, str(tmp_path), lib_path)
    assert rep["library_size_at_heldout"] == 1            # only the dev task was saved
    with open(lib_path) as f:
        assert len(f.readlines()) == 1                    # held-out runs added nothing


def test_budget_cap_truncates_and_flags(tmp_path):
    llm = MockLLM([GOOD] * 200)
    rep = run_eval(factory(llm), TASKS, ["single_shot"], 1, str(tmp_path), str(tmp_path / "l.jsonl"))
    assert rep["truncated"] is True
    assert rep["total_sim_runs"] <= 1 + 2    # may finish the in-flight task, never start another


def test_conditions_configured_as_documented():
    assert CONDITIONS["single_shot"] == dict(n_candidates=1, use_library=False, do_repair=False)
    assert CONDITIONS["best_of_5_repair"]["use_library"] is False and CONDITIONS["best_of_5_repair"]["do_repair"]
    assert CONDITIONS["best_of_5_repair_library"]["use_library"] is True


def test_report_written_with_per_task_rows(tmp_path):
    llm = MockLLM([GOOD] * 200)
    rep = run_eval(factory(llm), TASKS, ["single_shot", "best_of_5_repair"], 500, str(tmp_path), str(tmp_path / "l.jsonl"))
    with open(rep["path"]) as f:
        data = json.load(f)
    assert set(data["conditions"]) == {"single_shot", "best_of_5_repair"}
    assert data["conditions"]["single_shot"]["tasks_run"] == 2
    assert data["conditions"]["single_shot"]["verified_rate"] == 1.0


def test_rows_carry_run_meta_schema(tmp_path):
    llm = MockLLM([GOOD] * 200)
    rep = run_eval(factory(llm), TASKS, ["single_shot"], 500, str(tmp_path), str(tmp_path / "l.jsonl"),
                   extra_payload={"chat_template_kwargs": {"enable_thinking": False}}, workers=3,
                   config_hash="abc123")
    row = rep["conditions"]["single_shot"]["rows"][0]
    for key in ("condition", "model", "reasoning", "prompt_version", "classifier_version",
               "n_candidates", "max_repairs", "max_tokens", "temperature", "seeds",
               "workers", "config_hash", "git_commit", "date"):
        assert key in row, key
    assert row["condition"] == "single_shot"
    assert row["reasoning"] == "off"
    assert row["n_candidates"] == 1
    assert row["workers"] == 3
    assert row["config_hash"] == "abc123"
    assert rep["config_hash"] == "abc123"
    assert "n_candidates_generated" in row  # the actual count, distinct from the configured target
