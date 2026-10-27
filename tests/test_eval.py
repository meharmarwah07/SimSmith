import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.run_eval import (  # noqa: E402
    CONDITIONS, build_jobs, estimate_cost_per_task_run, run_eval,
)
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
    assert CONDITIONS["best_of_5"] == dict(n_candidates=5, use_library=False, do_repair=False)
    assert CONDITIONS["best_of_5_repair"]["use_library"] is False and CONDITIONS["best_of_5_repair"]["do_repair"]
    assert CONDITIONS["best_of_5_repair_library"]["use_library"] is True
    # best_of_5 isolates diversity from repair: same n_candidates as best_of_5_repair,
    # repair off -- the only difference between the two conditions.
    assert CONDITIONS["best_of_5"]["n_candidates"] == CONDITIONS["best_of_5_repair"]["n_candidates"]
    assert CONDITIONS["best_of_5"]["do_repair"] is False and CONDITIONS["best_of_5_repair"]["do_repair"] is True


def test_best_of_5_no_repair_condition_runs_with_mocked_llm(tmp_path):
    llm = MockLLM([GOOD] * 200)
    rep = run_eval(factory(llm), TASKS, ["best_of_5"], 500, str(tmp_path), str(tmp_path / "l.jsonl"))
    cond = rep["conditions"]["best_of_5"]
    assert cond["tasks_run"] == 2
    assert cond["verified_rate"] == 1.0
    row = cond["rows"][0]
    assert row["condition"] == "best_of_5"
    assert row["n_candidates"] == 5
    assert row["max_repairs"] == 0
    assert row["winner_stage"] == "generated"  # no repair possible in this condition


def test_estimate_cost_per_task_run_for_best_of_5_no_repair():
    reach_cost = estimate_cost_per_task_run("best_of_5", "reach", 5)
    assert reach_cost > 0
    # a real measurement (see _EMPIRICAL_MEAN_COST_PER_TASK_RUN), not the scaled fallback
    assert reach_cost == 0.0116
    pick_place_cost = estimate_cost_per_task_run("best_of_5", "pick_place", 5)
    assert 0 < pick_place_cost < estimate_cost_per_task_run("best_of_5_repair", "pick_place", 5)


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


# ---- step D: repeats, resume/checkpointing, cost estimate, budget cap ----------


def test_build_jobs_respects_repeats():
    jobs = build_jobs(TASKS, ["single_shot", "best_of_5_repair"], repeats=3)
    assert len(jobs) == 2 * 2 * 3  # 2 conditions * 2 heldout tasks * 3 repeats
    assert sum(1 for j in jobs if j["condition"] == "single_shot" and j["task_id"] == "h1") == 3


def test_estimate_cost_per_task_run_known_and_fallback():
    known = estimate_cost_per_task_run("best_of_5_repair", "reach", 5)
    assert known > 0
    fallback = estimate_cost_per_task_run("single_shot", "some_unmeasured_template", 1)
    assert fallback > 0
    # best_of_5_repair_library with no direct measurement falls back to
    # best_of_5_repair's own number (scaled), not the generic per-candidate fallback.
    lib_est = estimate_cost_per_task_run("best_of_5_repair_library", "reach", 5)
    assert lib_est > known


def test_resume_does_not_repeat_completed_task_runs(tmp_path):
    llm = MockLLM([GOOD] * 200)
    progress_path = str(tmp_path / "progress.jsonl")
    lib_path = str(tmp_path / "l.jsonl")

    rep1 = run_eval(factory(llm), TASKS, ["single_shot"], 500, str(tmp_path), lib_path,
                    progress_path=progress_path)
    assert rep1["conditions"]["single_shot"]["tasks_run"] == 2
    calls_after_first = dict(llm.calls)

    # A second invocation against the SAME progress file must not re-solve either
    # heldout task -- their rows are loaded from disk, not recomputed.
    rep2 = run_eval(factory(llm), TASKS, ["single_shot"], 500, str(tmp_path), lib_path,
                    progress_path=progress_path)
    assert rep2["conditions"]["single_shot"]["tasks_run"] == 2
    assert llm.calls == calls_after_first  # zero additional LLM calls on resume


def test_resume_does_not_rebuild_an_already_persisted_library_entry(tmp_path):
    llm = MockLLM([GOOD] * 200)
    progress_path = str(tmp_path / "progress.jsonl")
    lib_path = str(tmp_path / "l.jsonl")

    run_eval(factory(llm), TASKS, ["best_of_5_repair_library"], 500, str(tmp_path), lib_path,
            progress_path=progress_path)
    with open(lib_path) as f:
        n_lib_lines_after_first = len(f.readlines())
    assert n_lib_lines_after_first == 1  # the one dev task

    run_eval(factory(llm), TASKS, ["best_of_5_repair_library"], 500, str(tmp_path), lib_path,
            progress_path=progress_path)
    with open(lib_path) as f:
        assert len(f.readlines()) == n_lib_lines_after_first  # not duplicated on resume


def test_repeats_produces_n_rows_per_heldout_task(tmp_path):
    llm = MockLLM([GOOD] * 200)
    rep = run_eval(factory(llm), TASKS, ["single_shot"], 500, str(tmp_path), str(tmp_path / "l.jsonl"),
                   repeats=3)
    assert rep["conditions"]["single_shot"]["tasks_run"] == 2 * 3  # 2 heldout tasks * 3 repeats


def test_cost_is_read_back_from_the_llms_own_usage_log_path(tmp_path):
    # Regression test: run_eval() must resolve usage_path from pipe.llm.usage_log_path
    # itself, not invent its own -- a mismatch here silently makes every cost_usd None.
    usage_log_path = str(tmp_path / "usage.jsonl")
    llm = MockLLM([GOOD] * 200, usage_log_path=usage_log_path)
    rep = run_eval(factory(llm), TASKS, ["single_shot"], 500, str(tmp_path), str(tmp_path / "l.jsonl"))
    rows = rep["conditions"]["single_shot"]["rows"]
    assert all(r["price_known"] for r in rows), rows
    assert all(r["cost_usd"] is not None and r["cost_usd"] > 0 for r in rows), rows
    assert rep["conditions"]["single_shot"]["total_cost_usd"] > 0
    assert rep["total_cost_usd"] > 0


def test_bounded_submission_does_not_overshoot_budget_cap_by_more_than_one_window(tmp_path):
    # Many repeats, workers=1 (one in flight at a time), a budget tight enough to
    # stop partway through -- total_sim_runs must not run every repeat regardless.
    llm = MockLLM([GOOD] * 200, budget_usd=0.0001)
    llm.spent_usd = 0.0  # starts under budget; a real TokenFactoryLLM would raise
                        # BudgetExceeded on its first priced call instead of ever
                        # incrementing spent_usd past the cap -- MockLLM doesn't
                        # price anything, so this test exercises the OTHER cap
                        # (max_runs) as the actually-triggering one, confirming the
                        # bounded-submission loop (not a submit-everything-upfront
                        # loop) is what's enforcing it.
    rep = run_eval(factory(llm), TASKS, ["single_shot"], 2, str(tmp_path), str(tmp_path / "l.jsonl"),
                   repeats=10, workers=1)
    assert rep["truncated"] is True
    assert rep["total_sim_runs"] <= 2 + 2  # at most one task-run's worth of overshoot
