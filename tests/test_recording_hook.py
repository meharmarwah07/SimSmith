import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import simsmith.pipeline as pipeline_mod  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from tests.mocks import GOOD, WRONG, MockLLM  # noqa: E402


def make_pipe(tmp_path, llm, seeds=(0, 1)):
    cfg = Config(seeds=seeds, run_timeout_s=20)
    return Pipeline(llm, SubprocessRunner("", 20), Library(str(tmp_path / "lib.jsonl")),
                    DocIndex(str(tmp_path / "none")), cfg)


def test_record_clip_false_by_default_no_clip_path(tmp_path):
    pipe = make_pipe(tmp_path, MockLLM([GOOD]))
    res = pipe.solve("reach", n_candidates=1, do_repair=False)
    assert res.status == "VERIFIED"
    assert res.clip_path is None


def test_record_clip_true_sets_clip_path_on_verified(tmp_path):
    pipe = make_pipe(tmp_path, MockLLM([GOOD]))
    clip_dir = str(tmp_path / "clips")
    res = pipe.solve("reach", n_candidates=1, do_repair=False,
                     record_clip=True, clip_dir=clip_dir)
    assert res.status == "VERIFIED"
    assert res.clip_path is not None
    assert os.path.exists(res.clip_path)
    assert res.clip_path.startswith(clip_dir)


def test_record_clip_not_attempted_when_not_verified(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_mod, "record_clip",
                        lambda *a, **k: calls.append(1) or {"ok": True, "clip_path": "x"})
    pipe = make_pipe(tmp_path, MockLLM([WRONG]))
    res = pipe.solve("reach", n_candidates=1, do_repair=False, record_clip=True)
    assert res.status != "VERIFIED"
    assert res.clip_path is None
    assert calls == []  # never even attempted for a non-VERIFIED result


def test_record_clip_failure_never_changes_a_pass_to_a_fail(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("simulated recorder crash")
    monkeypatch.setattr(pipeline_mod, "record_clip", _raise)
    pipe = make_pipe(tmp_path, MockLLM([GOOD]))
    res = pipe.solve("reach", n_candidates=1, do_repair=False, record_clip=True)
    assert res.status == "VERIFIED"  # unaffected by the recorder raising
    assert res.clip_path is None     # just no clip, not a failed solve


def test_record_clip_ok_false_leaves_clip_path_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "record_clip",
                        lambda *a, **k: {"ok": False, "clip_path": None, "error": "no egl"})
    pipe = make_pipe(tmp_path, MockLLM([GOOD]))
    res = pipe.solve("reach", n_candidates=1, do_repair=False, record_clip=True)
    assert res.status == "VERIFIED"
    assert res.clip_path is None
