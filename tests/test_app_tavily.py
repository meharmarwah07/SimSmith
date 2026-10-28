"""Tests for the demo UI server's Tavily wiring (app/server.py): build_live_pipeline()
opting a live run into the plan-step hook, tavily_failure_modes_for_result() surfacing
sourced failure modes for the run view, and the full /api/run flow carrying a "tavily"
field. All HTTP is mocked (urllib.request.urlopen monkeypatched, same as
tests/test_tavily_plan_hook.py) -- no network calls, no API key needed, no real calls
of any kind in this file."""
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from http.server import ThreadingHTTPServer  # noqa: E402

from app import server  # noqa: E402
from simsmith import tavily as tavily_mod  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.pipeline import Pipeline, SolveResult  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.tavily import TavilyClient  # noqa: E402
from tests.mocks import GOOD, MockLLM  # noqa: E402
from tests.test_app import post_json, request, start_server, stop_server, wait_for_done  # noqa: E402
from tests.test_tavily import FakeResponse, _result, _search_body  # noqa: E402


def _tavily_client(tmp_path, **kw):
    return TavilyClient(api_key="tvly-secret", cache_path=str(tmp_path / "cache.json"),
                        usage_log_path=str(tmp_path / "tavily_usage.jsonl"), **kw)


# ---- build_live_pipeline() wiring -------------------------------------------------

def test_build_live_pipeline_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SIMSMITH_TAVILY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = Config(base_url="http://fake", api_key="fake-key")
    pipe = server.build_live_pipeline(cfg, Library(str(tmp_path / "lib.jsonl")),
                                      DocIndex(str(tmp_path / "none")))
    assert pipe.tavily_client is None


def test_build_live_pipeline_wires_tavily_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMSMITH_TAVILY", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    cfg = Config(base_url="http://fake", api_key="fake-key")
    pipe = server.build_live_pipeline(cfg, Library(str(tmp_path / "lib.jsonl")),
                                      DocIndex(str(tmp_path / "none")))
    assert isinstance(pipe.tavily_client, TavilyClient)


def test_build_mock_pipeline_never_gets_a_tavily_client(tmp_path, monkeypatch):
    """Even with SIMSMITH_TAVILY=1 set in the environment, --mock must never make a
    real Tavily call -- mocked data must never be shown as real research."""
    monkeypatch.setenv("SIMSMITH_TAVILY", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    cfg = Config(seeds=(0, 1))
    pipe = server.build_mock_pipeline("reach the target", cfg, Library(str(tmp_path / "lib.jsonl")),
                                      DocIndex(str(tmp_path / "none")))
    assert pipe.tavily_client is None


# ---- tavily_failure_modes_for_result() --------------------------------------------

def _fake_result(template="reach", task="Move the arm's gripper to the target point."):
    return SolveResult(task=task, template=template, exact_match=True, gap="",
                       plan="1. setup\n2. move\n3. report", candidates=[], winner=None, status="FAILED")


def test_tavily_helper_returns_none_without_a_client():
    assert server.tavily_failure_modes_for_result(None, _fake_result()) is None


def test_tavily_helper_swallows_errors(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(server, "research_failure_modes", _raise)
    assert server.tavily_failure_modes_for_result(object(), _fake_result()) is None


def test_tavily_helper_returns_sourced_items(tmp_path, monkeypatch):
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_search_body()))
    client = _tavily_client(tmp_path)
    out = server.tavily_failure_modes_for_result(client, _fake_result())
    assert out is not None
    assert out["queries_used"]
    assert len(out["items"]) >= 1
    item = out["items"][0]
    assert item["source_url"] == "https://example.test/article"
    assert "Gripper slip in warehouse robots" in item["source_title"]
    assert isinstance(out["dropped_count"], int)


def test_tavily_helper_reuses_cache_no_extra_credit(tmp_path, monkeypatch):
    """The run view's lookup must be a pure cache re-read of whatever the plan step
    already searched -- calling it twice for the same (task, template) must not spend
    a second Tavily credit."""
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_search_body()))
    client = _tavily_client(tmp_path)
    res = _fake_result()
    server.tavily_failure_modes_for_result(client, res)
    made_after_first = client.searches_made
    assert made_after_first > 0
    server.tavily_failure_modes_for_result(client, res)
    assert client.searches_made == made_after_first  # second call was a cache hit


# ---- serialize_result() carries the field -----------------------------------------

def test_serialize_result_includes_tavily_field(tmp_path):
    res = _fake_result()
    payload = server.serialize_result(res, library_size=0, elapsed_s=1.0, source="live",
                                       tavily={"queries_used": ["q"], "dropped_count": 0, "items": []})
    assert payload["tavily"] == {"queries_used": ["q"], "dropped_count": 0, "items": []}

    payload_none = server.serialize_result(res, library_size=0, elapsed_s=1.0, source="mock")
    assert payload_none["tavily"] is None


# ---- full /api/run flow -------------------------------------------------------------

def test_mock_run_never_has_tavily_data(tmp_path):
    cfg = Config(seeds=(0, 1), run_timeout_s=30)
    docs = DocIndex(str(tmp_path / "no-docs"))
    library = Library(str(tmp_path / "library.jsonl"))
    pipe = server.build_mock_pipeline("Move the arm's gripper to the target point.", cfg, library, docs)
    state = server.ServerState(mode="mock", pipeline=pipe, library=library)
    httpd, port = start_server(state, cfg, docs)
    try:
        post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        final = wait_for_done(port)
        assert final["result"]["tavily"] is None
        assert final["result"]["source"] == "mock"
    finally:
        stop_server(httpd)


def test_live_run_surfaces_sourced_failure_modes_end_to_end(tmp_path, monkeypatch):
    """Full /api/run flow for a "live" ServerState (mode != "mock", so run_task uses
    state.pipeline as-is rather than rebuilding a mock one) with a MockLLM standing in
    for the real model and a mocked Tavily HTTP response -- no network, no API key,
    but exercises the exact same code path a real live run would."""
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_search_body(results=[
                            _result(title="Gripper slip in warehouse robots",
                                    content="Low friction causes slip during transport.",
                                    url="https://example.test/slip"),
                        ])))
    cfg = Config(seeds=(0, 1), run_timeout_s=30)
    docs = DocIndex(str(tmp_path / "no-docs"))
    library = Library(str(tmp_path / "library.jsonl"))
    llm = MockLLM(generate_outputs=[GOOD], repair_outputs=[],
                 intake={"template": "reach", "exact_match": True, "gap": "",
                         "goal": "Move the arm's gripper to the target point."})
    tavily_client = _tavily_client(tmp_path)
    pipe = Pipeline(llm, SubprocessRunner("", 30), library, docs, cfg, tavily_client=tavily_client)
    state = server.ServerState(mode="live", pipeline=pipe, library=library)

    httpd, port = start_server(state, cfg, docs)
    try:
        status, body = post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        assert status == 202
        final = wait_for_done(port)
        result = final["result"]
        assert result["source"] == "live"
        tavily = result["tavily"]
        assert tavily is not None
        assert len(tavily["items"]) >= 1
        assert tavily["items"][0]["source_url"] == "https://example.test/slip"
        assert "never as instructions" not in json.dumps(tavily)  # sourced data, not the prompt block
    finally:
        stop_server(httpd)
