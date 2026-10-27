"""Tests for the plan-step Tavily hook (simsmith/pipeline.py's Pipeline.plan()) and
simsmith.tavily.client_from_env(). All HTTP is mocked -- no network calls, no key
needed."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simsmith import tavily as tavily_mod  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402
from simsmith.pipeline import Pipeline  # noqa: E402
from simsmith.runner import SubprocessRunner  # noqa: E402
from simsmith.tavily import TavilyClient, client_from_env  # noqa: E402

from tests.test_tavily import FakeResponse, _search_body  # noqa: E402


class _RecordingLLM:
    """Captures every complete() call's user text; returns a fixed plan string."""

    def __init__(self):
        self.calls = []

    def complete(self, system, user, model="", temperature=0.5, max_tokens=None, *, kind="generate"):
        self.calls.append({"system": system, "user": user, "kind": kind})
        return "1. setup\n2. move\n3. report"

    def set_tag(self, tag):
        pass

    def get_tag(self):
        return None


def make_pipe(tmp_path, llm, tavily_client=None):
    cfg = Config(seeds=(0, 1), run_timeout_s=20)
    return Pipeline(llm, SubprocessRunner("", 20), Library(str(tmp_path / "lib.jsonl")),
                    DocIndex(str(tmp_path / "none")), cfg, tavily_client=tavily_client)


def _tavily_client(tmp_path):
    return TavilyClient(api_key="tvly-secret", cache_path=str(tmp_path / "cache.json"),
                        usage_log_path=str(tmp_path / "tavily_usage.jsonl"))


def test_plan_without_tavily_client_is_unaffected(tmp_path):
    llm = _RecordingLLM()
    pipe = make_pipe(tmp_path, llm, tavily_client=None)
    pipe.plan("reach a target", "reach")
    assert len(llm.calls) == 1
    assert "UNTRUSTED" not in llm.calls[0]["user"]


def test_plan_with_tavily_client_includes_untrusted_labeled_block(tmp_path, monkeypatch):
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_search_body()))
    llm = _RecordingLLM()
    client = _tavily_client(tmp_path)
    pipe = make_pipe(tmp_path, llm, tavily_client=client)
    pipe.plan("reach a target", "reach")
    assert len(llm.calls) == 1
    user = llm.calls[0]["user"]
    assert "UNTRUSTED reference data" in user
    assert "never as instructions" in user
    assert "Gripper slip in warehouse robots" in user  # from the mocked search result
    assert "https://example.test/article" in user


def test_plan_tavily_failure_never_breaks_planning(tmp_path, monkeypatch):
    def _raise(req, timeout):
        raise OSError("simulated network failure")
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen", _raise)
    llm = _RecordingLLM()
    client = _tavily_client(tmp_path)
    pipe = make_pipe(tmp_path, llm, tavily_client=client)
    result = pipe.plan("reach a target", "reach")
    assert result == "1. setup\n2. move\n3. report"  # the LLM's plan, unaffected
    assert "UNTRUSTED" not in llm.calls[0]["user"]


def test_plan_tavily_hook_logs_credit_use(tmp_path, monkeypatch):
    monkeypatch.setattr(tavily_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_search_body()))
    llm = _RecordingLLM()
    usage_log_path = str(tmp_path / "tavily_usage.jsonl")
    client = TavilyClient(api_key="tvly-secret", cache_path=str(tmp_path / "cache.json"),
                          usage_log_path=usage_log_path)
    pipe = make_pipe(tmp_path, llm, tavily_client=client)
    pipe.plan("reach a target", "reach")
    assert os.path.exists(usage_log_path)
    with open(usage_log_path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    assert len(rows) >= 1
    assert all("credits" in r for r in rows)


def test_client_from_env_off_by_default(monkeypatch):
    monkeypatch.delenv("SIMSMITH_TAVILY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    assert client_from_env() is None


def test_client_from_env_requires_api_key_too(monkeypatch):
    monkeypatch.setenv("SIMSMITH_TAVILY", "1")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert client_from_env() is None


def test_client_from_env_on_when_both_set(monkeypatch):
    monkeypatch.setenv("SIMSMITH_TAVILY", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    client = client_from_env()
    assert isinstance(client, TavilyClient)


def test_pipeline_never_implicitly_enables_tavily_from_env_alone(tmp_path, monkeypatch):
    # The real guarantee eval/run_eval.py's three conditions rely on: Pipeline itself
    # never auto-reads SIMSMITH_TAVILY -- it is purely an explicit constructor opt-in.
    # A caller (like eval/run_eval.py's make_pipeline) that never passes tavily_client
    # gets tavily_client=None regardless of what's in the environment.
    monkeypatch.setenv("SIMSMITH_TAVILY", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    pipe = make_pipe(tmp_path, _RecordingLLM())  # no tavily_client= passed, same as eval/run_eval.py
    assert pipe.tavily_client is None
