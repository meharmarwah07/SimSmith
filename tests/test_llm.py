"""Tests for the hardened Token Factory client. All HTTP is mocked -- no network calls."""
import json
import os
import sys
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simsmith import llm as llm_mod  # noqa: E402
from simsmith.llm import BudgetExceeded, LLMError, TokenFactoryLLM  # noqa: E402


class FakeResponse:
    """Stands in for the context-managed object urllib.request.urlopen returns."""

    def __init__(self, body: dict):
        self._data = json.dumps(body).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok_body(content="print(1)", finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    return {
        "choices": [{
            "message": {"content": content, "reasoning_content": "<scratch space, must be ignored>"},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _llm(tmp_path, **kw):
    kw.setdefault("retries", 2)
    return TokenFactoryLLM("https://example.test/v1", "sk-test-secret",
                           usage_log_path=str(tmp_path / "usage.jsonl"), **kw)


def _http_error(code, reason="error"):
    return urllib.error.HTTPError("https://example.test/v1/chat/completions", code, reason, None, None)


# ---- content handling -------------------------------------------------------
def test_returns_content_and_ignores_reasoning_field(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_ok_body(content="  hello  ")))
    out = _llm(tmp_path).complete("sys", "user", "m")
    assert out == "hello"


def test_none_content_raises_llm_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_ok_body(content=None)))
    with pytest.raises(LLMError, match="empty content"):
        _llm(tmp_path).complete("sys", "user", "m")


def test_blank_content_raises_llm_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_ok_body(content="   ")))
    with pytest.raises(LLMError, match="empty content"):
        _llm(tmp_path).complete("sys", "user", "m")


def test_length_finish_reason_raises_llm_error(tmp_path, monkeypatch):
    # Non-empty content, but cut off: this must still be treated as a failure, not a win.
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_ok_body(content="def partial(", finish_reason="length")))
    with pytest.raises(LLMError, match="truncated"):
        _llm(tmp_path).complete("sys", "user", "m", max_tokens=50)


# ---- retries ------------------------------------------------------------------
def test_retries_5xx_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503, "Service Unavailable")
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(TokenFactoryLLM, "_backoff", staticmethod(lambda attempt: 0))
    out = _llm(tmp_path).complete("sys", "user", "m")
    assert out == "print(1)"
    assert calls["n"] == 3


def test_retries_429_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(429, "Too Many Requests")
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(TokenFactoryLLM, "_backoff", staticmethod(lambda attempt: 0))
    out = _llm(tmp_path).complete("sys", "user", "m")
    assert out == "print(1)"
    assert calls["n"] == 2


def test_non_retryable_4xx_fails_immediately(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise _http_error(400, "Bad Request")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(TokenFactoryLLM, "_backoff", staticmethod(lambda attempt: 0))
    with pytest.raises(LLMError, match="HTTP 400"):
        _llm(tmp_path).complete("sys", "user", "m")
    assert calls["n"] == 1  # no retry burned on a request that will fail the same way every time


def test_retry_cap_of_3_is_respected(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise _http_error(503, "Service Unavailable")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(TokenFactoryLLM, "_backoff", staticmethod(lambda attempt: 0))
    with pytest.raises(LLMError, match="HTTP 503"):
        _llm(tmp_path, retries=3).complete("sys", "user", "m")
    assert calls["n"] == 4  # the original attempt plus 3 retries, never more


# ---- budget guard ---------------------------------------------------------
def test_budget_exceeded_blocks_further_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMSMITH_PRICE_IN", "1.0")
    monkeypatch.setenv("SIMSMITH_PRICE_OUT", "1.0")
    monkeypatch.setattr(
        llm_mod.urllib.request, "urlopen",
        lambda req, timeout: FakeResponse(_ok_body(prompt_tokens=1_000_000, completion_tokens=1_000_000)))
    client = _llm(tmp_path, budget_usd=1.5)
    out = client.complete("sys", "user", "m")  # costs $1.00 + $1.00 = $2.00, now over the $1.50 cap
    assert out == "print(1)"
    with pytest.raises(BudgetExceeded):
        client.complete("sys", "user", "m")


def test_budget_exceeded_blocks_before_making_the_call(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    client = _llm(tmp_path, budget_usd=0.0)  # already exhausted
    with pytest.raises(BudgetExceeded):
        client.complete("sys", "user", "m")
    assert calls["n"] == 0  # never even made the HTTP request


def test_budget_from_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMSMITH_BUDGET_USD", "0")
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", lambda req, timeout: FakeResponse(_ok_body()))
    client = _llm(tmp_path)  # budget_usd not passed explicitly -> read from env
    with pytest.raises(BudgetExceeded):
        client.complete("sys", "user", "m")


def test_budget_fails_closed_when_model_has_no_price_configured(tmp_path, monkeypatch):
    # A budget cap with plenty of headroom (not "already exhausted") but NO price at
    # all for this model id must still block the call -- an unpriced call must never
    # be allowed to spend real money "for free" just because nothing is tracking it.
    for var in ("SIMSMITH_PRICE_IN", "SIMSMITH_PRICE_OUT", "SIMSMITH_PRICE_NANO_IN",
               "SIMSMITH_PRICE_NANO_OUT", "SIMSMITH_PRICE_SUPER_IN", "SIMSMITH_PRICE_SUPER_OUT",
               "SIMSMITH_PRICE_ULTRA_IN", "SIMSMITH_PRICE_ULTRA_OUT"):
        monkeypatch.delenv(var, raising=False)
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    client = _llm(tmp_path, budget_usd=1000.0)  # not exhausted -- the missing price must still block
    with pytest.raises(BudgetExceeded, match="no SIMSMITH_PRICE_"):
        client.complete("sys", "user", "some-totally-unpriced-model")
    assert calls["n"] == 0  # never even made the HTTP request
    # a priced model, same budget, same client: must succeed normally
    monkeypatch.setenv("SIMSMITH_PRICE_IN", "0.06")
    monkeypatch.setenv("SIMSMITH_PRICE_OUT", "0.24")
    assert client.complete("sys", "user", "m") == "print(1)"
    assert calls["n"] == 1


# ---- usage logging + max_tokens defaults ----------------------------------
def test_usage_logged_with_prompt_and_completion_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(_ok_body(prompt_tokens=42, completion_tokens=7)))
    log_path = tmp_path / "usage.jsonl"
    client = TokenFactoryLLM("https://example.test/v1", "sk-test-secret", usage_log_path=str(log_path))
    client.complete("sys", "user", "m", kind="plan")
    rec = json.loads(log_path.read_text().strip().splitlines()[0])
    assert rec["prompt_tokens"] == 42
    assert rec["completion_tokens"] == 7
    assert rec["kind"] == "plan"
    assert rec["model"] == "m"


def test_usage_logged_even_on_truncated_response(tmp_path, monkeypatch):
    # Tokens were still spent even though the content is unusable; cost tracking must see that.
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", lambda req, timeout: FakeResponse(
        _ok_body(content="partial", finish_reason="length", prompt_tokens=100, completion_tokens=50)))
    log_path = tmp_path / "usage.jsonl"
    client = TokenFactoryLLM("https://example.test/v1", "sk-test-secret", usage_log_path=str(log_path))
    with pytest.raises(LLMError):
        client.complete("sys", "user", "m")
    rec = json.loads(log_path.read_text().strip().splitlines()[0])
    assert rec["prompt_tokens"] == 100 and rec["completion_tokens"] == 50


def test_default_max_tokens_by_kind(tmp_path, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["payload"] = json.loads(req.data)
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    client = _llm(tmp_path)
    client.complete("sys", "user", "m", kind="plan")
    assert seen["payload"]["max_tokens"] == 3000
    client.complete("sys", "user", "m", kind="intake")
    assert seen["payload"]["max_tokens"] == 3000
    client.complete("sys", "user", "m", kind="generate")
    assert seen["payload"]["max_tokens"] == 24000
    client.complete("sys", "user", "m", kind="repair")
    assert seen["payload"]["max_tokens"] == 8000


def test_max_tokens_env_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMSMITH_MAX_TOKENS_GENERATE", "12345")
    seen = {}

    def fake_urlopen(req, timeout):
        seen["payload"] = json.loads(req.data)
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    _llm(tmp_path).complete("sys", "user", "m", kind="generate")
    assert seen["payload"]["max_tokens"] == 12345


def test_explicit_max_tokens_overrides_kind_default(tmp_path, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["payload"] = json.loads(req.data)
        return FakeResponse(_ok_body())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    _llm(tmp_path).complete("sys", "user", "m", max_tokens=99, kind="generate")
    assert seen["payload"]["max_tokens"] == 99


# ---- never logs the key or headers -----------------------------------------
def test_never_logs_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", lambda req, timeout: FakeResponse(_ok_body()))
    log_path = tmp_path / "usage.jsonl"
    secret = "sk-super-secret-value-should-never-appear"
    client = TokenFactoryLLM("https://example.test/v1", secret, usage_log_path=str(log_path))
    client.complete("sys", "user", "m")
    assert secret not in log_path.read_text()


def test_error_messages_never_include_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", lambda req, timeout: FakeResponse(_ok_body(content=None)))
    secret = "sk-super-secret-value-should-never-appear"
    client = TokenFactoryLLM("https://example.test/v1", secret, usage_log_path=str(tmp_path / "usage.jsonl"))
    with pytest.raises(LLMError) as exc_info:
        client.complete("sys", "user", "m")
    assert secret not in str(exc_info.value)
