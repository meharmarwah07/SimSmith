"""LLM access. Token Factory exposes an OpenAI-compatible chat API.

Hardening rules (learned the hard way against a real reasoning model):
  * Only `choices[0].message.content` is ever treated as the answer. A
    `reasoning`/`reasoning_content` field, if the model returns one, is
    ignored -- it is the model's scratch space, not its answer.
  * An empty/None content, or `finish_reason == "length"` (the response was
    cut off before it finished), is always a raised LLMError, never a
    silently-returned empty string a caller might extract_code() on.
  * Every call is logged (model, kind, tokens, cost, latency) to
    eval/results/usage.jsonl, and a $ budget guard (SIMSMITH_BUDGET_USD) can
    stop further calls before they run up a bill nobody noticed.
  * The API key and request headers are never written to a log or an
    exception message.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str, model: str = "", temperature: float = 0.5,
                 max_tokens: int | None = None, *, kind: str = "generate") -> str: ...


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    """Raised instead of making a call once accumulated spend hits SIMSMITH_BUDGET_USD."""


# Default max_tokens per call "kind", overridable per-kind with
# SIMSMITH_MAX_TOKENS_<KIND> (e.g. SIMSMITH_MAX_TOKENS_GENERATE=12000).
DEFAULT_MAX_TOKENS = {
    "generate": 24000,  # Nemotron Nano spends a large, variable chunk of this on its
                        # own "reasoning" field before any script content; 8000 was
                        # measured truncating 100% of pick_place attempts (mujoco-v1 baseline).
    "repair": 8000,
    "plan": 3000,
    "intake": 3000,
}
_FALLBACK_MAX_TOKENS = 4000

# Retrying 429 (rate limit) and 5xx (server-side) is almost always right; retrying
# any other 4xx (bad request, auth, not found) just burns time on a request that
# will fail identically every time, so those raise immediately instead.
_RETRYABLE_STATUS = lambda code: code == 429 or 500 <= code < 600  # noqa: E731


def _max_tokens_for(kind: str) -> int:
    env_key = f"SIMSMITH_MAX_TOKENS_{kind.upper()}"
    default = DEFAULT_MAX_TOKENS.get(kind, _FALLBACK_MAX_TOKENS)
    return int(os.environ.get(env_key, default))


def max_tokens_config() -> dict:
    """The actual max_tokens in effect for every call kind right now (after any
    SIMSMITH_MAX_TOKENS_<KIND> override) -- for stamping into a result row, see
    simsmith/run_meta.py."""
    return {kind: _max_tokens_for(kind) for kind in DEFAULT_MAX_TOKENS}


def _budget_from_env() -> float | None:
    raw = os.environ.get("SIMSMITH_BUDGET_USD", "")
    return float(raw) if raw else None


# Per-model price keys, checked as a case-insensitive substring of the model id (e.g.
# "nvidia/Nemotron-3-Ultra-550b-a55b" -> "ULTRA"), so SIMSMITH_PRICE_<KEY>_IN/_OUT can
# give each model tier its own real rate instead of one flat price applied to whichever
# model happens to be configured. Order matters only in that the first match wins; the
# three tiers don't overlap in practice.
_PRICE_KEY_PATTERNS = ("ultra", "super", "nano")


def _price_for_model(model: str) -> tuple[float, float] | None:
    """(price_in, price_out) per 1M tokens for `model`, or None if no price is
    configured for it -- callers must treat None as "unknown," never as $0."""
    low = model.lower()
    for pat in _PRICE_KEY_PATTERNS:
        if pat in low:
            key = pat.upper()
            in_raw = os.environ.get(f"SIMSMITH_PRICE_{key}_IN")
            out_raw = os.environ.get(f"SIMSMITH_PRICE_{key}_OUT")
            if in_raw is not None and out_raw is not None:
                return float(in_raw), float(out_raw)
            break
    # Fall back to one flat rate for any model (kept for simple single-model setups,
    # e.g. the test suite, and for a model id matching none of the tiers above).
    in_raw = os.environ.get("SIMSMITH_PRICE_IN")
    out_raw = os.environ.get("SIMSMITH_PRICE_OUT")
    if in_raw is not None and out_raw is not None:
        return float(in_raw), float(out_raw)
    return None


class TokenFactoryLLM:
    """Plain-HTTP client for an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str, retries: int = 3, timeout: int = 120,
                 usage_log_path: str | None = None, budget_usd: float | None = None,
                 extra_payload: dict | None = None):
        if not base_url or not api_key:
            raise LLMError("TOKEN_FACTORY_BASE_URL and TOKEN_FACTORY_API_KEY must be set")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.retries = retries
        self.timeout = timeout
        self.calls = 0
        self.usage_log_path = (
            usage_log_path if usage_log_path is not None
            else os.environ.get("SIMSMITH_USAGE_LOG", "eval/results/usage.jsonl")
        )
        self.budget_usd = budget_usd if budget_usd is not None else _budget_from_env()
        self.spent_usd = 0.0
        self._lock = threading.Lock()  # guards spent_usd/calls/usage-log across concurrent complete()s
        self._tls = threading.local()  # per-thread tag, so concurrent callers' usage rows can be told apart
        # Extra top-level JSON fields merged into every request, e.g.
        # {"chat_template_kwargs": {"enable_thinking": False}} to suppress a reasoning
        # model's "reasoning" field (Token Factory / Nemotron Nano specific; see
        # docs_index/mujoco_notes.md's notes on the "reasoning" field for why this
        # exists at all -- it's never read as the answer either way).
        self.extra_payload = extra_payload or {}

    def set_tag(self, tag) -> None:
        """Label every usage record this (calling) thread logs from here on with `tag`,
        e.g. an opaque generation id. Needed because with concurrent complete() calls
        (Pipeline.generate()'s ThreadPoolExecutor, or a caller parallelizing across
        whole generations) usage rows interleave in the log and can't be attributed by
        position; the tag is how a caller recovers "which generation cost what."

        NOTE: a thread's tag does NOT propagate to a ThreadPoolExecutor worker thread it
        spawns (thread-locals aren't inherited) -- a caller that wants per-generation
        attribution across its own nested pool (as Pipeline.generate() does) must read
        get_tag() in the parent and call set_tag() again inside each worker."""
        self._tls.tag = tag

    def get_tag(self):
        return getattr(self._tls, "tag", None)

    def complete(self, system: str, user: str, model: str = "", temperature: float = 0.5,
                 max_tokens: int | None = None, *, kind: str = "generate") -> str:
        if not model:
            raise LLMError("model id is empty; set SIMSMITH_MODEL_* in .env")
        if max_tokens is None:
            max_tokens = _max_tokens_for(kind)
        with self._lock:
            if self.budget_usd is not None:
                # Fail CLOSED, not open: a budget cap is worthless if it silently stops
                # counting the moment a model has no price configured (previously,
                # _price_for_model() returning None just meant cost_usd=None and
                # spent_usd never grew for that model's calls -- unpriced spend was
                # invisible to the guard, not blocked by it). If you set a budget, every
                # model it's meant to cover must have a real SIMSMITH_PRICE_* -- see
                # docs/diagnosis_2026-10-24.md's "High" finding on this.
                if _price_for_model(model) is None:
                    raise BudgetExceeded(
                        f"budget ${self.budget_usd:.4f} is set but no SIMSMITH_PRICE_* is "
                        f"configured for model {model!r}; failing closed rather than letting "
                        f"unpriced spend go uncounted against the cap (call blocked before it ran)")
                if self.spent_usd >= self.budget_usd:
                    raise BudgetExceeded(
                        f"budget ${self.budget_usd:.4f} reached (spent ${self.spent_usd:.4f}); call blocked before it ran")

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self.extra_payload)
        t0 = time.time()
        body = self._post_with_retries(payload)
        seconds = time.time() - t0
        with self._lock:
            self.calls += 1

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        content = message.get("content")  # "reasoning"/"reasoning_content", if present, is ignored

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        self._log_usage(model, kind, prompt_tokens, completion_tokens, seconds, finish_reason)

        if finish_reason == "length":
            raise LLMError(
                f"{model} response truncated (finish_reason='length'): max_tokens={max_tokens}, "
                f"completion_tokens={completion_tokens}. Raise max_tokens or shorten the prompt.")
        if content is None or not str(content).strip():
            raise LLMError(f"{model} returned empty content (finish_reason={finish_reason!r})")
        return content.strip()

    def _post_with_retries(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if not _RETRYABLE_STATUS(e.code) or attempt >= self.retries:
                    raise LLMError(f"HTTP {e.code} from Token Factory: {e.reason}") from e
                last = e
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                # connection-level failures (DNS, reset, timeout) and bad JSON bodies: same
                # transient-fault treatment as 5xx, up to the retry cap.
                if attempt >= self.retries:
                    raise LLMError(f"LLM call failed after {attempt + 1} attempts: {e}") from e
                last = e
            time.sleep(self._backoff(attempt))
        raise LLMError(f"LLM call failed after retries: {last}")  # unreachable in practice

    @staticmethod
    def _backoff(attempt: int) -> float:
        return (2 ** attempt) + random.uniform(0, 0.5)

    def _log_usage(self, model: str, kind: str, prompt_tokens: int, completion_tokens: int,
                    seconds: float, finish_reason) -> float | None:
        price = _price_for_model(model)
        cost = None
        if price is not None:
            price_in, price_out = price
            cost = prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out
        tag = getattr(self._tls, "tag", None)
        with self._lock:
            if cost is not None:
                self.spent_usd += cost
            if self.usage_log_path:
                record = {
                    "ts": time.time(), "model": model, "kind": kind, "tag": tag,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    # cost_usd is None (never a fake $0) when no SIMSMITH_PRICE_* is
                    # configured for this model; price_known tells a reader which it is
                    # without having to guess from cost_usd == 0 being a real zero cost.
                    "cost_usd": cost, "price_known": cost is not None,
                    "seconds": seconds, "finish_reason": finish_reason,
                    "cumulative_spent_usd": self.spent_usd,
                }
                dirname = os.path.dirname(self.usage_log_path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                with open(self.usage_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        return cost
