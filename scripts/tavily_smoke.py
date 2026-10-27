"""Manual smoke test: real Tavily search for both templates, an optional LLM extraction
pass over the results, then planned perturbations. Makes real HTTP calls and spends real
Tavily (and, if configured, Token Factory) credits -- run by hand, not from CI or any
test suite.

Usage:
  set -a; source .env; set +a   # needs TAVILY_API_KEY
  python scripts/tavily_smoke.py

Stays within 4 Tavily searches total: each template is capped at 2 (research_failure_
modes's own max_searches=2 default), and the shared TavilyClient's cap covers both.
Every call goes through TavilyClient's usual cache + credit log at
eval/results/tavily_usage.jsonl, so repeat runs of this script cost nothing.

If TOKEN_FACTORY_BASE_URL/TOKEN_FACTORY_API_KEY and SIMSMITH_MODEL_FAST ("nano") are also
set, failure-mode extraction uses that model; otherwise it's skipped and the deterministic
one-failure-mode-per-surviving-result mapping is used instead (see simsmith/tavily.py).
Perturbation planning itself always uses the deterministic fallback here, regardless.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith.config import Config  # noqa: E402
from simsmith.perturb_plan import plan_perturbations  # noqa: E402
from simsmith.tavily import TavilyClient, TavilyError, research_failure_modes  # noqa: E402

TASKS = {
    "reach": "Move the arm end effector to a target position.",
    "pick_place": "Pick up a cube and place it in a bin.",
}


def _build_llm(cfg: Config):
    if not (cfg.base_url and cfg.api_key and cfg.model_fast):
        print("(no TOKEN_FACTORY_BASE_URL/API_KEY/SIMSMITH_MODEL_FAST -- skipping LLM extraction)")
        return None, ""
    from simsmith.llm import TokenFactoryLLM
    return TokenFactoryLLM(cfg.base_url, cfg.api_key), cfg.model_fast


def main() -> None:
    if not os.environ.get("TAVILY_API_KEY"):
        sys.exit("Set TAVILY_API_KEY first (see .env.example).")

    cfg = Config.from_env()
    llm, model = _build_llm(cfg)

    # Two templates, capped at 2 searches each -- the client's cap is shared across the
    # whole run, so it must cover both (4), not just one.
    client = TavilyClient(max_searches=4)
    for template, task in TASKS.items():
        print(f"\n=== {template} ===")
        failure_modes = research_failure_modes(task, template, client, max_searches=2, llm=llm, model=model)

        print(f"queries used: {failure_modes.queries_used}")

        print(f"\n{len(failure_modes.dropped)} source(s) dropped:")
        for d in failure_modes.dropped:
            print(f"  - [{d.title}]({d.url}): {d.reason}")

        print(f"\n{len(failure_modes.items)} failure mode(s) extracted:")
        for item in failure_modes.items:
            print(f"  - {item.mode}")
            print(f"    why it matters: {item.why_it_matters}")
            print(f"    source: [{item.source_title}]({item.source_url})")

        specs = plan_perturbations(failure_modes, template)  # deterministic fallback, no LLM
        print(f"\n{len(specs)} perturbation(s) planned:")
        for spec in specs:
            print(f"  - {spec.kind}={spec.value} (seed_offset={spec.seed_offset}) "
                  f"from [{spec.source_title}]({spec.source_url}): {spec.rationale}")

    print(f"\nsearches made this run: {client.searches_made}")


if __name__ == "__main__":
    try:
        main()
    except TavilyError as e:
        sys.exit(str(e))
