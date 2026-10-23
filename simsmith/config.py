"""Configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so the project stays dependency-free."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Config:
    base_url: str = ""
    api_key: str = ""
    model_plan: str = ""
    model_generate: str = ""
    model_fast: str = ""
    sim_python: str = ""
    run_timeout_s: int = 180
    seeds: tuple = field(default_factory=lambda: (0, 1, 2))

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv()
        n_seeds = int(os.environ.get("SIMSMITH_SEEDS", "3"))
        return cls(
            base_url=os.environ.get("TOKEN_FACTORY_BASE_URL", ""),
            api_key=os.environ.get("TOKEN_FACTORY_API_KEY", ""),
            model_plan=os.environ.get("SIMSMITH_MODEL_PLAN", ""),
            model_generate=os.environ.get("SIMSMITH_MODEL_GENERATE", ""),
            model_fast=os.environ.get("SIMSMITH_MODEL_FAST", ""),
            sim_python=os.environ.get("SIMSMITH_SIM_PYTHON", ""),
            run_timeout_s=int(os.environ.get("SIMSMITH_RUN_TIMEOUT_S", "180")),
            seeds=tuple(range(n_seeds)),
        )
