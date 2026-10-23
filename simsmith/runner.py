"""Execution of candidate scripts in a separate process.

The same SubprocessRunner serves two cases:
  * real: sim_python is Isaac Sim's headless Python launcher (on a Nebius GPU)
  * local/test: sim_python is empty, so plain `python` runs mock scripts
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

CONTRACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")


@dataclass
class RunResult:
    ok: bool                      # process exited 0 and produced a result file
    final_state: dict = field(default_factory=dict)
    error: str = ""               # tail of stderr / reason for failure
    seconds: float = 0.0
    seed: int = 0
    perturb: dict = field(default_factory=dict)  # what was actually used for this run,
                                                   # so the verifier can recompute matching ground truth


class SubprocessRunner:
    def __init__(self, sim_python: str = "", timeout_s: int = 180):
        self.cmd = shlex.split(sim_python) if sim_python else [sys.executable]
        self.timeout_s = timeout_s
        self.runs = 0  # total executions, used for budget accounting

    def run(self, script: str, seed: int, perturb: dict | None = None) -> RunResult:
        self.runs += 1
        perturb = perturb or {}
        with tempfile.TemporaryDirectory(prefix="simsmith_") as tmp:
            script_path = os.path.join(tmp, "candidate.py")
            result_path = os.path.join(tmp, "result.json")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
            env = dict(os.environ)
            env["SIMSMITH_SEED"] = str(seed)
            env["SIMSMITH_RESULT"] = result_path
            env["SIMSMITH_PERTURB"] = json.dumps(perturb)
            env["PYTHONPATH"] = CONTRACT_DIR + os.pathsep + env.get("PYTHONPATH", "")
            env.setdefault("SIMSMITH_ROOT", os.path.dirname(CONTRACT_DIR))
            t0 = time.time()
            try:
                proc = subprocess.run(self.cmd + [script_path], cwd=tmp, env=env,
                                      capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return RunResult(False, error=f"timeout after {self.timeout_s}s",
                                 seconds=time.time() - t0, seed=seed, perturb=perturb)
            dt = time.time() - t0
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                return RunResult(False, error=f"exit {proc.returncode}: {tail}", seconds=dt, seed=seed, perturb=perturb)
            if not os.path.exists(result_path):
                return RunResult(False, error="script exited 0 but wrote no result file",
                                 seconds=dt, seed=seed, perturb=perturb)
            try:
                with open(result_path, encoding="utf-8") as f:
                    data = json.load(f)
            except ValueError as e:
                return RunResult(False, error=f"unreadable result file: {e}", seconds=dt, seed=seed, perturb=perturb)
            return RunResult(True, final_state=data.get("final_state", {}), seconds=dt, seed=seed, perturb=perturb)
