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
import threading
import time
from dataclasses import dataclass, field

CONTRACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")

# Only these survive into a candidate script's process -- explicitly NOT the parent's
# full os.environ, which would otherwise hand a generated script TOKEN_FACTORY_API_KEY,
# TAVILY_API_KEY, or anything else the harness process happens to have. None of PATH/
# HOME/LANG/LC_ALL are secret; they're just what the interpreter/mujoco/numpy expect to
# find (confirmed empirically: a bare PATH+HOME env is sufficient to import mujoco,
# build a model and mj_step it -- see docs/diagnosis_2026-10-24.md Part 2 finding #1).
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL")

# Best-effort network isolation for the candidate's subprocess. Plain `unshare --net`
# needs a privilege (CAP_SYS_ADMIN in the CURRENT user namespace) that an unprivileged
# process doesn't have here -- confirmed: `unshare --net true` fails with "Operation
# not permitted" in this environment. `unshare --net --map-root-user` instead first
# enters a new, unprivileged *user* namespace (mapping the calling user to root inside
# it, which is the part unprivileged users ARE allowed to do), and creating a network
# namespace from inside that succeeds -- confirmed: a script run this way gets a
# loopback-only interface and a real network call fails with "Temporary failure in
# name resolution." Probed once per process (results cached): if neither form works in
# some other environment, candidate scripts still run (unsandboxed, as before), with
# no network isolation at all -- `SubprocessRunner.network_isolated` says which case
# applies, and callers/reports should surface it rather than assume isolation happened.
_UNSHARE_NET_CANDIDATES = (["unshare", "--net", "--map-root-user"], ["unshare", "--net"])
_unshare_net_prefix: list | None = None
_unshare_probe_lock = threading.Lock()


def _probe_unshare_net() -> list:
    global _unshare_net_prefix
    with _unshare_probe_lock:
        if _unshare_net_prefix is not None:
            return _unshare_net_prefix
        for candidate in _UNSHARE_NET_CANDIDATES:
            try:
                r = subprocess.run(candidate + ["true"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    _unshare_net_prefix = candidate
                    return candidate
            except (OSError, subprocess.TimeoutExpired):
                continue
        _unshare_net_prefix = []
        return []


@dataclass
class RunResult:
    ok: bool                      # process exited 0 and produced a result file
    final_state: dict = field(default_factory=dict)
    error: str = ""               # tail of stderr / reason for failure
    seconds: float = 0.0
    seed: int = 0
    perturb: dict = field(default_factory=dict)  # what was actually used for this run,
                                                   # so the verifier can recompute matching ground truth
    exit_code: int | None = None  # None means the process never completed (e.g. timeout)
    stdout: str = ""               # full captured stdout, for post-hoc diagnosis
    stderr: str = ""               # full captured stderr, for post-hoc diagnosis


class SubprocessRunner:
    def __init__(self, sim_python: str = "", timeout_s: int = 180):
        self.cmd = shlex.split(sim_python) if sim_python else [sys.executable]
        self.timeout_s = timeout_s
        self.runs = 0  # total executions, used for budget accounting
        self._runs_lock = threading.Lock()  # a caller may run candidates concurrently
        self._unshare_prefix = _probe_unshare_net()  # probed once per runner, cheap after that

    @property
    def network_isolated(self) -> bool:
        return bool(self._unshare_prefix)

    def run(self, script: str, seed: int, perturb: dict | None = None, template: str = "") -> RunResult:
        with self._runs_lock:
            self.runs += 1
        perturb = perturb or {}
        with tempfile.TemporaryDirectory(prefix="simsmith_") as tmp:
            script_path = os.path.join(tmp, "candidate.py")
            result_path = os.path.join(tmp, "result.json")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
            env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
            env["SIMSMITH_SEED"] = str(seed)
            env["SIMSMITH_RESULT"] = result_path
            env["SIMSMITH_PERTURB"] = json.dumps(perturb)
            env["SIMSMITH_TEMPLATE"] = template  # lets the contract check model identity against the
                                                   # right official scene; empty = check skipped (unknown template)
            env["PYTHONPATH"] = CONTRACT_DIR
            env["SIMSMITH_ROOT"] = os.environ.get("SIMSMITH_ROOT", os.path.dirname(CONTRACT_DIR))
            cmd = self._unshare_prefix + self.cmd + [script_path]
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=tmp, env=env,
                                      capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired as e:
                return RunResult(False, error=f"timeout after {self.timeout_s}s",
                                 seconds=time.time() - t0, seed=seed, perturb=perturb,
                                 exit_code=None,
                                 stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                                 stderr=(e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
            dt = time.time() - t0
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                return RunResult(False, error=f"exit {proc.returncode}: {tail}", seconds=dt, seed=seed, perturb=perturb,
                                 exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
            if not os.path.exists(result_path):
                return RunResult(False, error="script exited 0 but wrote no result file",
                                 seconds=dt, seed=seed, perturb=perturb,
                                 exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
            try:
                with open(result_path, encoding="utf-8") as f:
                    data = json.load(f)
            except ValueError as e:
                return RunResult(False, error=f"unreadable result file: {e}", seconds=dt, seed=seed, perturb=perturb,
                                 exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
            return RunResult(True, final_state=data.get("final_state", {}), seconds=dt, seed=seed, perturb=perturb,
                             exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
