"""Cheap pre-flight checks that run BEFORE any GPU time is spent.

Rejects scripts that do not parse, import disallowed modules, call dangerous
builtins, never call the contract's write_result, never actually step the
MuJoCo simulation forward in time (import mujoco + a call to mj_step), or
reference a mujoco.<attr> chain that does not exist on the pinned mujoco
actually installed here (catches hallucinated API names before any sim time
is spent on them). If an allow-list of Isaac API names is supplied (from your
pinned version), attribute chains on the Isaac namespaces are checked against
it too -- that path is for an eventual Isaac Sim runner and is inert unless an
allow-list is actually configured.
"""
from __future__ import annotations

import ast
import difflib
import os
from dataclasses import dataclass, field

try:
    import mujoco as _mujoco_module
except ImportError:
    _mujoco_module = None

BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open_url", "input"}
BANNED_MODULES = {"subprocess", "socket", "requests", "urllib", "http", "ftplib",
                  "shutil", "ctypes", "multiprocessing", "pty"}
BANNED_ATTR_CALLS = {("os", "system"), ("os", "popen"), ("os", "remove"),
                     ("os", "rmdir"), ("os", "unlink"), ("os", "kill")}
ISAAC_ROOTS = ("omni", "isaacsim", "pxr", "isaaclab")


@dataclass
class PreflightResult:
    ok: bool
    problems: list = field(default_factory=list)


def load_allowlist(path: str) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _check_mujoco_attrs(tree: ast.AST, problems: list) -> None:
    """Resolve every mujoco.<attr> (and chained attrs, e.g. mujoco.mjtObj.mjOBJ_BODY)
    against the real, installed mujoco module by getattr-walking it -- not a
    hand-maintained symbol list, so it can't go stale against the pinned version.
    Rejects a name that doesn't exist, with difflib close matches at the point it
    broke (e.g. a typo'd mj_setp suggests mj_step)."""
    if _mujoco_module is None:
        return  # no mujoco in this process; the broader import/mj_step check still applies
    bound_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "mujoco":
                    bound_names.add(a.asname or "mujoco")
    if not bound_names:
        return
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted(node)
        if not dotted:
            continue
        parts = dotted.split(".")
        if parts[0] not in bound_names or len(parts) < 2 or dotted in seen:
            continue
        seen.add(dotted)
        obj = _mujoco_module
        for depth, part in enumerate(parts[1:], start=1):
            if not hasattr(obj, part):
                candidates = [n for n in dir(obj) if not n.startswith("_")]
                close = difflib.get_close_matches(part, candidates, n=3)
                where = ".".join(["mujoco"] + parts[1:depth])
                suggestion = f"; did you mean: {', '.join(close)}?" if close else ""
                problems.append(f"unknown attribute: {dotted} -- '{part}' does not exist "
                                f"on {where} in this mujoco version{suggestion}")
                break
            obj = getattr(obj, part)


def check(script: str, api_allowlist: set | None = None) -> PreflightResult:
    problems = []
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return PreflightResult(False, [f"syntax error: {e.msg} (line {e.lineno})"])

    calls_write = False
    imports_mujoco = False
    calls_mj_step = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_MODULES:
                    problems.append(f"banned import: {a.name}")
                if a.name.split(".")[0] == "mujoco":
                    imports_mujoco = True
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_MODULES:
                problems.append(f"banned import: {node.module}")
            if root == "mujoco":
                imports_mujoco = True
                # `from mujoco import mj_step` binds a bare name; the actual CALL (not
                # just the import) is what the ast.Call branch below looks for.
            if api_allowlist and root in ISAAC_ROOTS:
                full = node.module
                if not any(full == a or full.startswith(a + ".") or a.startswith(full + ".")
                           for a in api_allowlist):
                    problems.append(f"module not in pinned-version allow-list: {full}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_CALLS:
                    problems.append(f"banned call: {node.func.id}()")
                if node.func.id == "write_result":
                    calls_write = True
                if node.func.id == "mj_step":
                    calls_mj_step = True
            elif isinstance(node.func, ast.Attribute):
                d = _dotted(node.func)
                if d:
                    parts = d.split(".")
                    if len(parts) >= 2 and (parts[0], parts[1]) in BANNED_ATTR_CALLS:
                        problems.append(f"banned call: {d}()")
                    if parts[-1] == "write_result":
                        calls_write = True
                if node.func.attr == "mj_step":
                    calls_mj_step = True
    if not calls_write:
        problems.append("script never calls write_result(final_state)")
    _check_mujoco_attrs(tree, problems)
    if not (imports_mujoco and calls_mj_step):
        # A script cannot pass by faking the result -- it has to actually simulate.
        # This is the cheap, static half of that; the contract itself re-measures the
        # outcome at runtime and ignores whatever the script reports for those fields.
        problems.append("script must import mujoco and call mj_step(...) to advance "
                        "the simulation; reporting a position without ever stepping physics "
                        "forward in time is not accepted")
    return PreflightResult(not problems, problems)
