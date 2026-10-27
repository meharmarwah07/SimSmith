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

THREAT MODEL (see README.md "Security" section for the full statement): this is an
AST-literal pattern check, a cheap pre-filter against accidental/careless gaming, NOT a
sandbox and not a defense against a deliberately adversarial script. It can always be
defeated by one more level of indirection (string-built module names, reflection,
etc.) -- see docs/diagnosis_2026-10-24.md Part 2 for confirmed bypasses (importlib,
getattr) that this file's bans close, and the ones that remain open regardless
(anything requiring real process isolation). Real security against a hostile script
author requires sandboxed execution (simsmith/runner.py's network-namespace isolation
and stripped environment are a step in that direction, not a complete answer).
"""
from __future__ import annotations

import ast
import builtins
import difflib
import importlib
import os
from dataclasses import dataclass, field

try:
    import mujoco as _mujoco_module
except ImportError:
    _mujoco_module = None

# getattr/setattr are banned by default (not just "getattr(os, ...)") because the point
# (docs/diagnosis_2026-10-24.md Part 2 #6) is that ANY getattr/setattr call can reach a
# banned attribute via a non-literal string (getattr(os, "sys"+"tem")), which no AST
# pattern on the ARGUMENT can fully close -- banning the call itself was the only static
# fix at the time.
#
# diagnosis 2026-10-25: a blanket ban on the call also rejects common, harmless honest
# idioms that have nothing to do with os/sys/simsmith_contract -- e.g.
# getattr(args, "tol", 1e-3) on a plain argparse Namespace, or
# setattr(cfg, "max_steps", 1000) on a local SimpleNamespace. Measured on the real
# nemotron-super reach runs this never actually fired (the model doesn't spontaneously
# reach for getattr/setattr), so it isn't the cause of any known regression -- but it's
# a real false-positive tax on scripts that do use these idioms. _is_safe_getattr_setattr()
# below narrows the ban to exactly the dangerous shape: allow getattr/setattr only when
# (a) the attribute-name argument is a literal string, not a computed one (closes the
# "sys"+"tem" bypass outright -- a non-literal name is still unconditionally rejected),
# (b) that literal isn't a dunder name, and (c) the target is a plain local Name that
# isn't bound to os/sys/importlib/builtins/any BANNED_MODULES alias or a
# `import simsmith_contract` alias (those keep going through their own, stricter
# checks -- BANNED_ATTR_CALLS/BANNED_ATTR_ACCESS/_check_simsmith_contract_attrs --
# applied to ordinary `.attr` access, which this rule does not bypass since a sensitive
# target is rejected before the literal-name allowance is ever consulted). open()/vars()
# are NOT narrowed the same way: open() can read arbitrary files from an SIMSMITH_ROOT-
# derived path with no safe literal-shape to allow-list, and vars() can expose
# locals()/an object's full __dict__ -- neither has a clean static safety condition, so
# both stay banned outright (the honest open()-of-the-scene-XML / vars(args)-for-
# debug-printing idioms are real false positives too; sandboxing, not a smarter AST
# rule, is the actual fix -- see this file's THREAT MODEL note above).
#
# Same reasoning as before for globals/vars (exposes a module's/object's namespace, which
# can hold imported module objects or secrets) and builtins/importlib as modules
# (reachable via getattr(builtins, "ev"+"al") / importlib.import_module("socket") to
# bypass BANNED_CALLS / BANNED_MODULES otherwise). "open_url" was a typo for "open" in
# the previous version -- open was never actually banned.
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input",
                "getattr", "setattr", "globals", "vars"}
BANNED_MODULES = {"subprocess", "socket", "requests", "urllib", "http", "ftplib",
                  "shutil", "ctypes", "multiprocessing", "pty", "importlib", "builtins"}
BANNED_ATTR_CALLS = {("os", "system"), ("os", "popen"), ("os", "remove"),
                     ("os", "rmdir"), ("os", "unlink"), ("os", "kill")}
# Non-call attribute ACCESS (read OR write; ast.walk doesn't distinguish Load/Store,
# which is exactly what's needed here -- "sc._measure_from_mujoco = ..." is caught the
# same as reading it) that's banned regardless of whether it's then called.
BANNED_ATTR_ACCESS = {("sys", "modules")}
# Targets a getattr/setattr call is never allowed to touch, literal name or not --
# these keep going through their own dedicated, stricter checks instead (see
# _is_safe_getattr_setattr below).
GETATTR_SETATTR_SENSITIVE_MODULES = BANNED_MODULES | {"os", "sys"}
# Public contract surface a generated script is allowed to touch on simsmith_contract;
# everything else (_measure_from_mujoco, _MJ_TRACKED, _hook_mujoco, ...) is the
# harness's own "private" state -- see docs/diagnosis_2026-10-24.md Part 4 exploits #1
# ("monkeypatch_measurement") and #2 ("fake_mj_tracked_entry"), both of which require
# exactly this kind of out-of-contract attribute access on an `import simsmith_contract`
# reference (the normal `from simsmith_contract import get_task, ...` form never
# creates an Attribute node at all, so legitimate usage is unaffected).
SIMSMITH_CONTRACT_PUBLIC_API = {"get_task", "get_rng", "get_perturb", "write_result"}
ISAAC_ROOTS = ("omni", "isaacsim", "pxr", "isaaclab")

# --- skills ablation (pick_place-only; runtime/simsmith_skills.py) --------------
# move_to()/set_gripper() call the (hooked) mujoco.mj_step internally for every real
# step they take -- see that module's docstring -- so a script that only calls THEM,
# never mj_step itself, still genuinely simulates. This AST walk can't see inside
# simsmith_skills.py (it only parses the candidate script), so without naming these
# two here, a skills-only script would be statically (and wrongly) rejected below as
# "never steps physics." Scoped to exactly these two names from exactly this one
# module -- nothing else about simsmith_skills, or anything about simsmith_contract's
# own restricted public API above, is affected by this.
SIMSMITH_SKILLS_STEPPING_HELPERS = {"move_to", "set_gripper"}


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


def _bound_aliases_for_module(tree: ast.AST, module_name: str) -> set:
    """Names in the script bound to the MODULE OBJECT of module_name, via
    `import module_name` or `import module_name as alias` -- NOT `from module_name
    import x`, which binds specific names/functions, never the module object, and so
    never creates an Attribute node on it at all."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module_name:
                    aliases.add(a.asname or module_name)
    return aliases


def _sensitive_getattr_setattr_aliases(tree: ast.AST) -> set:
    aliases = set()
    for module_name in GETATTR_SETATTR_SENSITIVE_MODULES:
        aliases |= _bound_aliases_for_module(tree, module_name)
    return aliases


def _is_safe_getattr_setattr(node: ast.Call, sensitive_aliases: set, contract_aliases: set) -> bool:
    """getattr(obj, name[, default]) / setattr(obj, name, value) is allowed only when
    the attribute name is a literal string (not e.g. "sys" + "tem") AND the target is a
    plain local Name that isn't an alias for os/sys/importlib/builtins/any other
    BANNED_MODULES entry, or for `import simsmith_contract` -- see the module-level
    comment above BANNED_CALLS for the full reasoning. Anything that doesn't meet this
    stays rejected by the caller, same as before this narrowing existed."""
    if len(node.args) < 2:
        return False
    name_arg = node.args[1]
    if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
        return False
    literal = name_arg.value
    if literal.startswith("__") and literal.endswith("__"):
        return False
    target = node.args[0]
    if not isinstance(target, ast.Name):
        return False
    if target.id in sensitive_aliases or target.id in contract_aliases:
        return False
    return True


def _check_dunder_attrs(tree: ast.AST, problems: list) -> None:
    """Any `obj.__something__` attribute access (reflection/introspection escape
    hatches like __globals__, __builtins__, __subclasses__, __dict__, __loader__,
    __class__...) is banned outright, regardless of what obj is -- a generated script
    has no legitimate reason to touch one. Does NOT flag defining a dunder method
    (`def __init__(self):` is a FunctionDef, never an Attribute node)."""
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in seen:
                seen.add(node.attr)
                problems.append(f"banned dunder attribute access: .{node.attr}")


def _check_banned_attr_access(tree: ast.AST, problems: list) -> None:
    """Non-call attribute access against BANNED_ATTR_ACCESS, e.g. sys.modules --
    reachable even as a bare reference/subscript (sys.modules["os"]), not just a call,
    so this is separate from BANNED_ATTR_CALLS."""
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        d = _dotted(node)
        if not d:
            continue
        parts = d.split(".")
        if len(parts) >= 2 and (parts[0], parts[1]) in BANNED_ATTR_ACCESS and d not in seen:
            seen.add(d)
            problems.append(f"banned attribute access: {d}")


def _check_simsmith_contract_attrs(tree: ast.AST, problems: list) -> None:
    """A script that does `import simsmith_contract` (or `as alias`) and then touches
    any attribute on it other than the public get_task/get_rng/get_perturb/
    write_result is reaching into the harness's own "private" hooking/measurement
    state -- see SIMSMITH_CONTRACT_PUBLIC_API's comment above for the exploits this
    closes. The normal `from simsmith_contract import get_task, ...` form is
    unaffected (no Attribute node is ever created for it)."""
    aliases = _bound_aliases_for_module(tree, "simsmith_contract")
    if not aliases:
        return
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        d = _dotted(node)
        if not d:
            continue
        parts = d.split(".")
        if parts[0] not in aliases or len(parts) < 2 or d in seen:
            continue
        seen.add(d)
        if parts[1] not in SIMSMITH_CONTRACT_PUBLIC_API:
            problems.append(f"banned attribute access on simsmith_contract: {d} -- only "
                            f"{sorted(SIMSMITH_CONTRACT_PUBLIC_API)} may be used")


def _check_write_result_not_redefined(tree: ast.AST, problems: list) -> None:
    """A script that defines its own `write_result` (function def or plain
    assignment) shadows the real, imported one -- the contract relies on the name
    `write_result` resolving to its own module-level function; see
    docs/diagnosis_2026-10-24.md Part 4 exploit #3 ("shadow_write_result")."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "write_result":
            problems.append("banned: script defines its own write_result (shadows the real one)")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "write_result":
                    problems.append("banned: script reassigns the name write_result (shadows the real one)")


# Names available in a module's namespace without any import/assignment of its own.
_IMPLICIT_MODULE_NAMES = {
    "__name__", "__file__", "__doc__", "__builtins__", "__package__", "__spec__",
    "__loader__", "__debug__", "__annotations__", "__dict__",
}
_BUILTIN_NAMES = frozenset(dir(builtins)) | _IMPLICIT_MODULE_NAMES


def _check_undefined_names(tree: ast.AST, problems: list) -> None:
    """Flags a Name used (Load context) that is never imported, assigned/bound
    anywhere in the script (any scope -- see below), defined as a function/class/
    parameter, nor a builtin -- the static equivalent of the NameError this would
    raise at runtime (diagnosis 2026-10-25: forgetting `import numpy as np` while
    using `np.` throughout was the single largest crash cause in a regression run).

    Deliberately NOT scope-correct: every name bound ANYWHERE in the file (a nested
    function's parameter, a comprehension's loop variable, a sibling function's local
    assignment, ...) counts as "defined" for the WHOLE file, not just the scope that
    actually binds it. A real scope checker could additionally catch "defined in the
    wrong scope" bugs, but conflating scopes can only produce false NEGATIVES here,
    never false positives -- the one thing this check must not do, since a real
    reach/pick_place script is simple, mostly flat, top-level code where the
    distinction rarely matters anyway. `from X import *` makes the set of names it
    introduces unknowable statically, so the whole check is skipped in that case
    rather than risk a false positive.
    """
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    return  # can't know what names a star-import introduces
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arglist in (a.posonlyargs, a.args, a.kwonlyargs):
                bound.update(arg.arg for arg in arglist)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)

    first_line = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound and node.id not in _BUILTIN_NAMES:
                first_line.setdefault(node.id, node.lineno)
    for name, lineno in sorted(first_line.items(), key=lambda kv: kv[1]):
        problems.append(f"name '{name}' is not defined; add the import (line {lineno})")


def _check_mujoco_attrs(tree: ast.AST, problems: list) -> None:
    """Resolve every mujoco.<attr> (and chained attrs, e.g. mujoco.mjtObj.mjOBJ_BODY)
    against the real, installed mujoco module by getattr-walking it -- not a
    hand-maintained symbol list, so it can't go stale against the pinned version.
    Rejects a name that doesn't exist, with difflib close matches at the point it
    broke (e.g. a typo'd mj_setp suggests mj_step)."""
    if _mujoco_module is None:
        return  # no mujoco in this process; the broader import/mj_step check still applies
    bound_names = set()
    submodules = set()  # dotted paths like "mujoco.viewer" the script itself declares
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "mujoco":
                    bound_names.add(a.asname or "mujoco")
                elif a.name.startswith("mujoco."):
                    bound_names.add("mujoco")  # "import mujoco.viewer" binds "mujoco" too
                    submodules.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "mujoco":
                # "from mujoco import viewer" -- viewer may be a lazy submodule (like
                # mujoco.viewer) that a bare `import mujoco` never pulls in; try it as one.
                submodules.update(f"mujoco.{a.name}" for a in node.names)
            elif node.module and node.module.startswith("mujoco."):
                submodules.add(node.module)
    if not bound_names:
        return
    for sub in submodules:
        try:
            importlib.import_module(sub)  # makes it a real attribute of _mujoco_module below
        except ImportError:
            pass  # not a real submodule; the chain walk below will correctly flag it
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
    _getattr_sensitive_aliases = _sensitive_getattr_setattr_aliases(tree)
    _getattr_contract_aliases = _bound_aliases_for_module(tree, "simsmith_contract")
    # see SIMSMITH_SKILLS_STEPPING_HELPERS above: local names bound to move_to/
    # set_gripper via `from simsmith_skills import ...` (respecting `as` aliases), plus
    # any `import simsmith_skills [as x]` module alias for the `x.move_to(...)` form.
    _skills_stepping_aliases = set()
    _skills_module_aliases = _bound_aliases_for_module(tree, "simsmith_skills")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "simsmith_skills":
            for a in node.names:
                if a.name in SIMSMITH_SKILLS_STEPPING_HELPERS:
                    _skills_stepping_aliases.add(a.asname or a.name)
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
                if node.func.id in ("getattr", "setattr"):
                    if not _is_safe_getattr_setattr(node, _getattr_sensitive_aliases,
                                                    _getattr_contract_aliases):
                        problems.append(f"banned call: {node.func.id}() (non-literal "
                                        "attribute name, or target is os/sys/a banned "
                                        "module/simsmith_contract)")
                elif node.func.id in BANNED_CALLS:
                    problems.append(f"banned call: {node.func.id}()")
                if node.func.id == "write_result":
                    calls_write = True
                if node.func.id == "mj_step" or node.func.id in _skills_stepping_aliases:
                    calls_mj_step = True
            elif isinstance(node.func, ast.Attribute):
                d = _dotted(node.func)
                if d:
                    parts = d.split(".")
                    if len(parts) >= 2 and (parts[0], parts[1]) in BANNED_ATTR_CALLS:
                        problems.append(f"banned call: {d}()")
                    if parts[-1] == "write_result":
                        calls_write = True
                    if (len(parts) >= 2 and parts[0] in _skills_module_aliases
                            and parts[1] in SIMSMITH_SKILLS_STEPPING_HELPERS):
                        calls_mj_step = True
                if node.func.attr == "mj_step":
                    calls_mj_step = True
    if not calls_write:
        problems.append("script never calls write_result(final_state)")
    _check_mujoco_attrs(tree, problems)
    _check_dunder_attrs(tree, problems)
    _check_banned_attr_access(tree, problems)
    _check_simsmith_contract_attrs(tree, problems)
    _check_write_result_not_redefined(tree, problems)
    _check_undefined_names(tree, problems)
    if not (imports_mujoco and calls_mj_step):
        # A script cannot pass by faking the result -- it has to actually simulate.
        # This is the cheap, static half of that; the contract itself re-measures the
        # outcome at runtime and ignores whatever the script reports for those fields.
        problems.append("script must import mujoco and call mj_step(...) to advance "
                        "the simulation; reporting a position without ever stepping physics "
                        "forward in time is not accepted")
    return PreflightResult(not problems, problems)
