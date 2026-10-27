"""Re-renders an already-VERIFIED candidate script's run as an animated GIF, purely
for human viewing.

This NEVER affects verification: record_clip() is only ever called from
Pipeline.solve() AFTER a candidate has already been scored by simsmith.verifier (see
that function's `record_clip=` parameter), it runs the script again from scratch in a
brand-new subprocess, and every failure mode here (import error, render error,
timeout, zero frames captured) is caught and reported as "no clip" -- never raised up
to flip a pass to a fail, or a fail to a pass.

Technique: a small wrapper script (built from _WRAPPER_TEMPLATE below) is the
subprocess's actual entrypoint, not the candidate script directly, so it can:
  1. import simsmith_contract FIRST -- installs that module's own mj_step hook (used
     for honest independent re-measurement; completely untouched by anything here),
  2. wrap mujoco.mj_step ONE MORE TIME around that already-hooked version: the real
     step and the real measurement happen exactly as before, and every `frame_every`
     -th call additionally renders the scene from a fixed camera (a plain
     mujoco.MjvCamera with a hardcoded lookat/distance/azimuth/elevation -- not tied
     to any camera defined in the scene XML, since none of this project's scene
     files define one) and appends the frame to a list, capped at `max_frames` so a
     stuck script can't grow the GIF without bound,
  3. exec()s the candidate script's own source as __main__, so its own write_result()
     call runs completely normally (its result is discarded here -- verification
     already happened, this run is for a picture, not a score),
  4. assembles any captured frames into an animated GIF with Pillow and writes a
     status JSON next to it, which the parent process reads back.

MUJOCO_GL=egl is added to this subprocess's environment specifically, and only this
one -- a real verification run (simsmith/runner.py's SubprocessRunner) never sets it,
and its _ENV_PASSTHROUGH allow-list is untouched by this module -- because headless
rendering needs a GL backend and egl is what this sandbox actually has.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

from .runner import CONTRACT_DIR

DEFAULT_FRAME_EVERY = 15
DEFAULT_MAX_FRAMES = 100  # caps GIF length/size regardless of how long the script runs
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 10
DEFAULT_TIMEOUT_S = 90

# Not tied to the scene XML (none of assets/franka_emika_panda/*.xml define a camera)
# -- a fixed viewpoint good enough to see the Franka arm and, for pick_place, the cube/bin.
FIXED_CAMERA = {"lookat": (0.3, 0.0, 0.4), "distance": 1.6, "azimuth": 120.0, "elevation": -20.0}

_WRAPPER_TEMPLATE = '''\
import sys, os, json, time
sys.path.insert(0, {contract_dir!r})
import mujoco
import simsmith_contract  # installs the measurement hook before anything else runs

_frames = []
_step_count = [0]
_renderer_holder = {{"obj": None}}
_cam = mujoco.MjvCamera()
_cam.lookat[:] = {lookat!r}
_cam.distance = {distance!r}
_cam.azimuth = {azimuth!r}
_cam.elevation = {elevation!r}

_already_hooked_step = mujoco.mj_step  # simsmith_contract's own measurement hook

def _recording_step(model, data, nstep=1):
    result = _already_hooked_step(model, data, nstep)
    _step_count[0] += 1
    if _step_count[0] % {frame_every!r} == 0 and len(_frames) < {max_frames!r}:
        try:
            if _renderer_holder["obj"] is None:
                _renderer_holder["obj"] = mujoco.Renderer(model, height={height!r}, width={width!r})
            _renderer_holder["obj"].update_scene(data, camera=_cam)
            _frames.append(_renderer_holder["obj"].render().copy())
        except Exception:
            pass  # a render failure must never break the actual sim step
    return result

mujoco.mj_step = _recording_step

error = None
t0 = time.time()
try:
    with open({script_path!r}, encoding="utf-8") as f:
        _src = f.read()
    exec(compile(_src, {script_path!r}, "exec"), {{"__name__": "__main__", "__file__": {script_path!r}}})
except SystemExit:
    pass
except Exception as e:
    error = f"{{type(e).__name__}}: {{e}}"
render_seconds = time.time() - t0

status = {{"n_frames": len(_frames), "error": error, "render_seconds": render_seconds,
         "gif_written": False, "gif_bytes": None, "gif_error": None}}
try:
    if _frames:
        from PIL import Image
        images = [Image.fromarray(f) for f in _frames]
        images[0].save({out_gif!r}, save_all=True, append_images=images[1:],
                       duration=int(1000 / {fps!r}), loop=0)
        status["gif_written"] = True
        status["gif_bytes"] = os.path.getsize({out_gif!r})
except Exception as e:
    status["gif_error"] = f"{{type(e).__name__}}: {{e}}"

with open({status_path!r}, "w", encoding="utf-8") as f:
    json.dump(status, f)
'''


def record_clip(script: str, seed: int, template: str, cmd: list, out_path: str,
                perturb: dict | None = None, frame_every: int = DEFAULT_FRAME_EVERY,
                max_frames: int = DEFAULT_MAX_FRAMES, width: int = DEFAULT_WIDTH,
                height: int = DEFAULT_HEIGHT, fps: int = DEFAULT_FPS,
                timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """Best-effort -- NEVER raises. Returns
    {"ok": bool, "clip_path": str|None, "n_frames": int, "render_seconds": float,
     "gif_bytes": int|None, "error": str|None}. `cmd` is the already-resolved
    interpreter command (e.g. SubprocessRunner.cmd) that ran the real verification --
    reused here so the recording uses the exact same Python/mujoco install."""
    perturb = perturb or {}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="simsmith_record_") as tmp:
            script_path = os.path.join(tmp, "candidate.py")
            status_path = os.path.join(tmp, "status.json")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
            wrapper_path = os.path.join(tmp, "_record_wrapper.py")
            with open(wrapper_path, "w", encoding="utf-8") as f:
                f.write(_WRAPPER_TEMPLATE.format(
                    contract_dir=CONTRACT_DIR, script_path=script_path, out_gif=out_path,
                    status_path=status_path, frame_every=frame_every, max_frames=max_frames,
                    width=width, height=height, fps=fps, **FIXED_CAMERA))
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
            env["MUJOCO_GL"] = "egl"  # scoped to this one recording subprocess only
            env["SIMSMITH_SEED"] = str(seed)
            env["SIMSMITH_RESULT"] = os.path.join(tmp, "result.json")  # the script still calls
                                      # write_result(); its content is never read back here --
                                      # verification already happened before this was called
            env["SIMSMITH_PERTURB"] = json.dumps(perturb)
            env["SIMSMITH_TEMPLATE"] = template
            env["PYTHONPATH"] = CONTRACT_DIR
            env["SIMSMITH_ROOT"] = os.environ.get("SIMSMITH_ROOT", os.path.dirname(CONTRACT_DIR))

            proc = subprocess.run(list(cmd) + [wrapper_path], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout_s)
            if not os.path.exists(status_path):
                tail = (proc.stderr or proc.stdout or "")[-500:]
                return {"ok": False, "clip_path": None, "n_frames": 0,
                       "render_seconds": time.time() - t0, "gif_bytes": None,
                       "error": f"recording subprocess produced no status file "
                               f"(exit {proc.returncode}): {tail}"}
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
            ok = status.get("gif_written") is True and os.path.exists(out_path)
            return {"ok": ok, "clip_path": out_path if ok else None,
                   "n_frames": status.get("n_frames", 0),
                   "render_seconds": status.get("render_seconds") or (time.time() - t0),
                   "gif_bytes": status.get("gif_bytes"),
                   "error": status.get("error") or status.get("gif_error")}
    except subprocess.TimeoutExpired:
        return {"ok": False, "clip_path": None, "n_frames": 0,
               "render_seconds": time.time() - t0, "gif_bytes": None,
               "error": f"recording timed out after {timeout_s}s"}
    except Exception as e:  # recording must never raise into the caller -- see module docstring
        return {"ok": False, "clip_path": None, "n_frames": 0,
               "render_seconds": time.time() - t0, "gif_bytes": None,
               "error": f"{type(e).__name__}: {e}"}
