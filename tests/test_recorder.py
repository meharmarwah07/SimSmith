import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simsmith.recorder import record_clip  # noqa: E402

GOOD_SCRIPT = open(os.path.join(ROOT, "examples", "reach_mujoco.py"), encoding="utf-8").read()

BROKEN_SCRIPT = """
import os
import mujoco
import numpy as np
from simsmith_contract import get_task, get_rng, get_perturb, write_result
SIMSMITH_ROOT = os.environ.get("SIMSMITH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = mujoco.MjModel.from_xml_path(os.path.join(SIMSMITH_ROOT, "assets", "franka_emika_panda", "scene.xml"))
data = mujoco.MjData(model)
raise RuntimeError("deliberately broken, before any mj_step")
"""


def test_record_clip_produces_a_gif(tmp_path):
    out_path = str(tmp_path / "clip.gif")
    result = record_clip(GOOD_SCRIPT, seed=0, template="reach", cmd=[sys.executable],
                         out_path=out_path, frame_every=50, max_frames=4,
                         width=80, height=60, timeout_s=60)
    assert result["ok"] is True, result
    assert result["clip_path"] == out_path
    assert os.path.exists(out_path)
    assert result["n_frames"] > 0
    assert result["n_frames"] <= 4
    assert result["gif_bytes"] > 0
    assert result["render_seconds"] > 0
    assert result["error"] is None


def test_record_clip_frame_cap_respected(tmp_path):
    out_path = str(tmp_path / "clip.gif")
    result = record_clip(GOOD_SCRIPT, seed=0, template="reach", cmd=[sys.executable],
                         out_path=out_path, frame_every=1, max_frames=3,
                         width=80, height=60, timeout_s=60)
    assert result["ok"] is True, result
    assert result["n_frames"] <= 3


def test_record_clip_never_raises_on_a_broken_script(tmp_path):
    out_path = str(tmp_path / "clip.gif")
    # Must not raise -- a recording failure is reported, never propagated.
    result = record_clip(BROKEN_SCRIPT, seed=0, template="reach", cmd=[sys.executable],
                         out_path=out_path, frame_every=10, max_frames=5,
                         width=80, height=60, timeout_s=60)
    assert result["ok"] is False
    assert result["clip_path"] is None
    assert not os.path.exists(out_path)


def test_record_clip_never_raises_on_a_bogus_interpreter(tmp_path):
    out_path = str(tmp_path / "clip.gif")
    result = record_clip(GOOD_SCRIPT, seed=0, template="reach",
                         cmd=["/nonexistent/not-a-real-python"], out_path=out_path)
    assert result["ok"] is False
    assert result["error"]


def test_record_clip_never_raises_on_timeout(tmp_path):
    out_path = str(tmp_path / "clip.gif")
    hang_script = "import time\ntime.sleep(5)\n"
    result = record_clip(hang_script, seed=0, template="reach", cmd=[sys.executable],
                         out_path=out_path, timeout_s=1)
    assert result["ok"] is False
    assert "timed out" in result["error"]
