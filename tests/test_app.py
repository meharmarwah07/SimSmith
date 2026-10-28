"""Tests for the local demo UI server (app/server.py), using the mock
pipeline so these never touch the network or need an API key.

Each test starts a real ThreadingHTTPServer on 127.0.0.1:0 (an OS-assigned
free port) and talks to it over a real socket with http.client, so this
exercises the actual request parsing / routing, not a mocked-out handler.
"""
import http.client
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from http.server import ThreadingHTTPServer  # noqa: E402

from app import server  # noqa: E402
from simsmith.config import Config  # noqa: E402
from simsmith.docs import DocIndex  # noqa: E402
from simsmith.library import Library  # noqa: E402


def start_server(state, cfg=None, docs=None):
    cfg = cfg or Config(seeds=(0, 1))
    handler_cls = server.make_handler_class(state, cfg, docs)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


def stop_server(httpd):
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read()
        data = json.loads(raw) if raw else None
        return resp.status, data
    finally:
        conn.close()


def request_raw(port, method, path, body=None, headers=None):
    """Like request(), but returns the raw response bytes unparsed -- for
    endpoints that don't return JSON (static HTML/JS pages, media files)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, raw, dict(resp.getheaders())
    finally:
        conn.close()


def post_json(port, path, obj):
    body = json.dumps(obj).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    return request(port, "POST", path, body=body, headers=headers)


def make_mock_state(tmp_path, task="Move the arm's gripper to the target point."):
    cfg = Config(seeds=(0, 1), run_timeout_s=30)
    docs = DocIndex(str(tmp_path / "no-docs"))
    library = Library(str(tmp_path / "library.jsonl"))
    pipe = server.build_mock_pipeline(task, cfg, library, docs)
    state = server.ServerState(mode="mock", pipeline=pipe, library=library)
    return state, cfg, docs


def wait_for_done(port, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = request(port, "GET", "/api/status")
        assert status == 200
        if body["state"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError("run did not finish in time")


# ---- end-to-end run behaviour ----------------------------------------------

def test_run_end_to_end_mock(tmp_path):
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        status, body = post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        assert status == 202
        assert body["task"] == "Move the arm's gripper to the target point."

        final = wait_for_done(port)
        assert final["state"] == "done"
        result = final["result"]
        assert result["status"] in ("VERIFIED", "PARTIAL", "FAILED")
        assert len(result["candidates"]) == 5          # 5 generated candidate slots
        assert result["template"]["name"] == "reach"
        assert result["exact_match"] is True
        assert result["library_size"] >= 0
        assert isinstance(result["sim_runs"], int) and result["sim_runs"] > 0
        assert "repair_rounds" in result
    finally:
        stop_server(httpd)


def test_gap_statement_for_non_exact_task(tmp_path):
    state, cfg, docs = make_mock_state(tmp_path, task="Stack two cubes on top of each other.")
    httpd, port = start_server(state, cfg, docs)
    try:
        post_json(port, "/api/run", {"task": "Stack two cubes on top of each other."})
        final = wait_for_done(port)
        result = final["result"]
        assert result["exact_match"] is False
        assert result["gap"]
    finally:
        stop_server(httpd)


# ---- 409 when busy ----------------------------------------------------------

def test_second_run_rejected_while_busy(tmp_path):
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        # Force the busy state directly: deterministic, no timing race with
        # the background thread actually finishing before the second request.
        with state.lock:
            state.busy = True
            state.run_state = "running"
        status, body = post_json(port, "/api/run", {"task": "reach the target"})
        assert status == 409
        assert "already in progress" in body["error"]
    finally:
        stop_server(httpd)


def test_busy_then_free_allows_next_run(tmp_path):
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        status, _ = post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        assert status == 202
        status, body = post_json(port, "/api/run", {"task": "another task while busy"})
        assert status == 409
        wait_for_done(port)
        status, body = post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        assert status == 202
        wait_for_done(port)
    finally:
        stop_server(httpd)


# ---- oversized body ----------------------------------------------------------

def test_oversized_body_rejected(tmp_path):
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        huge_task = "x" * (11 * 1024)
        body = json.dumps({"task": huge_task}).encode("utf-8")
        assert len(body) > 10 * 1024
        status, resp = request(
            port, "POST", "/api/run", body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        assert status == 413
        # the oversized request must not have started a run
        status, stat = request(port, "GET", "/api/status")
        assert stat["busy"] is False
        assert stat["state"] == "idle"
    finally:
        stop_server(httpd)


def test_missing_task_field_rejected(tmp_path):
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        status, body = post_json(port, "/api/run", {})
        assert status == 400
        status, body = post_json(port, "/api/run", {"task": "   "})
        assert status == 400
    finally:
        stop_server(httpd)


# ---- replay mode --------------------------------------------------------------

FIXTURE = os.path.join(ROOT, "app", "fixtures", "sample_run_MOCK.json")


def test_replay_loading_and_read_only():
    with open(FIXTURE, encoding="utf-8") as f:
        recorded = json.load(f)["result"]
    state = server.ServerState(mode="replay", replay=recorded)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/replay")
        assert status == 200
        assert body["result"]["task"] == recorded["task"]
        assert body["result"]["status"] == recorded["status"]

        status, body = request(port, "GET", "/api/status")
        assert status == 200
        assert body["mode"] == "replay"
        assert body["state"] == "done"
        assert body["result"]["task"] == recorded["task"]

        status, body = post_json(port, "/api/run", {"task": "anything"})
        assert status == 409
        assert "replay" in body["error"]
    finally:
        stop_server(httpd)


def test_build_state_loads_replay_fixture(tmp_path):
    class Args:
        replay = FIXTURE
        mock = False
        port = 0

    state, cfg, docs = server.build_state(Args())
    assert state.mode == "replay"
    assert state.result["task"]


# ---- HTML escaping / XSS safety ----------------------------------------------

# Dotted/call form, so this doesn't false-positive on the plain-English
# words appearing inside app.js's own policy comment at the top of the file.
FORBIDDEN_SINKS = (".innerHTML", ".outerHTML", ".insertAdjacentHTML(", "document.write(")


def test_frontend_never_uses_html_injection_sinks():
    """The page must only ever attach model/user text via .textContent, never
    parse it as markup. We assert this structurally (no dangerous DOM sink is
    even present in the client code), which is what actually prevents a
    <script> tag embedded in a candidate's code or task text from ever
    executing, no matter what that text contains."""
    app_js = os.path.join(ROOT, "app", "static", "app.js")
    with open(app_js, encoding="utf-8") as f:
        src = f.read()
    for sink in FORBIDDEN_SINKS:
        assert sink not in src, f"app.js must not use {sink}"


def test_malicious_script_content_reaches_api_unexecuted_as_data(tmp_path, monkeypatch):
    """A candidate whose generated code contains a literal <script> tag (e.g.
    hidden in a comment or a string the model wrote) must still flow through
    the API as plain text data. Combined with
    test_frontend_never_uses_html_injection_sinks, there is no code path in
    the client that would ever execute it: it can only land in a .textContent
    assignment, which treats it as inert characters."""
    evil = (
        "```python\n"
        "import mujoco\n"
        "from simsmith_contract import get_task, write_result\n"
        "# <script>alert(1)</script>\n"
        "m = mujoco.MjModel.from_xml_string('<mujoco><option gravity=\"0 0 0\"/>"
        "<worldbody><body name=\"hand\"><freejoint/><geom type=\"sphere\" size=\"0.01\"/></body>"
        "</worldbody></mujoco>')\n"
        "d = mujoco.MjData(m)\n"
        "target = get_task('reach')['target_pos']\n"
        "d.qpos[:3] = target\n"
        "mujoco.mj_forward(m, d)\n"
        "for _ in range(20):\n"
        "    mujoco.mj_step(m, d)\n"
        "write_result({'ee_pos': d.qpos[:3].tolist(), 'target_pos': target, "
        "'note': '<script>alert(1)</script>'})\n"
        "```"
    )
    cfg = Config(seeds=(0, 1), run_timeout_s=30)
    docs = DocIndex(str(tmp_path / "no-docs"))
    library = Library(str(tmp_path / "library.jsonl"))

    from simsmith.pipeline import Pipeline
    from simsmith.runner import SubprocessRunner
    from tests.mocks import MockLLM

    llm = MockLLM(generate_outputs=[evil], repair_outputs=[],
                  intake={"template": "reach", "exact_match": True, "gap": "", "goal": "reach"})
    pipe = Pipeline(llm, SubprocessRunner("", cfg.run_timeout_s), library, docs, cfg)
    state = server.ServerState(mode="mock", pipeline=pipe, library=library)

    # run_task() would normally rebuild a mock pipeline per request; patch it
    # out so our hand-crafted single-candidate pipeline above is used as-is.
    monkeypatch.setattr(server, "build_mock_pipeline", lambda *a, **k: pipe)

    httpd, port = start_server(state, cfg, docs)
    try:
        status, body = post_json(port, "/api/run", {"task": "reach the target"})
        assert status == 202
        final = wait_for_done(port)
        result = final["result"]
        scripts = [a["script"] for g in result["candidates"] for a in g["attempts"]]
        assert any("<script>alert(1)</script>" in s for s in scripts)
    finally:
        stop_server(httpd)


# ---- (1) integrity badge: source field + fixture rename -----------------------

def test_mock_run_result_has_source_mock(tmp_path):
    """Every mock run must say so in its own payload -- the badge has
    nothing else to go on."""
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        final = wait_for_done(port)
        result = final["result"]
        assert result["source"] == "mock"
        assert result["model"] is None
        assert result["prompt_version"]
        assert result["date"]
    finally:
        stop_server(httpd)


def test_fixture_is_renamed_and_tagged_mock():
    with open(FIXTURE, encoding="utf-8") as f:
        recorded = json.load(f)["result"]
    assert recorded["source"] == "mock"
    assert os.path.basename(FIXTURE) == "sample_run_MOCK.json"


def test_build_state_refuses_replay_without_source(tmp_path):
    """A recording that doesn't say what it is must never be served -- this
    is the hard gate behind "a mock run can't render without the badge":
    if the field is missing, the server refuses to start in --replay mode
    at all, rather than falling back to an assumed/default source."""
    bad_fixture = tmp_path / "unlabeled.json"
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    del data["result"]["source"]
    bad_fixture.write_text(json.dumps(data), encoding="utf-8")

    class Args:
        replay = str(bad_fixture)
        mock = False
        port = 0

    try:
        server.build_state(Args())
        assert False, "build_state should have called sys.exit() for an unlabeled replay"
    except SystemExit as e:
        assert e.code == 1


def test_app_js_defines_and_calls_source_badge_unconditionally():
    """Structural guarantee mirroring test_frontend_never_uses_html_injection_sinks:
    renderResult() must call renderSourceBadge on every path, and the two
    required literal badge strings must exist in the client source."""
    app_js = os.path.join(ROOT, "app", "static", "app.js")
    with open(app_js, encoding="utf-8") as f:
        src = f.read()
    assert "function renderSourceBadge(" in src
    assert "renderSourceBadge(result);" in src
    # renderResult's own body must call it, not just some other function
    result_fn_start = src.index("function renderResult(result) {")
    result_fn_body = src[result_fn_start:result_fn_start + 400]
    assert "renderSourceBadge(result)" in result_fn_body
    assert "MOCK: scripted LLM, no model was called" in src
    assert "Recorded live run:" in src


def test_index_html_has_source_badge_element():
    index_html = os.path.join(ROOT, "app", "static", "index.html")
    with open(index_html, encoding="utf-8") as f:
        src = f.read()
    assert 'id="source-badge"' in src


# ---- (2) results page -----------------------------------------------------------

def _make_results_state(results_dir, media_dir=None):
    return server.ServerState(mode="replay", replay={"source": "mock"},
                              results_dir=str(results_dir), media_dir=str(media_dir or results_dir))


def test_results_route_serves_html(tmp_path):
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, raw, _ = request_raw(port, "GET", "/results")
        assert status == 200
        assert b"<!doctype html>" in raw
    finally:
        stop_server(httpd)


def test_api_results_missing_dir_reports_error_not_crash(tmp_path):
    state = _make_results_state(tmp_path / "does-not-exist")
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert body["rows"] == []
        assert any("not found" in e for e in body["errors"])
    finally:
        stop_server(httpd)


def test_api_results_eval_shape(tmp_path):
    """The hypothetical eval/run_eval.py "conditions" shape -- keyed by
    condition name, with a template on each individual task row. The keys
    eval/run_eval.py's own CONDITIONS dict actually uses are already
    canonical (single_shot / best_of_5_repair / best_of_5_repair_library),
    so they're trusted as-is, the same as an explicit top-level "condition"
    field would be. Each (condition, template) pair must come back as its
    own row, not merged across templates."""
    report = {
        "started": 1700000000.0, "max_runs": 300, "truncated": False,
        "conditions": {
            "single_shot": {
                "tasks_run": 4, "verified": 1, "verified_rate": 0.25,
                "mean_success_rate": 0.25,
                "rows": [
                    {"id": "t1", "template": "reach", "status": "VERIFIED", "success_rate": 1.0},
                    {"id": "t2", "template": "reach", "status": "FAILED", "success_rate": 0.0},
                    {"id": "t3", "template": "pick_place", "status": "FAILED", "success_rate": 0.0},
                    {"id": "t4", "template": "pick_place", "status": "PARTIAL", "success_rate": 0.5},
                ],
            },
            "best_of_5_repair": {
                "tasks_run": 4, "verified": 3, "verified_rate": 0.75,
                "mean_success_rate": 0.75,
                "rows": [{"id": f"t{i}", "template": "reach", "status": "VERIFIED"} for i in range(3)]
                       + [{"id": "t4", "template": "reach", "status": "FAILED"}],
            },
        },
        "library_size_at_heldout": 0, "total_sim_runs": 40, "path": "x",
    }
    (tmp_path / "eval_1700000000.json").write_text(json.dumps(report), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert body["errors"] == []
        by_key = {(r["condition"], r["template"]): r for r in body["rows"]}
        assert set(by_key.keys()) == {
            ("single_shot", "reach"), ("single_shot", "pick_place"), ("best_of_5_repair", "reach"),
        }

        ss_reach = by_key[("single_shot", "reach")]
        assert ss_reach["n"] == 2
        assert ss_reach["pass_rate"] == 0.5
        lo, hi = server.wilson_interval(1, 2)
        assert abs(ss_reach["wilson_lo"] - lo) < 1e-9
        assert abs(ss_reach["wilson_hi"] - hi) < 1e-9
        assert ss_reach["model"] == "unknown"
        assert ss_reach["failure_breakdown"].get("failed") == 1

        ss_pp = by_key[("single_shot", "pick_place")]
        assert ss_pp["n"] == 2
        assert ss_pp["pass_rate"] == 0.0  # neither pick_place row was VERIFIED
        assert ss_pp["failure_breakdown"].get("failed") == 1
        assert ss_pp["failure_breakdown"].get("partial") == 1

        bo5 = by_key[("best_of_5_repair", "reach")]
        assert bo5["n"] == 4
        assert bo5["pass_rate"] == 0.75
    finally:
        stop_server(httpd)


def test_api_results_baseline_shape(tmp_path):
    """Matches scripts/baseline_dev.py's real output shape: single_shot
    only, keyed by template under "summaries", with
    model/prompt_version/failure_breakdown already computed per template.
    Each template must come back as its own row -- a run covering two
    templates with very different pass rates must not be averaged together
    into one misleading number."""
    report = {
        "prompt_version": "mujoco-v1", "model": "nemotron-70b",
        "sim_python": "/usr/bin/python3", "sim_seeds": [0, 1, 2],
        "timestamp": 1700000000,
        "summaries": {
            "reach": {
                "template": "reach", "n": 10, "passes": 8, "pass_rate": 0.8,
                "wilson_95": [0.49, 0.95],
                "failure_breakdown": {"pass": 8, "crash": 1, "wrong_logic": 1},
                "mean_cost_usd": 0.001, "mean_wall_seconds": 3.2,
            },
            "pick_place": {
                "template": "pick_place", "n": 10, "passes": 6, "pass_rate": 0.6,
                "wilson_95": [0.31, 0.83],
                "failure_breakdown": {"pass": 6, "hallucinated_api": 2, "crash": 2},
                "mean_cost_usd": 0.002, "mean_wall_seconds": 4.0,
            },
        },
        "rows": {"reach": [], "pick_place": []},
    }
    (tmp_path / "baseline_1700000000.json").write_text(json.dumps(report), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert len(body["rows"]) == 2
        by_template = {r["template"]: r for r in body["rows"]}

        reach = by_template["reach"]
        assert reach["model"] == "nemotron-70b"
        assert reach["prompt_version"] == "mujoco-v1"
        assert reach["condition"] == "single_shot"  # no "condition" field -> filename prefix "baseline_"
        assert reach["n"] == 10
        assert abs(reach["pass_rate"] - 0.8) < 1e-9
        assert abs(reach["mean_seconds"] - 3.2) < 1e-9
        assert abs(reach["mean_cost_usd"] - 0.001) < 1e-9
        assert reach["failure_breakdown"]["crash"] == 1

        pick_place = by_template["pick_place"]
        assert pick_place["condition"] == "single_shot"
        assert pick_place["n"] == 10
        assert abs(pick_place["pass_rate"] - 0.6) < 1e-9
        assert abs(pick_place["mean_seconds"] - 4.0) < 1e-9
        assert pick_place["failure_breakdown"]["hallucinated_api"] == 2
    finally:
        stop_server(httpd)


def test_api_results_tolerates_malformed_and_unrecognized_files(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    good = {
        "prompt_version": "v2", "model": "m", "timestamp": 1700000000,
        "summaries": {"reach": {"template": "reach", "n": 2, "passes": 2, "pass_rate": 1.0,
                                "failure_breakdown": {"pass": 2}}},
        "rows": {"reach": []},
    }
    (tmp_path / "baseline_good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me, not .json", encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert len(body["rows"]) == 1
        assert any("broken.json" in e for e in body["errors"])
        assert any("unrelated.json" in e for e in body["errors"])
    finally:
        stop_server(httpd)


def test_wilson_interval_known_values():
    lo, hi = server.wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = server.wilson_interval(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0


# ---- (3) backend comparison panel -----------------------------------------------

def test_backend_comparison_endpoint_matches_fixture(tmp_path):
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/backend-comparison")
        assert status == 200
        assert "Real-world fidelity not tested" in body["note"]
        names = [b["backend"] for b in body["backends"]]
        assert any("MuJoCo" in n for n in names)
        measured = {b["backend"]: b["measured"] for b in body["backends"]}
        assert any(measured.values())
        assert not all(measured.values())  # not everything is claimed as measured
    finally:
        stop_server(httpd)


def test_backend_comparison_makes_no_best_claim():
    fixture_path = os.path.join(ROOT, "app", "fixtures", "backend_comparison.json")
    with open(fixture_path, encoding="utf-8") as f:
        text = f.read()
    for banned in ("best backend", "best choice", "recommended backend", "winner"):
        assert banned not in text.lower()

    index_html = os.path.join(ROOT, "app", "static", "index.html")
    with open(index_html, encoding="utf-8") as f:
        html = f.read()
    assert "Backend comparison (speed, coverage, stability)" in html


# ---- (4) media slot ---------------------------------------------------------------

MEDIA_DIR = os.path.join(ROOT, "app", "media")


def test_media_valid_file_served(tmp_path):
    state = _make_results_state(tmp_path, media_dir=MEDIA_DIR)
    httpd, port = start_server(state)
    try:
        status, raw, headers = request_raw(port, "GET", "/api/media?path=sample_clip.gif")
        assert status == 200
        assert headers.get("Content-Type") == "image/gif"
        with open(os.path.join(MEDIA_DIR, "sample_clip.gif"), "rb") as f:
            assert raw == f.read()
    finally:
        stop_server(httpd)


def test_media_rejects_unsupported_extension(tmp_path):
    state = _make_results_state(tmp_path, media_dir=MEDIA_DIR)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/media?path=../server.py")
        assert status == 400
        assert "unsupported" in body["error"]
    finally:
        stop_server(httpd)


def test_media_rejects_path_traversal_with_allowed_extension(tmp_path):
    """The real attack: an allow-listed extension (.gif) but a path that
    tries to climb out of --media-dir with '..'."""
    secret_dir = tmp_path / "secret_outside"
    secret_dir.mkdir()
    (secret_dir / "leak.gif").write_bytes(b"not actually a gif, just a probe")
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    state = _make_results_state(tmp_path, media_dir=media_dir)
    httpd, port = start_server(state)
    try:
        for attempt in ("../secret_outside/leak.gif", "..%2fsecret_outside%2fleak.gif",
                        "....//secret_outside/leak.gif"):
            status, body = request(port, "GET", f"/api/media?path={attempt}")
            assert status in (400, 404), f"traversal attempt {attempt!r} was not rejected"
    finally:
        stop_server(httpd)


def test_media_rejects_absolute_path(tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    outside = tmp_path / "absolute.gif"
    outside.write_bytes(b"probe")
    state = _make_results_state(tmp_path, media_dir=media_dir)
    httpd, port = start_server(state)
    try:
        status, _ = request(port, "GET", f"/api/media?path={outside}")
        assert status == 404
    finally:
        stop_server(httpd)


def test_safe_media_path_helper_directly(tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "clip.gif").write_bytes(b"x")
    assert server._safe_media_path(str(media_dir), "clip.gif") is not None
    assert server._safe_media_path(str(media_dir), "../clip.gif") is None
    assert server._safe_media_path(str(media_dir), "/etc/passwd") is None
    assert server._safe_media_path(str(media_dir), "a/../../clip.gif") is None


def test_serialize_candidate_surfaces_clip_path_when_present():
    class FakeScore:
        ok = True
        success_rate = 1.0
        crashes = 0
        mean_seconds = 0.1
        failure_text = ""
        details = []

    class FakeCandidateWithClip:
        idx = 0
        script = "pass"
        framing = "f"
        temperature = 0.2
        stage = "generated"
        preflight_ok = True
        preflight_problems = []
        score = FakeScore()
        clip_path = "reach_demo.gif"

    out_with = server.serialize_candidate(FakeCandidateWithClip())
    assert out_with["clip_path"] == "reach_demo.gif"

    class FakeCandidateNoClip2:
        idx = 1
        script = "pass"
        framing = "f"
        temperature = 0.2
        stage = "generated"
        preflight_ok = True
        preflight_problems = []
        score = FakeScore()

    out_without = server.serialize_candidate(FakeCandidateNoClip2())
    assert out_without["clip_path"] is None


def test_serialize_result_stamps_solve_result_clip_path_onto_winner_attempt():
    """SolveResult.clip_path (set only by solve(record_clip=True), see
    Pipeline._record_winner_clip) lives on the result, not on the winning
    Candidate -- serialize_result() must copy it onto the winner's own
    serialized attempt, since that's the dict app.js's renderMediaSlot()
    actually reads clip_path from."""
    from simsmith.pipeline import Candidate, SolveResult
    from simsmith.verifier import Score

    score = Score(ok=True, success_rate=1.0, crashes=0, details=[], failure_text="", mean_seconds=0.1)
    winner = Candidate(idx=0, script="pass", framing="f", temperature=0.2, score=score)
    other = Candidate(idx=1, script="pass", framing="f", temperature=0.2, score=score)
    res = SolveResult(task="t", template="reach", exact_match=True, gap="", plan="p",
                      candidates=[other, winner], winner=winner, status="VERIFIED",
                      clip_path="reach_abc123.gif")

    payload = server.serialize_result(res, library_size=0, elapsed_s=1.0, source="live")
    winner_group = next(g for g in payload["candidates"] if g["idx"] == payload["winner"]["idx"])
    winner_attempt = winner_group["attempts"][payload["winner"]["attempt"]]
    assert winner_attempt["clip_path"] == "reach_abc123.gif"
    # the non-winning candidate's attempt must be untouched
    other_group = next(g for g in payload["candidates"] if g["idx"] != payload["winner"]["idx"])
    assert other_group["attempts"][0]["clip_path"] is None


def test_serialize_result_no_clip_path_when_solve_result_has_none():
    from simsmith.pipeline import Candidate, SolveResult
    from simsmith.verifier import Score

    score = Score(ok=True, success_rate=1.0, crashes=0, details=[], failure_text="", mean_seconds=0.1)
    winner = Candidate(idx=0, script="pass", framing="f", temperature=0.2, score=score)
    res = SolveResult(task="t", template="reach", exact_match=True, gap="", plan="p",
                      candidates=[winner], winner=winner, status="VERIFIED")

    payload = server.serialize_result(res, library_size=0, elapsed_s=1.0, source="live")
    winner_group = next(g for g in payload["candidates"] if g["idx"] == payload["winner"]["idx"])
    assert winner_group["attempts"][payload["winner"]["attempt"]]["clip_path"] is None


# ---- (5) polish --------------------------------------------------------------------

def test_copy_to_clipboard_still_present():
    app_js = os.path.join(ROOT, "app", "static", "app.js")
    with open(app_js, encoding="utf-8") as f:
        src = f.read()
    assert "navigator.clipboard.writeText" in src
    assert "Copy code" in src


def test_failure_state_wording():
    app_js = os.path.join(ROOT, "app", "static", "app.js")
    with open(app_js, encoding="utf-8") as f:
        src = f.read()
    assert "No verified script. Closest template + gap:" in src


def test_mobile_width_rules_present():
    style_css = os.path.join(ROOT, "app", "static", "style.css")
    with open(style_css, encoding="utf-8") as f:
        src = f.read()
    assert "@media (max-width: 560px)" in src
    assert ".source-badge" in src
    assert ".data-table" in src


def test_keyboard_accessible_controls_are_native_elements():
    """Every clickable control the task cares about (examples, run, copy,
    seed dots, nav links) must be a real <button>/<a>, not a div with a
    click handler, so Tab/Enter/Space work without extra JS."""
    app_js = os.path.join(ROOT, "app", "static", "app.js")
    with open(app_js, encoding="utf-8") as f:
        src = f.read()
    assert 'el("button"' in src
    index_html = os.path.join(ROOT, "app", "static", "index.html")
    with open(index_html, encoding="utf-8") as f:
        html = f.read()
    assert '<a ' in html and 'href="/results"' in html


APP_JS = os.path.join(ROOT, "app", "static", "app.js")
INDEX_HTML = os.path.join(ROOT, "app", "static", "index.html")
STYLE_CSS = os.path.join(ROOT, "app", "static", "style.css")
RESULTS_JS = os.path.join(ROOT, "app", "static", "results.js")
RESULTS_HTML = os.path.join(ROOT, "app", "static", "results.html")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---- (6.1) classifier-version warning --------------------------------------

def test_api_results_flags_old_or_missing_classifier_version(tmp_path):
    """A result file with no classifier_version, or one below 2, gets
    low_confidence_labels=True; a file with classifier_version>=2 does not.
    The pass_rate number itself must be identical either way -- the
    classifier-version check only ever gates the warning, never the rate."""
    def make(classifier_version=None):
        data = {
            "prompt_version": "v1", "model": "m", "timestamp": 1700000000,
            "summaries": {
                "reach": {"template": "reach", "n": 4, "passes": 3, "pass_rate": 0.75,
                          "failure_breakdown": {"pass": 3, "hallucinated_api": 1}},
            },
            "rows": {"reach": []},
        }
        if classifier_version is not None:
            data["classifier_version"] = classifier_version
        return data

    (tmp_path / "baseline_missing.json").write_text(json.dumps(make(None)), encoding="utf-8")
    (tmp_path / "baseline_v1.json").write_text(json.dumps(make(1)), encoding="utf-8")
    (tmp_path / "baseline_v2.json").write_text(json.dumps(make(2)), encoding="utf-8")
    (tmp_path / "baseline_v3.json").write_text(json.dumps(make(3)), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        by_file = {r["file"]: r for r in body["rows"]}
        assert by_file["baseline_missing.json"]["low_confidence_labels"] is True
        assert by_file["baseline_v1.json"]["low_confidence_labels"] is True
        assert by_file["baseline_v2.json"]["low_confidence_labels"] is False
        assert by_file["baseline_v3.json"]["low_confidence_labels"] is False
        for name in by_file:
            assert abs(by_file[name]["pass_rate"] - 0.75) < 1e-9
    finally:
        stop_server(httpd)


def test_api_results_classifier_warning_eval_shape(tmp_path):
    report = {
        "started": 1700000000.0,
        "conditions": {
            "single_shot": {
                "tasks_run": 2, "verified": 1, "verified_rate": 0.5,
                "rows": [
                    {"id": "t1", "template": "reach", "status": "VERIFIED"},
                    {"id": "t2", "template": "reach", "status": "FAILED"},
                ],
            },
        },
    }
    (tmp_path / "eval_nowarn.json").write_text(json.dumps(report), encoding="utf-8")
    report2 = dict(report)
    report2["classifier_version"] = 2
    (tmp_path / "eval_warn_ok.json").write_text(json.dumps(report2), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        by_file = {r["file"]: r for r in body["rows"]}
        assert by_file["eval_nowarn.json"]["low_confidence_labels"] is True
        assert by_file["eval_warn_ok.json"]["low_confidence_labels"] is False
        assert by_file["eval_nowarn.json"]["pass_rate"] == by_file["eval_warn_ok.json"]["pass_rate"]
    finally:
        stop_server(httpd)


def test_api_results_classifier_banner_counts_affected_files(tmp_path):
    """Drives the banner's file count directly off real response data: two
    files with no classifier_version (old/unknown) and one with
    classifier_version=2 -- the banner text built client-side must be able
    to say '2 of 3 files', which means every row needs file+low_confidence
    data (already covered in the normalizer tests above); this test pins
    the exact counts the banner computation in results.js depends on."""
    def make(classifier_version):
        data = {
            "prompt_version": "v1", "model": "m", "timestamp": 1700000000,
            "summaries": {"reach": {"template": "reach", "n": 2, "passes": 1,
                                     "failure_breakdown": {"pass": 1, "crash": 1}}},
        }
        if classifier_version is not None:
            data["classifier_version"] = classifier_version
        return data

    (tmp_path / "baseline_a.json").write_text(json.dumps(make(None)), encoding="utf-8")
    (tmp_path / "baseline_b.json").write_text(json.dumps(make(1)), encoding="utf-8")
    (tmp_path / "baseline_c.json").write_text(json.dumps(make(2)), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        files_flagged = {r["file"]: r["low_confidence_labels"] for r in body["rows"]}
        affected = sum(1 for v in files_flagged.values() if v)
        assert affected == 2
        assert len(files_flagged) == 3
    finally:
        stop_server(httpd)


def test_results_html_has_classifier_banner_element():
    html = _read(RESULTS_HTML)
    assert 'id="classifier-banner"' in html


def test_results_js_renders_banner_and_per_row_warning_icon():
    src = _read(RESULTS_JS)
    assert "function renderClassifierBanner(" in src
    assert "low_confidence_labels" in src
    # the banner text names the actual count of affected files
    banner_start = src.index("function renderClassifierBanner(")
    banner_body = src[banner_start:banner_start + 1200]
    assert "affected" in banner_body and "total" in banner_body
    # the per-row marker is a small icon with the explanation in its title/tooltip,
    # not a full-text chip crowding the failure-breakdown cell
    assert "warn-icon" in src
    assert "Labels from an older failure classifier may be wrong (hallucinated_api over-counted)." in src
    icon_idx = src.index("warn-icon")
    fail_td_idx = src.index('const failTd = el("td"')
    ci_idx = src.index("const ci =")
    assert fail_td_idx < icon_idx
    assert ci_idx < fail_td_idx  # pass-rate cell is built first, before the warning icon exists


# ---- (6.2) backend comparison: Newton -------------------------------------

def test_backend_comparison_includes_newton_with_required_caveats(tmp_path):
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/backend-comparison")
        assert status == 200
        by_name = {b["backend"]: b for b in body["backends"]}
        newton = next(b for name, b in by_name.items() if "Newton" in name)
        assert newton["measured"] is True
        assert newton["pass_rate_handwritten"] == 1.0
        assert "reach" in newton["tasks_covered"]
        assert "pick_place" not in newton["tasks_covered"]
        notes = newton["setup_notes"].lower()
        assert "collapse_fixed_joints" in notes
        assert "hand-written" in notes
        assert "not llm-generated" in notes or "not llm generated" in notes
        assert "2026-10-22" in newton["setup_notes"] and "2026-10-22/23" in newton["setup_notes"]
        isaac = next(b for name, b in by_name.items() if "Isaac" in name)
        assert isaac["measured"] is False
    finally:
        stop_server(httpd)


def test_backend_comparison_still_makes_no_best_claim_with_three_backends():
    fixture_path = os.path.join(ROOT, "app", "fixtures", "backend_comparison.json")
    with open(fixture_path, encoding="utf-8") as f:
        text = f.read()
    for banned in ("best backend", "best choice", "recommended backend", "winner"):
        assert banned not in text.lower()
    assert "Real-world fidelity not tested" in text


# ---- (6.3) condition detection -- canonical vocabulary + real-shape fallbacks ----
#
# simsmith/run_meta.py's VALID_CONDITIONS (kept in sync by hand as
# server.VALID_CONDITIONS) is the real canonical vocabulary the main
# session's eval code writes: single_shot, best_of_5, best_of_5_repair,
# best_of_5_repair_library, best_of_5_repair_skills. An explicit top-level
# "condition" field is now trusted verbatim against that vocabulary --
# max_repairs/n_candidates are no longer used to second-guess or "correct"
# an explicit value, even an ambiguous-looking one like bare "best_of_5"
# with max_repairs>0 (see test_detect_condition_trusts_explicit_field_even_
# when_max_repairs_suggests_otherwise below). Those heuristics now only run
# as a fallback for files with NO "condition" field at all -- the shape
# scripts/baseline_dev.py wrote before simsmith/run_meta.py existed.

def test_detect_condition_real_shape_baseline_dev_no_condition_field(tmp_path):
    """The pre-run_meta.py baseline_dev.py shape: no "condition" key at
    all. Falls back to the filename-prefix heuristic."""
    data = {
        "model": "nemotron-70b", "prompt_version": "mujoco-v5", "timestamp": 1700000000,
        "sim_python": "/usr/bin/python3", "sim_seeds": [0, 1, 2],
        "summaries": {"reach": {"template": "reach", "n": 12, "passes": 3,
                                 "failure_breakdown": {"pass": 3, "crash": 9}}},
        "rows": {"reach": []},
    }
    assert server.detect_condition(data, "baseline_1792933477.json") == "single_shot"


def test_detect_condition_real_shape_best_of_5_no_repair(tmp_path):
    """An explicit "condition": "best_of_5" (bare, no repair) is trusted
    directly -- it's already one of the canonical VALID_CONDITIONS names."""
    data = {
        "condition": "best_of_5", "max_repairs": 0, "model": "m", "prompt_version": "mujoco-v2",
        "extra_payload": {"chat_template_kwargs": {"enable_thinking": False}},
        "tag": "mujoco-v2 best_of_5", "sim_python": "/x", "sim_seeds": [0, 1],
        "summaries": {"reach": {"template": "reach", "n": 5, "passes": 1,
                                 "failure_breakdown": {"pass": 1, "hallucinated_api": 4}}},
        "rows": {"reach": []},
    }
    assert server.detect_condition(data, "best_of_5_1792778096.json") == "best_of_5"


def test_detect_condition_real_shape_best_of_5_repair_canonical_value(tmp_path):
    """The real backfilled shape for a repair run: "condition":
    "best_of_5_repair" written directly (not inferred) -- trusted
    verbatim, same as any other canonical value."""
    data = {
        "condition": "best_of_5_repair", "max_repairs": 2, "model": "m", "prompt_version": "mujoco-v2",
        "tag": "mujoco-v2 best_of_5 (tag-fix rerun)", "sim_python": "/x", "sim_seeds": [0, 1],
        "summaries": {"reach": {"template": "reach", "n": 7, "passes": 0,
                                 "failure_breakdown": {"hallucinated_api": 7}}},
        "rows": {"reach": []},
    }
    assert server.detect_condition(data, "best_of_5_1792777760.json") == "best_of_5_repair"


def test_detect_condition_trusts_explicit_field_even_when_max_repairs_suggests_otherwise():
    """The behavior change from the old (pre-canonical-vocabulary)
    heuristic: an explicit "condition": "best_of_5" is trusted exactly as
    written even when max_repairs>0 looks like repair happened -- a result
    file's own condition field is the ground truth now that the full
    vocabulary is known, not something to be second-guessed from a side
    field. (A file whose condition field is simply wrong is a data bug to
    fix at the source, e.g. by backfilling it -- not something
    detect_condition should paper over.)"""
    data = {"condition": "best_of_5", "max_repairs": 2}
    assert server.detect_condition(data, "best_of_5_ambiguous.json") == "best_of_5"


def test_api_results_uses_detected_condition_for_each_real_shape(tmp_path):
    """End-to-end through /api/results, one file per real shape, confirming
    the condition each one is actually labeled with on the page -- not just
    the detect_condition() unit, but the full normalize+serve path."""
    baseline = {
        "model": "m", "prompt_version": "v5", "timestamp": 1700000000,
        "summaries": {"reach": {"template": "reach", "n": 12, "passes": 3}},
        "rows": {"reach": []},
    }
    bo5_no_repair = {
        "condition": "best_of_5", "max_repairs": 0, "model": "m", "prompt_version": "v2",
        "summaries": {"reach": {"template": "reach", "n": 5, "passes": 1}},
        "rows": {"reach": []},
    }
    bo5_repair = {
        "condition": "best_of_5_repair", "max_repairs": 2, "model": "m", "prompt_version": "v2",
        "summaries": {"reach": {"template": "reach", "n": 7, "passes": 2}},
        "rows": {"reach": []},
    }
    (tmp_path / "baseline_1.json").write_text(json.dumps(baseline), encoding="utf-8")
    (tmp_path / "best_of_5_1.json").write_text(json.dumps(bo5_no_repair), encoding="utf-8")
    (tmp_path / "best_of_5_2.json").write_text(json.dumps(bo5_repair), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        by_file = {r["file"]: r["condition"] for r in body["rows"]}
        assert by_file["baseline_1.json"] == "single_shot"
        assert by_file["best_of_5_1.json"] == "best_of_5"
        assert by_file["best_of_5_2.json"] == "best_of_5_repair"
    finally:
        stop_server(httpd)


def test_detect_condition_falls_back_to_n_candidates_when_no_explicit_fields():
    """Older/legacy file: no top-level "condition" at all, but a per-row
    n_candidates -- 1 reads as single_shot, >1 (with no max_repairs) as
    best_of_5, >1 with max_repairs>0 as the canonical best_of_5_repair.
    This is the one place the old max_repairs-driven heuristic still
    applies, precisely because there's no explicit condition field to
    defer to instead."""
    single = {"rows": {"reach": [{"n_candidates": 1}]}}
    assert server.detect_condition(single, "legacy_1.json") == "single_shot"

    multi_no_repair = {"rows": {"reach": [{"n_candidates": 5}]}}
    assert server.detect_condition(multi_no_repair, "legacy_2.json") == "best_of_5"

    multi_with_repair = {"rows": {"reach": [{"n_candidates": 7}]}, "max_repairs": 2}
    assert server.detect_condition(multi_with_repair, "legacy_3.json") == "best_of_5_repair"


def test_detect_condition_is_unknown_not_single_shot_when_no_evidence():
    """The bug being fixed: a file with no condition field, no n_candidates
    signal, and a filename that doesn't match this repo's own naming must
    be reported as "unknown", never silently defaulted to single_shot."""
    assert server.detect_condition({}, "mystery_report.json") == "unknown"
    assert server.detect_condition({"condition": "something_new"}, "x.json") == "unknown"


def test_detect_condition_rejects_unrecognized_explicit_value_as_unknown():
    """A canonical-looking but not-actually-valid condition string (e.g. a
    typo, or a future name not yet added to VALID_CONDITIONS) must come
    back "unknown", not be guessed at via the old heuristics -- an explicit
    field, even a bad one, always takes precedence over falling back."""
    assert server.detect_condition({"condition": "best_of_3"}, "x.json") == "unknown"
    assert server.detect_condition({"condition": ""}, "baseline_x.json") == "single_shot"


# ---- condition detection: tavily_on (flag or suffix), orthogonal to condition ----

def test_detect_tavily_on_flag_form():
    assert server.detect_tavily_on({"condition": "best_of_5_repair_skills", "tavily_on": True}) is True
    assert server.detect_tavily_on({"condition": "best_of_5_repair_skills", "tavily_on": False}) is False
    assert server.detect_tavily_on({"condition": "single_shot"}) is False


def test_detect_tavily_on_suffix_form_does_not_change_detected_condition():
    data = {"condition": "best_of_5_repair_skills_tavily_on"}
    assert server.detect_tavily_on(data) is True
    assert server.detect_condition(data, "x.json") == "best_of_5_repair_skills"

    data2 = {"condition": "best_of_5_repair_tavily"}
    assert server.detect_tavily_on(data2) is True
    assert server.detect_condition(data2, "x.json") == "best_of_5_repair"


def test_api_results_rows_carry_tavily_on_field(tmp_path):
    data = {
        "model": "m", "prompt_version": "v1", "condition": "best_of_5_repair_skills_tavily",
        "summaries": {"reach": {"template": "reach", "n": 3, "passes": 1}},
        "rows": {"reach": []},
    }
    (tmp_path / "best_of_5_tavily.json").write_text(json.dumps(data), encoding="utf-8")
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        row = body["rows"][0]
        assert row["condition"] == "best_of_5_repair_skills"
        assert row["tavily_on"] is True
    finally:
        stop_server(httpd)


# ---- condition detection: trimmed real files from eval/results/backfilled/ ----
#
# The main session has been backfilling eval/results/ files with an
# explicit, canonical "condition" field (plus max_repairs, classifier_version,
# n_candidates, reasoning, etc. all stamped per-row by simsmith/run_meta.py).
# /home/mehar/projects/simsmith/eval/results/backfilled/ (a sibling worktree,
# outside this one) currently has exactly three distinct condition values in
# use: single_shot, best_of_5, best_of_5_repair. Each fixture below is a
# trimmed copy of one real file for that condition (the heavy per-template
# "summaries" detail and the "rows" lists are cut down to the minimum needed
# to still be a valid "summaries"-shaped result file); none of them must
# render as "unknown".

def test_backfilled_real_file_single_shot_not_unknown(tmp_path):
    """Trimmed from backfilled/baseline_1792749261.json."""
    data = {
        "prompt_version": "mujoco-v1", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "sim_python": "/home/mehar/nw/bin/python", "sim_seeds": [0, 1, 2, 3, 4],
        "timestamp": 1792749261, "condition": "single_shot", "reasoning": "on",
        "classifier_version": None, "n_candidates": 1, "max_repairs": 0,
        "date": "2026-10-23T09:54:21+00:00",
        "summaries": {
            "reach": {"template": "reach", "n": 20, "passes": 0, "pass_rate": 0.0,
                      "failure_breakdown": {"hallucinated_api": 8, "truncation": 8, "crash": 4}},
        },
        "rows": {"reach": []},
    }
    (tmp_path / "baseline_1792749261.json").write_text(json.dumps(data), encoding="utf-8")
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert body["rows"][0]["condition"] == "single_shot"
        assert body["rows"][0]["condition"] != "unknown"
    finally:
        stop_server(httpd)


def test_backfilled_real_file_best_of_5_not_unknown(tmp_path):
    """Trimmed from backfilled/best_of_5_1792778096.json."""
    data = {
        "prompt_version": "mujoco-v2", "tag": "partC-A: reasoning_on nano (tag-fix rerun)",
        "condition": "best_of_5", "max_repairs": 0, "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "sim_python": "/home/mehar/nw/bin/python", "sim_seeds": [0, 1, 2, 3, 4],
        "timestamp": 1792778096, "reasoning": "on", "classifier_version": None, "n_candidates": 5,
        "date": "2026-10-23T17:54:56+00:00",
        "summaries": {
            "reach": {"template": "reach", "n": 10, "passes": 0, "pass_rate": 0.0,
                      "failure_breakdown": {"hallucinated_api": 10}},
        },
        "rows": {"reach": []},
    }
    (tmp_path / "best_of_5_1792778096.json").write_text(json.dumps(data), encoding="utf-8")
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert body["rows"][0]["condition"] == "best_of_5"
        assert body["rows"][0]["condition"] != "unknown"
    finally:
        stop_server(httpd)


def test_backfilled_real_file_best_of_5_repair_not_unknown(tmp_path):
    """Trimmed from backfilled/best_of_5_1792777760.json."""
    data = {
        "prompt_version": "mujoco-v2", "tag": "mujoco-v2 best_of_5 (tag-fix rerun)",
        "condition": "best_of_5_repair", "max_repairs": 2, "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "sim_python": "/home/mehar/nw/bin/python", "sim_seeds": [0, 1, 2, 3, 4],
        "timestamp": 1792777760, "reasoning": "off", "classifier_version": None, "n_candidates": 5,
        "date": "2026-10-23T17:49:20+00:00",
        "summaries": {
            "reach": {"template": "reach", "n": 12, "passes": 0, "pass_rate": 0.0,
                      "failure_breakdown": {"crash": 4, "hallucinated_api": 7, "no_simulation": 1}},
            "pick_place": {"template": "pick_place", "n": 12, "passes": 1, "pass_rate": 0.083},
        },
        "rows": {"reach": [], "pick_place": []},
    }
    (tmp_path / "best_of_5_1792777760.json").write_text(json.dumps(data), encoding="utf-8")
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert len(body["rows"]) == 2  # reach + pick_place, neither "unknown"
        for row in body["rows"]:
            assert row["condition"] == "best_of_5_repair"
            assert row["condition"] != "unknown"
    finally:
        stop_server(httpd)


def test_api_results_marks_unknown_condition_file_as_unknown(tmp_path):
    data = {
        "model": "m", "prompt_version": "v1",
        "summaries": {"reach": {"template": "reach", "n": 3, "passes": 1}},
        "rows": {"reach": []},
    }
    (tmp_path / "mystery_report.json").write_text(json.dumps(data), encoding="utf-8")
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert body["rows"][0]["condition"] == "unknown"
    finally:
        stop_server(httpd)


# ---- (6.4) grouped results table + Conditions chart blocks ------------------

def test_results_html_has_banner_toggle_and_chart_blocks_container():
    html = _read(RESULTS_HTML)
    assert 'id="classifier-banner"' in html
    assert 'id="show-older-toggle"' in html
    assert 'id="conditions-blocks"' in html
    # the old per-filter select controls are gone -- replaced by one block per model/template
    assert 'id="conditions-model"' not in html
    assert 'id="conditions-template"' not in html


def test_results_js_groups_table_by_model_template_prompt_condition():
    src = _read(RESULTS_JS)
    assert "function groupRows(" in src
    assert "function newestPromptVersion(" in src
    assert "function renderResultsTable(" in src
    assert "group-model" in src and "group-template" in src and "group-prompt" in src
    assert "show older runs" in src.lower() or "Show older runs" in src


def test_results_js_builds_svg_chart_blocks_without_innerhtml():
    src = _read(RESULTS_JS)
    for sink in (".innerHTML", ".outerHTML", ".insertAdjacentHTML(", "document.write("):
        assert sink not in src, f"results.js must not use {sink}"
    assert "createElementNS" in src
    assert "function renderConditionsBlock(" in src
    assert "function renderConditionsBlocks(" in src
    assert "chart-whisker" in src  # whisker drawn from the Wilson interval
    # one block per (model, template), not a single filtered chart
    assert "function groupByModelTemplate(" in src


# ---- canonical vocabulary: display names + least-to-most-machinery order ----

def test_results_js_condition_order_is_canonical_and_least_to_most_machinery():
    src = _read(RESULTS_JS)
    order_start = src.index("const CONDITION_ORDER = [")
    order_block = src[order_start:src.index("];", order_start)]
    # exact sequence, least machinery first -- a stray reorder would silently
    # put a bigger/more-expensive condition's bar ahead of a cheaper one
    for earlier, later in (
        ("single_shot", "best_of_5"),
        ("best_of_5", "best_of_5_repair"),
        ("best_of_5_repair", "best_of_5_repair_library"),
        ("best_of_5_repair_library", "best_of_5_repair_skills"),
    ):
        assert order_block.index(f'"{earlier}"') < order_block.index(f'"{later}"')
    # the old "+repair"-style strings must be gone entirely
    assert "+repair" not in src
    assert "best_of_5_library" not in src  # old (wrong) name for the library condition


def test_results_js_has_display_names_for_every_canonical_condition():
    src = _read(RESULTS_JS)
    assert "function conditionLabel(" in src
    for canonical, display in (
        ("single_shot", "Single-shot"),
        ("best_of_5", "Best-of-5"),
        ("best_of_5_repair", "Best-of-5 + repair"),
        ("best_of_5_repair_library", "Best-of-5 + repair + library"),
        ("best_of_5_repair_skills", "Best-of-5 + repair + skills"),
    ):
        assert f"{canonical}:" in src and display in src

    # the table and the chart row both render through the display-name
    # layer, not the raw canonical string
    assert "rowConditionLabel(row)" in src
    render_row_start = src.index("function renderConditionRow(")
    render_row_body = src[render_row_start:render_row_start + 400]
    assert "conditionLabel(condition)" in render_row_body


def test_results_js_tavily_shown_as_suffix_not_a_new_condition():
    src = _read(RESULTS_JS)
    assert '" (Tavily)"' in src
    assert "tavily_on" in src
    # tavily must not appear in the ordered condition vocabulary itself
    order_start = src.index("const CONDITION_ORDER = [")
    order_block = src[order_start:src.index("];", order_start)]
    assert "tavily" not in order_block.lower()


# ---- Conditions chart: a condition with zero runs still gets a "not run" row ----

def test_results_js_defines_not_run_placeholder_row():
    src = _read(RESULTS_JS)
    assert "function renderNotRunConditionRow(" in src
    not_run_start = src.index("function renderNotRunConditionRow(")
    not_run_body = src[not_run_start:not_run_start + 1200]
    assert '"Not run"' in not_run_body
    # same label layer as a real row, not a raw/untranslated condition string
    assert "conditionLabel(condition)" in not_run_body
    # a dashed/empty track class, not the normal filled chart-track alone
    assert "chart-track-empty" in not_run_body


def test_results_js_condition_rows_always_cover_the_canonical_vocabulary():
    """renderConditionRows() must consider every CONDITION_ORDER entry, not
    just whatever conditions happen to appear in the data -- a condition with
    no rows for this model/template/prompt-version renders the placeholder
    instead of being silently skipped."""
    src = _read(RESULTS_JS)
    rows_start = src.index("function renderConditionRows(")
    rows_end = src.index("\nfunction renderConditionsBlock(")
    body = src[rows_start:rows_end]
    assert "CONDITION_ORDER.concat(" in body
    assert "renderNotRunConditionRow(condition)" in body
    # the old byCondition-keys-only iteration (which silently dropped any
    # condition absent from the data) must be gone from this function
    assert "Array.from(byCondition.keys()).sort((a, b) => conditionRank(a) - conditionRank(b))" not in body


# ---- Conditions chart: never mix prompt versions in one block/bar ----------

def test_api_results_two_files_different_prompt_versions_stay_as_separate_rows(tmp_path):
    """The data-layer guarantee the chart depends on: two result files for
    the same model/template/condition but different prompt_version must
    come back as two distinct rows with their own n/pass_rate, never merged
    or averaged into one. (The renderer then additionally refuses to put
    them on the same bar -- see the results.js structural tests below.)"""
    older = {
        "model": "nemotron-70b", "prompt_version": "mujoco-v1", "timestamp": 1700000000,
        "summaries": {"reach": {"template": "reach", "n": 4, "passes": 1,
                                 "failure_breakdown": {"pass": 1, "crash": 3}}},
        "rows": {"reach": []},
    }
    newer = {
        "model": "nemotron-70b", "prompt_version": "mujoco-v2", "timestamp": 1700100000,
        "summaries": {"reach": {"template": "reach", "n": 4, "passes": 3,
                                 "failure_breakdown": {"pass": 3, "crash": 1}}},
        "rows": {"reach": []},
    }
    (tmp_path / "baseline_older.json").write_text(json.dumps(older), encoding="utf-8")
    (tmp_path / "baseline_newer.json").write_text(json.dumps(newer), encoding="utf-8")

    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        rows = [r for r in body["rows"] if r["model"] == "nemotron-70b" and r["template"] == "reach"]
        assert len(rows) == 2
        by_version = {r["prompt_version"]: r for r in rows}
        assert set(by_version.keys()) == {"mujoco-v1", "mujoco-v2"}
        # each version keeps its own, unmerged pass rate -- never averaged (which would be 0.5)
        assert by_version["mujoco-v1"]["pass_rate"] == 0.25
        assert by_version["mujoco-v2"]["pass_rate"] == 0.75
        assert by_version["mujoco-v1"]["n"] == 4
        assert by_version["mujoco-v2"]["n"] == 4
    finally:
        stop_server(httpd)


def test_results_js_default_view_filters_to_single_prompt_version_by_exact_match():
    """The default/selected (non-"All versions") path must filter rows by
    an exact prompt_version equality check before grouping into condition
    rows -- this is what guarantees a bar never silently combines two
    different prompt versions."""
    src = _read(RESULTS_JS)
    assert "function pickDefaultPromptVersion(" in src
    render_block_start = src.index("function renderConditionsBlock(")
    render_block_body = src[render_block_start:render_block_start + 2400]
    assert 'rows.filter((r) => r.prompt_version === selected)' in render_block_body
    assert "const defaultVersion = pickDefaultPromptVersion(rows)" in render_block_body


def test_results_js_mismatch_warning_only_shown_in_all_versions_mode():
    src = _read(RESULTS_JS)
    render_block_start = src.index("function renderConditionsBlock(")
    render_fn_start = src.index("function render() {", render_block_start)
    render_fn_body = src[render_fn_start:render_fn_start + 1000]
    assert "not directly comparable" in render_fn_body
    # the warning text only appears inside a branch gated on showingAll
    guard_idx = render_fn_body.index("if (showingAll")
    warning_idx = render_fn_body.index("not directly comparable")
    next_brace_close = render_fn_body.index("}", guard_idx)
    assert guard_idx < warning_idx < next_brace_close + 200  # warning sits inside that if-block


def test_results_js_never_pools_multiple_runs_into_one_averaged_bar():
    """pickLatestPerCondition (which silently dropped every run but the
    newest) must be gone entirely -- every run for a condition now gets its
    own marker and its own line in the run list, never averaged together."""
    src = _read(RESULTS_JS)
    assert "pickLatestPerCondition" not in src
    assert "function renderConditionRow(" in src
    assert "run-list" in src
    render_row_start = src.index("function renderConditionRow(")
    render_row_body = src[render_row_start:render_row_start + 2000]
    assert "runs.forEach((r) => {" in render_row_body
    # no averaging/summing arithmetic over multiple runs' pass_rate in this function
    assert "reduce" not in render_row_body


def test_results_js_version_select_offers_all_versions_sentinel():
    src = _read(RESULTS_JS)
    assert 'const ALL_VERSIONS = "__all__";' in src
    assert '"All versions"' in src
    assert "conditions-version-select" in src


# ---- failure-breakdown color legend -----------------------------------------

def test_results_html_has_failure_breakdown_legend():
    html = _read(RESULTS_HTML)
    assert 'id="fail-legend-key"' in html
    for category in (
        "pass", "crash", "hallucinated_api", "wrong_logic", "no_simulation",
        "truncation", "preflight_other", "task_param_mismatch",
    ):
        assert f"legend-{category}" in html


def test_style_css_legend_swatch_colors_match_fail_seg_colors():
    """The legend's colors must actually match what renders in the bars --
    checked by pinning both the .fail-seg.fail-<x> and the
    .legend-swatch.legend-<x> rules to the same literal color value for a
    representative sample of categories."""
    css = _read(STYLE_CSS)
    for category, color in (
        ("pass", "var(--green)"),
        ("crash", "var(--red)"),
        ("hallucinated_api", "#d16bff"),
        ("wrong_logic", "var(--amber)"),
    ):
        seg_rule = f".fail-seg.fail-{category}"
        legend_rule = f".legend-swatch.legend-{category}"
        assert seg_rule in css or f".fail-seg.fail-{category}," in css
        assert legend_rule in css
        legend_idx = css.index(legend_rule)
        legend_block = css[legend_idx:legend_idx + 80]
        assert color in legend_block


def test_api_results_malformed_file_still_reported_in_errors(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    state = _make_results_state(tmp_path)
    httpd, port = start_server(state)
    try:
        status, body = request(port, "GET", "/api/results")
        assert status == 200
        assert body["rows"] == []
        assert any("broken.json" in e for e in body["errors"])
        assert any("unrelated.json" in e for e in body["errors"])
    finally:
        stop_server(httpd)


# ---- (6.4) demo / presentation mode -----------------------------------------

def test_index_html_has_present_toggle_and_hidden_controls():
    html = _read(INDEX_HTML)
    assert 'id="present-toggle"' in html
    assert html.count("present-hide") >= 2  # at least two controls hidden in present mode


def test_app_js_present_mode_reads_query_param_and_toggles_class():
    src = _read(APP_JS)
    assert "function initPresentMode(" in src
    assert "URLSearchParams" in src
    assert '"present"' in src
    assert 'classList.toggle("present-mode"' in src
    # init() must actually wire it up, not just define it
    init_start = src.index("async function init() {")
    init_body = src[init_start:init_start + 400]
    assert "initPresentMode()" in init_body


def test_style_css_present_mode_rules_and_bigger_mock_badge():
    src = _read(STYLE_CSS)
    assert "body.present-mode" in src
    assert "body.present-mode .source-badge.source-mock" in src
    # the MOCK badge rule in present mode must be larger than the generic present-mode badge rule
    generic = src.index("body.present-mode .source-badge {")
    generic_block = src[generic:generic + 200]
    mock = src.index("body.present-mode .source-badge.source-mock {")
    mock_block = src[mock:mock + 200]
    import re as _re
    generic_size = float(_re.search(r"font-size:\s*(\d+)", generic_block).group(1))
    mock_size = float(_re.search(r"font-size:\s*(\d+)", mock_block).group(1))
    assert mock_size > generic_size


# ---- (6.5) hero + pipeline stage strip ---------------------------------------

STAGE_NAMES = ("intake", "generate", "preflight", "execute", "verify", "repair")


def test_index_html_has_hero_and_stage_strip():
    html = _read(INDEX_HTML)
    assert "hero-title" in html
    assert 'id="stage-strip"' in html
    for stage in STAGE_NAMES:
        assert f'data-stage="{stage}"' in html


def test_app_js_computes_and_renders_stage_activity_from_result():
    src = _read(APP_JS)
    assert "function computeActiveStages(" in src
    assert "function renderStageStrip(" in src
    assert "function clearStages(" in src
    for stage in STAGE_NAMES:
        assert f'"{stage}"' in src
    # renderResult() must call renderStageStrip on every path
    result_fn_start = src.index("function renderResult(result) {")
    result_fn_body = src[result_fn_start:result_fn_start + 500]
    assert "renderStageStrip(result)" in result_fn_body
    # repair only lights up if repair rounds actually happened
    compute_start = src.index("function computeActiveStages(")
    compute_body = src[compute_start:compute_start + 600]
    assert "repair_rounds" in compute_body


def test_app_js_clears_stages_before_starting_a_new_run():
    src = _read(APP_JS)
    run_click_start = src.index("async function onRunClick() {")
    run_click_body = src[run_click_start:run_click_start + 300]
    assert "clearStages()" in run_click_body


# ---- (6.6) Winner panel: green "Winner" styling/wording is VERIFIED-only ----

def test_serialize_result_partial_status_still_carries_a_winner(tmp_path):
    """Backend contract check: pipeline.py sets a winner on a PARTIAL run
    (the best-scoring candidate, even though it didn't clear the bar) --
    serialize_result() must pass that straight through so the frontend has
    both result.status == "PARTIAL" and result.winner to react to."""
    from simsmith.pipeline import Candidate, SolveResult
    from simsmith.verifier import Score

    score = Score(ok=False, success_rate=0.5, crashes=0,
                   details=["seed 0: PASS", "seed 1: FAIL off by 0.5m"],
                   failure_text="seed 1 missed the target by 0.5m", mean_seconds=0.2)
    candidate = Candidate(idx=0, script="print('best attempt')", framing="f", temperature=0.2,
                          stage="generated", preflight_ok=True, preflight_problems=[], score=score)
    res = SolveResult(task="reach the target", template="reach", exact_match=True, gap="",
                      plan="1. move", candidates=[candidate], winner=candidate,
                      status="PARTIAL", notes=["no candidate passed every seed"])

    payload = server.serialize_result(res, library_size=0, elapsed_s=1.0, source="mock")
    assert payload["status"] == "PARTIAL"
    assert payload["winner"] is not None
    assert payload["winner"]["idx"] == 0


def test_app_js_winner_panel_uses_verified_style_and_wording_only_when_verified():
    """Structural check mirroring the rest of this file's app.js assertions
    (no JS runtime available here): the green "winner-panel" class and the
    literal "Winner" heading text must both be conditioned on
    result.status === "VERIFIED", never applied unconditionally -- that was
    the bug (every run with a winner, VERIFIED or not, got the green
    "Winner" panel)."""
    src = _read(APP_JS)
    render_winner_start = src.index("function renderWinner(result) {")
    render_winner_body = src[render_winner_start:render_winner_start + 2200]

    assert 'const verified = result.status === "VERIFIED";' in render_winner_body
    assert '"Best attempt (not verified)"' in render_winner_body
    assert 'box.className = verified ? "winner-panel" : "winner-panel not-verified";' in render_winner_body
    assert "Why this is not verified:" in render_winner_body
    # the plain, green "winner-panel" class literal must only appear inside that ternary,
    # never assigned unconditionally elsewhere in the function
    unconditioned = render_winner_body.replace('box.className = verified ? "winner-panel" : "winner-panel not-verified";', "")
    assert 'box.className = "winner-panel";' not in unconditioned


def test_index_html_winner_title_is_dynamic_not_hardcoded():
    html = _read(INDEX_HTML)
    assert 'id="winner-title"' in html


def test_style_css_not_verified_winner_panel_is_not_green():
    src = _read(STYLE_CSS)
    assert ".winner-panel.not-verified" in src
    block_start = src.index(".winner-panel.not-verified {")
    block = src[block_start:block_start + 200]
    assert "var(--green)" not in block


def test_stage_strip_lights_up_repair_only_when_repair_happened_end_to_end(tmp_path):
    """End-to-end sanity check (via the real mock pipeline) that a finished
    run's data distinguishes the repair case: the 'reach' example always
    produces a repair round in this demo's scripted mock, and the server-side
    payload used by computeActiveStages() must say so explicitly."""
    state, cfg, docs = make_mock_state(tmp_path)
    httpd, port = start_server(state, cfg, docs)
    try:
        post_json(port, "/api/run", {"task": "Move the arm's gripper to the target point."})
        final = wait_for_done(port)
        result = final["result"]
        assert len(result["repair_rounds"]) > 0
        assert result["sim_runs"] > 0
        assert any(a["score"] for g in result["candidates"] for a in g["attempts"])
    finally:
        stop_server(httpd)
