"""API tests against a real TuningServer on an ephemeral port, with a temp
copy of the real config.yaml and a fake kiosk child. The UI is not the trust
boundary: hand-crafted bad POSTs must be rejected server-side."""

import http.client
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

from tune.kiosk_proc import KioskProcess
from tune.server import TuningServer

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture()
def srv(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(REAL_CONFIG, cfg)
    kproc = KioskProcess(cmd=["bash", "-c", "echo kiosk-up; sleep 30"],
                         foreign_pids=lambda: [])
    server = TuningServer(config_path=str(cfg), kproc=kproc, port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server, cfg
    kproc.stop()
    server.shutdown()


def _req(server, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data) if data else None


def test_state_has_knobs_with_fresh_values(srv):
    server, cfg = srv
    status, state = _req(server, "GET", "/api/state")
    assert status == 200
    by_path = {k["path"]: k for k in state["knobs"]}
    assert by_path["kiosk.talkback.turn_gate.doa.cone_deg"]["value"] == 20
    assert state["kiosk"]["running"] is False
    assert "reachable" in state["llm"]
    # values are read fresh: hand-edit the file, state reflects it
    cfg.write_text(cfg.read_text().replace("cone_deg: 20", "cone_deg: 30"))
    _, state2 = _req(server, "GET", "/api/state")
    by_path2 = {k["path"]: k for k in state2["knobs"]}
    assert by_path2["kiosk.talkback.turn_gate.doa.cone_deg"]["value"] == 30


def test_save_writes_file_and_preserves_comments(srv):
    server, cfg = srv
    before = cfg.read_text()
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0}})
    assert status == 200 and out == {"saved": ["kiosk.talkback.turn_gate.doa.cone_deg"]}
    after = cfg.read_text()
    assert yaml.safe_load(after)["kiosk"]["talkback"]["turn_gate"]["doa"]["cone_deg"] == 25.0
    diff = [1 for a, b in zip(before.split("\n"), after.split("\n")) if a != b]
    assert len(diff) == 1


def test_save_rejects_unregistered_path(srv):
    server, cfg = srv
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.output_device": "hax"}})
    assert status == 400 and "output_device" in out["error"]
    assert "hax" not in cfg.read_text()


def test_save_rejects_out_of_range_and_wrong_kind(srv):
    server, cfg = srv
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": 500}})
    assert status == 400
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.reject_bystanders": "yes"}})
    assert status == 400
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.wake_phrase": "hey_hacker"}})
    assert status == 400


def test_save_nullable_accepts_null_others_do_not(srv):
    server, cfg = srv
    status, _ = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.barge_in.proximity.rms_threshold": None}})
    assert status == 200
    status, _ = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": None}})
    assert status == 400


def test_save_all_or_nothing(srv):
    server, cfg = srv
    before = cfg.read_text()
    status, _ = _req(server, "POST", "/api/save", {"changes": {
        "kiosk.talkback.turn_gate.doa.cone_deg": 25.0,
        "kiosk.talkback.output_device": "hax"}})
    assert status == 400
    assert cfg.read_text() == before


def test_kiosk_start_stop_restart_roundtrip(srv):
    server, cfg = srv
    status, st = _req(server, "POST", "/api/kiosk/start", {"diag": True})
    assert status == 200 and st["running"] is True and st["diag"] is True
    status, _ = _req(server, "POST", "/api/kiosk/start", {"diag": True})
    assert status == 409
    status, st = _req(server, "POST", "/api/kiosk/restart", None)
    assert status == 200 and st["running"] is True
    status, st = _req(server, "POST", "/api/kiosk/stop", None)
    assert status == 200 and st["running"] is False
    status, st = _req(server, "POST", "/api/kiosk/stop", None)  # idempotent
    assert status == 200


def test_logs_sse_replays_ring(srv):
    server, cfg = srv
    _req(server, "POST", "/api/kiosk/start", {"diag": False})
    time.sleep(0.5)  # let the echo land in the ring
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as s:
        s.sendall(b"GET /api/logs HTTP/1.1\r\nHost: x\r\n\r\n")
        buf = b""
        deadline = time.monotonic() + 5
        while b"kiosk-up" not in buf and time.monotonic() < deadline:
            buf += s.recv(4096)
    assert b"text/event-stream" in buf
    assert b"data: kiosk-up" in buf


def test_unknown_route_404s_with_json(srv):
    server, cfg = srv
    status, out = _req(server, "GET", "/api/nope")
    assert status == 404 and "error" in out


def test_unexpected_error_returns_500_json(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(REAL_CONFIG, cfg)

    def boom():
        raise RuntimeError("boom")

    kproc = KioskProcess(cmd=["bash", "-c", "sleep 30"], foreign_pids=boom)
    server = TuningServer(config_path=str(cfg), kproc=kproc, port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, out = _req(server, "POST", "/api/kiosk/start", {"diag": False})
        assert status == 500 and "boom" in out["error"]
    finally:
        server.shutdown()


def test_index_served_with_expected_ui_hooks(srv):
    server, cfg = srv
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    assert resp.status == 200
    for hook in ("id=\"tabs\"", "id=\"panes\"", "id=\"savebar\"",
                 "id=\"logpane\"", "/api/state", "/api/save", "/api/logs",
                 "/api/kiosk/"):
        assert hook in body, hook


def test_save_preserves_file_mode(srv):
    server, cfg = srv
    os.chmod(cfg, 0o664)
    status, _ = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0}})
    assert status == 200
    assert (cfg.stat().st_mode & 0o777) == 0o664


def test_sigterm_handler_raises_systemexit():
    from tune.__main__ import _sigterm
    with pytest.raises(SystemExit):
        _sigterm(15, None)
