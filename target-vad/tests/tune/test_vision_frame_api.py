"""GET /api/vision/frame — the tuning console's camera panel source.

Kiosk running: relay the fresh preview JPEG the vision loop publishes
(atomic file); stale/missing => 503 (never a torn or ancient frame).
Kiosk stopped: fall back to a direct grabber the server owns — which MUST
be released before the kiosk child starts (V4L2 access is exclusive)."""
import http.client
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from tune.kiosk_proc import KioskProcess
from tune.server import TuningServer

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
JPEG = b"\xff\xd8\xff\xe0fakejpegbytes"


class FakeGrabber:
    def __init__(self, jpeg=JPEG):
        self.jpeg = jpeg
        self.closed = False

    def grab_jpeg(self):
        return self.jpeg

    def close(self):
        self.closed = True


@pytest.fixture()
def srv(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(REAL_CONFIG, cfg)
    preview = tmp_path / "pv.jpg"
    cfg.write_text(cfg.read_text().replace(
        "/dev/shm/tvad-vision-preview.jpg", str(preview)))
    kproc = KioskProcess(cmd=["bash", "-c", "echo kiosk-up; sleep 30"],
                         foreign_pids=lambda: [])
    grabber = FakeGrabber()
    server = TuningServer(config_path=str(cfg), kproc=kproc, port=0,
                          grabber_factory=lambda: grabber)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server, preview, grabber, kproc
    kproc.stop()
    server.shutdown()


def _get_raw(server, path):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    ctype = resp.getheader("Content-Type")
    conn.close()
    return resp.status, ctype, data


def _post(server, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("POST", path, body=json.dumps(body or {}),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data) if data else None


def test_running_with_fresh_file_relays_jpeg(srv):
    server, preview, grabber, kproc = srv
    assert _post(server, "/api/kiosk/start")[0] == 200
    preview.write_bytes(JPEG)
    status, ctype, data = _get_raw(server, "/api/vision/frame?t=1")
    assert status == 200 and ctype == "image/jpeg" and data == JPEG
    assert grabber.closed is False and not hasattr(grabber, "grabbed")


def test_running_with_stale_file_is_503(srv):
    server, preview, grabber, kproc = srv
    assert _post(server, "/api/kiosk/start")[0] == 200
    preview.write_bytes(JPEG)
    old = time.time() - 60
    os.utime(preview, (old, old))
    status, ctype, data = _get_raw(server, "/api/vision/frame")
    assert status == 503 and b"error" in data


def test_running_with_missing_file_is_503(srv):
    server, preview, grabber, kproc = srv
    assert _post(server, "/api/kiosk/start")[0] == 200
    status, _, data = _get_raw(server, "/api/vision/frame")
    assert status == 503


def test_stopped_serves_direct_grabber(srv):
    server, preview, grabber, kproc = srv
    status, ctype, data = _get_raw(server, "/api/vision/frame")
    assert status == 200 and ctype == "image/jpeg" and data == JPEG


def test_stopped_and_grabber_fails_is_503(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(REAL_CONFIG, cfg)

    class DeadGrabber:
        def grab_jpeg(self):
            return None

        def close(self):
            pass

    kproc = KioskProcess(cmd=["bash", "-c", "sleep 30"], foreign_pids=lambda: [])
    server = TuningServer(config_path=str(cfg), kproc=kproc, port=0,
                          grabber_factory=lambda: DeadGrabber())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, _, data = _get_raw(server, "/api/vision/frame")
        assert status == 503
    finally:
        server.shutdown()


def test_kiosk_start_releases_direct_grabber(srv):
    server, preview, grabber, kproc = srv
    # Direct grab first => the server now holds the camera.
    assert _get_raw(server, "/api/vision/frame")[0] == 200
    assert grabber.closed is False
    # Starting the kiosk MUST release it before the child can want the device.
    assert _post(server, "/api/kiosk/start")[0] == 200
    assert grabber.closed is True


def test_kiosk_restart_releases_direct_grabber(srv):
    server, preview, grabber, kproc = srv
    assert _get_raw(server, "/api/vision/frame")[0] == 200   # server holds camera
    assert grabber.closed is False
    # Restart is the tune-a-vision-knob-then-apply workflow: it MUST release
    # the grabber before the kiosk child spawns, same as start.
    assert _post(server, "/api/kiosk/restart")[0] == 200
    assert grabber.closed is True


def test_running_poll_self_heals_a_leaked_grabber(srv):
    server, preview, grabber, kproc = srv
    # Simulate the start/poll race: grabber acquired, then kiosk begins running
    # WITHOUT the start route (e.g. --start-kiosk, or the TOCTOU window).
    assert _get_raw(server, "/api/vision/frame")[0] == 200
    assert grabber.closed is False
    kproc.start(diag=False)
    preview.write_bytes(JPEG)
    # First frame poll while running must release the held grabber — the kiosk
    # only opens the camera at SESSION start, so this closes the race for real.
    assert _get_raw(server, "/api/vision/frame")[0] == 200
    assert grabber.closed is True
