"""KioskProcess lifecycle against fake children (bash one-liners): start/stop/
restart, TERM-then-KILL, foreign-process refusal, ring buffer + SSE fan-out,
exit announcement. Nothing here touches the real kiosk."""

import queue
import time

import pytest

from tune.kiosk_proc import KioskProcError, KioskProcess


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _proc(script, **kw):
    kw.setdefault("foreign_pids", lambda: [])
    return KioskProcess(cmd=["bash", "-c", script], **kw)


def test_start_stop_lifecycle():
    p = _proc("sleep 30")
    p.start(diag=False)
    st = p.status()
    assert st["running"] is True and isinstance(st["pid"], int)
    p.stop()
    assert p.status() == {"running": False, "pid": None, "diag": False}


def test_double_start_refused():
    p = _proc("sleep 30")
    p.start(diag=False)
    try:
        with pytest.raises(KioskProcError, match="already running"):
            p.start(diag=False)
    finally:
        p.stop()


def test_foreign_kiosk_refused():
    p = KioskProcess(cmd=["bash", "-c", "sleep 30"], foreign_pids=lambda: [4242])
    with pytest.raises(KioskProcError, match="4242"):
        p.start(diag=False)
    assert p.status()["running"] is False


def test_stop_escalates_to_kill_when_term_ignored():
    p = _proc("trap '' TERM; echo up; sleep 60", term_grace_s=0.3)
    p.start(diag=False)
    snapshot, q = p.attach()
    assert _wait(lambda: "up" in "\n".join(p.attach()[0]))
    t0 = time.monotonic()
    p.stop()
    assert time.monotonic() - t0 < 5.0          # grace 0.3 + KILL, not 60s
    assert p.status()["running"] is False


def test_output_lands_in_ring_ansi_stripped():
    p = _proc(r"printf '\033[1mhello\033[0m world\n'; sleep 30")
    p.start(diag=False)
    try:
        assert _wait(lambda: any("hello world" == l for l in p.attach()[0]))
    finally:
        p.stop()


def test_exit_is_announced_with_code():
    p = _proc("echo bye; exit 3")
    p.start(diag=False)
    assert _wait(lambda: any("exited (code 3)" in l for l in p.attach()[0]))
    assert p.status()["running"] is False


def test_attach_replays_ring_then_streams_live():
    p = _proc("echo first; sleep 0.3; echo second; sleep 30")
    p.start(diag=False)
    try:
        assert _wait(lambda: any("first" in l for l in p.attach()[0]))
        snapshot, q = p.attach()
        assert any("first" in l for l in snapshot)
        line = q.get(timeout=5.0)
        assert "second" in line
        p.detach(q)
    finally:
        p.stop()


def test_diag_env_reaches_the_child():
    p = _proc('echo "diag=${TVAD_DIAG:-unset}"; sleep 30')
    p.start(diag=True)
    try:
        assert _wait(lambda: any("diag=1" in l for l in p.attach()[0]))
        assert p.status()["diag"] is True
    finally:
        p.stop()
    p2 = _proc('echo "diag=${TVAD_DIAG:-unset}"; sleep 30')
    p2.start(diag=False)
    try:
        assert _wait(lambda: any("diag=unset" in l for l in p2.attach()[0]))
    finally:
        p2.stop()


def test_restart_reuses_last_diag_flag():
    p = _proc('echo "diag=${TVAD_DIAG:-unset}"; sleep 30')
    p.start(diag=True)
    p.restart()
    try:
        assert p.status() == {"running": True, "pid": p.status()["pid"], "diag": True}
        assert _wait(lambda: "\n".join(p.attach()[0]).count("diag=1") >= 2)
    finally:
        p.stop()


def test_ring_is_bounded():
    p = _proc("for i in $(seq 1 50); do echo line$i; done; sleep 30", ring_size=10)
    p.start(diag=False)
    try:
        assert _wait(lambda: any("line50" in l for l in p.attach()[0]))
        snapshot, q = p.attach()
        p.detach(q)
        assert len(snapshot) <= 10
        assert not any("line1" == l for l in snapshot)
    finally:
        p.stop()
