"""--start-kiosk entry-point helper: auto-start is best-effort — a refusal
(foreign kiosk already running, double start) must be reported, not raised,
so the console still comes up and the browser's Start button stays usable."""

from tune.__main__ import start_kiosk_if_requested
from tune.kiosk_proc import KioskProcess


def _kproc(**kw):
    kw.setdefault("foreign_pids", lambda: [])
    return KioskProcess(cmd=["bash", "-c", "sleep 30"], **kw)


def test_not_requested_does_nothing():
    p = _kproc()
    out = []
    start_kiosk_if_requested(p, False, out=out.append)
    assert p.status()["running"] is False and out == []


def test_requested_starts_with_diag_on():
    p = _kproc()
    out = []
    try:
        start_kiosk_if_requested(p, True, out=out.append)
        st = p.status()
        assert st["running"] is True and st["diag"] is True
        assert any("auto-started" in line for line in out)
    finally:
        p.stop()


def test_refusal_is_reported_not_raised():
    p = KioskProcess(cmd=["bash", "-c", "sleep 30"],
                     foreign_pids=lambda: [4242])
    out = []
    start_kiosk_if_requested(p, True, out=out.append)  # must not raise
    assert p.status()["running"] is False
    assert any("4242" in line for line in out)
