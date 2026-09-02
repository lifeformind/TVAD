"""DirectGrabber — the tune server's own camera path while the kiosk is
stopped. Wraps the vision backend: lazy open, annotated JPEG out, clean close."""
import numpy as np

from tune.vision_preview import DirectGrabber


class FakeBackend:
    def __init__(self, open_ok=True, frames=None):
        self._open_ok = open_ok
        self._frames = frames if frames is not None else []
        self.opened = 0
        self.closed = False

    def open(self):
        self.opened += 1
        return self._open_ok

    def grab(self):
        return self._frames.pop(0) if self._frames else None

    def detect_box(self, frame):
        return (5.0, 5.0, 20.0, 20.0)

    def close(self):
        self.closed = True


def _frame():
    return np.zeros((60, 80, 3), dtype=np.uint8)


def test_grab_jpeg_opens_once_and_returns_bytes():
    be = FakeBackend(frames=[_frame(), _frame()])
    g = DirectGrabber({}, _backend=be)
    a, b = g.grab_jpeg(), g.grab_jpeg()
    assert a[:2] == b"\xff\xd8" and b[:2] == b"\xff\xd8"
    assert be.opened == 1


def test_grab_jpeg_none_when_open_fails():
    g = DirectGrabber({}, _backend=FakeBackend(open_ok=False))
    assert g.grab_jpeg() is None


def test_grab_jpeg_none_when_no_frame():
    g = DirectGrabber({}, _backend=FakeBackend(frames=[]))
    assert g.grab_jpeg() is None


def test_close_releases_backend_and_reopens_after():
    be = FakeBackend(frames=[_frame(), _frame()])
    g = DirectGrabber({}, _backend=be)
    assert g.grab_jpeg() is not None
    g.close()
    assert be.closed is True
    assert g.grab_jpeg() is not None      # lazy re-open on next use
    assert be.opened == 2
