"""Tests for VisionWorker — camera thread that enrolls, monitors, emits presence."""
from modes.director.events import OwnerPresenceEvent, PresenceStatus as PS
from modes.director.workers.vision import VisionWorker


class FakeBackend:
    """grab/embed/make_classify_fn driven by scripted sequences."""
    def __init__(self, grabs, embeds, classifies):
        self._grabs = list(grabs)
        self._embeds = list(embeds)
        self._classifies = list(classifies)
        self.opened = False
        self.closed = False

    def open(self): self.opened = True; return True
    def grab(self): return self._grabs.pop(0) if self._grabs else None
    def embed(self, frame): return self._embeds.pop(0) if self._embeds else None

    def make_classify_fn(self, reference):
        seq = self._classifies

        def fn(frame):
            return seq.pop(0)
        return fn

    def close(self): self.closed = True


def _worker(backend):
    return VisionWorker(backend, bus=None, fps=10.0, present_after_s=1.0,
                        absent_after_s=2.0, enroll_frames=2)


def test_enrolls_then_monitors_and_emits_present():
    import numpy as np
    be = FakeBackend(grabs=["f", "f", "f", "f"],
                     embeds=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
                     classifies=[PS.PRESENT, PS.PRESENT, PS.PRESENT])
    w = _worker(be)
    # First _run_once enrolls (consumes enroll grabs+embeds), returns no event yet.
    assert w._run_once(0.0) is None
    assert w._run_once(0.1) is None     # monitor: still debouncing
    ev = w._run_once(1.2)               # debounced present
    assert isinstance(ev, OwnerPresenceEvent) and ev.status is PS.PRESENT


def test_failed_enroll_reports_unavailable_not_absent():
    be = FakeBackend(grabs=["f", "f", "f", "f", "f"], embeds=[None, None, None, None, None],
                     classifies=[])
    w = _worker(be)
    ev = w._run_once(0.0)
    assert isinstance(ev, OwnerPresenceEvent) and ev.status is PS.UNAVAILABLE
    # stays unavailable; never emits ABSENT off a failed enroll
    assert w._run_once(0.1) is None
