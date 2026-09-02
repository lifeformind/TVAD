"""Detail capture for the camera preview: the backend stashes what the last
classify saw (box/score/raw verdict) and the VisionWorker feeds an injected
preview sink after each monitor step — never during enroll, never fatally."""
import numpy as np

from modes.director.events import PresenceStatus as PS
from modes.director.vision.opencv_backend import OpenCvBackend
from modes.director.workers.vision import VisionWorker


class _Rec:
    def alignCrop(self, frame, f):
        return frame

    def feature(self, aligned):
        return np.array([[1.0, 0.0]])


def _bare_backend():
    b = OpenCvBackend.__new__(OpenCvBackend)
    b._thr = 0.4
    b._min_area = 0.015
    b._rec = _Rec()
    return b


def test_classify_fn_stashes_box_score_and_raw_verdict():
    b = _bare_backend()
    b._largest_face = lambda frame: np.array([300.0, 100.0, 80.0, 100.0])
    fn = b.make_classify_fn(reference=np.array([1.0, 0.0]))
    status = fn(np.zeros((360, 640, 3), dtype=np.uint8))
    assert status is PS.PRESENT
    assert b.last_box == (300.0, 100.0, 80.0, 100.0)
    assert abs(b.last_score - 1.0) < 1e-6
    assert b.last_raw_present is True


def test_classify_fn_stashes_none_when_no_face():
    b = _bare_backend()
    b._largest_face = lambda frame: None
    fn = b.make_classify_fn(reference=np.array([1.0, 0.0]))
    status = fn(np.zeros((360, 640, 3), dtype=np.uint8))
    assert status is PS.ABSENT
    assert b.last_box is None and b.last_score is None
    assert b.last_raw_present is False


class _SinkBackend:
    """Worker-level fake following test_vision_worker.py's FakeBackend shape."""
    def __init__(self, classifies):
        self._classifies = list(classifies)
        self.last_box = (1.0, 2.0, 3.0, 4.0)
        self.last_score = 0.77
        self.last_raw_present = True

    def open(self):
        return True

    def grab(self):
        return "frame"

    def embed(self, frame):
        return np.array([1.0, 0.0])

    def make_classify_fn(self, reference):
        seq = self._classifies
        return lambda frame: seq.pop(0)

    def close(self):
        pass


def _worker(backend, sink):
    return VisionWorker(backend, bus=None, fps=10.0, present_after_s=1.0,
                        absent_after_s=2.0, enroll_frames=1, preview_sink=sink)


def test_sink_receives_frame_and_detail_after_monitor_step():
    calls = []
    w = _worker(_SinkBackend([PS.PRESENT, PS.PRESENT]),
                lambda frame, detail: calls.append((frame, detail)))
    w._run_once(0.0)                      # enroll — no sink call
    assert calls == []
    w._run_once(0.1)                      # monitor step — sink fed
    assert len(calls) == 1
    frame, detail = calls[0]
    assert frame == "frame"
    assert detail["box"] == (1.0, 2.0, 3.0, 4.0)
    assert detail["score"] == 0.77
    assert detail["raw_present"] is True
    assert detail["stable"] in ("PRESENT", "ABSENT")


def test_sink_error_never_breaks_monitoring():
    def bad_sink(frame, detail):
        raise RuntimeError("boom")
    w = _worker(_SinkBackend([PS.PRESENT, PS.PRESENT, PS.PRESENT]), bad_sink)
    w._run_once(0.0)
    w._run_once(0.1)
    ev = w._run_once(1.2)                 # debounced PRESENT still emitted
    assert ev is not None and ev.status is PS.PRESENT


def test_detect_box_public_helper():
    b = _bare_backend()
    b._largest_face = lambda frame: np.array([10.0, 20.0, 30.0, 40.0])
    assert b.detect_box("frame") == (10.0, 20.0, 30.0, 40.0)
    b._largest_face = lambda frame: None
    assert b.detect_box("frame") is None
