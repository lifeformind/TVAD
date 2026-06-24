from modes.director.events import PresenceStatus as PS
from modes.director.vision.classify import PresenceDebouncer
from modes.director.vision.monitor import PresenceMonitor


def _monitor(seq):
    """classify_fn pops from seq; an Exception instance is raised."""
    box = list(seq)

    def classify_fn(frame):
        v = box.pop(0)
        if isinstance(v, Exception):
            raise v
        return v
    return PresenceMonitor(classify_fn, PresenceDebouncer(present_after_s=1.0,
                                                          absent_after_s=2.0))


def test_emits_present_once_on_debounced_change():
    m = _monitor([PS.PRESENT, PS.PRESENT, PS.PRESENT])
    assert m.observe("f", 0.0) is None        # absent->absent (no change from init)
    assert m.observe("f", 0.6) is None        # still debouncing
    assert m.observe("f", 1.2) is PS.PRESENT  # debounced present: emit once
    # no re-emit on subsequent presents (next observe would need another frame)


def test_classify_error_is_unavailable():
    m = _monitor([RuntimeError("yunet boom")])
    assert m.observe("f", 0.0) is PS.UNAVAILABLE


def test_none_frame_is_unavailable():
    m = PresenceMonitor(lambda f: PS.PRESENT, PresenceDebouncer())
    assert m.observe(None, 0.0) is PS.UNAVAILABLE


def test_recovery_from_unavailable_re_debounces():
    m = _monitor([RuntimeError("x"), PS.PRESENT, PS.PRESENT])
    assert m.observe("f", 0.0) is PS.UNAVAILABLE
    assert m.observe("f", 0.1) is None        # present_after not yet met after reset
    assert m.observe("f", 1.2) is PS.PRESENT  # re-accrued from the recovery instant
