"""DoaTracker (Director-11): continuous DOA sampling with a hard fail-open
contract — any USB error latches the tracker unavailable and every read
returns None (the cone gate abstains; the kiosk degrades to D10 behavior)."""

import time

import pytest

from core.audio.doa_tracker import DoaTracker


class _Reader:
    """Scripted read_param: returns queued (DOAANGLE, SPEECHDETECTED) pairs;
    a pair of Exception instances raises instead."""
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __call__(self, dev, name):
        if name == "DOAANGLE":
            self._current = self._pairs.pop(0)
            val = self._current[0]
        else:
            val = self._current[1]
        if isinstance(val, Exception):
            raise val
        return val


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _tracker(pairs, clock=None):
    t = DoaTracker(reader=_Reader(pairs), finder=lambda: object(),
                   clock=clock or _Clock())
    return t


def test_start_probe_failure_latches_unavailable():
    t = DoaTracker(reader=_Reader([(RuntimeError("errno 13"), 0)]),
                   finder=lambda: object())
    t.start()
    assert t.available is False
    assert t.latest() is None
    assert t.median_between(0.0, 99.0) is None
    t.stop()   # must be safe even though no thread ever started


def test_missing_device_is_unavailable():
    t = DoaTracker(reader=_Reader([]), finder=lambda: None)
    t.start()
    assert t.available is False
    t.stop()


def test_sample_once_appends_and_latest_reads():
    clock = _Clock(5.0)
    t = _tracker([(97, 1), (140, 0)], clock)
    t._dev, t._available = object(), True      # bypass start(): no thread, no probe read
    t.sample_once()
    assert t.latest() == (5.0, 97.0, 1)
    clock.t = 6.0
    t.sample_once()
    assert t.latest() == (6.0, 140.0, 0)


def test_median_between_filters_time_and_speech_flag():
    clock = _Clock()
    t = _tracker([(90, 1), (200, 0), (100, 1), (95, 1)], clock)
    t._dev, t._available = object(), True
    for ts in (1.0, 2.0, 3.0, 4.0):
        clock.t = ts
        t.sample_once()
    # window [2.0, 4.0]: samples 200(speech=0, dropped), 100, 95 -> median 100 or 95
    assert t.median_between(2.0, 4.0) in (95.0, 100.0)
    # window [0.5, 1.5]: only the 90/speech=1 sample
    assert t.median_between(0.5, 1.5) == 90.0
    # window with no qualifying samples
    assert t.median_between(10.0, 20.0) is None
    # raw-sample access (Director-11 fraction vote): same filters, tuple out
    assert t.angles_between(2.0, 4.0) == (100.0, 95.0)
    assert t.angles_between(10.0, 20.0) == ()


def test_read_error_latches_unavailable_forever():
    clock = _Clock()
    t = _tracker([(90, 1), (RuntimeError("unplugged"), 0)], clock)
    t._dev, t._available = object(), True
    clock.t = 1.0
    t.sample_once()
    assert t.available is True and t.latest() is not None
    clock.t = 2.0
    t.sample_once()                            # raises inside -> latch
    assert t.available is False
    assert t.latest() is None                  # even though a sample exists
    assert t.median_between(0.0, 9.0) is None
    assert t.angles_between(0.0, 9.0) is None


def test_thread_lifecycle_samples_and_stops():
    pairs = [(97, 1)] * 50                     # probe + plenty of polls
    t = DoaTracker(poll_s=0.02, reader=_Reader(pairs), finder=lambda: object())
    t.start()
    assert t.available is True
    time.sleep(0.1)
    t.stop()
    assert t.latest() is not None              # the thread actually sampled
    n = len(t._samples)
    time.sleep(0.05)
    assert len(t._samples) == n                # and actually stopped
