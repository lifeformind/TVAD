"""Tests for PvadWorker — VADStream wrapper with the RMS crash-fallback."""

import numpy as np

from modes.director.pvad.worker import PvadWorker
from modes.director.pvad.types import SpeakerFrame


class _OkStream:
    def update_speaker(self, emb):
        self.emb = emb

    def push(self, chunk, ts):
        return [SpeakerFrame(ts=ts, is_target=True, confidence=0.9, rms=0.1)]


class _BoomStream:
    def update_speaker(self, emb):
        pass

    def push(self, chunk, ts):
        raise RuntimeError("pvad died")


def test_normal_path_passes_frames_through():
    events = []
    w = PvadWorker(_OkStream(), proximity_rms=0.02, emit=lambda e, p: events.append((e, p)))
    w.update_speaker(np.ones(192, dtype=np.float32))
    out = w.process(np.ones(3200, dtype=np.float32), ts=1.0)
    assert out == [SpeakerFrame(ts=1.0, is_target=True, confidence=0.9, rms=0.1)]
    assert events == []


def test_crash_falls_back_to_rms_proximity_gate():
    events = []
    w = PvadWorker(_BoomStream(), proximity_rms=0.02, emit=lambda e, p: events.append((e, p)))
    w.update_speaker(np.ones(192, dtype=np.float32))
    loud = np.ones(3200, dtype=np.float32) * 0.5     # rms >= proximity_rms
    out = w.process(loud, ts=2.0)
    assert len(out) == 1 and out[0].is_target is True
    assert any(e == "worker_failed" for e, _ in events)


def test_crash_fallback_quiet_chunk_is_not_target():
    w = PvadWorker(_BoomStream(), proximity_rms=0.2, emit=lambda e, p: None)
    w.update_speaker(np.ones(192, dtype=np.float32))
    quiet = np.ones(3200, dtype=np.float32) * 0.01   # rms < proximity_rms
    out = w.process(quiet, ts=3.0)
    assert out[0].is_target is False


def test_worker_failed_emitted_only_once():
    events = []
    w = PvadWorker(_BoomStream(), proximity_rms=0.02, emit=lambda e, p: events.append(e))
    w.update_speaker(np.ones(192, dtype=np.float32))
    w.process(np.ones(3200, dtype=np.float32), ts=1.0)
    w.process(np.ones(3200, dtype=np.float32), ts=2.0)
    assert events.count("worker_failed") == 1
