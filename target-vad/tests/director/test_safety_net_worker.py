from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.director.bus import EventBus
from modes.director.safety_net import SafetyNet
from modes.director.workers.safety_net import SafetyNetWorker
from modes.director import commands as C
from modes.director import events as E


class _Emb:
    def extract(self, audio, sample_rate=16000):
        return np.ones(4, dtype=np.float32)      # cosine vs ones-primary == 1.0


def _worker(verify_window_ms=100):
    bus = EventBus()
    net = SafetyNet(_Emb(), np.ones(4, dtype=np.float32),
                    verify_window_ms=verify_window_ms, threshold=0.30, sr=16000)
    return SafetyNetWorker(net, bus), bus


async def _events(bus):
    return [await bus.get() for _ in range(bus.qsize())]


@pytest.mark.asyncio
async def test_execute_without_staged_audio_is_a_noop():
    worker, bus = _worker()
    await worker.execute(C.AccumulateSpeakerAudio())
    assert await _events(bus) == []


@pytest.mark.asyncio
async def test_pending_is_consumed_once():
    worker, bus = _worker(verify_window_ms=100)   # 1600 samples fills a window
    worker.set_pending_audio(np.full(1600, 0.5, dtype=np.float32))
    await worker.execute(C.AccumulateSpeakerAudio())
    assert len(await _events(bus)) == 1
    await worker.execute(C.AccumulateSpeakerAudio())          # nothing staged now
    assert await _events(bus) == []


@pytest.mark.asyncio
async def test_subwindow_audio_emits_nothing_until_window_fills():
    worker, bus = _worker(verify_window_ms=100)
    worker.set_pending_audio(np.full(800, 0.5, dtype=np.float32))   # half a window
    await worker.execute(C.AccumulateSpeakerAudio())
    assert await _events(bus) == []
    worker.set_pending_audio(np.full(800, 0.5, dtype=np.float32))   # completes it
    await worker.execute(C.AccumulateSpeakerAudio())
    events = await _events(bus)
    assert len(events) == 1 and isinstance(events[0], E.SpeakerWindowVerdict)
    assert events[0].smoother_ok is True and events[0].score > 0.99
    assert events[0].window_rms == pytest.approx(0.5, abs=1e-6)


@pytest.mark.asyncio
async def test_long_audio_drains_multiple_windows_in_order():
    worker, bus = _worker(verify_window_ms=100)
    worker.set_pending_audio(np.full(3300, 0.5, dtype=np.float32))  # 2 windows + rest
    await worker.execute(C.AccumulateSpeakerAudio())
    events = await _events(bus)
    assert len(events) == 2
    assert all(isinstance(e, E.SpeakerWindowVerdict) for e in events)


@pytest.mark.asyncio
async def test_non_accumulate_commands_are_ignored():
    worker, bus = _worker()
    worker.set_pending_audio(np.full(1600, 0.5, dtype=np.float32))
    await worker.execute(C.TranscribeUserTurn())
    assert await _events(bus) == []
