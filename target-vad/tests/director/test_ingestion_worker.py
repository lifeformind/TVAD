# tests/director/test_ingestion_worker.py
import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.director.workers.ingestion import IngestionWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director.state import State
from core.vad.silero_vad import SpeechSegment
from modes.director import events as E


class FakeMic:
    """Delivers a fixed batch of chunks once via read_available(), then empties
    (mirrors MicrophoneStream.read_available: a non-blocking buffer drain)."""
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._delivered = False

    def read_available(self):
        if self._delivered:
            return []
        self._delivered = True
        return list(self._chunks)


async def _run_briefly(w):
    """Run the (now infinite, non-blocking) ingestion loop just long enough to
    drain the FakeMic's one batch + process it, then stop it cleanly."""
    task = asyncio.create_task(w.run())
    await asyncio.sleep(0.05)
    w.stop()
    await task


class FakeVad:
    """Returns queued segment-lists per chunk; is_speaking is settable."""
    def __init__(self, per_chunk_segments, is_speaking=False):
        self._per_chunk = list(per_chunk_segments)
        self.is_speaking = is_speaking

    def process_chunk(self, chunk):
        return self._per_chunk.pop(0) if self._per_chunk else []


class FakeTurn:
    def __init__(self, prob):
        self._prob = prob

    def endpoint_prob(self, audio, sample_rate):
        return self._prob


def _seg(duration_ms=900.0, level=0.5):
    n = int(duration_ms / 1000 * 16000)
    return SpeechSegment(audio=np.full(n, level, dtype=np.float32),
                         start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms)


def make_worker(mic, vad, state, turn_prob=0.9, embedder_score=0.9):
    bus = EventBus()
    stt = MagicMock()
    stt.set_pending_user_audio = MagicMock()
    stt.set_pending_interjection_audio = MagicMock()
    embedder = MagicMock()
    embedder.extract = MagicMock(return_value=np.ones(192, dtype=np.float32))
    playback = MagicMock()
    playback.get_reference_frame = MagicMock(return_value=None)
    w = IngestionWorker(
        mic=mic, vad=vad, aec=None,
        turn_detector=FakeTurn(turn_prob), embedder=embedder,
        primary_embedding=np.ones(192, dtype=np.float32),
        stt_worker=stt, playback=playback, bus=bus,
        cfg=DirectorConfig(), proximity_rms=0.02,
        state_getter=lambda: state,
        score_fn=lambda a, b: embedder_score,    # injected cosine (no real ECAPA)
    )
    return w, bus, stt


@pytest.mark.asyncio
async def test_listening_segment_emits_segment_endpointed_and_stages_audio():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.LISTENING, turn_prob=0.8)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1
    assert seps[0].is_target is True and seps[0].endpoint_prob == 0.8
    assert seps[0].duration_ms == 900.0 and seps[0].rms > 0.0
    stt.set_pending_user_audio.assert_called_once()


@pytest.mark.asyncio
async def test_evaluating_segment_emits_interjection_with_speaker_score():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.EVALUATING, embedder_score=0.77)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    inter = [e for e in evs if isinstance(e, E.InterjectionSegment)]
    assert len(inter) == 1 and inter[0].speaker_score == 0.77
    stt.set_pending_interjection_audio.assert_called_once()


@pytest.mark.asyncio
async def test_speaking_onset_emits_near_field_onset_once():
    seg_audio = np.full(512, 0.5, dtype=np.float32)
    vad = FakeVad([[]], is_speaking=True)        # speaking, no endpointed segment yet
    w, bus, stt = make_worker(FakeMic([seg_audio]), vad, State.SPEAKING)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].is_target is True and onsets[0].rms > 0.0


@pytest.mark.asyncio
async def test_far_onset_below_proximity_does_not_emit():
    quiet = np.full(512, 0.001, dtype=np.float32)
    vad = FakeVad([[]], is_speaking=True)
    w, bus, stt = make_worker(FakeMic([quiet]), vad, State.SPEAKING)
    await _run_briefly(w)
    assert bus.qsize() == 0
