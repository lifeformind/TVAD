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
from modes.director.pvad.types import SpeakerFrame


class _FakePvad:
    """Stand-in PvadWorker: every chunk yields one frame with a fixed verdict."""
    def __init__(self, is_target, conf=0.8):
        self._is_target = is_target
        self._conf = conf

    def update_speaker(self, emb):
        pass

    def process(self, chunk, ts):
        return [SpeakerFrame(ts=ts, is_target=self._is_target,
                             confidence=self._conf, rms=0.1)]


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


def make_worker(mic, vad, state, turn_prob=0.9, embedder_score=0.9, pvad=None,
                safety=None, doa_tracker=None):
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
        pvad=pvad,
        safety_worker=safety,
        doa_tracker=doa_tracker,
    )
    return w, bus, stt


@pytest.mark.asyncio
async def test_pvad_target_makes_segment_is_target_true():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.LISTENING, pvad=_FakePvad(is_target=True))
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1 and seps[0].is_target is True


@pytest.mark.asyncio
async def test_pvad_bystander_makes_segment_is_target_false():
    # A non-target speaker's turn must be marked is_target=False so the reducer
    # ignores it (crowd focus on normal turns).
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.LISTENING, pvad=_FakePvad(is_target=False))
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1 and seps[0].is_target is False


@pytest.mark.asyncio
async def test_pvad_bystander_onset_is_not_target():
    # A bystander onset during SPEAKING must NOT be flagged target (no duck).
    seg_audio = np.full(512, 0.5, dtype=np.float32)
    vad = FakeVad([[]], is_speaking=True)
    w, bus, stt = make_worker(FakeMic([seg_audio]), vad, State.SPEAKING,
                              pvad=_FakePvad(is_target=False))
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].is_target is False


@pytest.mark.asyncio
async def test_pvad_interjection_speaker_score_comes_from_pvad():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.EVALUATING, pvad=_FakePvad(is_target=True, conf=0.83))
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    inter = [e for e in evs if isinstance(e, E.InterjectionSegment)]
    assert len(inter) == 1 and inter[0].speaker_score == pytest.approx(0.83)


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


class _FlippingVad(FakeVad):
    """FakeVad whose is_speaking flips per chunk, popped from a queue inside
    process_chunk (mirrors the real VAD: is_speaking tracks the run and can go
    False between two runs of the same SPEAKING state)."""
    def __init__(self, is_speaking_sequence):
        super().__init__([[] for _ in is_speaking_sequence])
        self._is_speaking_seq = list(is_speaking_sequence)

    def process_chunk(self, chunk):
        segs = super().process_chunk(chunk)
        self.is_speaking = self._is_speaking_seq.pop(0)
        return segs


@pytest.mark.asyncio
async def test_onset_latch_rearms_after_speech_run_ends_same_state():
    # Director-11: a suppressed (out-of-cone) onset leaves state at SPEAKING
    # (no Duck), so the old "reset only on leaving SPEAKING" latch would never
    # re-arm and the owner's next barge-in in the same reply would be silent.
    # Once-per-speech-RUN semantics: chunk 1 speaking -> onset #1; chunk 2 the
    # run ends (is_speaking False, no onset); chunk 3 a new run starts while
    # state is STILL SPEAKING -> onset #2 must fire.
    loud = np.full(512, 0.5, dtype=np.float32)
    vad = _FlippingVad([True, False, True])
    w, bus, stt = make_worker(FakeMic([loud, loud, loud]), vad, State.SPEAKING)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 2


@pytest.mark.asyncio
async def test_apply_aec_uses_real_playback_worker_reference(monkeypatch):
    """Regression: the ingestion AEC path (state==SPEAKING) calls
    playback.get_reference_frame — the REAL PlaybackWorker must expose it
    (delegating to its Player). The other tests pass a MagicMock playback, which
    masked the missing method; on a box where AEC actually loads this crashed the
    ingestion task and starved the mic."""
    from modes.director.workers.playback import PlaybackWorker

    player = MagicMock()
    player.get_reference_frame = MagicMock(return_value=None)
    pw = PlaybackWorker(tts=MagicMock(), player=player, cfg=DirectorConfig(), bus=EventBus())

    assert hasattr(pw, "get_reference_frame")
    assert pw.get_reference_frame(160) is None
    player.get_reference_frame.assert_called_once_with(160)

    fake_aec = MagicMock()
    fake_aec.frame_samples = 160
    fake_aec.process_frame = MagicMock(side_effect=lambda f, r: f)
    w = IngestionWorker(
        mic=FakeMic([]), vad=FakeVad([]), aec=fake_aec,
        turn_detector=FakeTurn(0.9), embedder=MagicMock(),
        primary_embedding=np.ones(192, dtype=np.float32),
        stt_worker=MagicMock(), playback=pw, bus=EventBus(),
        cfg=DirectorConfig(), proximity_rms=0.02,
        state_getter=lambda: State.SPEAKING, score_fn=lambda a, b: 0.9,
    )
    # Must not raise AttributeError; returns processed audio of the same length.
    out = w._apply_aec(np.zeros(480, dtype=np.float32), State.SPEAKING)
    assert out is not None and len(out) == 480


@pytest.mark.asyncio
async def test_listening_segment_staged_into_safety_worker():
    seg = _seg()
    safety = MagicMock()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.LISTENING, safety=safety)
    await _run_briefly(w)
    safety.set_pending_audio.assert_called_once()
    assert np.array_equal(safety.set_pending_audio.call_args[0][0], seg.audio)


@pytest.mark.asyncio
async def test_evaluating_segment_staged_into_safety_worker():
    seg = _seg()
    safety = MagicMock()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.EVALUATING, safety=safety)
    await _run_briefly(w)
    safety.set_pending_audio.assert_called_once()


@pytest.mark.asyncio
async def test_listening_segments_stage_with_matching_monotonic_seq():
    # Each segment gets a fresh seq (from 1 — the assembly seed owns 0), and
    # the SAME seq goes to the event and both staging slots, so the reducer's
    # echoed commands can match staged audio to their segment.
    seg1, seg2 = _seg(), _seg()
    safety = MagicMock()
    w, bus, stt = make_worker(FakeMic([seg1.audio, seg2.audio]),
                              FakeVad([[seg1], [seg2]]),
                              State.LISTENING, safety=safety)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert [e.seq for e in seps] == [1, 2]
    assert [c.args[1] for c in safety.set_pending_audio.call_args_list] == [1, 2]
    assert [c.args[1] for c in stt.set_pending_user_audio.call_args_list] == [1, 2]


@pytest.mark.asyncio
async def test_interjection_seq_matches_staging():
    seg = _seg()
    safety = MagicMock()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.EVALUATING, safety=safety)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    inter = [e for e in evs if isinstance(e, E.InterjectionSegment)]
    assert len(inter) == 1 and inter[0].seq == 1
    assert safety.set_pending_audio.call_args[0][1] == 1
    assert stt.set_pending_interjection_audio.call_args[0][1] == 1


class _FakeDoa:
    """Records the median_between window; returns a scripted angle."""
    def __init__(self, median=97.0, latest=None):
        self._median = median
        self._latest = latest          # (t, angle, speech) or None
        self.windows = []

    def latest(self):
        return self._latest

    def median_between(self, t0, t1):
        self.windows.append((t0, t1))
        return self._median


@pytest.mark.asyncio
async def test_segment_carries_doa_median_over_its_own_span():
    seg = _seg(duration_ms=900.0)
    doa = _FakeDoa(median=97.0)
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.LISTENING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1 and seps[0].doa_angle == 97.0
    (t0, t1), = doa.windows
    assert (t1 - t0) == pytest.approx(0.9, abs=0.05)   # the segment's own span


@pytest.mark.asyncio
async def test_interjection_carries_doa_median():
    seg = _seg(duration_ms=900.0)
    doa = _FakeDoa(median=193.0)
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.EVALUATING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    inters = [e for e in evs if isinstance(e, E.InterjectionSegment)]
    assert len(inters) == 1 and inters[0].doa_angle == 193.0


@pytest.mark.asyncio
async def test_onset_carries_latest_speech_flagged_angle():
    chunk = np.full(512, 0.5, dtype=np.float32)
    doa = _FakeDoa(latest=(1.0, 140.0, 1))
    w, bus, stt = make_worker(FakeMic([chunk]), FakeVad([[]], is_speaking=True),
                              State.SPEAKING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].doa_angle == 140.0


@pytest.mark.asyncio
async def test_onset_nonspeech_latest_sample_stamps_none():
    chunk = np.full(512, 0.5, dtype=np.float32)
    doa = _FakeDoa(latest=(1.0, 140.0, 0))              # speech_flag = 0
    w, bus, stt = make_worker(FakeMic([chunk]), FakeVad([[]], is_speaking=True),
                              State.SPEAKING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].doa_angle is None


@pytest.mark.asyncio
async def test_no_tracker_stamps_none_everywhere():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.LISTENING, doa_tracker=None)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1 and seps[0].doa_angle is None
