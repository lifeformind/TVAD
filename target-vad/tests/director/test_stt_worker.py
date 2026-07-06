# tests/director/test_stt_worker.py
import numpy as np
import pytest

from modes.director.bus import EventBus
from modes.director.transcript import TranscriptResult
from modes.director.workers.stt_worker import SttWorker
from modes.director import events as E
from modes.director import commands as C


class FakeStt:
    """Mirrors today's bare-str transcribe_segment; Plan 04 returns TranscriptResult."""
    def __init__(self, returns):
        self._returns = returns
        self.calls = []

    async def transcribe_segment(self, audio):
        self.calls.append(audio)
        return self._returns


@pytest.mark.asyncio
async def test_transcribe_user_turn_emits_user_event_via_shim():
    bus = EventBus()
    stt = FakeStt("tell me a story")          # bare str -> shim => prob 1.0
    w = SttWorker(stt, bus)
    audio = np.ones(16000, dtype=np.float32)
    w.set_pending_user_audio(audio)
    await w.execute(C.TranscribeUserTurn())
    ev = await bus.get()
    assert isinstance(ev, E.UserTurnTranscribed)
    assert ev.text == "tell me a story" and ev.mean_word_prob == 1.0
    assert stt.calls and stt.calls[0] is audio


@pytest.mark.asyncio
async def test_transcribe_interjection_passes_real_confidence_through():
    bus = EventBus()
    stt = FakeStt(TranscriptResult(text="wait why", mean_word_prob=0.42))
    w = SttWorker(stt, bus)
    w.set_pending_interjection_audio(np.ones(8000, dtype=np.float32))
    await w.execute(C.TranscribeInterjection())
    ev = await bus.get()
    assert isinstance(ev, E.InterjectionTranscribed)
    assert ev.text == "wait why" and ev.mean_word_prob == 0.42


@pytest.mark.asyncio
async def test_missing_audio_emits_empty_low_noop_transcript():
    """If no audio was staged, emit an empty transcript (reducer RESTOREs/keeps
    listening) rather than crashing the worker."""
    bus = EventBus()
    w = SttWorker(FakeStt("ignored"), bus)
    await w.execute(C.TranscribeUserTurn())          # no set_pending_user_audio
    ev = await bus.get()
    assert isinstance(ev, E.UserTurnTranscribed) and ev.text == ""


@pytest.mark.asyncio
async def test_user_seq_mismatch_emits_empty_and_preserves_staged_audio():
    """Overwrite-last staging race: a stale TranscribeUserTurn must not consume
    a NEWER segment's staged audio. It emits an empty transcript (reducer keeps
    listening) and the staged audio survives for its matching command."""
    bus = EventBus()
    stt = FakeStt("real words")
    w = SttWorker(stt, bus)
    audio = np.ones(16000, dtype=np.float32)
    w.set_pending_user_audio(audio, seq=2)
    await w.execute(C.TranscribeUserTurn(seq=1))          # stale command
    ev = await bus.get()
    assert isinstance(ev, E.UserTurnTranscribed) and ev.text == ""
    assert stt.calls == []                                 # audio not consumed
    await w.execute(C.TranscribeUserTurn(seq=2))          # the matching one
    ev = await bus.get()
    assert ev.text == "real words" and stt.calls[0] is audio


@pytest.mark.asyncio
async def test_interjection_seq_mismatch_emits_empty_and_preserves_staged_audio():
    bus = EventBus()
    stt = FakeStt(TranscriptResult(text="wait why", mean_word_prob=0.42))
    w = SttWorker(stt, bus)
    w.set_pending_interjection_audio(np.ones(8000, dtype=np.float32), seq=5)
    await w.execute(C.TranscribeInterjection(seq=4))      # stale command
    ev = await bus.get()
    assert isinstance(ev, E.InterjectionTranscribed) and ev.text == ""
    await w.execute(C.TranscribeInterjection(seq=5))
    ev = await bus.get()
    assert ev.text == "wait why"


@pytest.mark.asyncio
async def test_unknown_command_is_ignored():
    bus = EventBus()
    w = SttWorker(FakeStt("x"), bus)
    await w.execute(C.Restore())     # not an STT command
    assert bus.qsize() == 0
