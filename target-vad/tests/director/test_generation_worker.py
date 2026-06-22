# tests/director/test_generation_worker.py
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.director.workers.generation import GenerationWorker
from modes.director.workers.playback import PlaybackWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.talkback.player import Player
from modes.talkback.chunker import SentenceChunker
from modes.director import events as E
from modes.director import commands as C


class FakeLlm:
    def __init__(self, tokens):
        self._tokens = tokens
        self.cancelled = False

    async def stream(self, messages):
        for t in self._tokens:
            await asyncio.sleep(0)
            yield t

    def cancel(self):
        self.cancelled = True


def make_playback():
    pw = PlaybackWorker(tts=MagicMock(), player=Player(16000, 2.0),
                        cfg=DirectorConfig(), bus=EventBus())
    pw._out_stream = MagicMock()
    pw._running = True
    return pw


def make_worker(tokens):
    bus = EventBus()
    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
    pw = make_playback()
    w = GenerationWorker(
        llm=FakeLlm(tokens), tts=tts,
        chunker_factory=lambda: SentenceChunker(max_chunk_chars=120),
        playback=pw, bus=bus,
    )
    return w, bus, tts, pw


@pytest.mark.asyncio
async def test_start_generation_emits_first_frame_then_reply_complete():
    w, bus, tts, pw = make_worker(["Hello there. ", "How are you?"])
    await w.execute(C.StartGeneration(gen_id=1,
                                      messages=[{"role": "user", "content": "hi"}],
                                      steer=None))
    events = []
    while bus.qsize():
        events.append(await bus.get())
    first = [e for e in events if isinstance(e, E.FirstTtsFrame)]
    done = [e for e in events if isinstance(e, E.ReplyComplete)]
    assert len(first) == 1 and first[0].gen_id == 1
    assert len(done) == 1 and done[0].gen_id == 1
    assert done[0].assistant_text == "Hello there. How are you?"


@pytest.mark.asyncio
async def test_steer_is_appended_as_system_note_for_this_generation_only():
    w, bus, tts, pw = make_worker(["ok."])
    captured = {}
    orig = w._llm.stream

    async def spy(messages):
        captured["messages"] = list(messages)
        async for t in orig(messages):
            yield t

    w._llm.stream = spy
    await w.execute(C.StartGeneration(gen_id=1, messages=[{"role": "user", "content": "q"}],
                                      steer="continue the earlier topic"))
    assert captured["messages"][-1] == {"role": "system",
                                        "content": "continue the earlier topic"}


@pytest.mark.asyncio
async def test_cut_drains_playback_cancels_llm_and_bumps_gen():
    w, bus, tts, pw = make_worker(["a long answer that streams. "])
    pw.set_gen(1)
    await w.execute(C.Cut(gen_id=1))
    assert w._llm.cancelled is True
    assert pw._play_gen == 2          # drain() bumped the play gen


@pytest.mark.asyncio
async def test_markdown_is_stripped_before_tts_and_in_partials():
    # gemma emits markdown emphasis; it must never reach the TTS (it would be
    # vocalized) nor the spoken-so-far partials (they feed barge-in history).
    w, bus, tts, pw = make_worker(["Find the *area* ", "under **that** line. "])
    await w.execute(C.StartGeneration(gen_id=1, messages=[], steer=None))
    spoken = [call.args[0] for call in tts.synthesize.call_args_list]
    assert spoken, "expected at least one synthesized chunk"
    assert all("*" not in s for s in spoken), spoken
    assert any("area" in s and "that" in s for s in spoken)
    events = []
    while bus.qsize():
        events.append(await bus.get())
    partials = [e for e in events if isinstance(e, E.AssistantPartial)]
    assert partials and all("*" not in p.text for p in partials)


@pytest.mark.asyncio
async def test_chunk_that_is_only_markup_is_not_spoken():
    # a chunk that strips down to empty (e.g. a lone bullet) must not be synthesized
    w, bus, tts, pw = make_worker(["* ", "real content here. "])
    await w.execute(C.StartGeneration(gen_id=1, messages=[], steer=None))
    spoken = [call.args[0] for call in tts.synthesize.call_args_list]
    assert all(s.strip() for s in spoken), spoken  # no empty/whitespace-only synth


@pytest.mark.asyncio
async def test_no_first_frame_when_llm_yields_nothing():
    w, bus, tts, pw = make_worker([])
    await w.execute(C.StartGeneration(gen_id=3, messages=[], steer=None))
    events = []
    while bus.qsize():
        events.append(await bus.get())
    assert not [e for e in events if isinstance(e, E.FirstTtsFrame)]
    done = [e for e in events if isinstance(e, E.ReplyComplete)]
    assert len(done) == 1 and done[0].gen_id == 3 and done[0].assistant_text == ""
