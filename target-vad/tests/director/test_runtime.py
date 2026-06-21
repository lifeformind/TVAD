# tests/director/test_runtime.py
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.director.runtime import DirectorRuntime
from modes.director.director import Director
from modes.director.config import DirectorConfig
from modes.director.bus import EventBus
from modes.director.watchdog import AsyncWatchdog
from modes.director.workers.stt_worker import SttWorker
from modes.director.workers.generation import GenerationWorker
from modes.director.workers.playback import PlaybackWorker
from modes.director.state import State
from modes.talkback.conversation import ConversationManager
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


class FakeStt:
    async def transcribe_segment(self, audio):
        return "tell me a story"


def build_runtime(clock):
    bus = EventBus()
    conv = ConversationManager(system_prompt="s")
    director = Director(DirectorConfig(), conv, now=clock(), proximity_rms=0.02)
    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
    playback = PlaybackWorker(tts=tts, player=Player(16000, 2.0),
                              cfg=DirectorConfig(), bus=bus)
    playback._out_stream = MagicMock()
    playback._running = True
    stt_worker = SttWorker(FakeStt(), bus)
    generation = GenerationWorker(
        llm=FakeLlm(["Once upon a time. ", "The end."]), tts=tts,
        chunker_factory=lambda: SentenceChunker(max_chunk_chars=120),
        playback=playback, bus=bus)
    ingestion = MagicMock()           # driven manually in the test, not run
    ingestion.run = AsyncMock()
    ingestion.stop = MagicMock()
    watchdog = AsyncWatchdog(tick_s=1.0, clock=clock, bus=bus,
                             on_session_end=lambda r: None)
    rt = DirectorRuntime(director=director, bus=bus, watchdog=watchdog,
                         ingestion=ingestion, stt_worker=stt_worker,
                         generation=generation, playback=playback, clock=clock)
    return rt, bus, director, playback, generation


@pytest.mark.asyncio
async def test_full_synthetic_turn_round_trip_and_clean_teardown():
    t = [0.0]
    rt, bus, director, playback, generation = build_runtime(lambda: t[0])

    async def drive():
        # 1. user turn transcribed -> StartGeneration (THINKING)
        await bus.emit(E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
        # 2. let the runtime process: it will route StartGeneration, which emits
        #    FirstTtsFrame (-> SPEAKING) and ReplyComplete (-> LISTENING).
        await asyncio.sleep(0.05)
        assert director.state in (State.SPEAKING, State.LISTENING)
        await asyncio.sleep(0.05)
        assert director.state is State.LISTENING
        # 3. assistant turn recorded by the reducer on ReplyComplete
        msgs = director.ctx.conversation.get_messages()
        assert {"role": "assistant", "content": "Once upon a time. The end."} in msgs
        # 4. advance the clock past silence_timeout while LISTENING -> EndSession.
        t[0] = 100.0
        await bus.emit(E.Tick(now=100.0))

    driver = asyncio.create_task(drive())
    result = await asyncio.wait_for(rt.run_async(), timeout=5.0)
    await driver
    assert result.reason == "silence_timeout"
    assert result.total_duration_s >= 0.0
    # clean teardown: stream closed, llm cancelled-or-done, no orphan task.
    assert playback._out_stream is None
    assert generation._task is None or generation._task.done()


@pytest.mark.asyncio
async def test_duck_command_routes_to_playback():
    t = [0.0]
    rt, bus, director, playback, generation = build_runtime(lambda: t[0])
    await rt._route(C.Duck(level=0.15))
    assert playback.gain == 0.15
    await rt._route(C.Restore())
    assert playback.gain == 1.0


@pytest.mark.asyncio
async def test_end_session_sets_result_and_stops():
    t = [0.0]
    rt, bus, director, playback, generation = build_runtime(lambda: t[0])

    async def drive():
        await asyncio.sleep(0.01)
        await bus.emit(E.Tick(now=400.0))     # past hard_timeout (300s)

    asyncio.create_task(drive())
    result = await asyncio.wait_for(rt.run_async(), timeout=5.0)
    assert result.reason == "hard_timeout"
