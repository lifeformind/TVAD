"""Regression: the GenerationWorker must drop a stale aiohttp session (bound to a
previous, now-closed loop — e.g. the startup ping's asyncio.run loop) on its FIRST
generation, so llm.stream() binds fresh to the runtime's loop. Without this the
live kiosk hung after [SESSION STARTED] with no response."""
import numpy as np
import pytest
from unittest.mock import AsyncMock

from modes.director.bus import EventBus
from modes.director.workers.generation import GenerationWorker
from modes.director import commands as C
from modes.talkback.chunker import SentenceChunker


class _FakeLlm:
    def __init__(self):
        self.close = AsyncMock()

    async def stream(self, messages):
        for tok in ["hi", "."]:
            yield tok

    def cancel(self):
        pass


class _FakeTts:
    async def synthesize(self, text):
        return np.zeros(800, dtype=np.float32)


class _FakePlayback:
    def set_gen(self, gen_id):
        pass

    async def play(self, audio, gen_id):
        pass

    async def drain(self):
        pass


def _worker(llm):
    return GenerationWorker(
        llm=llm, tts=_FakeTts(),
        chunker_factory=lambda: SentenceChunker(
            sentence_terminators=[".", "?", "!"], max_chunk_chars=120),
        playback=_FakePlayback(), bus=EventBus(),
    )


def _gen(n):
    return C.StartGeneration(gen_id=n, messages=[{"role": "user", "content": "hi"}],
                             steer=None)


@pytest.mark.asyncio
async def test_llm_session_rebound_once_per_worker():
    llm = _FakeLlm()
    g = _worker(llm)
    await g.execute(_gen(1))
    await g.execute(_gen(2))
    assert llm.close.await_count == 1   # rebound once, NOT once per generation


@pytest.mark.asyncio
async def test_fresh_worker_rebinds_again():
    # A new GenerationWorker is built per session; each must rebind for its loop.
    llm = _FakeLlm()
    g = _worker(llm)
    await g.execute(_gen(1))
    assert llm.close.await_count == 1


@pytest.mark.asyncio
async def test_aclose_closes_llm_session_at_teardown():
    llm = _FakeLlm()
    g = _worker(llm)
    await g.execute(_gen(1))      # rebind close #1
    await g.aclose()             # teardown close #2
    assert llm.close.await_count == 2
