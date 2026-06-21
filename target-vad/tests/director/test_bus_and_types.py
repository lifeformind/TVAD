# tests/director/test_bus_and_types.py
import asyncio

import pytest

from modes.director.bus import EventBus
from modes.director.result import DirectorResult
from modes.director.transcript import TranscriptResult, wrap_transcript


@pytest.mark.asyncio
async def test_bus_round_trips_events_in_fifo_order():
    bus = EventBus()
    await bus.emit("a")
    await bus.emit("b")
    assert bus.qsize() == 2
    assert await bus.get() == "a"
    assert await bus.get() == "b"
    assert bus.qsize() == 0


@pytest.mark.asyncio
async def test_bus_get_blocks_until_emit():
    bus = EventBus()

    async def producer():
        await asyncio.sleep(0.01)
        await bus.emit("late")

    asyncio.create_task(producer())
    # get() must wait for the producer rather than raising QueueEmpty.
    assert await asyncio.wait_for(bus.get(), timeout=1.0) == "late"


def test_director_result_is_frozen_and_carries_fields():
    r = DirectorResult(reason="silence_timeout", turns=3, total_duration_s=12.5)
    assert r.reason == "silence_timeout" and r.turns == 3 and r.total_duration_s == 12.5
    with pytest.raises(Exception):
        r.reason = "x"  # frozen


def test_transcript_result_fields_and_shim():
    tr = TranscriptResult(text="hello", mean_word_prob=0.8)
    assert tr.text == "hello" and tr.mean_word_prob == 0.8
    # The shim coerces today's bare-str return into a full TranscriptResult.
    wrapped = wrap_transcript("bare string")
    assert isinstance(wrapped, TranscriptResult)
    assert wrapped.text == "bare string" and wrapped.mean_word_prob == 1.0
    # A real TranscriptResult passes through untouched.
    assert wrap_transcript(tr) is tr
    # None / empty coerces to empty text, full confidence (never crashes).
    assert wrap_transcript(None) == TranscriptResult(text="", mean_word_prob=1.0)
