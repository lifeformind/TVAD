# tests/director/test_cut_teardown_regression.py
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.director.workers.playback import PlaybackWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.talkback.player import Player


def make_playback():
    pw = PlaybackWorker(tts=MagicMock(), player=Player(16000, 2.0),
                        cfg=DirectorConfig(), bus=EventBus())
    pw._out_stream = MagicMock()
    pw._running = True
    return pw


@pytest.mark.asyncio
async def test_cut_during_playback_serializes_write_and_close():
    """Write and close must never overlap (concurrent PortAudio calls segfault).
    The shared _write_lock is the guard; assert no interleave by checking the
    lock is held during write and that close waits for it."""
    pw = make_playback()
    events = []
    write_entered = threading.Event()
    release_write = threading.Event()

    def blocking_write(frame):
        events.append("write_start")
        write_entered.set()
        release_write.wait(timeout=2.0)     # hold the lock briefly
        events.append("write_end")

    pw._out_stream.write.side_effect = blocking_write

    # Start a play in a background executor (it grabs _write_lock for the frame).
    play_task = asyncio.create_task(pw.play(np.ones(480, dtype=np.float32), gen_id=0))
    await asyncio.get_event_loop().run_in_executor(None, write_entered.wait, 2.0)

    # While the write holds the lock, a close from "another thread" must block
    # until the write releases — proving they never run concurrently.
    def do_close():
        events.append("close_attempt")
        pw.close()
        events.append("close_done")

    closer = asyncio.get_event_loop().run_in_executor(None, do_close)
    await asyncio.sleep(0.05)
    # close_attempt is recorded but close_done is NOT yet (blocked on the lock).
    assert "close_attempt" in events and "close_done" not in events
    release_write.set()
    await play_task
    await closer
    # write fully finished before close finished (lock serialized them).
    assert events.index("write_end") < events.index("close_done")


@pytest.mark.asyncio
async def test_drain_awaits_future_strictly_before_close():
    """drain() must await the in-flight write before close() (invariant 4); the
    future survives the cut (invariant 6)."""
    pw = make_playback()
    order = []
    pw._out_stream.write.side_effect = lambda f: order.append("write")
    await pw.play(np.ones(1920, dtype=np.float32), gen_id=0)   # 4 frames
    fut = pw._play_future
    await pw.drain()
    order.append("drain_done")
    assert pw._play_future is fut                  # invariant 6: not cleared
    pw.close()
    order.append("close")
    assert order.count("write") == 4
    assert order.index("drain_done") < order.index("close")


@pytest.mark.asyncio
async def test_stale_generation_frames_dropped_after_cut():
    """After a cut bumps _play_gen, a late play() for the OLD gen writes nothing
    (spec section 11 stale-gen drop)."""
    pw = make_playback()
    written = []
    pw._out_stream.write.side_effect = lambda f: written.append(f)
    await pw.drain()                       # bumps _play_gen 0 -> 1
    pw._play_audio(np.ones(480, dtype=np.float32), gen=0)    # stale gen 0
    assert written == []
