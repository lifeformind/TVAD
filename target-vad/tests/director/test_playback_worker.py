# tests/director/test_playback_worker.py
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.director.workers.playback import PlaybackWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.talkback.player import Player
from modes.director import commands as C


def make_worker(gain=1.0):
    w = PlaybackWorker(
        tts=MagicMock(),
        player=Player(sample_rate=16000, ring_buffer_seconds=2.0),
        cfg=DirectorConfig(),
        bus=EventBus(),
    )
    w._out_stream = MagicMock()       # fake OutputStream (no real device)
    w._running = True
    w._gain = gain
    w._play_gen = 0
    return w


class TestPlayAudioInvariants:
    def test_writes_frames_to_output_stream(self):
        w = make_worker()
        w._play_audio(np.ones(960, dtype=np.float32), 0)
        assert w._out_stream.write.call_count == 2      # 960 / 480 (invariant: framed)

    def test_applies_gain(self):
        w = make_worker(gain=0.15)
        w._play_audio(np.ones(480, dtype=np.float32), 0)
        written = w._out_stream.write.call_args[0][0]
        np.testing.assert_array_almost_equal(written, np.full(480, 0.15, np.float32))

    def test_records_post_gain_frame_as_aec_reference(self):
        # Invariant 3: record + write co-located under one lock; ducked gain recorded.
        w = make_worker(gain=0.15)
        w._play_audio(np.ones(480, dtype=np.float32), 0)
        ref = w._player.get_reference_frame(160)
        np.testing.assert_array_almost_equal(ref, np.full(160, 0.15, np.float32))

    def test_superseded_generation_writes_nothing(self):
        # Invariant 2: _play_gen mismatch stops a stale playback immediately.
        w = make_worker()
        w._play_gen = 5
        w._play_audio(np.ones(960, dtype=np.float32), 0)   # gen 0 stale
        w._out_stream.write.assert_not_called()

    def test_no_output_stream_is_noop(self):
        w = make_worker()
        w._out_stream = None
        w._play_audio(np.ones(960, dtype=np.float32), 0)   # must not raise

    def test_write_and_close_share_one_lock(self):
        w = make_worker()
        assert isinstance(w._write_lock, type(threading.Lock()))


class TestTeardown:
    def test_close_stops_clears_and_halts_playback(self):
        # Invariant 1 + 5: lock-guarded synchronous close; late frame writes nothing.
        w = make_worker()
        stream = w._out_stream
        w.close()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert w._out_stream is None and w._running is False
        w._play_audio(np.ones(480, dtype=np.float32), w._play_gen)
        stream.write.assert_not_called()

    def test_close_is_idempotent(self):
        w = make_worker()
        w.close()
        w.close()     # must not raise

    @pytest.mark.asyncio
    async def test_drain_awaits_future_before_close(self):
        # Invariant 4 + 6: drain bumps gen, awaits the in-flight write, does NOT
        # clear _play_future; close after drain segfault-free (no concurrent write).
        w = make_worker()
        order = []

        def slow_write(frame):
            order.append("write")

        w._out_stream.write.side_effect = slow_write
        await w.play(np.ones(1440, dtype=np.float32), gen_id=0)   # 3 frames
        assert order.count("write") == 3
        future_before = w._play_future
        await w.drain()
        assert w._play_future is future_before          # NOT cleared on barge-in
        w.close()
        order.append("close")
        assert order[-1] == "close"                     # close strictly after drain

    @pytest.mark.asyncio
    async def test_drain_survives_cancelled_play_future(self):
        # Live crash (2026-07-06 Check 2): a session end during active TTS
        # cancels the generation task, which is suspended awaiting
        # _play_future — task cancellation cancels that future's asyncio
        # wrapper (the executor write thread keeps running). drain() then
        # shields an already-cancelled future, which raises CancelledError;
        # as a BaseException (Python 3.8+) it escaped `except Exception`
        # and killed the kiosk in teardown. drain() must swallow it —
        # stream safety still holds via close()'s gen-bump + write lock.
        w = make_worker()
        import asyncio
        fut = asyncio.get_running_loop().create_future()
        fut.cancel()
        w._play_future = fut
        await w.drain()                                 # must not raise
        w.close()                                       # close still safe after


class TestCommands:
    @pytest.mark.asyncio
    async def test_duck_sets_gain_and_restore_resets(self):
        w = make_worker()
        await w.execute(C.Duck(level=0.15))
        assert w.gain == 0.15
        await w.execute(C.Restore())
        assert w.gain == 1.0

    @pytest.mark.asyncio
    async def test_speak_nudge_synthesizes_are_you_still_there(self):
        from unittest.mock import AsyncMock
        w = make_worker()
        w._tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
        await w.execute(C.SpeakNudge())
        w._tts.synthesize.assert_awaited_once()
        assert "still there" in w._tts.synthesize.await_args[0][0].lower()
        # nudge audio went to the stream
        assert w._out_stream.write.called
