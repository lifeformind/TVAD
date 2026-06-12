"""Tests for TalkbackController._play_audio — gain-controlled streaming playback.

Regression coverage for the AEC no-op bug: playback must record each actually
played (post-gain) frame as the AEC reference, so get_reference_frame() returns
real audio instead of silence.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.controller import TalkbackController
from modes.talkback.player import Player


def make_ctrl(gain=1.0):
    ctrl = TalkbackController(
        stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
        player=Player(sample_rate=16000, ring_buffer_seconds=2.0),
        logger=MagicMock(),
    )
    ctrl._running = True
    ctrl._out_stream = MagicMock()
    ctrl._gain = gain
    ctrl._play_gen = 0
    return ctrl


class TestPlayAudio:
    def test_writes_frames_to_output_stream(self):
        ctrl = make_ctrl()
        ctrl._play_audio(np.ones(960, dtype=np.float32), 0)
        assert ctrl._out_stream.write.call_count == 2   # 960 / 480

    def test_applies_gain_to_written_audio(self):
        ctrl = make_ctrl(gain=0.15)
        ctrl._play_audio(np.ones(480, dtype=np.float32), 0)
        written = ctrl._out_stream.write.call_args[0][0]
        np.testing.assert_array_almost_equal(written, np.full(480, 0.15, np.float32))

    def test_records_played_frame_as_aec_reference(self):
        """The ring buffer must hold what was actually played (post-gain), so
        get_reference_frame returns real audio — not the silence that made AEC
        a no-op."""
        ctrl = make_ctrl(gain=1.0)
        ctrl._play_audio(np.ones(480, dtype=np.float32), 0)
        ref = ctrl._player.get_reference_frame(160)
        assert ref is not None
        assert np.any(ref != 0.0)
        np.testing.assert_array_almost_equal(ref, np.ones(160, np.float32))

    def test_records_ducked_reference_when_gain_reduced(self):
        ctrl = make_ctrl(gain=0.15)
        ctrl._play_audio(np.ones(480, dtype=np.float32), 0)
        ref = ctrl._player.get_reference_frame(160)
        np.testing.assert_array_almost_equal(ref, np.full(160, 0.15, np.float32))

    def test_superseded_generation_writes_nothing(self):
        """A barge-in bumps _play_gen; stale playback (older gen) must stop
        immediately so it never writes concurrently with the new response."""
        ctrl = make_ctrl()
        ctrl._play_gen = 5
        ctrl._play_audio(np.ones(960, dtype=np.float32), 0)   # gen 0 is stale
        ctrl._out_stream.write.assert_not_called()

    def test_no_output_stream_is_noop(self):
        ctrl = make_ctrl()
        ctrl._out_stream = None
        ctrl._play_audio(np.ones(960, dtype=np.float32), 0)  # must not raise


class TestTeardown:
    """The output stream must never be closed while a write is in flight
    (concurrent PortAudio calls across threads segfault)."""

    def test_close_stops_clears_and_halts_playback(self):
        ctrl = make_ctrl()
        stream = ctrl._out_stream
        ctrl._close_out_stream()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert ctrl._out_stream is None
        assert ctrl._running is False
        # A late playback frame after close writes nothing (stream is gone).
        ctrl._play_audio(np.ones(480, dtype=np.float32), ctrl._play_gen)
        stream.write.assert_not_called()

    def test_close_is_idempotent(self):
        ctrl = make_ctrl()
        ctrl._close_out_stream()
        ctrl._close_out_stream()        # must not raise on already-closed stream

    def test_write_and_close_share_one_lock(self):
        """Guards against a future refactor dropping the shared lock that makes
        write and close mutually exclusive."""
        import threading
        ctrl = make_ctrl()
        assert isinstance(ctrl._write_lock, type(threading.Lock()))
