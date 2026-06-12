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
    ctrl._playback_cancelled = False
    return ctrl


class TestPlayAudio:
    def test_writes_frames_to_output_stream(self):
        ctrl = make_ctrl()
        ctrl._play_audio(np.ones(960, dtype=np.float32))
        assert ctrl._out_stream.write.call_count == 2   # 960 / 480

    def test_applies_gain_to_written_audio(self):
        ctrl = make_ctrl(gain=0.15)
        ctrl._play_audio(np.ones(480, dtype=np.float32))
        written = ctrl._out_stream.write.call_args[0][0]
        np.testing.assert_array_almost_equal(written, np.full(480, 0.15, np.float32))

    def test_records_played_frame_as_aec_reference(self):
        """The ring buffer must hold what was actually played (post-gain), so
        get_reference_frame returns real audio — not the silence that made AEC
        a no-op."""
        ctrl = make_ctrl(gain=1.0)
        ctrl._play_audio(np.ones(480, dtype=np.float32))
        ref = ctrl._player.get_reference_frame(160)
        assert ref is not None
        assert np.any(ref != 0.0)
        np.testing.assert_array_almost_equal(ref, np.ones(160, np.float32))

    def test_records_ducked_reference_when_gain_reduced(self):
        ctrl = make_ctrl(gain=0.15)
        ctrl._play_audio(np.ones(480, dtype=np.float32))
        ref = ctrl._player.get_reference_frame(160)
        np.testing.assert_array_almost_equal(ref, np.full(160, 0.15, np.float32))

    def test_cancelled_playback_writes_nothing(self):
        ctrl = make_ctrl()
        ctrl._playback_cancelled = True
        ctrl._play_audio(np.ones(960, dtype=np.float32))
        ctrl._out_stream.write.assert_not_called()

    def test_no_output_stream_is_noop(self):
        ctrl = make_ctrl()
        ctrl._out_stream = None
        ctrl._play_audio(np.ones(960, dtype=np.float32))  # must not raise
