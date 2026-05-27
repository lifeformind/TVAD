"""Tests for TalkbackController._listen_loop.

Covers: chunk delivery, AEC during SPEAKING, no AEC during LISTENING,
and _last_speech_at timestamp update on segment detection.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.controller import TalkbackController, TalkbackState


def make_controller() -> TalkbackController:
    """Return a minimally wired TalkbackController for unit tests."""
    ctrl = TalkbackController(
        stt=MagicMock(),
        llm=MagicMock(),
        tts=MagicMock(),
        player=MagicMock(),
        logger=MagicMock(),
    )
    ctrl._segment_queue = asyncio.Queue()
    ctrl._running = True
    return ctrl


class TestListenLoop:
    @pytest.mark.asyncio
    async def test_segments_land_in_queue(self):
        """VAD segment on first chunk ends up in _segment_queue."""
        ctrl = make_controller()

        chunk1 = np.zeros(480, dtype=np.float32)
        chunk2 = np.zeros(480, dtype=np.float32)

        fake_segment = MagicMock()

        call_count = 0

        def vad_side_effect(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [fake_segment]
            # Stop the loop after second chunk
            ctrl._running = False
            return []

        mic = MagicMock()
        mic.stream.return_value = iter([chunk1, chunk2])

        vad = MagicMock()
        vad.process_chunk.side_effect = vad_side_effect

        embedder = MagicMock()
        primary_embedding = np.ones(192, dtype=np.float32)

        await ctrl._listen_loop(mic, vad, embedder, primary_embedding)

        assert ctrl._segment_queue.qsize() == 1
        seg = ctrl._segment_queue.get_nowait()
        assert seg is fake_segment

    @pytest.mark.asyncio
    async def test_aec_applied_during_speaking(self):
        """AEC.process_frame called once per 160-sample frame when SPEAKING."""
        ctrl = make_controller()
        ctrl.state = TalkbackState.SPEAKING

        aec_mock = MagicMock()
        aec_mock.frame_samples = 160
        # Return a zero frame so np.concatenate works
        aec_mock.process_frame.return_value = np.zeros(160, dtype=np.float32)
        ctrl._aec = aec_mock

        # Player returns a reference frame for AEC
        ctrl._player.get_reference_frame.return_value = np.zeros(160, dtype=np.float32)

        # 480-sample chunk → 3 frames of 160 samples
        chunk = np.zeros(480, dtype=np.float32)

        def vad_side_effect(chunk):
            ctrl._running = False
            return []

        mic = MagicMock()
        mic.stream.return_value = iter([chunk])

        vad = MagicMock()
        vad.process_chunk.side_effect = vad_side_effect

        embedder = MagicMock()
        primary_embedding = np.ones(192, dtype=np.float32)

        await ctrl._listen_loop(mic, vad, embedder, primary_embedding)

        assert aec_mock.process_frame.call_count == 3

    @pytest.mark.asyncio
    async def test_no_aec_during_listening(self):
        """AEC.process_frame NOT called when state is LISTENING."""
        ctrl = make_controller()
        ctrl.state = TalkbackState.LISTENING

        aec_mock = MagicMock()
        aec_mock.frame_samples = 160
        aec_mock.process_frame.return_value = np.zeros(160, dtype=np.float32)
        ctrl._aec = aec_mock

        chunk = np.zeros(480, dtype=np.float32)

        def vad_side_effect(chunk):
            ctrl._running = False
            return []

        mic = MagicMock()
        mic.stream.return_value = iter([chunk])

        vad = MagicMock()
        vad.process_chunk.side_effect = vad_side_effect

        embedder = MagicMock()
        primary_embedding = np.ones(192, dtype=np.float32)

        await ctrl._listen_loop(mic, vad, embedder, primary_embedding)

        aec_mock.process_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_last_speech_at_updated_on_segment(self):
        """_last_speech_at is updated when VAD returns a segment."""
        ctrl = make_controller()
        ctrl._last_speech_at = 0.0

        fake_segment = MagicMock()

        def vad_side_effect(chunk):
            ctrl._running = False
            return [fake_segment]

        mic = MagicMock()
        mic.stream.return_value = iter([np.zeros(480, dtype=np.float32)])

        vad = MagicMock()
        vad.process_chunk.side_effect = vad_side_effect

        embedder = MagicMock()
        primary_embedding = np.ones(192, dtype=np.float32)

        await ctrl._listen_loop(mic, vad, embedder, primary_embedding)

        assert ctrl._last_speech_at > 0.0
