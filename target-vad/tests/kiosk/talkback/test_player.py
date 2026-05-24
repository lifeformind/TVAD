"""Tests for Player — async audio output with ring buffer for AEC reference."""

import numpy as np
import pytest

from modes.talkback.player import Player


@pytest.fixture
def player():
    return Player(sample_rate=16000, ring_buffer_seconds=2.0)


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_adds_to_queue(self, player):
        audio = np.zeros(1600, dtype=np.float32)
        await player.enqueue(audio)
        assert player.pending_frames > 0

    @pytest.mark.asyncio
    async def test_enqueue_multiple(self, player):
        for _ in range(3):
            await player.enqueue(np.zeros(1600, dtype=np.float32))
        assert player.pending_frames == 3


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_clears_queue(self, player):
        await player.enqueue(np.zeros(1600, dtype=np.float32))
        await player.enqueue(np.zeros(1600, dtype=np.float32))
        player.flush()
        assert player.pending_frames == 0

    @pytest.mark.asyncio
    async def test_flush_sets_flushed_flag(self, player):
        player.flush()
        assert player.is_flushed


class TestRingBuffer:
    @pytest.mark.asyncio
    async def test_ring_buffer_captures_played_frames(self, player):
        audio = np.ones(160, dtype=np.float32) * 0.5
        player._record_to_ring_buffer(audio)
        ref = player.get_reference_frame(160)
        assert ref is not None
        np.testing.assert_array_almost_equal(ref, audio)

    @pytest.mark.asyncio
    async def test_ring_buffer_wraps(self, player):
        frame = np.ones(160, dtype=np.float32) * 0.25
        total_frames = int(player._ring_buffer_size / 160) + 5
        for _ in range(total_frames):
            player._record_to_ring_buffer(frame)
        ref = player.get_reference_frame(160)
        assert ref is not None
        assert len(ref) == 160

    def test_reference_frame_returns_none_when_empty(self, player):
        ref = player.get_reference_frame(160)
        assert ref is None


class TestPlayingState:
    @pytest.mark.asyncio
    async def test_is_playing_when_frames_queued(self, player):
        await player.enqueue(np.zeros(1600, dtype=np.float32))
        assert player.is_playing

    def test_not_playing_when_empty(self, player):
        assert not player.is_playing
