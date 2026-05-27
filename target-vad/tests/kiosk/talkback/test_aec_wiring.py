"""Tests for AEC initialization in the multi-turn controller."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.handoff import TalkbackHandoff


def make_segment(duration_ms=500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestAecWiring:
    @pytest.mark.asyncio
    async def test_aec_disabled_in_config(self):
        """When aec.enabled is false, _aec stays None."""
        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="hi")

        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=np.zeros(1600, dtype=np.float32))

        player = MagicMock()
        player.enqueue = AsyncMock()

        logger = MagicMock()
        logger.log = MagicMock()
        logger.start_session = MagicMock()

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=tts, player=player, logger=logger,
        )

        handoff = TalkbackHandoff(
            mic=MagicMock(stream=MagicMock(return_value=iter([]))),
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=make_segment(),
            config={
                "aec": {"enabled": False},
                "silence_timeout_s": 0.1,
                "hard_timeout_s": 1.0,
                "chunker": {"max_chunk_chars": 120},
            },
            vad=MagicMock(process_chunk=MagicMock(return_value=[])),
            embedder=MagicMock(),
        )

        async def fake_stream(messages):
            yield "ok"
        llm.stream = fake_stream

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._run_async(handoff)

        assert ctrl._aec is None

    @pytest.mark.asyncio
    async def test_aec_enabled_but_init_fails_degrades(self):
        """When AEC init throws, _aec is None and controller continues."""
        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="hi")

        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=np.zeros(1600, dtype=np.float32))

        player = MagicMock()
        player.enqueue = AsyncMock()

        logger = MagicMock()
        logger.log = MagicMock()
        logger.start_session = MagicMock()

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=tts, player=player, logger=logger,
        )

        handoff = TalkbackHandoff(
            mic=MagicMock(stream=MagicMock(return_value=iter([]))),
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=make_segment(),
            config={
                "aec": {"enabled": True},
                "silence_timeout_s": 0.1,
                "hard_timeout_s": 1.0,
                "chunker": {"max_chunk_chars": 120},
            },
            vad=MagicMock(process_chunk=MagicMock(return_value=[])),
            embedder=MagicMock(),
        )

        async def fake_stream(messages):
            yield "ok"
        llm.stream = fake_stream

        with patch("modes.talkback.controller.AecProcessor", side_effect=RuntimeError("no shim")):
            with patch("modes.talkback.controller.sd"):
                result = await ctrl._run_async(handoff)

        assert ctrl._aec is None
        assert result.reason in ("silence_timeout", "stopped")
