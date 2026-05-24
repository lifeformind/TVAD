"""Tests for barge-in — speaker-verified TTS cut on primary speech."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.player import Player


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_flushes_player(self):
        player = Player(sample_rate=16000)
        await player.enqueue(np.zeros(1600, dtype=np.float32))
        assert player.is_playing

        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=player, logger=MagicMock(),
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._llm.cancel = MagicMock()

        ctrl._handle_barge_in(primary_score=0.85, speech_ms=200)

        assert player.pending_frames == 0
        assert ctrl.state == TalkbackState.BARGED_IN

    @pytest.mark.asyncio
    async def test_barge_in_cancels_llm(self):
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._llm.cancel = MagicMock()
        ctrl._player.flush = MagicMock()

        ctrl._handle_barge_in(primary_score=0.75, speech_ms=150)

        ctrl._llm.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_barge_in_logs_event(self):
        logger = MagicMock()
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=logger,
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._llm.cancel = MagicMock()
        ctrl._player.flush = MagicMock()

        ctrl._handle_barge_in(primary_score=0.82, speech_ms=180)

        logger.log.assert_called()
        call_args = logger.log.call_args
        assert call_args[0][0] == "barge_in"
        assert call_args[0][1]["primary_score"] == 0.82
        assert call_args[0][1]["during_state"] == "SPEAKING"

    @pytest.mark.asyncio
    async def test_barge_in_ignored_when_not_speaking(self):
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.LISTENING
        ctrl._llm.cancel = MagicMock()

        ctrl._handle_barge_in(primary_score=0.85, speech_ms=200)
        assert ctrl.state == TalkbackState.LISTENING
        ctrl._llm.cancel.assert_not_called()


class TestSpeakerVerifiedBargeIn:
    def test_non_primary_does_not_trigger_barge_in(self):
        """When require_speaker_match is True, non-primary speech is ignored."""
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._barge_in_require_speaker_match = True

        primary_emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        other_emb = np.zeros(192, dtype=np.float32)
        other_emb[0] = 1.0

        from core.speaker.verifier import cosine_similarity
        score = cosine_similarity(other_emb, primary_emb)
        assert score < 0.3
