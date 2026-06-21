"""Tests for barge-in — speaker-verified TTS cut on primary speech."""

from unittest.mock import AsyncMock, MagicMock

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


class TestBackchannelVsQuestion:
    """The SPEAKING-branch reorder: classify the interjection before cutting."""

    def _make_ctrl(self):
        from modes.talkback.conversation import ConversationManager
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._conversation = ConversationManager(system_prompt="sys")
        ctrl._primary_embedding = np.ones(192, dtype=np.float32) / np.sqrt(192)
        ctrl._embedder = MagicMock()
        ctrl._embedder.extract = MagicMock(
            return_value=np.ones(192, dtype=np.float32) / np.sqrt(192))
        ctrl._talkback_config = {
            "barge_in": {"enabled": True, "require_speaker_match": True,
                         "min_speech_ms": 120, "verify_window_ms": 700,
                         "speaker_threshold": 0.20},
            "resume": {"enabled": True},
        }
        ctrl._barge_duck_enabled = True
        ctrl._proximity_rms = 0.0  # disable proximity gate for the test
        ctrl._restore_volume = MagicMock()
        ctrl._drain_playback = AsyncMock()
        ctrl._response_task = None
        return ctrl

    def _segment(self, ms=900, rms=0.5):
        n = int(16000 * ms / 1000)
        audio = np.full(n, rms, dtype=np.float32)
        seg = MagicMock()
        seg.audio = audio
        seg.duration_ms = ms
        return seg

    @pytest.mark.asyncio
    async def test_backchannel_keeps_speaking_and_does_not_pollute_history(self):
        ctrl = self._make_ctrl()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="yeah got it")

        task = await ctrl._handle_segment(self._segment())

        assert task is None
        assert ctrl.state == TalkbackState.SPEAKING       # never cut
        ctrl._restore_volume.assert_called_once()          # un-ducked
        ctrl._drain_playback.assert_not_called()           # no cut
        assert all(m["content"] != "yeah got it"
                   for m in ctrl._conversation.get_messages())

    @pytest.mark.asyncio
    async def test_question_cuts_and_starts_new_turn(self):
        ctrl = self._make_ctrl()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="wait why is that")
        ctrl._generate_and_speak = AsyncMock()

        task = await ctrl._handle_segment(self._segment())

        ctrl._drain_playback.assert_awaited_once()         # cut happened
        assert any(m["content"] == "wait why is that"
                   for m in ctrl._conversation.get_messages())
