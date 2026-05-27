"""Tests for TalkbackController state machine with fake components."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.conversation import ConversationManager
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult


def make_segment(duration_ms: float = 1000.0) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestTalkbackControllerStates:
    def test_initial_state_is_idle(self):
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        assert ctrl.state == TalkbackState.IDLE

    @pytest.mark.asyncio
    async def test_handle_timeout_sets_result(self):
        fake_logger = MagicMock()
        fake_logger.log = MagicMock()

        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=fake_logger,
        )
        ctrl._started_at = time.monotonic() - 5.0
        ctrl._running = True

        conv = ConversationManager(system_prompt="sys")
        conv.add_user_turn("a")
        conv.add_assistant_turn("b")
        conv.add_user_turn("c")
        conv.add_assistant_turn("d")
        ctrl._conversation = conv

        ctrl._handle_timeout("silence_timeout")

        assert ctrl._running is False
        assert ctrl._run_result is not None
        assert ctrl._run_result.reason == "silence_timeout"
        assert ctrl._run_result.turns == 2
        assert ctrl._run_result.total_duration_s >= 5.0
        fake_logger.log.assert_called_with("watchdog_fired", {"reason": "silence_timeout"})


class TestTalkbackControllerTransitions:
    def test_state_enum_values(self):
        assert TalkbackState.IDLE.value == "IDLE"
        assert TalkbackState.LISTENING.value == "LISTENING"
        assert TalkbackState.SPEAKING.value == "SPEAKING"
        assert TalkbackState.BARGED_IN.value == "BARGED_IN"

    def test_transition_listening_to_speaking(self):
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.LISTENING
        ctrl._transition(TalkbackState.SPEAKING)
        assert ctrl.state == TalkbackState.SPEAKING

    def test_transition_speaking_to_barged_in(self):
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._transition(TalkbackState.BARGED_IN)
        assert ctrl.state == TalkbackState.BARGED_IN


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handle_timeout_logs_watchdog_fired(self):
        logger = MagicMock()
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=logger,
        )
        ctrl._started_at = time.monotonic()
        ctrl._running = True
        ctrl._conversation = ConversationManager(system_prompt="sys")

        ctrl._handle_timeout("hard_timeout")

        logger.log.assert_called_with("watchdog_fired", {"reason": "hard_timeout"})
        assert ctrl._run_result.reason == "hard_timeout"

    @pytest.mark.asyncio
    async def test_llm_unavailable_returns_false(self):
        fake_llm = MagicMock()
        fake_llm.ping = AsyncMock(return_value=False)
        logger = MagicMock()

        ctrl = TalkbackController(
            stt=MagicMock(), llm=fake_llm, tts=MagicMock(),
            player=MagicMock(), logger=logger,
        )

        result = await ctrl._check_llm_available()
        assert result is False


class TestMultiTurnRunAsync:
    @pytest.mark.asyncio
    async def test_two_turn_conversation(self):
        """Controller processes first segment, then a second from the mic."""
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(side_effect=["hello", "thanks"])

        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        llm.cancel = MagicMock()

        async def fake_stream(messages):
            yield "response"
        llm.stream = fake_stream

        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=np.zeros(1600, dtype=np.float32))

        player = MagicMock()
        player.enqueue = AsyncMock()
        player.flush = MagicMock()
        player.get_reference_frame = MagicMock(return_value=None)

        logger = MagicMock()
        logger.log = MagicMock()
        logger.start_session = MagicMock()

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=tts, player=player, logger=logger,
        )

        first_seg = make_segment(500)
        second_seg = make_segment(500)

        # Mic yields one chunk that produces second_seg via VAD, then stops
        vad = MagicMock()
        call_count = 0
        def vad_process(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [second_seg]
            return []
        vad.process_chunk = vad_process

        embedder = MagicMock()

        chunk = np.zeros(480, dtype=np.float32)
        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([chunk]))

        config = {
            "silence_timeout_s": 1.0,
            "hard_timeout_s": 300.0,
            "watchdog": {"tick_ms": 100},
            "llm": {"system_prompt": "test"},
            "chunker": {"max_chunk_chars": 120},
            "barge_in": {"enabled": False},
            "aec": {"enabled": False},
        }

        handoff = TalkbackHandoff(
            mic=mic,
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=first_seg,
            config=config,
            vad=vad,
            embedder=embedder,
        )

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._run_async(handoff)

        assert result.turns >= 2
        assert stt.transcribe_segment.await_count == 2
