"""Tests for TalkbackController._handle_segment multi-turn and barge-in handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.conversation import ConversationManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def make_segment(duration_ms: float = 500.0) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0,
        end_ms=duration_ms,
        duration_ms=duration_ms,
    )


def make_ctrl() -> TalkbackController:
    """Return a TalkbackController with all deps mocked."""
    fake_logger = MagicMock()
    fake_logger.log = MagicMock()

    ctrl = TalkbackController(
        stt=AsyncMock(),
        llm=MagicMock(),
        tts=AsyncMock(),
        player=MagicMock(),
        logger=fake_logger,
    )

    # Attach required Task 3 attributes — always overwrite to ensure
    # correct types regardless of __init__ defaults.
    ctrl._talkback_config = {}
    ctrl._primary_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ctrl._embedder = MagicMock()
    ctrl._response_task = None

    # Wire up a fresh ConversationManager
    ctrl._conversation = ConversationManager(system_prompt="You are a test assistant.")
    ctrl._running = True

    return ctrl


# ---------------------------------------------------------------------------
# TestHandleSegmentListening
# ---------------------------------------------------------------------------

class TestHandleSegmentListening:

    @pytest.mark.asyncio
    async def test_listening_segment_transcribed_and_response_started(self):
        """LISTENING → transcribes text, starts response task, state → SPEAKING."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        # Primary speaker passes the turn gate (embedding matches primary).
        ctrl._embedder.extract = MagicMock(return_value=ctrl._primary_embedding.copy())
        ctrl._stt.transcribe_segment = AsyncMock(return_value="hello there")
        ctrl._generate_response = AsyncMock(return_value="Hi!")

        segment = make_segment(500.0)

        with patch("modes.talkback.controller.sd"):
            task = await ctrl._handle_segment(segment)

        try:
            assert task is not None
            ctrl._stt.transcribe_segment.assert_awaited_once_with(segment.audio)
            assert ctrl.state == TalkbackState.SPEAKING
            # Conversation should have user turn
            messages = ctrl._conversation.get_messages()
            user_msgs = [m for m in messages if m["role"] == "user"]
            assert len(user_msgs) == 1
            assert user_msgs[0]["content"] == "hello there"
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_listening_empty_transcript_ignored(self):
        """LISTENING → empty transcript returns None, state stays LISTENING."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        ctrl._embedder.extract = MagicMock(return_value=ctrl._primary_embedding.copy())
        ctrl._stt.transcribe_segment = AsyncMock(return_value="")

        segment = make_segment(500.0)

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._handle_segment(segment)

        assert result is None
        assert ctrl.state == TalkbackState.LISTENING


# ---------------------------------------------------------------------------
# TestHandleSegmentTurnGate — speaker lock on NEW turns (not just barge-in)
# ---------------------------------------------------------------------------

class TestHandleSegmentTurnGate:
    """A new turn in LISTENING state must come from the enrolled primary speaker.

    Regression: previously the LISTENING branch transcribed and answered ANY
    speech, so a second (non-enrolled) person was served between TTS replies.
    """

    # Segments must be >= min_verify_ms (default 2000) to be scored at all.
    VERIFY_MS = 2500.0

    def _config(self, **gate):
        cfg = {"require_speaker_match": True, "speaker_threshold": 0.5,
               "min_verify_ms": 2000}
        cfg.update(gate)
        return {"turn_gate": cfg}

    @pytest.mark.asyncio
    async def test_non_primary_turn_rejected(self):
        """LISTENING + stranger (long turn) → no transcription, stays LISTENING."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        primary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        stranger = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # cosine 0
        ctrl._primary_embedding = primary
        ctrl._embedder.extract = MagicMock(return_value=stranger)
        ctrl._talkback_config = self._config()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="who are you")

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._handle_segment(make_segment(self.VERIFY_MS))

        assert result is None
        assert ctrl.state == TalkbackState.LISTENING
        # Stranger's speech must never reach STT or the conversation.
        ctrl._stt.transcribe_segment.assert_not_awaited()
        assert ctrl._conversation.turn_count == 0
        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert "turn_gate" in events
        assert "user_turn_complete" not in events

    @pytest.mark.asyncio
    async def test_short_turn_accepted_unverified(self):
        """LISTENING + short turn (< min_verify_ms) → accepted without scoring."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        ctrl._embedder.extract = MagicMock()
        ctrl._talkback_config = self._config()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="okay")
        ctrl._generate_response = AsyncMock(return_value="Sure!")

        with patch("modes.talkback.controller.sd"):
            task = await ctrl._handle_segment(make_segment(900.0))

        try:
            assert task is not None              # short turn still answered
            ctrl._embedder.extract.assert_not_called()  # ECAPA never invoked
            events = [c[0][0] for c in ctrl._logger.log.call_args_list]
            assert "turn_gate_skipped" in events
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_primary_turn_accepted(self):
        """LISTENING + primary speaker (long turn) → transcribed and answered."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        primary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ctrl._primary_embedding = primary
        ctrl._embedder.extract = MagicMock(return_value=primary.copy())
        ctrl._talkback_config = self._config()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="hello there")
        ctrl._generate_response = AsyncMock(return_value="Hi!")

        with patch("modes.talkback.controller.sd"):
            task = await ctrl._handle_segment(make_segment(self.VERIFY_MS))

        try:
            assert task is not None
            assert ctrl.state == TalkbackState.SPEAKING
            user_msgs = [m for m in ctrl._conversation.get_messages()
                         if m["role"] == "user"]
            assert [m["content"] for m in user_msgs] == ["hello there"]
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_gate_disabled_skips_verification(self):
        """require_speaker_match False → no embedding call, turn proceeds."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        ctrl._embedder.extract = MagicMock()
        ctrl._talkback_config = self._config(require_speaker_match=False)
        ctrl._stt.transcribe_segment = AsyncMock(return_value="hello")
        ctrl._generate_response = AsyncMock(return_value="Hi!")

        with patch("modes.talkback.controller.sd"):
            task = await ctrl._handle_segment(make_segment(500.0))

        try:
            assert task is not None
            ctrl._embedder.extract.assert_not_called()
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


# ---------------------------------------------------------------------------
# TestHandleSegmentSpeaking
# ---------------------------------------------------------------------------

class TestHandleSegmentSpeaking:

    @pytest.mark.asyncio
    async def test_speaking_primary_speaker_triggers_barge_in(self):
        """SPEAKING + primary speaker → cancels response task, starts new one."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.SPEAKING

        primary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ctrl._primary_embedding = primary
        ctrl._embedder.extract = MagicMock(return_value=primary.copy())

        ctrl._talkback_config = {
            "barge_in": {
                "enabled": True,
                "min_speech_ms": 120,
                "require_speaker_match": True,
                "speaker_threshold": 0.5,
            }
        }

        ctrl._generate_response = AsyncMock(return_value="New response")
        ctrl._stt.transcribe_segment = AsyncMock(return_value="interrupt")
        ctrl._player.flush = MagicMock()
        ctrl._llm.cancel = MagicMock()

        # Create a dummy existing response task
        dummy_task = asyncio.create_task(asyncio.sleep(100))
        ctrl._response_task = dummy_task

        segment = make_segment(500.0)

        with patch("modes.talkback.controller.sd"):
            new_task = await ctrl._handle_segment(segment)

        try:
            # Dummy task should have been cancelled
            assert dummy_task.cancelled()
            # A new task should have been started
            assert new_task is not None
        finally:
            if new_task and not new_task.done():
                new_task.cancel()
                try:
                    await new_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_speaking_non_primary_ignored(self):
        """SPEAKING + non-primary speaker → barge-in rejected, returns None."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.SPEAKING

        primary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Orthogonal embedding — cosine similarity will be 0
        stranger = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ctrl._primary_embedding = primary
        ctrl._embedder.extract = MagicMock(return_value=stranger)

        ctrl._talkback_config = {
            "barge_in": {
                "enabled": True,
                "min_speech_ms": 120,
                "require_speaker_match": True,
                "speaker_threshold": 0.5,
            }
        }

        ctrl._generate_response = AsyncMock(return_value="Response")
        # Set up a pending response task to clean up later
        ctrl._response_task = asyncio.create_task(asyncio.sleep(100))

        segment = make_segment(500.0)

        try:
            with patch("modes.talkback.controller.sd"):
                result = await ctrl._handle_segment(segment)

            assert result is None
            assert ctrl.state == TalkbackState.SPEAKING
        finally:
            if ctrl._response_task and not ctrl._response_task.done():
                ctrl._response_task.cancel()
                try:
                    await ctrl._response_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_speaking_short_segment_ignored(self):
        """SPEAKING + segment below min_speech_ms → ignored, returns None."""
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.SPEAKING

        ctrl._talkback_config = {
            "barge_in": {
                "enabled": True,
                "min_speech_ms": 300,
                "require_speaker_match": True,
                "speaker_threshold": 0.5,
            }
        }

        ctrl._generate_response = AsyncMock(return_value="Response")
        ctrl._response_task = asyncio.create_task(asyncio.sleep(100))

        # Segment shorter than min_speech_ms
        segment = make_segment(duration_ms=100.0)

        try:
            with patch("modes.talkback.controller.sd"):
                result = await ctrl._handle_segment(segment)

            assert result is None
        finally:
            if ctrl._response_task and not ctrl._response_task.done():
                ctrl._response_task.cancel()
                try:
                    await ctrl._response_task
                except asyncio.CancelledError:
                    pass
