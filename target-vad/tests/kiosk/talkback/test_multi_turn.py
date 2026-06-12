"""Tests for TalkbackController._handle_segment multi-turn and barge-in handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.speaker.decision_smoother import DecisionSmoother
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

    # A single turn this long fills the verify window (2000) on its own, so it
    # is scored immediately rather than accumulated.
    VERIFY_MS = 2500.0

    def _config(self, **gate):
        cfg = {"require_speaker_match": True, "speaker_threshold": 0.5,
               "verify_window_ms": 2000}
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
    async def test_short_turn_accepted_provisionally(self):
        """LISTENING + sub-window turn → served now, audio buffered for the window.

        A single short turn cannot be embedded reliably (ECAPA needs ~2s), so it
        is accepted provisionally and its audio accumulates toward the next
        verification window rather than being scored on its own.
        """
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
            ctrl._embedder.extract.assert_not_called()  # ECAPA never invoked yet
            events = [c[0][0] for c in ctrl._logger.log.call_args_list]
            assert "turn_gate_pending" in events
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
# TestRollingWindowGate — accumulate consecutive turns to a reliable window
# ---------------------------------------------------------------------------

class TestRollingWindowGate:
    """Per-turn ECAPA can't separate self from a bystander at conversational
    lengths (~0.3-1.5s). Instead, consecutive turn audio accumulates to a
    >=verify_window_ms window where ECAPA is reliable, scored once, then reset.
    """

    PRIMARY = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    STRANGER = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # cosine 0 → reject

    def _ctrl(self, embedding, window_ms=2000, threshold=0.3):
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        ctrl._primary_embedding = self.PRIMARY
        ctrl._embedder.extract = MagicMock(return_value=embedding)
        ctrl._talkback_config = {"turn_gate": {
            "require_speaker_match": True, "speaker_threshold": threshold,
            "verify_window_ms": window_ms}}
        return ctrl

    @pytest.mark.asyncio
    async def test_subwindow_turns_accumulate_then_verify_once(self):
        """Three 800ms primary turns: first two pending, the third fills the
        2000ms window → ECAPA runs ONCE on the concatenation and accepts."""
        ctrl = self._ctrl(self.PRIMARY.copy())

        assert await ctrl._passes_turn_gate(make_segment(800.0)) is True
        assert await ctrl._passes_turn_gate(make_segment(800.0)) is True
        ctrl._embedder.extract.assert_not_called()        # 1600ms < window

        assert await ctrl._passes_turn_gate(make_segment(800.0)) is True
        ctrl._embedder.extract.assert_called_once()       # window full → scored
        # Scored on the full accumulated audio (~2400ms == 3 * 12800 samples).
        scored_audio = ctrl._embedder.extract.call_args[0][0]
        assert len(scored_audio) == 3 * int(800.0 / 1000 * 16000)

        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert events.count("turn_gate_pending") == 2
        assert events.count("turn_gate") == 1

    @pytest.mark.asyncio
    async def test_window_rejects_stranger_after_accumulation(self):
        """A bystander's sub-window turns are served provisionally, but once the
        window fills the accumulated audio is scored and rejected."""
        ctrl = self._ctrl(self.STRANGER)

        assert await ctrl._passes_turn_gate(make_segment(800.0)) is True   # pending
        assert await ctrl._passes_turn_gate(make_segment(800.0)) is True   # pending
        assert await ctrl._passes_turn_gate(make_segment(800.0)) is False  # window → reject

    @pytest.mark.asyncio
    async def test_buffer_resets_after_each_window(self):
        """After a window is scored the buffer clears, so the next window
        accumulates fresh rather than re-scoring stale audio."""
        ctrl = self._ctrl(self.PRIMARY.copy())

        await ctrl._passes_turn_gate(make_segment(2500.0))   # fills + scores window
        assert ctrl._gate_audio_ms == 0.0
        assert ctrl._gate_audio == []

    @pytest.mark.asyncio
    async def test_single_long_turn_scored_immediately(self):
        """A turn already >= window is scored on its own, no accumulation wait."""
        ctrl = self._ctrl(self.PRIMARY.copy())
        assert await ctrl._passes_turn_gate(make_segment(2200.0)) is True
        ctrl._embedder.extract.assert_called_once()


# ---------------------------------------------------------------------------
# TestSpeakerLockout — end the session on sustained verified mismatch
# ---------------------------------------------------------------------------

class TestSpeakerLockout:
    PRIMARY = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    STRANGER = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # cosine 0 → reject
    LONG_MS = 2500.0

    def _ctrl(self, window=3, min_matches=1, threshold=0.5):
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        ctrl._primary_embedding = self.PRIMARY
        ctrl._talkback_config = {"turn_gate": {
            "require_speaker_match": True, "speaker_threshold": threshold,
            "verify_window_ms": 2000}}
        ctrl._gate_smoother = DecisionSmoother(window, min_matches, threshold)
        ctrl._gate_scored = 0
        return ctrl

    async def _gate(self, ctrl, emb):
        ctrl._embedder.extract = MagicMock(return_value=emb)
        return await ctrl._passes_turn_gate(make_segment(self.LONG_MS))

    @pytest.mark.asyncio
    async def test_lockout_after_three_consecutive_rejects(self):
        ctrl = self._ctrl()
        for _ in range(2):
            assert await self._gate(ctrl, self.STRANGER) is False
            assert ctrl._running is True            # window not full yet
            assert ctrl._run_result is None
        # 3rd verified reject fills the window → lockout
        assert await self._gate(ctrl, self.STRANGER) is False
        assert ctrl._running is False
        assert ctrl._run_result is not None
        assert ctrl._run_result.reason == "speaker_lockout"
        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert "speaker_lockout" in events

    @pytest.mark.asyncio
    async def test_accept_keeps_session_alive(self):
        ctrl = self._ctrl()
        await self._gate(ctrl, self.STRANGER)            # reject
        await self._gate(ctrl, self.PRIMARY.copy())      # accept
        await self._gate(ctrl, self.STRANGER)            # reject (window has 1 match)
        await self._gate(ctrl, self.STRANGER)            # reject (window: a,r,r → 1 match)
        assert ctrl._running is True
        assert ctrl._run_result is None

    @pytest.mark.asyncio
    async def test_lockout_disabled_never_fires(self):
        ctrl = self._ctrl()
        ctrl._gate_smoother = None                       # lockout off
        for _ in range(5):
            assert await self._gate(ctrl, self.STRANGER) is False
        assert ctrl._running is True
        assert ctrl._run_result is None


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
                "verify_window_ms": 1200,
            }
        }

        ctrl._generate_response = AsyncMock(return_value="New response")
        ctrl._stt.transcribe_segment = AsyncMock(return_value="interrupt")
        ctrl._player.flush = MagicMock()
        ctrl._llm.cancel = MagicMock()

        # Create a dummy existing response task
        dummy_task = asyncio.create_task(asyncio.sleep(100))
        ctrl._response_task = dummy_task

        # >= verify_window_ms so the interruption is long enough to verify.
        segment = make_segment(1500.0)

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
                "verify_window_ms": 1200,
            }
        }

        ctrl._generate_response = AsyncMock(return_value="Response")
        # Set up a pending response task to clean up later
        ctrl._response_task = asyncio.create_task(asyncio.sleep(100))

        segment = make_segment(1500.0)

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


# ---------------------------------------------------------------------------
# TestBargeInGate — proximity pre-gate + verify-window + duck/restore
# ---------------------------------------------------------------------------

def make_segment_amp(duration_ms: float, amplitude: float) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.full(samples, amplitude, dtype=np.float32),
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestBargeInGate:
    PRIMARY = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    STRANGER = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def _ctrl(self, emb, *, threshold=0.3, verify_window_ms=1200):
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.SPEAKING
        ctrl._primary_embedding = self.PRIMARY
        ctrl._embedder.extract = MagicMock(return_value=emb)
        ctrl._stt.transcribe_segment = AsyncMock(return_value="interrupt")
        ctrl._generate_response = AsyncMock(return_value="resp")
        ctrl._talkback_config = {"barge_in": {
            "enabled": True, "min_speech_ms": 120, "require_speaker_match": True,
            "speaker_threshold": threshold, "verify_window_ms": verify_window_ms,
        }}
        return ctrl

    @pytest.mark.asyncio
    async def test_too_short_to_verify_does_not_cut(self):
        """A sub-verify_window interruption can't be verified → never cuts."""
        ctrl = self._ctrl(self.PRIMARY.copy())
        with patch("modes.talkback.controller.sd"):
            result = await ctrl._handle_segment(make_segment(600.0))
        assert result is None
        assert ctrl.state == TalkbackState.SPEAKING
        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert "barge_in" not in events

    @pytest.mark.asyncio
    async def test_far_speech_ignored_by_proximity(self):
        """Speech too quiet to be at the kiosk is ignored before verification."""
        ctrl = self._ctrl(self.PRIMARY.copy())
        ctrl._barge_duck_enabled = True
        ctrl._proximity_rms = 0.05
        with patch("modes.talkback.controller.sd"):
            result = await ctrl._handle_segment(make_segment_amp(1500.0, 0.01))
        assert result is None
        ctrl._embedder.extract.assert_not_called()       # never even verified
        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert "barge_in_ignored_far" in events

    @pytest.mark.asyncio
    async def test_verified_user_cuts_playback(self):
        ctrl = self._ctrl(self.PRIMARY.copy())
        ctrl._playback_cancelled = False
        ctrl._response_task = asyncio.create_task(asyncio.sleep(100))
        with patch("modes.talkback.controller.sd"):
            task = await ctrl._handle_segment(make_segment(1500.0))
        try:
            assert task is not None
            assert ctrl._playback_cancelled is True
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_reject_restores_volume(self):
        """A ducked, then rejected, interruption restores full volume."""
        ctrl = self._ctrl(self.STRANGER)
        ctrl._gain = 0.15
        ctrl._ducked = True
        with patch("modes.talkback.controller.sd"):
            result = await ctrl._handle_segment(make_segment(1500.0))
        assert result is None
        assert ctrl._gain == 1.0
        assert ctrl._ducked is False
        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert "barge_in_rejected" in events
