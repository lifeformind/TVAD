"""Tests for interruption-resume: remember the interrupted topic, offer to
continue it, and resume on the next turn (controller state + LLM phrasing)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.conversation import ConversationManager


def make_segment(duration_ms=1500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def make_ctrl():
    ctrl = TalkbackController(
        stt=AsyncMock(), llm=MagicMock(), tts=AsyncMock(),
        player=MagicMock(), logger=MagicMock(),
    )
    ctrl._talkback_config = {"resume": {"enabled": True}}
    ctrl._conversation = ConversationManager(system_prompt="test")
    ctrl._running = True
    ctrl._primary_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ctrl._embedder = MagicMock()
    return ctrl


class TestStoreInterruption:
    def test_captures_query_and_partial(self):
        ctrl = make_ctrl()
        ctrl._current_query = "how do I get to the conference room"
        ctrl._partial_response = "go from the cafe toward the blue door and"
        ctrl._store_interruption()

        assert ctrl._interrupted["query"] == "how do I get to the conference room"
        assert ctrl._interrupted["partial"].startswith("go from the cafe")
        # The partial is preserved in history, marked interrupted.
        assistant = [m for m in ctrl._conversation.get_messages()
                     if m["role"] == "assistant"]
        assert assistant and "[interrupted]" in assistant[-1]["content"]
        # A steer is queued so the next generation answers + offers to resume.
        assert ctrl._pending_steer is not None
        assert "conference room" in ctrl._pending_steer
        events = [c[0][0] for c in ctrl._logger.log.call_args_list]
        assert "interruption_stored" in events
        assert "resume_pending" in events

    def test_disabled_stores_nothing(self):
        ctrl = make_ctrl()
        ctrl._talkback_config = {"resume": {"enabled": False}}
        ctrl._current_query = "directions please"
        ctrl._partial_response = "go toward"
        ctrl._store_interruption()
        assert ctrl._interrupted is None
        assert ctrl._pending_steer is None

    def test_no_query_stores_nothing(self):
        ctrl = make_ctrl()
        ctrl._current_query = ""
        ctrl._partial_response = "something"
        ctrl._store_interruption()
        assert ctrl._interrupted is None


class TestSteerInjection:
    @pytest.mark.asyncio
    async def test_generate_injects_pending_steer_once(self):
        ctrl = make_ctrl()
        ctrl._conversation.add_user_turn("which side is the cafe")
        ctrl._pending_steer = "STEER: answer then offer to continue directions"

        captured = {}

        async def fake_stream(messages):
            captured["msgs"] = list(messages)
            yield "ok"

        ctrl._llm.stream = fake_stream
        ctrl._tts.synthesize = AsyncMock(return_value=np.zeros(480, dtype=np.float32))
        ctrl._out_stream = None
        config = {"chunker": {"max_chunk_chars": 120}, "llm": {}}

        with patch("modes.talkback.controller.sd"):
            await ctrl._generate_response(ctrl._conversation, config)

        steer_msgs = [m for m in captured["msgs"]
                      if m["role"] == "system" and m["content"].startswith("STEER")]
        assert len(steer_msgs) == 1
        assert ctrl._pending_steer is None        # consumed, not re-applied


class TestResumeOnNextTurn:
    @pytest.mark.asyncio
    async def test_listening_turn_injects_resume_steer_then_clears(self):
        ctrl = make_ctrl()
        ctrl.state = TalkbackState.LISTENING
        ctrl._interrupted = {"query": "how do I get to the conference room",
                             "partial": "go from the cafe toward the blue door"}
        # Primary speaker, long enough turn → accepted by the gate.
        ctrl._embedder.extract = MagicMock(return_value=ctrl._primary_embedding.copy())
        ctrl._talkback_config = {
            "resume": {"enabled": True},
            "turn_gate": {"require_speaker_match": True,
                          "speaker_threshold": 0.3, "verify_window_ms": 1200},
        }
        ctrl._stt.transcribe_segment = AsyncMock(return_value="yes")
        ctrl._generate_response = AsyncMock(return_value="As I was saying...")

        with patch("modes.talkback.controller.sd"):
            task = await ctrl._handle_segment(make_segment(1500.0))

        try:
            assert ctrl._pending_steer is not None
            assert "conference room" in ctrl._pending_steer
            assert ctrl._interrupted is None       # one-shot, cleared
            events = [c[0][0] for c in ctrl._logger.log.call_args_list]
            assert "resume_continued" in events
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
