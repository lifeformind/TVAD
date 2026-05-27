"""End-to-end multi-turn test with mocked backends."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult


def make_segment(duration_ms=500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class AsyncTokenStream:
    def __init__(self, tokens):
        self._tokens = tokens
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        return self._tokens.pop(0)


class TestE2EMultiTurn:
    @pytest.mark.asyncio
    async def test_two_turns_then_silence_timeout(self):
        transcripts = iter(["What is Python?", "Thanks!"])
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(side_effect=lambda audio: next(transcripts))

        responses = [
            ["Python ", "is a ", "programming language."],
            ["You're ", "welcome!"],
        ]
        response_iter = iter(responses)
        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        llm.cancel = MagicMock()
        def make_stream(messages):
            return AsyncTokenStream(next(response_iter))
        llm.stream = make_stream

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

        # VAD produces second_seg on first chunk, nothing after
        vad = MagicMock()
        vad_call = 0
        def vad_process(chunk):
            nonlocal vad_call
            vad_call += 1
            if vad_call == 1:
                return [second_seg]
            return []
        vad.process_chunk = vad_process

        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([np.zeros(480, dtype=np.float32)]))

        config = {
            "silence_timeout_s": 0.5,
            "hard_timeout_s": 60.0,
            "watchdog": {"tick_ms": 100},
            "llm": {"system_prompt": "You are helpful."},
            "chunker": {"max_chunk_chars": 200},
            "barge_in": {"enabled": False},
            "aec": {"enabled": False},
        }

        handoff = TalkbackHandoff(
            mic=mic,
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=first_seg,
            config=config,
            vad=vad,
            embedder=MagicMock(),
        )

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._run_async(handoff)

        assert result.turns == 2
        assert result.reason == "silence_timeout"

        # Verify both transcripts were processed
        assert stt.transcribe_segment.await_count == 2

        # Verify conversation has 4 messages (2 user + 2 assistant) + system
        msgs = ctrl._conversation.get_messages()
        # msgs[0] is system prompt
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is Python?"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "Thanks!"
        assert msgs[4]["role"] == "assistant"

        # Verify key events logged
        log_events = [call[0][0] for call in logger.log.call_args_list]
        assert "handoff_to_talkback" in log_events
        assert log_events.count("user_turn_complete") == 2
        assert "session_ended" in log_events
