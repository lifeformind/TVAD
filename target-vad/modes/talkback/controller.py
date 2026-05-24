"""TalkbackController — full-duplex voice assistant state machine.

Owns the LISTENING ⇄ SPEAKING ⇄ BARGED_IN lifecycle for one conversation.
Called by KioskPipeline.run() via TalkbackHandoff; returns TalkbackResult.
"""

import asyncio
import enum
import time
from typing import Optional

import numpy as np

from core.logging.jsonl_logger import EventLogger
from modes.talkback.chunker import SentenceChunker
from modes.talkback.conversation import ConversationManager
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult
from modes.talkback.llm import LlmClient
from modes.talkback.player import Player
from modes.talkback.stt import StreamingStt
from modes.talkback.tts import TtsEngine
from modes.talkback.watchdog import AsyncWatchdog


class TalkbackState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"
    BARGED_IN = "BARGED_IN"


class TalkbackController:
    """Full-duplex talkback controller.

    Sync entry point `run(handoff)` starts an asyncio loop internally.
    """

    def __init__(
        self,
        stt: StreamingStt,
        llm: LlmClient,
        tts: TtsEngine,
        player: Player,
        logger: EventLogger,
    ):
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._player = player
        self._logger = logger
        self.state = TalkbackState.IDLE
        self._run_result: Optional[TalkbackResult] = None
        self._started_at: float = 0.0
        self._last_speech_at: float = 0.0
        self._running = False
        self._conversation: Optional[ConversationManager] = None
        self._barge_in_require_speaker_match = True

    def _transition(self, new_state: TalkbackState) -> None:
        self.state = new_state

    def run(self, handoff: TalkbackHandoff) -> TalkbackResult:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._run_async(handoff))
        finally:
            loop.close()

    async def _run_async(self, handoff: TalkbackHandoff) -> TalkbackResult:
        self._started_at = time.monotonic()
        self._last_speech_at = self._started_at
        self._running = True
        self._transition(TalkbackState.LISTENING)

        config = handoff.config
        silence_timeout = config.get("silence_timeout_s", 10.0)
        hard_timeout = config.get("hard_timeout_s", 300.0)

        self._conversation = ConversationManager(
            system_prompt=config.get("llm", {}).get(
                "system_prompt",
                "You are a concise voice assistant.",
            )
        )
        conversation = self._conversation

        if not await self._check_llm_available():
            self._logger.log("session_ended", {
                "reason": "llm_unavailable", "turns": 0,
                "total_duration_ms": 0,
            })
            return TalkbackResult(reason="llm_unavailable", turns=0, total_duration_s=0.0)

        watchdog_tick = config.get("watchdog", {}).get("tick_ms", 500) / 1000.0

        watchdog = AsyncWatchdog(
            tick_s=watchdog_tick,
            on_timeout=self._handle_timeout,
            get_silence_duration=lambda: time.monotonic() - self._last_speech_at,
            get_session_duration=lambda: time.monotonic() - self._started_at,
            silence_timeout_s=silence_timeout,
            hard_timeout_s=hard_timeout,
        )

        self._logger.log("handoff_to_talkback", {
            "primary_embedding_norm": float(np.linalg.norm(handoff.primary_embedding)),
        })

        first_text = await self._stt.transcribe_segment(handoff.first_segment.audio)
        if first_text:
            self._last_speech_at = time.monotonic()
            self._logger.log("user_turn_complete", {"text": first_text, "turn_number": 1})
            conversation.add_user_turn(first_text)

            self._transition(TalkbackState.SPEAKING)
            self._logger.log("turn_started", {"turn_number": 1})

            assistant_text = await self._generate_response(conversation, config)
            if assistant_text:
                conversation.add_assistant_turn(assistant_text)

            self._transition(TalkbackState.LISTENING)

        watchdog.start()
        try:
            while self._running:
                await asyncio.sleep(0.05)
                if self._run_result is not None:
                    break
        finally:
            await watchdog.stop()

        await self._llm.close()

        if self._run_result is None:
            self._run_result = TalkbackResult(
                reason="stopped",
                turns=conversation.turn_count,
                total_duration_s=time.monotonic() - self._started_at,
            )

        self._transition(TalkbackState.IDLE)
        self._logger.log("session_ended", {
            "reason": self._run_result.reason,
            "turns": self._run_result.turns,
            "total_duration_ms": self._run_result.total_duration_s * 1000,
        })

        return self._run_result

    async def _generate_response(
        self, conversation: ConversationManager, config: dict
    ) -> str:
        messages = conversation.get_messages()
        self._logger.log("llm_request_sent", {
            "messages_count": len(messages),
            "model": config.get("llm", {}).get("model", "unknown"),
        })

        chunker = SentenceChunker(
            max_chunk_chars=config.get("chunker", {}).get("max_chunk_chars", 120),
        )

        full_response = []
        t0 = time.monotonic()
        first_token = True

        async for token in self._llm.stream(messages):
            if not self._running:
                break
            full_response.append(token)
            if first_token:
                self._logger.log("llm_response_started", {
                    "time_to_first_token_ms": (time.monotonic() - t0) * 1000,
                })
                first_token = False

            chunk = chunker.feed(token)
            if chunk:
                audio = await self._tts.synthesize(chunk)
                if len(audio) > 0:
                    await self._player.enqueue(audio)

        remaining = chunker.flush()
        if remaining:
            audio = await self._tts.synthesize(remaining)
            if len(audio) > 0:
                await self._player.enqueue(audio)

        response_text = "".join(full_response)
        self._logger.log("llm_response_complete", {
            "tokens": len(full_response),
            "latency_ms": (time.monotonic() - t0) * 1000,
        })

        return response_text

    def _handle_timeout(self, reason: str) -> None:
        self._running = False
        self._logger.log("watchdog_fired", {"reason": reason})
        turns = self._conversation.turn_count if self._conversation else 0
        self._run_result = TalkbackResult(
            reason=reason,
            turns=turns,
            total_duration_s=time.monotonic() - self._started_at,
        )

    def _handle_barge_in(self, primary_score: float, speech_ms: float) -> None:
        if self.state != TalkbackState.SPEAKING:
            return
        prior_state = self.state.value
        self._player.flush()
        self._llm.cancel()
        self._transition(TalkbackState.BARGED_IN)
        self._logger.log("barge_in", {
            "during_state": prior_state,
            "primary_score": primary_score,
            "cut_at_ms": speech_ms,
        })

    async def _check_llm_available(self) -> bool:
        try:
            available = await self._llm.ping()
            if not available:
                self._logger.log("llm_unavailable", {})
            return available
        except Exception:
            self._logger.log("llm_unavailable", {})
            return False

    def stop(self) -> None:
        self._running = False
