"""TalkbackController — full-duplex voice assistant state machine.

Owns the LISTENING ⇄ SPEAKING ⇄ BARGED_IN lifecycle for one conversation.
Called by KioskPipeline.run() via TalkbackHandoff; returns TalkbackResult.
"""

import asyncio
import enum
import time
import uuid
from typing import Optional

import numpy as np
import sounddevice as sd

from core.logging.jsonl_logger import EventLogger
from core.speaker.verifier import cosine_similarity
from modes.talkback.chunker import SentenceChunker
from modes.talkback.conversation import ConversationManager
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult
from modes.talkback.llm import LlmClient
from modes.talkback.player import Player
from modes.talkback.stt import StreamingStt
from modes.talkback.tts import TtsEngine
from modes.talkback.watchdog import AsyncWatchdog

try:
    from modes.talkback.aec import AecProcessor
except Exception:  # pragma: no cover
    AecProcessor = None  # type: ignore[assignment,misc]


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
        on_event: Optional[callable] = None,
    ):
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._player = player
        self._logger = logger
        self._on_event = on_event
        self.state = TalkbackState.IDLE
        self._run_result: Optional[TalkbackResult] = None
        self._started_at: float = 0.0
        self._last_speech_at: float = 0.0
        self._running = False
        self._conversation: Optional[ConversationManager] = None
        self._barge_in_require_speaker_match = True
        self._aec: Optional['AecProcessor'] = None
        self._segment_queue: asyncio.Queue = asyncio.Queue()
        self._primary_embedding: Optional[np.ndarray] = None
        self._embedder = None
        self._talkback_config: dict = {}
        self._response_task: Optional[asyncio.Task] = None

    def _emit(self, event: str, payload: dict) -> None:
        self._logger.log(event, payload)
        if self._on_event:
            try:
                self._on_event(event, payload)
            except Exception:
                pass

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

        # Force a fresh aiohttp session — the warmup ping ran on a
        # different event loop that's now closed.
        await self._llm.close()

        session_id = uuid.uuid4().hex[:12]
        self._logger.start_session(session_id)

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
            self._emit("session_ended", {
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

        self._emit("handoff_to_talkback", {
            "primary_embedding_norm": float(np.linalg.norm(handoff.primary_embedding)),
        })

        first_text = await self._stt.transcribe_segment(handoff.first_segment.audio)
        if first_text:
            self._last_speech_at = time.monotonic()
            self._emit("user_turn_complete", {"text": first_text, "turn_number": 1})
            conversation.add_user_turn(first_text)

            self._transition(TalkbackState.SPEAKING)
            self._emit("turn_started", {"turn_number": 1})

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
        self._emit("session_ended", {
            "reason": self._run_result.reason,
            "turns": self._run_result.turns,
            "total_duration_ms": self._run_result.total_duration_s * 1000,
        })

        return self._run_result

    async def _generate_response(
        self, conversation: ConversationManager, config: dict
    ) -> str:
        messages = conversation.get_messages()
        self._emit("llm_request_sent", {
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
                self._emit("llm_response_started", {
                    "time_to_first_token_ms": (time.monotonic() - t0) * 1000,
                })
                first_token = False

            chunk = chunker.feed(token)
            if chunk:
                audio = await self._tts.synthesize(chunk)
                if len(audio) > 0:
                    await self._player.enqueue(audio)
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda a=audio: sd.play(a, samplerate=16000, blocking=True)
                    )

        remaining = chunker.flush()
        if remaining:
            audio = await self._tts.synthesize(remaining)
            if len(audio) > 0:
                await self._player.enqueue(audio)
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda a=audio: sd.play(a, samplerate=16000, blocking=True)
                )

        response_text = "".join(full_response)
        self._emit("llm_response_complete", {
            "tokens": len(full_response),
            "latency_ms": (time.monotonic() - t0) * 1000,
            "text": response_text,
        })

        return response_text

    async def _generate_and_speak(
        self, conversation: ConversationManager, config: dict
    ) -> str:
        """Cancellable wrapper around _generate_response."""
        try:
            return await self._generate_response(conversation, config)
        except asyncio.CancelledError:
            self._llm.cancel()
            raise

    async def _handle_segment(self, segment) -> Optional[asyncio.Task]:
        """Process a speech segment based on current state.

        Returns an asyncio.Task if a new response was started, None otherwise.
        """
        if self.state == TalkbackState.LISTENING:
            text = await self._stt.transcribe_segment(segment.audio)
            if not text:
                return None
            turn = self._conversation.turn_count + 1
            self._emit("user_turn_complete", {"text": text, "turn_number": turn})
            self._conversation.add_user_turn(text)
            self._transition(TalkbackState.SPEAKING)
            self._emit("turn_started", {"turn_number": turn})
            task = asyncio.create_task(
                self._generate_and_speak(self._conversation, self._talkback_config)
            )
            self._response_task = task
            return task

        elif self.state == TalkbackState.SPEAKING:
            barge_cfg = self._talkback_config.get("barge_in", {})
            if not barge_cfg.get("enabled", True):
                return None
            if segment.duration_ms < barge_cfg.get("min_speech_ms", 120):
                return None

            if barge_cfg.get("require_speaker_match", True):
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self._embedder.extract, segment.audio
                )
                score = cosine_similarity(embedding, self._primary_embedding)
                threshold = barge_cfg.get("speaker_threshold", 0.5)
                if score < threshold:
                    self._emit("barge_in_rejected", {
                        "score": float(score), "threshold": threshold,
                    })
                    return None
            else:
                score = 1.0

            # Cancel current response
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                try:
                    await self._response_task
                except asyncio.CancelledError:
                    pass

            sd.stop()
            self._handle_barge_in(primary_score=float(score), speech_ms=segment.duration_ms)

            # Transcribe barge-in speech and start new response
            text = await self._stt.transcribe_segment(segment.audio)
            if not text:
                self._transition(TalkbackState.LISTENING)
                return None

            turn = self._conversation.turn_count + 1
            self._emit("user_turn_complete", {"text": text, "turn_number": turn})
            self._conversation.add_user_turn(text)
            self._transition(TalkbackState.SPEAKING)
            self._emit("turn_started", {"turn_number": turn})
            task = asyncio.create_task(
                self._generate_and_speak(self._conversation, self._talkback_config)
            )
            self._response_task = task
            return task

        return None

    async def _listen_loop(
        self,
        mic,
        vad,
        embedder,
        primary_embedding: np.ndarray,
    ) -> None:
        loop = asyncio.get_event_loop()
        mic_iter = mic.stream()

        def _next_chunk():
            try:
                return next(mic_iter)
            except StopIteration:
                return None

        while self._running:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                break

            # Apply AEC during playback to remove echo
            if self.state == TalkbackState.SPEAKING and self._aec is not None:
                frame_samples = self._aec.frame_samples
                cleaned_frames = []
                for i in range(0, len(chunk), frame_samples):
                    frame = chunk[i:i + frame_samples]
                    if len(frame) < frame_samples:
                        break
                    ref = self._player.get_reference_frame(frame_samples)
                    if ref is not None:
                        frame = self._aec.process_frame(frame, ref)
                    cleaned_frames.append(frame)
                if cleaned_frames:
                    chunk = np.concatenate(cleaned_frames)

            segments = vad.process_chunk(chunk)
            for seg in segments:
                self._last_speech_at = time.monotonic()
                await self._segment_queue.put(seg)

    def _handle_timeout(self, reason: str) -> None:
        self._running = False
        self._emit("watchdog_fired", {"reason": reason})
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
        self._emit("barge_in", {
            "during_state": prior_state,
            "primary_score": primary_score,
            "cut_at_ms": speech_ms,
        })

    async def _check_llm_available(self) -> bool:
        try:
            available = await self._llm.ping()
            if not available:
                self._emit("llm_unavailable", {})
            return available
        except Exception:
            self._emit("llm_unavailable", {})
            return False

    def stop(self) -> None:
        self._running = False
