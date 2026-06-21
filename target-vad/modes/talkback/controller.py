"""TalkbackController — full-duplex voice assistant state machine.

Owns the LISTENING ⇄ SPEAKING ⇄ BARGED_IN lifecycle for one conversation.
Called by KioskPipeline.run() via TalkbackHandoff; returns TalkbackResult.
"""

import asyncio
import enum
import os
import threading
import time
import uuid
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

from core.logging.jsonl_logger import EventLogger
from core.speaker.decision_smoother import DecisionSmoother
from core.speaker.verifier import cosine_similarity
from modes.talkback.chunker import SentenceChunker
from modes.talkback.conversation import ConversationManager
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult
from modes.talkback.llm import LlmClient
from modes.talkback.player import Player
from modes.talkback.stt import StreamingStt
from modes.talkback.tts import TtsEngine
from modes.talkback.watchdog import AsyncWatchdog
from modes.talkback.intent import Interjection, classify_interjection

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
        # Debug: when TVAD_DEBUG_AUDIO_DIR is set, dump the primary snapshot and
        # every gated turn segment to WAV so live audio can be re-embedded
        # offline (bench/ecapa_selftest.py) and compared to a clean recording.
        self._dump_dir = os.environ.get("TVAD_DEBUG_AUDIO_DIR")
        self._dump_n = 0
        self._session_id = ""
        # M-of-N speaker lockout: a sliding window over VERIFIED window scores.
        # Sustained mismatch (a hijacker) ends the session. Set in _run_async.
        self._gate_smoother: Optional[DecisionSmoother] = None
        self._gate_scored = 0
        # Rolling verification window: consecutive LISTENING turn audio
        # accumulates here until it reaches verify_window_ms, then it's embedded
        # and scored as one (ECAPA is unreliable below ~2s). See _passes_turn_gate.
        self._gate_audio: list = []
        self._gate_audio_ms = 0.0
        # Streaming playback: TTS audio is written frame-by-frame to a persistent
        # output stream so each played (post-gain) frame can be recorded as the
        # AEC reference (without this, AEC cancels against silence — a no-op).
        # _gain scales output for barge-in ducking; the generation counter
        # (below) cuts the in-flight utterance. Set up in _run_async.
        self._out_stream = None
        self._gain = 1.0
        # Playback runs one utterance at a time in an executor. _play_gen tags
        # each response; a barge-in bumps it so an in-flight (now stale) playback
        # stops at its next frame. _play_future is the running write job — a cut
        # awaits it before starting the new response, so the old and new writes
        # never touch the (non-thread-safe) output stream concurrently.
        self._play_gen = 0
        self._play_future = None
        # Held around every output-stream write AND around stream close, so the
        # stream is never closed (nor the session torn down) while a write is in
        # flight — concurrent PortAudio calls across threads segfault. This is
        # the interrupt-safe backstop: it works even when a KeyboardInterrupt
        # kills the event loop before the async drain can run.
        self._write_lock = threading.Lock()
        # Barge-in ducking/proximity, configured in _run_async from barge_in cfg.
        self._barge_duck_enabled = False
        self._proximity_enabled = False
        self._duck_level = 0.15
        self._proximity_rms = 0.0
        self._ducked = False
        # Interruption-resume state, set when a verified barge-in cuts an
        # in-progress answer. None when no resume is pending. _partial_response
        # tracks the answer text spoken so far (for the resume reference);
        # _current_query is the request being answered; _pending_steer is a
        # one-shot system instruction injected into the next LLM generation.
        self._interrupted: Optional[dict] = None
        self._partial_response = ""
        self._current_query = ""
        self._pending_steer: Optional[str] = None

    def _emit(self, event: str, payload: dict) -> None:
        self._logger.log(event, payload)
        if self._on_event:
            try:
                self._on_event(event, payload)
            except Exception:
                pass

    def _transition(self, new_state: TalkbackState) -> None:
        self.state = new_state
        if new_state == TalkbackState.LISTENING:
            # Kiosk is (re)entering the wait-for-user state: start the silence
            # grace window fresh, so a long prior reply never shortens it.
            self._last_speech_at = time.monotonic()

    def _silence_duration(self) -> float:
        """User-silence seconds — only accrues while we're waiting for the user
        (LISTENING). While the kiosk is THINKING/SPEAKING/BARGED_IN the user is
        expected to be quiet, so silence must not accrue (else a long reply ends
        the session mid-conversation)."""
        if self.state != TalkbackState.LISTENING:
            return 0.0
        return time.monotonic() - self._last_speech_at

    def _dump_wav(self, audio: np.ndarray, tag: str) -> None:
        """Write a float32 [-1,1] segment to 16kHz mono int16 WAV (debug only)."""
        if not self._dump_dir:
            return
        try:
            os.makedirs(self._dump_dir, exist_ok=True)
            path = os.path.join(self._dump_dir, f"{self._session_id}_{tag}.wav")
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
        except Exception:
            pass

    # 30ms playback frames: small enough for ~30ms duck latency, large enough
    # to keep output-stream writes cheap.
    PLAYBACK_FRAME_SAMPLES = 480

    async def _speak(self, audio: np.ndarray) -> None:
        """Play one utterance in an executor, tracking the job so a barge-in or
        shutdown can wait for it to finish before the stream is touched again.

        _play_future is intentionally NOT cleared on cancellation: cancelling the
        await does not stop the executor thread (run_in_executor jobs can't be
        cancelled once running), so the reference must survive for _drain_playback
        to await the thread's real completion before any stream teardown.
        """
        gen = self._play_gen
        self._play_future = asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio, gen
        )
        await self._play_future

    async def _drain_playback(self) -> None:
        """Stop in-flight playback and wait for the write thread to exit.

        Bumping the generation makes _play_audio break at its next frame; then we
        wait for the executor thread to actually return, so no sd.write is running
        when the output stream is closed or the mic stream is stopped (concurrent
        PortAudio calls across threads segfault).
        """
        self._play_gen += 1
        fut = self._play_future
        if fut is not None:
            try:
                await asyncio.shield(fut)
            except Exception:
                pass

    def _play_audio(self, audio: np.ndarray, gen: int) -> None:
        """Write one utterance to the output stream frame-by-frame (blocking).

        Runs in an executor. Applies the current gain (for barge-in ducking),
        records each played frame as the AEC reference, and bails immediately if
        a barge-in superseded this generation or the session ended.
        """
        if self._out_stream is None or len(audio) == 0:
            return
        frame = self.PLAYBACK_FRAME_SAMPLES
        for i in range(0, len(audio), frame):
            if not self._running or gen != self._play_gen:
                break
            gained = (audio[i:i + frame] * self._gain).astype(np.float32)
            # The lock makes write and close mutually exclusive; re-check the
            # stream inside it since teardown may have closed it while we waited.
            with self._write_lock:
                if self._out_stream is None or gen != self._play_gen:
                    break
                if self._player is not None:
                    self._player.record_reference(gained)
                try:
                    self._out_stream.write(gained)
                except Exception:
                    break

    def _close_out_stream(self) -> None:
        """Stop playback and close the output stream — synchronous and lock-
        guarded so it's safe even when a KeyboardInterrupt has killed the event
        loop (the async drain can't run then). Idempotent."""
        self._running = False
        self._play_gen += 1
        with self._write_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None

    def run(self, handoff: TalkbackHandoff) -> TalkbackResult:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._run_async(handoff))
        finally:
            # Interrupt-safe backstop: if KeyboardInterrupt unwound the loop, the
            # async finally may not have run — make sure no playback thread is
            # still writing the (about-to-be-torn-down) audio device.
            self._close_out_stream()
            loop.close()

    async def _run_async(self, handoff: TalkbackHandoff) -> TalkbackResult:
        self._started_at = time.monotonic()
        self._last_speech_at = self._started_at
        self._running = True
        self._transition(TalkbackState.LISTENING)
        self._segment_queue = asyncio.Queue()

        await self._llm.close()

        session_id = uuid.uuid4().hex[:12]
        self._session_id = session_id
        self._logger.start_session(session_id)
        self._dump_wav(handoff.first_segment.audio, "primary")

        config = handoff.config
        self._talkback_config = config
        self._primary_embedding = handoff.primary_embedding
        self._embedder = handoff.embedder
        silence_timeout = config.get("silence_timeout_s", 10.0)
        hard_timeout = config.get("hard_timeout_s", 300.0)

        # Per-session speaker lockout over verified window scores.
        gate_cfg = config.get("turn_gate", {})
        lockout_cfg = gate_cfg.get("lockout", {})
        self._gate_scored = 0
        self._gate_smoother = None
        self._gate_audio = []
        self._gate_audio_ms = 0.0
        if lockout_cfg.get("enabled", True):
            self._gate_smoother = DecisionSmoother(
                window_size=lockout_cfg.get("window_size", 3),
                min_matches=lockout_cfg.get("min_matches", 1),
                threshold=gate_cfg.get("speaker_threshold", 0.3),
            )

        aec_cfg = config.get("aec", {})
        if aec_cfg.get("enabled", False) and AecProcessor is not None:
            try:
                self._aec = AecProcessor(
                    sample_rate=config.get("sample_rate_hz", 16000),
                    frame_ms=config.get("frame_ms", 10),
                )
            except Exception:
                self._aec = None
        else:
            self._aec = None

        # Persistent output stream: playback writes frames here so each played
        # frame becomes the AEC reference and gain (ducking) can change mid-reply.
        self._gain = 1.0
        self._ducked = False
        self._play_gen = 0
        self._play_future = None
        self._interrupted = None
        self._partial_response = ""
        self._current_query = ""
        self._pending_steer = None
        sr = config.get("sample_rate_hz", 16000)
        self._out_stream = None
        try:
            self._out_stream = sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32",
                device=config.get("output_device"),
            )
            self._out_stream.start()
        except Exception:
            self._out_stream = None

        # Barge-in ducking + proximity pre-gate.
        barge_cfg = config.get("barge_in", {})
        prox_cfg = barge_cfg.get("proximity", {})
        self._duck_level = barge_cfg.get("duck_level", 0.15)
        self._proximity_enabled = prox_cfg.get("enabled", True)
        self._barge_duck_enabled = (
            barge_cfg.get("enabled", True) and self._proximity_enabled
        )
        # Ignore speech too quiet to be someone AT the kiosk. Auto-calibrate the
        # RMS floor from the primary enrollment segment (the user is at the
        # kiosk) unless an explicit threshold is configured.
        thr = prox_cfg.get("rms_threshold")
        if thr is None:
            first = handoff.first_segment
            primary_rms = (
                float(np.sqrt(np.mean(np.square(first.audio))))
                if first is not None and len(first.audio) else 0.0
            )
            thr = primary_rms * prox_cfg.get("rms_factor", 0.5)
        self._proximity_rms = thr

        self._conversation = ConversationManager(
            system_prompt=config.get("llm", {}).get(
                "system_prompt", "You are a concise voice assistant.",
            )
        )
        conversation = self._conversation

        if not await self._check_llm_available():
            self._emit("session_ended", {
                "reason": "llm_unavailable", "turns": 0, "total_duration_ms": 0,
            })
            return TalkbackResult(reason="llm_unavailable", turns=0, total_duration_s=0.0)

        watchdog_tick = config.get("watchdog", {}).get("tick_ms", 500) / 1000.0
        watchdog = AsyncWatchdog(
            tick_s=watchdog_tick,
            on_timeout=self._handle_timeout,
            get_silence_duration=self._silence_duration,
            get_session_duration=lambda: time.monotonic() - self._started_at,
            silence_timeout_s=silence_timeout,
            hard_timeout_s=hard_timeout,
        )

        self._emit("handoff_to_talkback", {
            "primary_embedding_norm": float(np.linalg.norm(handoff.primary_embedding)),
        })

        # Process first speech segment synchronously before starting listen loop
        first_text = await self._stt.transcribe_segment(handoff.first_segment.audio)
        response_task = None

        if first_text:
            self._last_speech_at = time.monotonic()
            self._emit("user_turn_complete", {"text": first_text, "turn_number": 1})
            conversation.add_user_turn(first_text)
            self._current_query = first_text
            self._transition(TalkbackState.SPEAKING)
            self._emit("turn_started", {"turn_number": 1})
            response_task = asyncio.create_task(
                self._generate_and_speak(conversation, config)
            )
            self._response_task = response_task

        # Start continuous mic listener and watchdog
        listen_task = asyncio.create_task(
            self._listen_loop(handoff.mic, handoff.vad, handoff.embedder, handoff.primary_embedding)
        )
        watchdog.start()

        try:
            while self._running:
                # Check if response task completed
                if response_task and response_task.done():
                    try:
                        assistant_text = response_task.result()
                        if assistant_text:
                            conversation.add_assistant_turn(assistant_text)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    self._transition(TalkbackState.LISTENING)
                    response_task = None
                    self._response_task = None

                # Check for new speech segments
                try:
                    segment = self._segment_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
                    continue

                new_task = await self._handle_segment(segment)
                if new_task is not None:
                    response_task = new_task

                if self._run_result is not None:
                    break
        finally:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
            if response_task and not response_task.done():
                response_task.cancel()
                try:
                    await response_task
                except asyncio.CancelledError:
                    pass
            # Wait for any in-flight playback write to finish BEFORE closing the
            # output stream (and before the pipeline stops the mic on return) —
            # concurrent PortAudio calls across threads segfault.
            await self._drain_playback()
            await watchdog.stop()
            self._close_out_stream()

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
        # One-shot steering (interruption answer / resume continuation): a
        # transient system note appended for THIS generation only.
        if self._pending_steer:
            messages = messages + [{"role": "system", "content": self._pending_steer}]
            self._pending_steer = None
        self._emit("llm_request_sent", {
            "messages_count": len(messages),
            "model": config.get("llm", {}).get("model", "unknown"),
        })

        chunker = SentenceChunker(
            max_chunk_chars=config.get("chunker", {}).get("max_chunk_chars", 120),
        )

        # New answer starts at full volume.
        self._gain = 1.0
        self._ducked = False
        self._partial_response = ""

        full_response = []
        t0 = time.monotonic()
        first_token = True

        async for token in self._llm.stream(messages):
            if not self._running:
                break
            full_response.append(token)
            self._partial_response = "".join(full_response)
            if first_token:
                self._emit("llm_response_started", {
                    "time_to_first_token_ms": (time.monotonic() - t0) * 1000,
                })
                first_token = False

            chunk = chunker.feed(token)
            if chunk:
                audio = await self._tts.synthesize(chunk)
                if len(audio) > 0:
                    await self._speak(audio)

        remaining = chunker.flush()
        if remaining:
            audio = await self._tts.synthesize(remaining)
            if len(audio) > 0:
                await self._speak(audio)

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

    async def _passes_turn_gate(self, segment) -> bool:
        """Speaker-lock NEW turns (LISTENING state) to the enrolled primary.

        ECAPA cosine on a single conversational turn (~0.3-1.5s) can't separate
        the enrolled speaker from a bystander — scores scatter for both. So we
        don't judge turns in isolation: consecutive turn audio accumulates into
        a rolling buffer, and once it reaches verify_window_ms (~2s, where ECAPA
        IS reliable) the whole window is embedded and scored at once. See
        docs/speaker-gate-measurement.md.

        Sub-window turns are accepted provisionally (served now, their audio fed
        into the next window) so the real speaker is never false-rejected for
        speaking briefly. A bystander's brief turns leak until the window fills,
        then the window rejects and the M-of-N lockout ends the session.
        """
        gate_cfg = self._talkback_config.get("turn_gate", {})
        if not gate_cfg.get("require_speaker_match", True):
            return True
        if self._primary_embedding is None or self._embedder is None:
            return True

        # Proximity pre-gate: a turn too quiet to be someone AT the kiosk is a
        # bystander — ignore it rather than serve it unverified while we wait for
        # the rolling window to fill. Closes the short-turn leak for far speakers.
        if self._proximity_enabled and self._proximity_rms > 0.0:
            rms = (
                float(np.sqrt(np.mean(np.square(segment.audio))))
                if len(segment.audio) else 0.0
            )
            if rms < self._proximity_rms:
                self._emit("turn_gate_ignored_far", {
                    "rms": rms, "threshold": self._proximity_rms,
                })
                return False

        self._gate_audio.append(segment.audio)
        self._gate_audio_ms += float(segment.duration_ms)
        window_ms = gate_cfg.get("verify_window_ms", 2000)

        # Not enough accumulated audio to embed reliably yet — serve the turn
        # provisionally; it still counts toward the next window's judgment.
        if self._gate_audio_ms < window_ms:
            self._emit("turn_gate_pending", {
                "accumulated_ms": self._gate_audio_ms,
                "window_ms": float(window_ms),
            })
            return True

        window_audio = np.concatenate(self._gate_audio)
        window_audio_ms = self._gate_audio_ms
        self._gate_audio = []
        self._gate_audio_ms = 0.0

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._embedder.extract, window_audio
        )
        score = float(cosine_similarity(embedding, self._primary_embedding))
        threshold = gate_cfg.get("speaker_threshold", 0.3)
        decision = "accept" if score >= threshold else "reject"
        self._dump_n += 1
        self._dump_wav(window_audio, f"win{self._dump_n:02d}_{score:.2f}")
        self._emit("turn_gate", {
            "score": score, "threshold": threshold, "decision": decision,
            "duration_ms": window_audio_ms,
        })

        # M-of-N lockout: feed every verified window score into the window. Once
        # it is full, a sustained mismatch (smoother no longer sees min_matches
        # accepts) means a different speaker has taken over — end the session
        # instead of letting them chip through one slip at a time.
        if self._gate_smoother is not None:
            matched = self._gate_smoother.update(score)
            self._gate_scored += 1
            if (not matched
                    and self._gate_scored >= self._gate_smoother.window_size):
                self._handle_speaker_lockout()
                return False

        return decision == "accept"

    def _handle_speaker_lockout(self) -> None:
        """End the session: recent verified turns no longer match the primary."""
        self._running = False
        self._emit("speaker_lockout", {
            "window_size": self._gate_smoother.window_size,
            "min_matches": self._gate_smoother.min_matches,
        })
        turns = self._conversation.turn_count if self._conversation else 0
        self._run_result = TalkbackResult(
            reason="speaker_lockout",
            turns=turns,
            total_duration_s=time.monotonic() - self._started_at,
        )

    async def _handle_segment(self, segment) -> Optional[asyncio.Task]:
        """Process a speech segment based on current state.

        Returns an asyncio.Task if a new response was started, None otherwise.
        """
        if self.state == TalkbackState.LISTENING:
            if not await self._passes_turn_gate(segment):
                return None
            text = await self._stt.transcribe_segment(segment.audio)
            if not text:
                return None
            turn = self._conversation.turn_count + 1
            self._emit("user_turn_complete", {"text": text, "turn_number": turn})
            self._conversation.add_user_turn(text)
            self._current_query = text
            # If we owe the user a resume offer, steer this reply to continue it.
            self._maybe_inject_resume_steer()
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

            # Proximity pre-gate: in a crowd, ignore speech too quiet to be
            # someone AT the kiosk (we may have ducked at onset — restore).
            if self._barge_duck_enabled and self._proximity_rms > 0.0:
                rms = (
                    float(np.sqrt(np.mean(np.square(segment.audio))))
                    if len(segment.audio) else 0.0
                )
                if rms < self._proximity_rms:
                    self._restore_volume()
                    self._emit("barge_in_ignored_far", {
                        "rms": rms, "threshold": self._proximity_rms,
                    })
                    return None

            # ECAPA can't verify a short interruption (see turn_gate); don't cut
            # the AI for un-verifiable audio — crowd-safe default.
            if segment.duration_ms < barge_cfg.get("verify_window_ms", 1200):
                self._restore_volume()
                self._emit("barge_in_rejected", {
                    "reason": "too_short_to_verify",
                    "duration_ms": float(segment.duration_ms),
                })
                return None

            if barge_cfg.get("require_speaker_match", True):
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self._embedder.extract, segment.audio
                )
                score = cosine_similarity(embedding, self._primary_embedding)
                threshold = barge_cfg.get("speaker_threshold", 0.3)
                if score < threshold:
                    self._restore_volume()
                    self._emit("barge_in_rejected", {
                        "score": float(score), "threshold": threshold,
                    })
                    return None
            else:
                score = 1.0

            # Speaker-verified primary interjection. Transcribe BEFORE deciding
            # to cut so backchannels ("okay") keep us talking while questions
            # ("why?") cut. (Reorder: STT used to run only after the cut.)
            text = await self._stt.transcribe_segment(segment.audio)

            # Note: empty/garbage STT classifies as BACKCHANNEL (stays SPEAKING),
            # a deliberate change from the old "empty -> LISTENING" cut: never
            # cut the reply on an un-transcribable blip.
            if classify_interjection(text) is Interjection.BACKCHANNEL:
                # Acknowledgment, not an interruption: un-duck, keep talking.
                # No cut, no history pollution, no resume offer.
                self._restore_volume()
                self._emit("barge_in_backchannel_ignored", {"text": text})
                return None

            # Genuine question/command — CUT the in-flight reply. Stop the old
            # playback thread and wait for it to exit before the new response
            # writes the stream, so two threads never touch it at once.
            await self._drain_playback()
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                try:
                    await self._response_task
                except asyncio.CancelledError:
                    pass

            self._handle_barge_in(
                primary_score=float(score), speech_ms=segment.duration_ms
            )

            # Remember the interrupted answer so we can offer to resume it.
            self._store_interruption()

            self._gain = 1.0
            self._ducked = False
            turn = self._conversation.turn_count + 1
            self._emit("user_turn_complete", {"text": text, "turn_number": turn})
            self._conversation.add_user_turn(text)
            self._current_query = text
            self._transition(TalkbackState.SPEAKING)
            self._emit("turn_started", {"turn_number": turn})
            task = asyncio.create_task(
                self._generate_and_speak(self._conversation, self._talkback_config)
            )
            self._response_task = task
            return task

        return None

    def _restore_volume(self) -> None:
        """Un-duck: return TTS to full volume after a non-cut interruption."""
        if self._ducked or self._gain != 1.0:
            self._gain = 1.0
            self._ducked = False
            self._emit("barge_in_restored", {})

    def _store_interruption(self) -> None:
        """Remember the answer that was cut off so we can offer to resume it.

        Captures the request being answered + the partial answer, preserves the
        partial in history (marked interrupted), and queues a steer so the
        upcoming reply answers the interjection and then offers to continue.
        """
        if not self._talkback_config.get("resume", {}).get("enabled", True):
            return
        query = (self._current_query or "").strip()
        partial = (self._partial_response or "").strip()
        if not query:
            return
        if partial and self._conversation is not None:
            self._conversation.add_assistant_turn(partial + " [interrupted]")
        self._interrupted = {"query": query, "partial": partial}
        self._pending_steer = (
            f'You were answering the user\'s request: "{query}". So far you had '
            f'said: "{partial}". The user interrupted with a new question. Answer '
            f'their new question briefly, then ask if they would like you to '
            f'continue with the earlier topic.'
        )
        self._emit("interruption_stored",
                   {"query": query, "partial_chars": len(partial)})
        self._emit("resume_pending", {})

    def _maybe_inject_resume_steer(self) -> None:
        """If a resume is pending, steer the next reply to continue (or drop) it.

        The LLM interprets the user's reply (yes/no) and either resumes the
        earlier explanation naturally or just answers normally. One-shot.
        """
        if self._interrupted is None:
            return
        query = self._interrupted["query"]
        partial = self._interrupted["partial"]
        self._pending_steer = (
            f'Earlier you offered to continue explaining "{query}" (you had '
            f'said: "{partial}"). Interpret the user\'s latest reply: if they '
            f'want you to continue, resume that explanation naturally (e.g. "As '
            f'I was saying...") and finish it; otherwise just respond normally.'
        )
        self._interrupted = None
        self._emit("resume_continued", {"query": query})

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

            # Duck at speech onset (before the segment endpoints) so the
            # interruption is captured at low echo and can be verified. Only for
            # near-field speech — distant crowd chatter never ducks the AI.
            if (self.state == TalkbackState.SPEAKING
                    and self._barge_duck_enabled
                    and not self._ducked
                    and getattr(vad, "is_speaking", False) is True):
                rms = (
                    float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
                )
                if rms >= self._proximity_rms:
                    self._gain = self._duck_level
                    self._ducked = True
                    self._emit("barge_in_ducked", {"rms": rms})

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
