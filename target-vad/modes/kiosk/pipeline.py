"""KioskPipeline — state machine for the wake-word talkback kiosk."""

import sys
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np

from core.audio.mic_stream import MicrophoneStream
from core.speaker.decision_smoother import DecisionSmoother
from core.speaker.embedder import EmbeddingExtractor
from core.speaker.verifier import cosine_similarity
from core.vad.silero_vad import SileroVAD, SpeechSegment
from modes.kiosk.session import Session
from modes.kiosk.wake_word import WakeWordDetector


class KioskPipeline:
    """Wake-word activated streaming kiosk with session-locked speaker filter.

    States:
      IDLE              — feeding chunks to wake detector
      AWAITING_SPEECH   — wake fired; feeding chunks to VAD; first speech segment
                          becomes the session-primary snapshot
      ACTIVE_SESSION    — feeding chunks to VAD, scoring segments against snapshot
    """

    def __init__(
        self,
        config: dict,
        on_primary_speech: Callable[[SpeechSegment, np.ndarray], None],
        on_session_started: Callable[[], None] = lambda: None,
        on_session_ended: Callable[[str], None] = lambda reason: None,
        on_event: Callable[[str, dict], None] = lambda event, payload: None,
        # Underscore kwargs are for test injection. Production code omits them.
        _mic: Optional[Any] = None,
        _vad: Optional[Any] = None,
        _embedder: Optional[Any] = None,
        _wake_detector: Optional[Any] = None,
    ):
        self.config = config
        self.on_primary_speech = on_primary_speech
        self.on_session_started = on_session_started
        self.on_session_ended = on_session_ended
        self.on_event = on_event

        kiosk_cfg = config["kiosk"]
        self._awaiting_speech_timeout_s = kiosk_cfg["awaiting_speech_timeout_s"]
        self._silence_timeout_s = kiosk_cfg["session_silence_timeout_s"]
        self._hard_timeout_s = kiosk_cfg["session_hard_timeout_s"]
        self._smoother_cfg = kiosk_cfg["decision_smoother"]

        self.mic = _mic or MicrophoneStream(config["core"]["audio"])
        self.vad = _vad or SileroVAD(config["core"]["vad"])
        self.embedder = _embedder or EmbeddingExtractor()
        self.wake_detector = _wake_detector or WakeWordDetector(
            kiosk_cfg["wake_phrase"], kiosk_cfg["wake_threshold"]
        )

        self._session: Optional[Session] = None
        self._wake_time: Optional[float] = None
        self._running = False

        # Warm up the ECAPA model so the first wake → session_started transition
        # doesn't include the model's cold-start latency (~1.3s on CPU).
        # Skip during tests (when _embedder was injected) — tests don't need warmup.
        if _embedder is None:
            try:
                _ = self.embedder.extract(np.zeros(12800, dtype=np.float32))
            except Exception:
                # Warmup failure is non-fatal; the model will lazy-load on first real use.
                pass

        self._state = "IDLE"

    def stop(self) -> None:
        """Signal run() to exit cleanly on the next loop iteration."""
        self._running = False

    def run(self) -> None:
        """Main mic loop. Blocks until stop() is called or KeyboardInterrupt.

        Note: this loop is driven entirely by mic chunks. Session timeouts
        (silence and hard) are checked on each chunk arrival in
        _handle_active_chunk, so they cannot fire if the mic stops producing
        chunks (e.g., disconnected). In practice the C10 always emits chunks
        (even silence), so this is a latent risk rather than a current bug.
        """
        self._running = True
        try:
            with self.mic:
                for chunk in self.mic.stream():
                    if not self._running:
                        break
                    self._handle_chunk(chunk)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            if self._state == "ACTIVE_SESSION":
                self._end_session("stopped")

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        """Route a single mic chunk based on current state."""
        if self._state == "IDLE":
            self._handle_idle_chunk(chunk)
        elif self._state == "AWAITING_SPEECH":
            self._handle_awaiting_speech_chunk(chunk)
        elif self._state == "ACTIVE_SESSION":
            self._handle_active_chunk(chunk)

    def _handle_idle_chunk(self, chunk: np.ndarray) -> None:
        wake_score = self.wake_detector.process(chunk)
        if wake_score is not None:
            self._safe_callback(self.on_event, "wake_detected",
                                {"phrase": self.config["kiosk"]["wake_phrase"], "score": wake_score})
            self._state = "AWAITING_SPEECH"
            self._wake_time = time.monotonic()
            self.vad.reset()

    def _handle_awaiting_speech_chunk(self, chunk: np.ndarray) -> None:
        # Check awaiting-speech timeout first
        assert self._wake_time is not None
        if time.monotonic() - self._wake_time >= self._awaiting_speech_timeout_s:
            # No speech within timeout window; abort back to IDLE
            self._state = "IDLE"
            self._wake_time = None
            self.wake_detector.reset()
            self._safe_callback(self.on_event, "session_ended",
                                {"reason": "awaiting_speech_timeout"})
            self._safe_callback(self.on_session_ended, "awaiting_speech_timeout")
            return

        # Feed chunk to VAD; first completed segment becomes the snapshot
        for segment in self.vad.process_chunk(chunk):
            self._start_session_from_segment(segment)
            return  # only the first segment matters; ignore any others (shouldn't be more in one chunk)

    def _start_session_from_segment(self, segment: SpeechSegment) -> None:
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception:
            # Embedding the first segment failed; abort to IDLE.
            self._state = "IDLE"
            self._wake_time = None
            self.wake_detector.reset()
            return
        now = time.monotonic()
        self._session = Session(
            primary_embedding=embedding,
            smoother=DecisionSmoother(**self._smoother_cfg),
            started_at=now,
            last_speech_at=now,
        )
        self._state = "ACTIVE_SESSION"
        self._wake_time = None
        self._safe_callback(self.on_event, "session_started",
                            {"snapshot_norm": float(np.linalg.norm(embedding))})
        self._safe_callback(self.on_session_started)
        # The first segment IS primary speech by definition (the speaker who just spoke
        # after the wake word). Fire on_primary_speech immediately without smoother voting,
        # so single-utterance sessions still produce a primary event.
        self._safe_callback(self.on_event, "segment_scored", {
            "score": 1.0,  # vs itself, by definition
            "duration_ms": float(segment.duration_ms),
            "decision": "match",
        })
        self._safe_callback(self.on_primary_speech, segment, embedding)

    def _handle_active_chunk(self, chunk: np.ndarray) -> None:
        # Check timeouts before processing more audio.
        # Hard is checked before silence: when both expire on the same chunk
        # (silence_timeout < hard_timeout), the absolute wall-clock limit wins.
        # This is a deliberate deviation from the spec's listed order.
        now = time.monotonic()
        assert self._session is not None
        if self._session.session_duration(now) >= self._hard_timeout_s:
            self._end_session("hard_timeout")
            return
        if self._session.silence_duration(now) >= self._silence_timeout_s:
            self._end_session("silence_timeout")
            return

        # Feed chunk to VAD, process any completed speech segments
        for segment in self.vad.process_chunk(chunk):
            self._process_session_segment(segment)

    def _process_session_segment(self, segment: SpeechSegment) -> None:
        assert self._session is not None
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception:
            return  # skip this segment, session continues
        score = cosine_similarity(embedding, self._session.primary_embedding)
        matched = self._session.smoother.update(score)
        self._safe_callback(self.on_event, "segment_scored", {
            "score": float(score),
            "duration_ms": float(segment.duration_ms),
            "decision": "match" if matched else "no_match",
        })
        if matched:
            self._session.last_speech_at = time.monotonic()
            self._safe_callback(self.on_primary_speech, segment, embedding)

    def _end_session(self, reason: str) -> None:
        self._session = None
        self._state = "IDLE"
        self._wake_time = None
        self.wake_detector.reset()
        self._safe_callback(self.on_event, "session_ended", {"reason": reason})
        self._safe_callback(self.on_session_ended, reason)

    def _safe_callback(self, fn: Callable, *args) -> None:
        """Invoke a callback; swallow + log exceptions so a buggy downstream
        handler doesn't crash the pipeline (per spec)."""
        try:
            fn(*args)
        except Exception as e:
            # In production, route to logger. Print to stderr for now.
            print(f"[kiosk callback error] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
