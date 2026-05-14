"""KioskPipeline — state machine for the wake-word talkback kiosk."""

import time
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
      IDLE             — feeding chunks to wake detector
      CAPTURING        — collecting wake_capture_tail_seconds of audio for snapshot
      ACTIVE_SESSION   — feeding chunks to VAD, scoring segments against snapshot
      ENDING           — transient cleanup before returning to IDLE
    """

    def __init__(
        self,
        config: dict,
        on_primary_speech: Callable[[SpeechSegment, np.ndarray], None],
        on_session_started: Callable[[], None] = lambda: None,
        on_session_ended: Callable[[str], None] = lambda reason: None,
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

        kiosk_cfg = config["kiosk"]
        self._tail_samples = int(
            kiosk_cfg["wake_capture_tail_seconds"] * config["core"]["audio"]["sample_rate"]
        )
        self._session_primary_threshold = kiosk_cfg["session_primary_threshold"]
        self._silence_timeout_s = kiosk_cfg["session_silence_timeout_s"]
        self._hard_timeout_s = kiosk_cfg["session_hard_timeout_s"]
        self._smoother_cfg = kiosk_cfg["decision_smoother"]

        self.mic = _mic or MicrophoneStream(config["core"]["audio"])
        self.vad = _vad or SileroVAD(config["core"]["vad"])
        self.embedder = _embedder or EmbeddingExtractor()
        self.wake_detector = _wake_detector or WakeWordDetector(
            kiosk_cfg["wake_phrase"], kiosk_cfg["wake_threshold"]
        )

        self._state = "IDLE"
        self._session: Optional[Session] = None
        self._capture_buffer = np.array([], dtype=np.float32)
        self._running = False

    def stop(self) -> None:
        """Signal run() to exit cleanly on the next loop iteration."""
        self._running = False

    def run(self) -> None:
        """Main mic loop. Blocks until stop() is called or KeyboardInterrupt."""
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
        elif self._state == "CAPTURING":
            self._handle_capturing_chunk(chunk)
        elif self._state == "ACTIVE_SESSION":
            self._handle_active_chunk(chunk)
        # ENDING is transient — handled inline by _end_session

    def _handle_idle_chunk(self, chunk: np.ndarray) -> None:
        wake_score = self.wake_detector.process(chunk)
        if wake_score is not None:
            self._state = "CAPTURING"
            self._capture_buffer = np.array([], dtype=np.float32)

    def _handle_capturing_chunk(self, chunk: np.ndarray) -> None:
        self._capture_buffer = np.concatenate([self._capture_buffer, chunk])
        if len(self._capture_buffer) >= self._tail_samples:
            snapshot_audio = self._capture_buffer[: self._tail_samples]
            self._start_session(snapshot_audio)

    def _start_session(self, snapshot_audio: np.ndarray) -> None:
        try:
            embedding = self.embedder.extract(snapshot_audio)
        except Exception:
            # Snapshot failed (e.g. all silence) — abort session, return to IDLE
            self._state = "IDLE"
            self._capture_buffer = np.array([], dtype=np.float32)
            self.wake_detector.reset()
            return
        now = time.monotonic()
        self._session = Session(
            primary_embedding=embedding,
            smoother=DecisionSmoother(**self._smoother_cfg),
            started_at=now,
            last_speech_at=now,
        )
        self.vad.reset()
        self._state = "ACTIVE_SESSION"
        self._safe_callback(self.on_session_started)

    def _handle_active_chunk(self, chunk: np.ndarray) -> None:
        # Check timeouts before processing more audio
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
        if matched:
            self._session.last_speech_at = time.monotonic()
            self._safe_callback(self.on_primary_speech, segment, embedding)

    def _end_session(self, reason: str) -> None:
        self._session = None
        self._state = "IDLE"
        self._capture_buffer = np.array([], dtype=np.float32)
        self.wake_detector.reset()
        self._safe_callback(self.on_session_ended, reason)

    def _safe_callback(self, fn: Callable, *args) -> None:
        """Invoke a callback; swallow + log exceptions so a buggy downstream
        handler doesn't crash the pipeline (per spec)."""
        try:
            fn(*args)
        except Exception as e:
            # In production, route to logger. Print to stderr for now.
            import sys
            print(f"[kiosk callback error] {type(e).__name__}: {e}", file=sys.stderr)
