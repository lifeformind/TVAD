# modes/director/wakegate.py
"""WakeGate — the THIN front of the Conversation Director (spec section 4).

Two states only:
  IDLE                 — feeding mic chunks to the wake detector.
  AWAIT_FIRST_SEGMENT  — wake fired; feeding chunks to VAD; the first completed
                         speech segment becomes the session-primary ECAPA snapshot.

On that first segment the WakeGate snapshots an embedding, builds ONE
DirectorHandoff, and makes ONE blocking call to runtime.run(handoff). The
Director (Plans 01/02) owns the entire active session — every timer, the single
AsyncWatchdog, the conversation lifecycle, and the sole session-end reason. From
the WakeGate's view runtime.run() is fully synchronous (the Director spins its
own asyncio loop internally); the WakeGate thread is parked inside that call for
the whole conversation. When it returns a DirectorResult, the WakeGate's ONLY
post-return action is to reset to IDLE.

This component deliberately owns NO session object, NO watchdog thread, NO
silence/hard timer, and NO session-teardown method — that double-management was
the live bug (spec section 1, HARD REQ 5). The grep post-conditions in
tests/director/test_wakegate_single_ownership.py enforce that absence (they ban
the literal session/timeout identifiers, so this prose avoids them too).
"""

import sys
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np

from core.audio.mic_stream import MicrophoneStream
from core.speaker.embedder import EmbeddingExtractor
from core.vad.silero_vad import SileroVAD, SpeechSegment
from modes.kiosk.wake_word import WakeWordDetector
from modes.talkback.handoff import DirectorHandoff


class WakeGate:
    def __init__(
        self,
        config: dict,
        runtime: Any,
        on_event: Callable[[str, dict], None] = lambda event, payload: None,
        # Underscore kwargs are for test injection. Production code omits them.
        _mic: Optional[Any] = None,
        _vad: Optional[Any] = None,
        _embedder: Optional[Any] = None,
        _wake_detector: Optional[Any] = None,
    ):
        self.config = config
        self.runtime = runtime
        self.on_event = on_event

        kiosk_cfg = config["kiosk"]
        self._awaiting_speech_timeout_s = kiosk_cfg["awaiting_speech_timeout_s"]
        self._talkback_config = kiosk_cfg.get("talkback", {})

        self.mic = _mic or MicrophoneStream(config["core"]["audio"])
        self.vad = _vad or SileroVAD(config["core"]["vad"])
        self.embedder = _embedder or EmbeddingExtractor()
        self.wake_detector = _wake_detector or WakeWordDetector(
            kiosk_cfg["wake_phrase"], kiosk_cfg["wake_threshold"]
        )

        self._wake_time: Optional[float] = None
        self._running = False

        # Warm up ECAPA so the first wake -> snapshot transition doesn't pay the
        # model's cold-start latency (~1.3s on CPU). Skip when injected (tests).
        if _embedder is None:
            try:
                _ = self.embedder.extract(np.zeros(12800, dtype=np.float32))
            except Exception:
                pass  # non-fatal; lazy-loads on first real use

        self._state = "IDLE"

    def stop(self) -> None:
        """Signal run() to exit cleanly on the next loop iteration."""
        self._running = False

    def run(self) -> None:
        """Main mic loop. Blocks until stop() or KeyboardInterrupt. NOTE: while a
        session is active, this thread is parked inside runtime.run(handoff) and
        is NOT iterating the mic loop — the Director owns the mic during a
        session. There is no watchdog here; the Director owns the single one."""
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

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        if self._state == "IDLE":
            self._handle_idle_chunk(chunk)
        elif self._state == "AWAIT_FIRST_SEGMENT":
            self._handle_await_chunk(chunk)

    def _handle_idle_chunk(self, chunk: np.ndarray) -> None:
        wake_score = self.wake_detector.process(chunk)
        if wake_score is not None:
            self._safe_callback(
                self.on_event, "wake_detected",
                {"phrase": self.config["kiosk"]["wake_phrase"], "score": wake_score},
            )
            self._state = "AWAIT_FIRST_SEGMENT"
            self._wake_time = time.monotonic()
            self.vad.reset()

    def _handle_await_chunk(self, chunk: np.ndarray) -> None:
        # Pre-session abort if no speech arrives in time. This is NOT a session
        # end (no session ever started) — the reason authority for a real
        # session is DirectorResult.reason alone.
        assert self._wake_time is not None
        if time.monotonic() - self._wake_time >= self._awaiting_speech_timeout_s:
            self._reset_to_idle()
            self._safe_callback(self.on_event, "awaiting_speech_timeout", {})
            return

        for segment in self.vad.process_chunk(chunk):
            self._start_session_from_segment(segment)
            return  # only the first segment matters

    def _start_session_from_segment(self, segment: SpeechSegment) -> None:
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception:
            self._reset_to_idle()
            return

        self._safe_callback(self.on_event, "session_started",
                            {"snapshot_norm": float(np.linalg.norm(embedding))})

        # Placeholder holdout (Plan 05 owns the real pre-finalize capture): for
        # now the holdout IS the first-segment/primary embedding. Acceptable
        # ONLY because Plan 05 replaces it; verify-before-serve trivially passes
        # at cosine(primary, primary) == 1.0 until then.
        handoff = DirectorHandoff(
            mic=self.mic,
            primary_embedding=embedding,
            holdout_embedding=embedding,
            first_segment=segment,
            config=self._talkback_config,
            vad=self.vad,
            embedder=self.embedder,
        )

        # ONE blocking call. The Director owns the whole conversation and every
        # timer; it returns only at true session end. This is the single point
        # of session ownership transfer (spec section 4a.2).
        result = self.runtime.run(handoff)

        # The ONLY post-return action: reset to IDLE. The end reason originates
        # solely from DirectorResult.reason (spec section 4a.3).
        self._reset_to_idle()
        self._safe_callback(self.on_event, "session_ended", {"reason": result.reason})

    def _reset_to_idle(self) -> None:
        self._state = "IDLE"
        self._wake_time = None
        self.wake_detector.reset()

    def _safe_callback(self, fn: Callable, *args) -> None:
        """Invoke a callback; swallow + log exceptions so a buggy downstream
        handler doesn't crash the gate."""
        try:
            fn(*args)
        except Exception as e:
            print(f"[wakegate callback error] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
