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

import os
import sys
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np

from core.audio.doa_math import (circular_distance, circular_median,
                                 fraction_vote)
from core.audio.mic_stream import MicrophoneStream
from core.speaker.embedder import EmbeddingExtractor
from core.vad.silero_vad import SileroVAD, SpeechSegment
from modes.director.verify import verify_before_serve
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
        doa_tracker: Optional[Any] = None,
    ):
        self.config = config
        self.runtime = runtime
        self.on_event = on_event

        kiosk_cfg = config["kiosk"]
        self._awaiting_speech_timeout_s = kiosk_cfg["awaiting_speech_timeout_s"]
        self._talkback_config = kiosk_cfg.get("talkback", {})

        # Director-09 post-eject quiet-hold (port of lockout.py's idle half):
        # after a speaker_mismatch end, ignore wakes until the near field has
        # been quiet this long. Never permanent — quiet always clears it.
        self._hold_idle_after_s = float(
            self._talkback_config.get("lockout_idle_after_s", 5.0))
        self._hold_floor: Optional[float] = None
        self._hold_quiet_since: float = 0.0

        self.mic = _mic or MicrophoneStream(config["core"]["audio"])
        self.vad = _vad or SileroVAD(config["core"]["vad"])
        self.embedder = _embedder or EmbeddingExtractor()
        self.wake_detector = _wake_detector or WakeWordDetector(
            kiosk_cfg["wake_phrase"], kiosk_cfg["wake_threshold"]
        )

        self._wake_time: Optional[float] = None
        # Director-11 seed direction filter (live 2026-07-07 19:04: waking
        # while a podcast played let the podcast become the enrollment seed —
        # floor, voiceprint AND bearing enrolled on the bystander). The wake
        # phrase is the one utterance guaranteed to be the owner, so the
        # bearing is measured over ITS time window and candidate seeds must
        # come from that direction. No tracker / no samples = fail open
        # (first-segment-wins, exactly pre-D11).
        self._doa = doa_tracker
        self._doa_cfg = self._talkback_config.get("turn_gate", {}).get("doa", {})
        self._wake_bearing: Optional[float] = None
        self._wake_bearing_known = False   # None is a real outcome; compute once
        self._running = False
        # Set by _start_session_from_segment; consumed by run(). Lets us CLOSE the
        # wake mic generator before runtime.run so the Director's ingestion worker
        # is the SOLE mic consumer during the session (two concurrent stream()
        # generators on one MicrophoneStream starve each other on a real device).
        self._pending_handoff: Optional[Any] = None

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
        """Main loop. Blocks until stop() or KeyboardInterrupt. Each cycle: read
        the mic ONLY to detect wake + the first speech segment, then CLOSE that
        mic generator and hand off to runtime.run(handoff). Closing the wake
        generator first means the Director's ingestion worker is the sole mic
        consumer during the session — there is never a second parked stream()
        generator competing for the shared buffer. The mic device stays open
        across sessions (outer `with`); only the per-cycle wake generator closes.
        There is no watchdog here; the Director owns the single one."""
        self._running = True
        try:
            with self.mic:
                while self._running:
                    handoff = self._collect_handoff()
                    if handoff is None:
                        break
                    result = self.runtime.run(handoff)
                    self._reset_to_idle()
                    if (result.reason == "speaker_mismatch"
                            and getattr(result, "proximity_rms", 0.0) > 0.0):
                        self._hold_floor = result.proximity_rms
                        self._hold_quiet_since = time.monotonic()
                        if os.environ.get("TVAD_DIAG"):
                            print(f"[DIAG wakegate] hold engaged "
                                  f"floor={self._hold_floor:.4f} "
                                  f"quiet_needed={self._hold_idle_after_s}s",
                                  file=sys.stderr, flush=True)
                    self._safe_callback(self.on_event, "session_ended",
                                        {"reason": result.reason})
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def _collect_handoff(self) -> Optional[Any]:
        """Iterate the mic until wake + first speech segment produces a handoff,
        then return it. The local generator is CLOSED in `finally` so it is NOT
        left parked during the session (single mic consumer — see run())."""
        self._pending_handoff = None
        gen = self.mic.stream()
        try:
            for chunk in gen:
                if not self._running:
                    return None
                self._handle_chunk(chunk)
                if self._pending_handoff is not None:
                    handoff, self._pending_handoff = self._pending_handoff, None
                    return handoff
            return None
        finally:
            # Terminate the suspended generator (GeneratorExit) so it is not left
            # parked. Defensive: plain iterators (test fakes) have no close().
            close = getattr(gen, "close", None)
            if close is not None:
                close()

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        if self._state == "IDLE":
            self._handle_idle_chunk(chunk)
        elif self._state == "AWAIT_FIRST_SEGMENT":
            self._handle_await_chunk(chunk)

    def _handle_idle_chunk(self, chunk: np.ndarray) -> None:
        if self._hold_floor is not None:
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            now = time.monotonic()
            if rms >= self._hold_floor:
                self._hold_quiet_since = now      # still someone close -> keep holding
                return
            if now - self._hold_quiet_since < self._hold_idle_after_s:
                return                            # quiet, but not long enough yet
            self._hold_floor = None               # cleared; fall through to wake
            if os.environ.get("TVAD_DIAG"):
                print("[DIAG wakegate] hold cleared (near field quiet)",
                      file=sys.stderr, flush=True)

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
            ok = self._seed_from_wake_direction(segment)
            if os.environ.get("TVAD_DIAG"):
                print(f"[DIAG wakegate] seed candidate dur={segment.duration_ms:.0f}ms "
                      f"direction_ok={ok}", file=sys.stderr, flush=True)
            if not ok:
                continue                 # bystander-direction seed: keep waiting
            self._start_session_from_segment(segment)
            return  # only the first OWNER-DIRECTION segment matters

    def _seed_from_wake_direction(self, segment: SpeechSegment) -> bool:
        """Direction-filter the enrollment seed against the wake bearing
        (Director-11). True = enroll this segment; False = skip it and keep
        waiting (awaiting_speech_timeout is the backstop). Abstain (no
        tracker, no bearing, thin evidence) enrolls — fail open."""
        if self._doa is None:
            return True
        if not self._wake_bearing_known:
            # Lazily, once per wake: the phrase ends at _wake_time, so its
            # samples exist by the time the first candidate segment arrives.
            self._wake_bearing = self._compute_wake_bearing()
            self._wake_bearing_known = True
        now = time.monotonic()
        angles = self._doa.angles_between(now - segment.duration_ms / 1000.0, now)
        vote = fraction_vote(
            angles or (), self._wake_bearing,
            float(self._doa_cfg.get("cone_deg", 20.0)),
            float(self._doa_cfg.get("min_in_cone_fraction", 0.25)),
            int(self._doa_cfg.get("min_in_cone_samples", 3)))
        if vote is False and os.environ.get("TVAD_DIAG"):
            print(f"[DIAG wakegate] seed rejected: out of wake cone "
                  f"(bearing={self._wake_bearing:.0f}°, n={len(angles or ())})",
                  file=sys.stderr, flush=True)
        return vote is not False

    # Wake-bearing windows (s): the phrase spans ~WAKE_WINDOW before the
    # detector fires (+MARGIN for detector latency); AMBIENT is the baseline
    # span before the phrase.
    WAKE_WINDOW_S = 1.5
    WAKE_MARGIN_S = 0.3
    AMBIENT_WINDOW_S = 6.0

    def _compute_wake_bearing(self) -> Optional[float]:
        """Owner bearing from the wake phrase. A plain median over the wake
        window gets OUTVOTED by continuous background speech (live 2026-07-07
        19:46: bearing landed on the podcast, 90° from the user, while the
        array's LED tracked the user whenever they spoke). The wake event
        proves the OWNER spoke in this window, so when ambient speech exists
        the owner is the NOVEL cluster — samples deviating from the pre-wake
        ambient bearing by more than the cone. No novel cluster (owner
        co-located with the source, or unseen) -> None: a wrong bearing locks
        the owner out; an abstaining one degrades to D10 behavior."""
        w = self._wake_time
        ambient = self._doa.median_between(w - self.AMBIENT_WINDOW_S,
                                           w - self.WAKE_WINDOW_S)
        if ambient is None:
            bearing = self._doa.median_between(w - self.WAKE_WINDOW_S,
                                               w + self.WAKE_MARGIN_S)
            if os.environ.get("TVAD_DIAG"):
                shown = f"{bearing:.0f}°" if bearing is not None else "abstain"
                print(f"[DIAG wakegate] wake bearing: quiet ambient -> {shown}",
                      file=sys.stderr, flush=True)
            return bearing
        cone = float(self._doa_cfg.get("cone_deg", 20.0))
        angles = self._doa.angles_between(w - self.WAKE_WINDOW_S,
                                          w + self.WAKE_MARGIN_S) or ()
        novel = [a for a in angles if circular_distance(a, ambient) > cone]
        bearing = circular_median(novel) if len(novel) >= 2 else None
        if os.environ.get("TVAD_DIAG"):
            shown = f"{bearing:.0f}°" if bearing is not None else "abstain"
            print(f"[DIAG wakegate] wake bearing: ambient={ambient:.0f}° "
                  f"novel={len(novel)}/{len(angles)} -> {shown}",
                  file=sys.stderr, flush=True)
        return bearing

    def _start_session_from_segment(self, segment: SpeechSegment) -> None:
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception as e:
            # Infra failure on THIS segment only: stay in AWAIT so the next
            # utterance retries without a re-wake (live 2026-07-07 19:38: the
            # old silent reset-to-IDLE made the kiosk deaf after the wake —
            # three wakes, zero sessions, nothing printed). The
            # awaiting_speech_timeout backstop still bounds the phase.
            if os.environ.get("TVAD_DIAG"):
                print(f"[DIAG wakegate] seed embed failed ({e}); awaiting retry",
                      file=sys.stderr, flush=True)
            return

        # Verify-before-serve (Director-09 spec s5): split-half self-similarity.
        # Same-utterance halves of one speaker are highly self-similar; a noise/
        # garbage first segment is not — so 0.80 is honest HERE, while cross-
        # utterance verification (too noisy on short audio) is window 1's job.
        # Only for segments >= 1.0s: halves off the 300ms VAD floor are too
        # short to compare honestly and would false-refuse real users.
        sr = int(self.config["core"]["audio"]["sample_rate"])
        if len(segment.audio) >= sr:
            thr = self._talkback_config.get("verify_before_serve_threshold", 0.80)
            half = len(segment.audio) // 2
            try:
                emb_a = self.embedder.extract(segment.audio[:half])
                emb_b = self.embedder.extract(segment.audio[half:])
            except Exception as e:
                if os.environ.get("TVAD_DIAG"):     # infra failure == embed-fail path
                    print(f"[DIAG wakegate] half embed failed ({e}); awaiting retry",
                          file=sys.stderr, flush=True)
                return
            ok, score = verify_before_serve(emb_a, emb_b, thr)
            if not ok:
                # Contaminated/garbage seed (a user utterance mixed with
                # background speech fails split-half self-similarity). Stay in
                # AWAIT — the next utterance retries without a re-wake; the
                # refusal is LOUD via the verify_refused event (the old silent
                # reset left the user talking to a kiosk that had given up).
                self._safe_callback(self.on_event, "verify_refused",
                                    {"score": float(score)})
                return

        self._safe_callback(self.on_event, "session_started",
                            {"snapshot_norm": float(np.linalg.norm(embedding))})

        # Stage the handoff for run() to execute AFTER _collect_handoff closes the
        # wake mic generator — so runtime.run is NOT called from inside a parked
        # generator and the Director's ingestion is the sole mic consumer. run()
        # makes the single blocking runtime.run call and emits session_ended from
        # DirectorResult.reason (spec section 4a — the Director is the sole session
        # owner and the only end-reason authority).
        self._pending_handoff = DirectorHandoff(
            mic=self.mic,
            primary_embedding=embedding,
            first_segment=segment,
            config=self._talkback_config,
            vad=self.vad,
            embedder=self.embedder,
            wake_bearing=self._wake_bearing,
        )

    def _reset_to_idle(self) -> None:
        self._state = "IDLE"
        self._wake_time = None
        self._wake_bearing = None
        self._wake_bearing_known = False
        self.wake_detector.reset()

    def _safe_callback(self, fn: Callable, *args) -> None:
        """Invoke a callback; swallow + log exceptions so a buggy downstream
        handler doesn't crash the gate."""
        try:
            fn(*args)
        except Exception as e:
            print(f"[wakegate callback error] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
