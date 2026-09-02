"""VisionWorker — the only camera-touching component. A background thread that
(1) opens the backend, (2) self-enrolls the owner's face at session start, then
(3) monitors presence at ~fps and emits OwnerPresenceEvent on debounced changes.
Cross-thread emit uses run_coroutine_threadsafe onto the runtime's loop. Any
failure degrades to UNAVAILABLE (fail-safe); it never raises into the session.

Camera ownership lives entirely here — the WakeGate stays camera-free."""
import asyncio
import sys
import threading
import time
from typing import Optional

from modes.director.events import OwnerPresenceEvent, PresenceStatus
from modes.director.vision.classify import PresenceDebouncer
from modes.director.vision.enroll import enroll_reference
from modes.director.vision.monitor import PresenceMonitor


class VisionWorker:
    def __init__(self, backend, bus, *, fps, present_after_s, absent_after_s,
                 enroll_frames, clock=time.monotonic, preview_sink=None,
                 open_attempts=5, open_retry_delay_s=0.6):
        self._backend = backend
        self._bus = bus
        self._preview_sink = preview_sink   # callable(frame, detail) | None
        self._open_attempts = open_attempts
        self._open_retry_delay_s = open_retry_delay_s
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._present_after_s = present_after_s
        self._absent_after_s = absent_after_s
        self._enroll_frames = enroll_frames
        self._clock = clock
        self._monitor: Optional[PresenceMonitor] = None   # None until enrolled
        self._enrolled = False
        self._unavailable_emitted = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop = None

    def _run_once(self, now: float) -> Optional[OwnerPresenceEvent]:
        """One synchronous step (testable). Enroll on first call; then monitor."""
        if not self._enrolled:
            ref = enroll_reference(self._backend.grab, self._backend.embed,
                                   n_frames=self._enroll_frames,
                                   max_attempts=self._enroll_frames * 5)
            self._enrolled = True
            if ref is None:
                # Can't see the owner -> UNAVAILABLE (never ABSENT off a bad enroll).
                if not self._unavailable_emitted:
                    self._unavailable_emitted = True
                    print("[vision] UNAVAILABLE for this session: owner face "
                          "enrollment failed (no face found) — walk-away "
                          "protection is OFF", file=sys.stderr, flush=True)
                    return OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, now)
                return None
            self._monitor = PresenceMonitor(
                self._backend.make_classify_fn(ref),
                PresenceDebouncer(self._present_after_s, self._absent_after_s))
            return None
        if self._monitor is None:
            return None                          # enroll failed; stay UNAVAILABLE
        frame = self._backend.grab()
        status = self._monitor.observe(frame, now)
        if self._preview_sink is not None and frame is not None:
            try:
                self._preview_sink(frame, {
                    "box": getattr(self._backend, "last_box", None),
                    "score": getattr(self._backend, "last_score", None),
                    "raw_present": getattr(self._backend, "last_raw_present", False),
                    "stable": self._monitor.current,
                })
            except Exception:              # noqa: BLE001 — preview is best-effort
                pass
        return OwnerPresenceEvent(status, now) if status is not None else None

    def _emit(self, ev: OwnerPresenceEvent) -> None:
        if self._bus is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._bus.emit(ev), self._loop)
        except Exception:                        # noqa: BLE001 — loop closing
            pass

    def _open_with_retry(self) -> bool:
        """V4L2 often refuses a re-open for a moment after the previous session
        released the device (live: one failed open() = UNAVAILABLE for the whole
        session, silently). Retry briefly; be LOUD on final failure."""
        for attempt in range(1, self._open_attempts + 1):
            if self._backend.open():
                return True
            if attempt < self._open_attempts and self._open_retry_delay_s > 0:
                time.sleep(self._open_retry_delay_s)
        print(f"[vision] UNAVAILABLE for this session: camera open failed "
              f"after {self._open_attempts} attempts — walk-away protection "
              "is OFF", file=sys.stderr, flush=True)
        return False

    def _loop_body(self) -> None:
        try:
            if not self._open_with_retry():
                self._emit(OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, self._clock()))
                return
            while self._running:
                t0 = self._clock()
                ev = self._run_once(t0)
                if ev is not None:
                    self._emit(ev)
                dt = self._period - (self._clock() - t0)
                if dt > 0:
                    time.sleep(dt)
        except Exception as exc:                 # noqa: BLE001 — never crash the session
            print(f"[vision] worker thread died: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            self._emit(OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, self._clock()))
        finally:
            self._backend.close()

    def start(self, loop) -> None:
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._loop_body, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
