"""PresenceMonitor — the pure per-frame core of the VisionWorker. Wraps a
classify_fn (frame -> PRESENT/ABSENT, may raise) and a PresenceDebouncer, and
returns a status ONLY when the emitted status changes (edge), so the worker emits
one OwnerPresenceEvent per change. A None frame or a classify error is UNAVAILABLE
(fail-safe: the reducer then leans on the audio silence timeout)."""
from typing import Optional

from modes.director.events import PresenceStatus


class PresenceMonitor:
    def __init__(self, classify_fn, debouncer):
        self._classify = classify_fn
        self._deb = debouncer
        self._emitted: PresenceStatus = PresenceStatus.ABSENT  # last status we returned as a change
        self._unavailable = False

    @property
    def current(self) -> str:
        """Latest stable status name (preview display; ABSENT until first emit)."""
        return self._emitted.name

    def observe(self, frame, now: float) -> Optional[PresenceStatus]:
        if frame is None:
            return self._go_unavailable()
        try:
            raw = self._classify(frame)           # PRESENT or ABSENT
        except Exception:                         # noqa: BLE001 — detector glitch
            return self._go_unavailable()
        if self._unavailable:
            # Recover: reset the debouncer so present/absent re-accrue from now.
            self._unavailable = False
            self._deb.reset()
            # On recovery, don't emit yet — just reset and let debouncer re-accrue.
            # The next status change (from re-accrual) will emit normally.
            self._emitted = PresenceStatus.ABSENT  # restore to initial state after reset
        detected = raw is PresenceStatus.PRESENT
        stable = (PresenceStatus.PRESENT if self._deb.update(detected, now) == "present"
                  else PresenceStatus.ABSENT)
        if stable is not self._emitted:
            self._emitted = stable
            return stable
        return None

    def _go_unavailable(self) -> Optional[PresenceStatus]:
        self._unavailable = True
        if self._emitted is not PresenceStatus.UNAVAILABLE:
            self._emitted = PresenceStatus.UNAVAILABLE
            return PresenceStatus.UNAVAILABLE
        return None
