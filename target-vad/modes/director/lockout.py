"""De-risked session-hijack lockout (spec section 7).

Never a permanent lockout: the real user must not be ejected by short-segment
ECAPA noise (MEMORY: ecapa-short-segment-unreliable.md).
- 1st failed window      -> WARN (duck + caution), NOT eject.
- 2 consecutive failed windows AND a failed RMS proximity check -> EJECT.
- After EJECT, idle_after_s of continuous no-near-field-RMS -> IDLE (accept a
  fresh wake), so the user is never permanently locked out.
A passing window resets the miss streak.
"""

import enum
from typing import Optional

from .safety_net import SafetyVerdict


class LockoutAction(enum.Enum):
    NONE = "NONE"
    WARN = "WARN"
    EJECT = "EJECT"
    IDLE = "IDLE"


class Lockout:
    def __init__(self, idle_after_s: float = 5.0):
        self._idle_after_s = idle_after_s
        self._miss_streak = 0
        self._ejected = False
        self._quiet_since: Optional[float] = None

    def on_verdict(self, verdict: SafetyVerdict, rms_ok: bool) -> LockoutAction:
        if verdict.smoother_ok:
            self._miss_streak = 0
            return LockoutAction.NONE
        self._miss_streak += 1
        if self._miss_streak >= 2 and not rms_ok:
            self._ejected = True
            return LockoutAction.EJECT
        return LockoutAction.WARN

    def note_ejected_at(self, now: float) -> None:
        self._ejected = True
        self._quiet_since = now

    def on_idle_tick(self, now: float, near_field_rms_active: bool) -> Optional[LockoutAction]:
        if not self._ejected:
            return None
        if near_field_rms_active:
            self._quiet_since = now           # activity resets the quiet clock
            return None
        if self._quiet_since is None:
            self._quiet_since = now
            return None
        if now - self._quiet_since >= self._idle_after_s:
            return LockoutAction.IDLE
        return None
