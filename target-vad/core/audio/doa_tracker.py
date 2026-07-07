"""DoaTracker — continuous DOA sampling over the ReSpeaker USB control path.

DOAANGLE is the bearing of the CURRENT dominant sound (spike 2026-07-06), so
direction must be sampled WHILE speech happens and segments scored over their
own time span — a read at decision time sees whatever is loud NOW. A daemon
thread polls DOAANGLE + SPEECHDETECTED every poll_s into a bounded buffer;
readers take the circular median of speech-flagged samples in a window.

Fail-open contract (Director-11 spec s6): ANY USB error — at the start()
probe or mid-session — latches the tracker unavailable (logged once to
stderr) and every read returns None from then on; the cone gate abstains and
the kiosk degrades to Director-10 behavior. No retry: a dead control path
stays dead for the process; the next startup's probe reports it.

Process-lifetime, owned by kiosk.py: the owner bearing is calibrated from
the wake utterance, so the tracker must already be sampling before any
session exists."""

import sys
import threading
import time
from collections import deque

from core.audio import respeaker
from core.audio.doa_math import circular_median


class DoaTracker:
    def __init__(self, poll_s: float = 0.15, maxlen: int = 600,
                 reader=None, finder=None, clock=time.monotonic):
        self._poll_s = poll_s
        self._samples = deque(maxlen=maxlen)   # (t, angle_deg, speech_flag)
        self._reader = reader or respeaker.read_param
        self._finder = finder or respeaker.find
        self._clock = clock
        self._lock = threading.Lock()
        self._dev = None
        self._available = False
        self._running = False
        self._thread = None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        try:
            self._dev = self._finder()
            if self._dev is None:
                raise RuntimeError("ReSpeaker not found on USB (2886:0018)")
            self._reader(self._dev, "DOAANGLE")            # probe read
        except Exception as e:
            print(f"[doa] unavailable ({e}) — cone gate will abstain",
                  file=sys.stderr, flush=True)
            self._available = False
            return
        self._available = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="doa-tracker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            self.sample_once()
            time.sleep(self._poll_s)

    def sample_once(self) -> None:
        if not self._available:
            return
        try:
            angle = float(self._reader(self._dev, "DOAANGLE"))
            speech = int(self._reader(self._dev, "SPEECHDETECTED"))
        except Exception as e:
            print(f"[doa] read failed ({e}) — latching unavailable, "
                  "cone gate will abstain", file=sys.stderr, flush=True)
            self._available = False
            self._running = False
            return
        with self._lock:
            self._samples.append((self._clock(), angle, speech))

    def latest(self):
        """Newest (t, angle_deg, speech_flag) sample, or None."""
        if not self._available:
            return None
        with self._lock:
            return self._samples[-1] if self._samples else None

    def median_between(self, t0: float, t1: float):
        """Circular median of speech-flagged angles in [t0, t1], or None."""
        if not self._available:
            return None
        with self._lock:
            angles = [a for (t, a, s) in self._samples if t0 <= t <= t1 and s]
        return circular_median(angles) if angles else None
