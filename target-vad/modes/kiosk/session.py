"""Session — per-session state for the active kiosk speaker lock."""

from dataclasses import dataclass

import numpy as np

from core.speaker.decision_smoother import DecisionSmoother


@dataclass
class Session:
    """All state for a single ACTIVE_SESSION. Discarded on session end."""
    primary_embedding: np.ndarray   # 192-dim L2-normalized snapshot from wake-word audio
    smoother: DecisionSmoother
    started_at: float               # time.monotonic() at session start
    last_speech_at: float           # time.monotonic() at most recent matched speech

    def silence_duration(self, now: float) -> float:
        """Seconds since last matched primary-speech segment."""
        return now - self.last_speech_at

    def session_duration(self, now: float) -> float:
        """Seconds since session start."""
        return now - self.started_at
