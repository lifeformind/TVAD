"""Tests for the Session dataclass — holds session-primary state."""

import numpy as np

from core.speaker.decision_smoother import DecisionSmoother
from modes.kiosk.session import Session


def make_session(now: float = 1000.0) -> Session:
    return Session(
        primary_embedding=np.ones(192, dtype=np.float32) / np.sqrt(192),
        smoother=DecisionSmoother(window_size=3, min_matches=2, threshold=0.6),
        started_at=now,
        last_speech_at=now,
    )


class TestSession:
    def test_silence_duration(self):
        s = make_session(now=1000.0)
        assert s.silence_duration(now=1003.5) == 3.5

    def test_session_duration(self):
        s = make_session(now=1000.0)
        assert s.session_duration(now=1042.0) == 42.0

    def test_silence_resets_when_last_speech_advances(self):
        s = make_session(now=1000.0)
        s.last_speech_at = 1010.0
        assert s.silence_duration(now=1012.0) == 2.0
