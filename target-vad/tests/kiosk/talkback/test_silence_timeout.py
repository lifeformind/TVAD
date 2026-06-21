"""Silence timeout must only accrue while LISTENING, and reset when the kiosk
yields the floor — so a long assistant turn never ends the session mid-reply."""

import time
from unittest.mock import MagicMock

from modes.talkback.controller import TalkbackController, TalkbackState


def _ctrl():
    return TalkbackController(
        stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
        player=MagicMock(), logger=MagicMock(),
    )


def test_silence_does_not_accrue_while_speaking_or_thinking():
    c = _ctrl()
    c._last_speech_at = time.monotonic() - 60.0  # 60s since user spoke
    c.state = TalkbackState.SPEAKING
    assert c._silence_duration() == 0.0
    c.state = TalkbackState.BARGED_IN
    assert c._silence_duration() == 0.0


def test_silence_accrues_while_listening():
    c = _ctrl()
    c.state = TalkbackState.LISTENING
    c._last_speech_at = time.monotonic() - 5.0
    assert c._silence_duration() >= 4.0


def test_entering_listening_resets_the_silence_clock():
    c = _ctrl()
    c._last_speech_at = time.monotonic() - 60.0
    c._transition(TalkbackState.LISTENING)        # kiosk yields the floor
    assert c._silence_duration() < 1.0
