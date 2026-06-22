# tests/director/test_kiosk_entrypoint.py
"""kiosk.py wiring: build a WakeGate around a DirectorRuntime, with console
event prints emitted from ONE owner (no double [HANDOFF])."""

from unittest.mock import MagicMock

import numpy as np

from core.vad.silero_vad import SpeechSegment
from modes.talkback.handoff import DirectorResult


def _config():
    return {
        "core": {
            "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 480},
            "vad": {"sample_rate": 16000, "speech_threshold": 0.5,
                    "min_speech_duration_ms": 300, "padding_ms": 200},
        },
        "kiosk": {
            "wake_phrase": "hey_jarvis", "wake_threshold": 0.5,
            "awaiting_speech_timeout_s": 5,
            "talkback": {"sample_rate_hz": 16000},
        },
    }


def _segment(duration_ms=1000.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def test_build_wakegate_attaches_runtime_and_emits_events_from_one_owner():
    import kiosk

    runtime = MagicMock(run=MagicMock(
        return_value=DirectorResult(reason="silence_timeout", turns=1, total_duration_s=3.0)))
    console = MagicMock()

    fake_mic = MagicMock()
    fake_mic.__enter__ = MagicMock(return_value=fake_mic)
    fake_mic.__exit__ = MagicMock(return_value=None)
    fake_vad = MagicMock(process_chunk=MagicMock(return_value=[]), reset=MagicMock())
    fake_embedder = MagicMock(
        extract=MagicMock(return_value=np.ones(192, dtype=np.float32) / np.sqrt(192)))
    fake_wake = MagicMock(process=MagicMock(return_value=None), reset=MagicMock())

    gate = kiosk.build_wakegate(
        _config(), console, runtime=runtime,
        _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
    )

    # Drive one full cycle through run() with a finite mic:
    # wake -> snapshot -> (close wake gen) -> blocking handoff -> IDLE.
    gate.mic.stream = MagicMock(return_value=iter([
        np.zeros(480, dtype=np.float32),   # chunk 1 -> wake
        np.zeros(480, dtype=np.float32),   # chunk 2 -> first segment
    ]))
    fake_wake.process.return_value = 0.9
    fake_vad.process_chunk.return_value = [_segment()]
    gate.run()

    runtime.run.assert_called_once()
    assert gate._state == "IDLE"

    # The console saw the event tags, emitted from the single WakeGate owner.
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "[WAKE]" in printed
    assert "[SESSION STARTED]" in printed
    assert "[SESSION ENDED]" in printed
    assert "[IDLE]" in printed
    # No legacy double-handoff tag.
    assert "[HANDOFF]" not in printed
