# tests/director/conftest.py
"""Shared WakeGate fixtures/helpers for test_wakegate.py and
test_wakegate_hold.py (deduped fast-follow — the hold file used to carry
copies). The fixtures are pytest-injected; the plain helpers (make_segment,
make_gate, drive_one_cycle) are imported explicitly by the test modules."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.handoff import DirectorResult


@pytest.fixture
def base_config():
    return {
        "core": {
            "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 480},
            "vad": {
                "sample_rate": 16000,
                "speech_threshold": 0.5,
                "min_speech_duration_ms": 300,
                "padding_ms": 200,
            },
        },
        "kiosk": {
            "wake_phrase": "hey_jarvis",
            "wake_threshold": 0.5,
            "awaiting_speech_timeout_s": 5,
            "talkback": {"sample_rate_hz": 16000},
        },
    }


@pytest.fixture
def fake_mic():
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=None)
    return m


@pytest.fixture
def fake_vad():
    m = MagicMock()
    m.process_chunk = MagicMock(return_value=[])
    m.reset = MagicMock()
    return m


@pytest.fixture
def fake_embedder():
    m = MagicMock()
    m.extract = MagicMock(return_value=np.ones(192, dtype=np.float32) / np.sqrt(192))
    return m


@pytest.fixture
def fake_wake():
    m = MagicMock()
    m.process = MagicMock(return_value=None)
    m.reset = MagicMock()
    return m


@pytest.fixture
def fake_runtime():
    """A DirectorRuntime stub: .run(handoff) returns a DirectorResult."""
    m = MagicMock()
    m.run = MagicMock(
        return_value=DirectorResult(reason="silence_timeout", turns=2, total_duration_s=10.0)
    )
    return m


def make_segment(duration_ms: float = 1000.0) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, on_event=None):
    from modes.director.wakegate import WakeGate
    return WakeGate(
        config=base_config,
        runtime=fake_runtime,
        on_event=on_event or (lambda et, pl: None),
        _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder,
        _wake_detector=fake_wake,
    )


def drive_one_cycle(g, fake_wake, fake_vad, seg=None):
    """Drive ONE full wake->session cycle through g.run() with a finite mic.
    The mic yields a wake chunk then a first-segment chunk, then is exhausted, so
    g.run() runs the single session and exits. runtime.run is called from run()
    AFTER the wake mic generator is closed (single-consumer handoff)."""
    seg = seg or make_segment()
    g.mic.stream = MagicMock(return_value=iter([
        np.zeros(480, dtype=np.float32),   # chunk 1 -> wake detected
        np.zeros(480, dtype=np.float32),   # chunk 2 -> first speech segment
    ]))
    fake_wake.process.return_value = 0.87
    fake_vad.process_chunk.return_value = [seg]
    g.run()
    return seg
