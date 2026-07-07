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


class _FakeSd:
    """Stand-in sounddevice module for _assert_array_startup."""
    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return self._devices


_ARRAY_DEVICES = [
    {"name": "NVIDIA: HDMI 1", "max_output_channels": 8},
    {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio", "max_output_channels": 2},
]


def _cfg_with_output(output_device="ReSpeaker"):
    cfg = _config()
    cfg["kiosk"]["talkback"]["output_device"] = output_device
    return cfg


_ARRAY_SINKS = [
    {"name": "alsa_output.platform-NVDA2014_00.hdmi-stereo",
     "description": "Built-in Audio Digital Stereo (HDMI)"},
    {"name": "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.iec958-stereo",
     "description": "ReSpeaker 4 Mic Array (UAC1.0) Digital Stereo (IEC958)"},
]


def test_startup_assert_happy_path_pins_output_and_kills_agc(monkeypatch):
    # String pin resolves against PipeWire sinks (pw-dump), NOT PortAudio:
    # sounddevice is poisoned to prove the string path never imports it
    # (live 2026-07-06: the direct-ALSA open races PipeWire's card
    # reservation; pinning now happens via PIPEWIRE_NODE at stream open).
    import sys as _sys
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice", None)
    monkeypatch.setattr("modes.director.assembly._pipewire_sinks",
                        lambda: _ARRAY_SINKS)
    fake_dev = object()
    calls = []
    monkeypatch.setattr("core.audio.respeaker.find", lambda: fake_dev)
    monkeypatch.setattr("core.audio.respeaker.write_param",
                        lambda dev, name, value: calls.append((dev, name, value)))
    console = MagicMock()
    kiosk._assert_array_startup(_cfg_with_output(), console)
    assert calls == [(fake_dev, "AGCONOFF", 0)]
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "SEEED_ReSpeaker" in printed          # the resolved node name is shown


def test_startup_assert_missing_output_device_exits_4(monkeypatch):
    import pytest
    import kiosk

    monkeypatch.setattr("modes.director.assembly._pipewire_sinks",
                        lambda: _ARRAY_SINKS[:1])        # no ReSpeaker sink
    console = MagicMock()
    with pytest.raises(SystemExit) as exc:
        kiosk._assert_array_startup(_cfg_with_output(), console)
    assert exc.value.code == 4


def test_startup_assert_pwdump_failure_exits_4(monkeypatch):
    # pw-dump absent/broken -> the pin cannot be VERIFIED -> refuse to start
    # (unverifiable routing is the same hazard as wrong routing).
    import pytest
    import kiosk

    def _boom():
        raise OSError("pw-dump not found")

    monkeypatch.setattr("modes.director.assembly._pipewire_sinks", _boom)
    console = MagicMock()
    with pytest.raises(SystemExit) as exc:
        kiosk._assert_array_startup(_cfg_with_output(), console)
    assert exc.value.code == 4


def test_startup_assert_out_of_range_int_device_exits_4(monkeypatch):
    import sys as _sys
    import pytest
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice", _FakeSd(_ARRAY_DEVICES))
    console = MagicMock()
    with pytest.raises(SystemExit) as exc:
        kiosk._assert_array_startup(_cfg_with_output(output_device=99), console)
    assert exc.value.code == 4


def test_startup_assert_negative_int_device_exits_4(monkeypatch):
    # Python negative indexing must not wrap to a real device and print a
    # false "pinned" checkmark.
    import sys as _sys
    import pytest
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice", _FakeSd(_ARRAY_DEVICES))
    console = MagicMock()
    with pytest.raises(SystemExit) as exc:
        kiosk._assert_array_startup(_cfg_with_output(output_device=-1), console)
    assert exc.value.code == 4


def test_startup_assert_agc_failure_is_nonfatal(monkeypatch):
    import sys as _sys
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice", _FakeSd(_ARRAY_DEVICES))
    monkeypatch.setattr("core.audio.respeaker.find", lambda: None)  # array USB absent
    console = MagicMock()
    kiosk._assert_array_startup(_cfg_with_output(), console)        # must not raise
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "AGC" in printed                                          # loud warning


def test_doa_probe_failure_warns_but_does_not_exit(monkeypatch):
    # Sink resolve + AGC fine; DOAANGLE read raises -> startup must NOT exit
    # (fail-open: the cone abstains; D10 behavior).
    import kiosk

    cfg = _config()
    cfg["kiosk"]["talkback"]["turn_gate"] = {"doa": {"enabled": True}}
    fake_dev = object()
    monkeypatch.setattr("core.audio.respeaker.find", lambda: fake_dev)
    monkeypatch.setattr("core.audio.respeaker.write_param",
                        lambda dev, name, value: None)

    def _read_param(dev, name):
        raise RuntimeError("errno 13")
    monkeypatch.setattr("core.audio.respeaker.read_param", _read_param)

    console = MagicMock()
    kiosk._assert_array_startup(cfg, console)        # no SystemExit
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "DOA unavailable" in printed


def test_doa_probe_success_prints_bearing(monkeypatch):
    import kiosk

    cfg = _config()
    cfg["kiosk"]["talkback"]["turn_gate"] = {"doa": {"enabled": True}}
    fake_dev = object()
    monkeypatch.setattr("core.audio.respeaker.find", lambda: fake_dev)
    monkeypatch.setattr("core.audio.respeaker.write_param",
                        lambda dev, name, value: None)
    monkeypatch.setattr("core.audio.respeaker.read_param",
                        lambda dev, name: 143 if name == "DOAANGLE" else 0)

    console = MagicMock()
    kiosk._assert_array_startup(cfg, console)
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "DOA control readable" in printed


def test_startup_assert_null_output_device_skips_pin_check(monkeypatch):
    import sys as _sys
    import kiosk

    # sys.modules["sounddevice"] = None makes ANY `import sounddevice`
    # raise ImportError — structurally proving the null-output_device path
    # never touches it (on the real kiosk the package imports fine, so a
    # regression would otherwise pass silently). AGC assert still runs and
    # here fails softly (array absent).
    monkeypatch.setitem(_sys.modules, "sounddevice", None)
    monkeypatch.setattr("core.audio.respeaker.find", lambda: None)
    console = MagicMock()
    kiosk._assert_array_startup(_cfg_with_output(output_device=None), console)
