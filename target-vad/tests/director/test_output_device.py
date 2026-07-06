# tests/director/test_output_device.py
"""TTS output pinning (Director-10): resolve the ReSpeaker's PipeWire sink,
fail loud. Live validation 2026-07-06 found the original direct-ALSA open
races PipeWire's card reservation (works only while the sink is suspended),
so production pins via PIPEWIRE_NODE over the 'pipewire' PortAudio device —
and because a bogus PIPEWIRE_NODE silently falls back to the default sink,
the fail-loud guarantee lives HERE, in resolution against live pw-dump state.

Pure-function tests over canned pw-dump JSON — no PipeWire, no PortAudio."""

import json
from unittest.mock import MagicMock

import pytest

from modes.director.assembly import _pipewire_sinks, resolve_pipewire_sink

SINKS = [
    {"name": "alsa_output.platform-NVDA2014_00.hdmi-stereo",
     "description": "Built-in Audio Digital Stereo (HDMI)"},
    {"name": "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.iec958-stereo",
     "description": "ReSpeaker 4 Mic Array (UAC1.0) Digital Stereo (IEC958)"},
]


def test_matches_description_substring_case_insensitive():
    assert resolve_pipewire_sink("respeaker", SINKS) == SINKS[1]["name"]


def test_matches_node_name_substring():
    assert resolve_pipewire_sink("SEEED_ReSpeaker", SINKS) == SINKS[1]["name"]


def test_first_match_wins_deterministically():
    assert resolve_pipewire_sink("alsa_output", SINKS) == SINKS[0]["name"]


def test_no_match_raises_actionable_runtimeerror():
    with pytest.raises(RuntimeError) as exc:
        resolve_pipewire_sink("ReSpeaker", SINKS[:1])
    assert "ReSpeaker" in str(exc.value)
    assert "output_device" in str(exc.value)


PW_DUMP = json.dumps([
    {"type": "PipeWire:Interface:Node",
     "info": {"props": {"media.class": "Audio/Source",
                        "node.name": "alsa_input.usb-SEEED_ReSpeaker...analog-surround-51.6",
                        "node.description": "ReSpeaker 4 Mic Array (UAC1.0) Analog Surround 5.1"}}},
    {"type": "PipeWire:Interface:Node",
     "info": {"props": {"media.class": "Audio/Sink",
                        "node.name": "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.iec958-stereo",
                        "node.description": "ReSpeaker 4 Mic Array (UAC1.0) Digital Stereo (IEC958)"}}},
    {"type": "PipeWire:Interface:Device",
     "info": {"props": {"media.class": "Audio/Device"}}},
    {"type": "PipeWire:Interface:Node", "info": None},
])


def test_pipewire_sinks_filters_audio_sink_nodes(monkeypatch):
    import subprocess

    run = MagicMock(return_value=MagicMock(stdout=PW_DUMP))
    monkeypatch.setattr(subprocess, "run", run)
    sinks = _pipewire_sinks()
    assert sinks == [{
        "name": "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.iec958-stereo",
        "description": "ReSpeaker 4 Mic Array (UAC1.0) Digital Stereo (IEC958)",
    }]
    assert run.call_args[0][0][0] == "pw-dump"


def test_build_aec_disabled_returns_none():
    # Spec s3.3: aec.enabled false -> assembly passes aec=None to ingestion
    # (whose _apply_aec no-ops on None — existing tested path). Pin the
    # config edge here since the shipped config now relies on it.
    from modes.director.assembly import _build_aec
    assert _build_aec({"aec": {"enabled": False}}) is None
    assert _build_aec({}) is None
