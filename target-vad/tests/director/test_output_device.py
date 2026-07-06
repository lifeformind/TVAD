# tests/director/test_output_device.py
"""resolve_output_device (Director-10): pin TTS to the array, fail loud.

Pure-function tests over a fake sd.query_devices() table — no PortAudio."""

import pytest

from modes.director.assembly import resolve_output_device

DEVICES = [
    {"name": "NVIDIA: LG SDQHD (hw:0,3)", "max_output_channels": 2},
    {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)",
     "max_output_channels": 2},
    {"name": "pipewire", "max_output_channels": 64},
]


def test_none_passes_through():
    assert resolve_output_device(None, DEVICES) is None


def test_int_passes_through():
    assert resolve_output_device(5, DEVICES) == 5


def test_substring_match_is_case_insensitive():
    assert resolve_output_device("respeaker", DEVICES) == 1


def test_input_only_device_with_matching_name_is_skipped():
    devices = [
        {"name": "ReSpeaker 4 Mic Array (capture only)", "max_output_channels": 0},
        {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio", "max_output_channels": 2},
    ]
    assert resolve_output_device("ReSpeaker", devices) == 1


def test_first_output_capable_match_wins():
    devices = [
        {"name": "ReSpeaker A", "max_output_channels": 2},
        {"name": "ReSpeaker B", "max_output_channels": 2},
    ]
    assert resolve_output_device("ReSpeaker", devices) == 0


def test_no_match_raises_actionable_runtimeerror():
    with pytest.raises(RuntimeError) as exc:
        resolve_output_device("ReSpeaker", DEVICES[:1])
    assert "ReSpeaker" in str(exc.value)
    assert "output_device" in str(exc.value)


def test_build_aec_disabled_returns_none():
    # Spec s3.3: aec.enabled false -> assembly passes aec=None to ingestion
    # (whose _apply_aec no-ops on None — existing tested path). Pin the
    # config edge here since the shipped config now relies on it.
    from modes.director.assembly import _build_aec
    assert _build_aec({"aec": {"enabled": False}}) is None
    assert _build_aec({}) is None
