"""Tests for diarize.main() early-exit error paths.

These tests exercise the argument-validation and config-validation paths that
fire before any model loading, so they run fast without requiring HF_TOKEN or
a GPU.
"""

import struct
import tempfile
import wave
import os

import pytest
import yaml

from diarize import main, EXIT_BAD_INPUT, EXIT_CONFIG_OR_MODEL


def _write_tiny_wav(path: str) -> None:
    """Write a minimal 1-channel 16 kHz PCM WAV (10 ms, silence)."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 160, *([0] * 160)))


def _write_config(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


@pytest.fixture
def tiny_wav(tmp_path):
    p = tmp_path / "test.wav"
    _write_tiny_wav(str(p))
    return str(p)


@pytest.fixture
def base_config() -> dict:
    """Minimal config with all required sections except diarization."""
    return {
        "core": {
            "paths": {"voiceprints_dir": str(tempfile.mkdtemp())},
            "vad": {},
            "audio": {},
            "speaker": {},
        }
    }


class TestMissingDiarizationBlock:
    def test_exits_3_when_diarization_block_absent(self, tiny_wav, tmp_path, base_config):
        cfg_path = str(tmp_path / "config.yaml")
        _write_config(cfg_path, base_config)  # no 'diarization:' key

        code = main([tiny_wav, "--config", cfg_path])

        assert code == EXIT_CONFIG_OR_MODEL  # == 3

    def test_friendly_message_on_missing_diarization_block(self, tiny_wav, tmp_path, base_config, capsys):
        cfg_path = str(tmp_path / "config.yaml")
        _write_config(cfg_path, base_config)

        main([tiny_wav, "--config", cfg_path])

        out = capsys.readouterr().out
        assert "diarization" in out.lower()
        # Must not be a raw Python traceback
        assert "KeyError" not in out


class TestMissingInputFile:
    def test_exits_2_when_input_file_missing(self, tmp_path, base_config):
        cfg_path = str(tmp_path / "config.yaml")
        _write_config(cfg_path, base_config)

        code = main([str(tmp_path / "nonexistent.wav"), "--config", cfg_path])

        assert code == EXIT_BAD_INPUT  # == 2
