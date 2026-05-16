"""Tests for prosody.py CLI orchestration with a stubbed analyzer + tiny synthetic WAV."""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import soundfile as sf

import prosody


def _word(start: float, end: float, w: str = "x") -> dict:
    return {"start": start, "end": end, "word": w, "probability": 0.9}


@pytest.fixture
def tmp_workspace():
    """A tmp dir with a tiny WAV + post-2A diarization JSON pointing at it."""
    d = tempfile.mkdtemp()
    wav_path = os.path.join(d, "session.wav")
    json_path = os.path.join(d, "session.diarization.json")
    config_path = os.path.join(d, "config.yaml")

    # 5 seconds of silence - analyzer is stubbed, so audio content doesn't matter.
    sf.write(wav_path, np.zeros(int(16000 * 5.0), dtype=np.float32), 16000)

    with open(config_path, "w") as f:
        f.write(
            "prosody:\n"
            "  pitch_min_hz: 80\n"
            "  pitch_max_hz: 400\n"
            "  frame_length_ms: 25\n"
            "  hop_length_ms: 10\n"
        )

    data = {
        "audio_file": wav_path,
        "duration_s": 5.0,
        "diarized_at": "2026-05-16T00:00:00Z",
        "config": {},
        "enrolled_users_matched": [{"id": "alice", "name": "Alice"}, {"id": "bob", "name": "Bob"}],
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker_id": "alice", "speaker": "Alice",
             "text": "hello", "words": [_word(0, 0.5)]},
            {"start": 2.0, "end": 4.0, "speaker_id": "bob", "speaker": "Bob",
             "text": "hi", "words": [_word(2.0, 2.3)]},
            {"start": 4.0, "end": 5.0, "speaker_id": "alice", "speaker": "Alice",
             "text": "ok", "words": [_word(4.0, 4.2)]},
        ],
        "passes_run": ["diarization", "transcription"],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)

    yield {"dir": d, "json": json_path, "wav": wav_path, "config": config_path}
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stub_analyzer(monkeypatch):
    """Replace analyzer.analyze_segment with a deterministic stub.

    Returns prosody dicts that vary by index so we can verify per-segment write.
    """
    calls = []

    def fake_analyze(audio_chunk, sample_rate, words, segment_duration, cfg):
        calls.append({"len_audio": len(audio_chunk), "n_words": len(words), "duration": segment_duration})
        idx = len(calls)
        return {
            "pitch_hz_median": 140.0 + idx * 10.0,
            "pitch_hz_std": 5.0,
            "pitch_range_hz": 20.0,
            "energy_db_mean": -25.0 + idx,
            "energy_db_range": 8.0,
            "speech_rate_wps": 2.0,
            "pause_ratio": 0.2,
        }

    monkeypatch.setattr(prosody, "analyze_segment", fake_analyze)
    return calls


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class TestProsodyOrchestration:
    def test_happy_path_attaches_prosody_per_segment(self, tmp_workspace, stub_analyzer):
        rc = prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 0
        data = _read(tmp_workspace["json"])
        assert all("prosody" in s for s in data["segments"])
        assert data["segments"][0]["prosody"]["pitch_hz_median"] == 150.0  # idx 1
        assert "prosody_baselines" in data
        assert "prosody_config" in data
        assert "prosody" in data["passes_run"]
        # Stub was called once per segment.
        assert len(stub_analyzer) == 3

    def test_out_path_leaves_input_unchanged(self, tmp_workspace, stub_analyzer):
        original = _read(tmp_workspace["json"])
        out_json = os.path.join(tmp_workspace["dir"], "out.json")
        rc = prosody.main([tmp_workspace["json"], "--out", out_json, "--config", tmp_workspace["config"]])
        assert rc == 0
        assert _read(tmp_workspace["json"]) == original
        assert "prosody_baselines" in _read(out_json)

    def test_audio_override_used_when_json_path_wrong(self, tmp_workspace, stub_analyzer):
        # Mutate JSON to have a wrong audio_file path; pass --audio explicitly.
        data = _read(tmp_workspace["json"])
        data["audio_file"] = "/does/not/exist.wav"
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = prosody.main(
            [tmp_workspace["json"], "--audio", tmp_workspace["wav"], "--config", tmp_workspace["config"]]
        )
        assert rc == 0

    def test_idempotent_rerun_skips_already_analyzed(self, tmp_workspace, stub_analyzer):
        prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        # After first run, stub_analyzer has 3 calls.
        prior_calls = len(stub_analyzer)
        prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        # Second run skips all 3 because prosody is already populated.
        assert len(stub_analyzer) == prior_calls

    def test_rerun_flag_forces_full_reanalysis(self, tmp_workspace, stub_analyzer):
        prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        prior_calls = len(stub_analyzer)
        prosody.main([tmp_workspace["json"], "--rerun", "--config", tmp_workspace["config"]])
        # --rerun re-analyzes all 3 segments.
        assert len(stub_analyzer) == prior_calls + 3

    def test_missing_transcription_pass_exit_2(self, tmp_workspace, stub_analyzer):
        data = _read(tmp_workspace["json"])
        data["passes_run"] = ["diarization"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_audio_file_missing_exit_2(self, tmp_workspace, stub_analyzer):
        os.remove(tmp_workspace["wav"])
        rc = prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2
