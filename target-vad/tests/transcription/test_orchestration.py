"""Tests for transcribe.py orchestration with a stub WhisperRunner.

The CLI's `main()` is invoked directly with argv lists. The WhisperRunner class
is replaced with a stub that records calls and returns canned (text, words)
tuples — this isolates the JSON-read-modify-write flow from any real model.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import soundfile as sf

import transcribe


SR = 16000


@pytest.fixture
def tmp_workspace():
    """A tmp dir with a 90s mono 16kHz WAV and a 2-segment diarization JSON next to it."""
    d = tempfile.mkdtemp()
    wav_path = os.path.join(d, "session.wav")
    sf.write(wav_path, np.zeros(90 * SR, dtype=np.float32), SR)
    json_path = os.path.join(d, "session.diarization.json")
    json_data = {
        "audio_file": wav_path,
        "duration_s": 90.0,
        "diarized_at": "2026-05-15T10:00:00Z",
        "config": {"pyannote_pipeline": "p", "identification_threshold": 0.55},
        "enrolled_users_matched": [],
        "segments": [
            {"start": 0.0, "end": 10.0, "speaker_id": "siddharth", "speaker": "Siddharth Jain"},
            {"start": 10.0, "end": 25.0, "speaker_id": "unknown", "speaker": "unknown"},
        ],
        "passes_run": ["diarization"],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    yield {"dir": d, "wav": wav_path, "json": json_path}
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stub_runner(monkeypatch):
    """Patch WhisperRunner so no real model loads. Returns a list of call records."""
    calls = []

    class StubRunner:
        def __init__(self, model, language, device, compute_type):
            self.model = model
            self.language = language

        def load(self):
            pass

        def transcribe(self, audio, initial_prompt):
            calls.append({
                "audio_samples": int(len(audio)),
                "initial_prompt": initial_prompt,
            })
            # Return a deterministic transcription based on the call index
            idx = len(calls)
            return (f"text_{idx}", [
                {"start": 0.0, "end": 1.0, "word": f"word{idx}", "probability": 0.9}
            ])

    monkeypatch.setattr(transcribe, "WhisperRunner", StubRunner)
    return calls


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class TestTranscribeOrchestration:
    def test_fresh_transcription_attaches_text_and_words(self, tmp_workspace, stub_runner):
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["text"] == "text_1"
        assert data["segments"][1]["text"] == "text_2"
        assert data["segments"][0]["words"][0]["word"] == "word1"
        # Word timestamps are WAV-absolute: segment 0 starts at 0.0, so word stays at 0.0
        assert data["segments"][0]["words"][0]["start"] == 0.0
        # Segment 1 starts at 10.0, so the clip-relative 0.0 becomes 10.0 absolute
        assert data["segments"][1]["words"][0]["start"] == 10.0
        assert data["segments"][1]["words"][0]["end"] == 11.0
        assert len(stub_runner) == 2

    def test_passes_run_gains_transcription(self, tmp_workspace, stub_runner):
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["passes_run"] == ["diarization", "transcription"]

    def test_transcription_config_block_written(self, tmp_workspace, stub_runner):
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml", "--model", "small"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        cfg = data["transcription_config"]
        assert cfg["model"] == "small"
        assert cfg["language"] == "en"
        assert cfg["initial_prompt_chars"] == 200
        assert "transcribed_at" in cfg

    def test_skip_already_transcribed_segments(self, tmp_workspace, stub_runner):
        """A segment with existing non-null text is skipped; only the un-transcribed one runs."""
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["text"] = "already done"
        data["segments"][0]["words"] = []
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["text"] == "already done"
        assert data["segments"][1]["text"] == "text_1"   # only the second segment was transcribed
        assert len(stub_runner) == 1

    def test_retranscribe_flag_processes_all_segments(self, tmp_workspace, stub_runner):
        """--retranscribe overrides skipping; both segments re-run from empty context."""
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["text"] = "stale"
        data["segments"][0]["words"] = []
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml", "--retranscribe"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["text"] == "text_1"
        assert data["segments"][1]["text"] == "text_2"
        assert len(stub_runner) == 2

    def test_rolling_context_carries_across_segments(self, tmp_workspace, stub_runner):
        """The second transcribe call's initial_prompt contains the first segment's text."""
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        assert stub_runner[0]["initial_prompt"] == ""
        assert "text_1" in stub_runner[1]["initial_prompt"]

    def test_skipped_segments_still_contribute_to_context(self, tmp_workspace, stub_runner):
        """Segment 0 already has text 'hello world' — segment 1's prompt should include it."""
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["text"] = "hello world"
        data["segments"][0]["words"] = []
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        assert "hello world" in stub_runner[0]["initial_prompt"]

    def test_missing_audio_file_exits_2(self, tmp_workspace, stub_runner):
        os.remove(tmp_workspace["wav"])
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_explicit_audio_override(self, tmp_workspace, stub_runner):
        """--audio overrides the path embedded in the JSON."""
        alt_wav = os.path.join(tmp_workspace["dir"], "alt.wav")
        sf.write(alt_wav, np.zeros(90 * SR, dtype=np.float32), SR)
        os.remove(tmp_workspace["wav"])  # original WAV gone
        rc = transcribe.main([tmp_workspace["json"], "--audio", alt_wav, "--config", "config.yaml"])
        assert rc == 0

    def test_explicit_out_path_does_not_modify_input(self, tmp_workspace, stub_runner):
        out_path = os.path.join(tmp_workspace["dir"], "out.json")
        rc = transcribe.main([
            tmp_workspace["json"], "--out", out_path, "--config", "config.yaml",
        ])
        assert rc == 0
        assert os.path.exists(out_path)
        # Original input untouched (no text added)
        original = _read_json(tmp_workspace["json"])
        assert "text" not in original["segments"][0]

    def test_malformed_json_exits_2(self, tmp_workspace, stub_runner):
        with open(tmp_workspace["json"], "w") as f:
            f.write("{not json")
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_missing_segments_field_exits_2(self, tmp_workspace, stub_runner):
        bad = {"audio_file": tmp_workspace["wav"], "duration_s": 90.0}
        with open(tmp_workspace["json"], "w") as f:
            json.dump(bad, f)
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_per_segment_exception_marks_segment_null(self, tmp_workspace, monkeypatch):
        """If runner.transcribe raises, that segment gets text=None, words=[], exit 0 (other segments proceed)."""
        call_count = [0]

        class PartialFailingRunner:
            def __init__(self, *a, **kw):
                pass

            def load(self):
                pass

            def transcribe(self, audio, initial_prompt):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("simulated whisper crash on segment 0")
                return ("ok_text", [{"start": 0.0, "end": 1.0, "word": "ok", "probability": 0.9}])

        monkeypatch.setattr(transcribe, "WhisperRunner", PartialFailingRunner)
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["text"] is None
        assert data["segments"][0]["words"] == []
        # Second segment still transcribed normally
        assert data["segments"][1]["text"] == "ok_text"

    def test_model_load_failure_exits_3(self, tmp_workspace, monkeypatch):
        """Model load failure (HF down, bad model) surfaces as exit 3, not silent all-null."""
        class FailingLoadRunner:
            def __init__(self, *a, **kw):
                pass

            def load(self):
                raise RuntimeError("simulated model download failure")

            def transcribe(self, audio, initial_prompt):
                raise AssertionError("transcribe should never be called when load fails")

        monkeypatch.setattr(transcribe, "WhisperRunner", FailingLoadRunner)
        rc = transcribe.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 3
        # JSON should NOT have been modified (load failure exits before write)
        data = _read_json(tmp_workspace["json"])
        assert "text" not in data["segments"][0]
