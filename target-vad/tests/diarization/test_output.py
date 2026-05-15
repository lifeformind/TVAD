"""Tests for JSON and RTTM output writers (post in-session enrollment refactor)."""

import json
import os
import tempfile

import pytest

from modes.diarization.output import (
    DiarizationSegment,
    write_json,
    write_rttm,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_segments():
    return [
        DiarizationSegment(start=0.42, end=3.81, speaker_id="siddharth", speaker="Siddharth Jain"),
        DiarizationSegment(start=3.81, end=5.10, speaker_id="unknown", speaker="unknown"),
        DiarizationSegment(start=5.20, end=7.45, speaker_id="siddharth", speaker="Siddharth Jain"),
    ]


class TestWriteJson:
    def test_roundtrip_basic(self, temp_dir, sample_segments):
        out = os.path.join(temp_dir, "out.json")
        write_json(
            out,
            audio_file="session.wav",
            duration_s=2734.51,
            diarized_at="2026-05-15T10:23:01Z",
            config={"pyannote_pipeline": "pyannote/speaker-diarization-3.1", "identification_threshold": 0.55},
            segments=sample_segments,
        )
        with open(out) as f:
            data = json.load(f)

        assert data["audio_file"] == "session.wav"
        assert data["duration_s"] == pytest.approx(2734.51)
        assert data["diarized_at"] == "2026-05-15T10:23:01Z"
        assert data["segments"] == [
            {"start": 0.42, "end": 3.81, "speaker_id": "siddharth", "speaker": "Siddharth Jain"},
            {"start": 3.81, "end": 5.10, "speaker_id": "unknown", "speaker": "unknown"},
            {"start": 5.20, "end": 7.45, "speaker_id": "siddharth", "speaker": "Siddharth Jain"},
        ]

    def test_enrolled_users_matched_objects(self, temp_dir):
        segments = [
            DiarizationSegment(0.0, 1.0, "alice_smith", "Alice"),
            DiarizationSegment(1.0, 2.0, "bob", "Bob"),
            DiarizationSegment(2.0, 3.0, "alice_smith", "Alice"),
            DiarizationSegment(3.0, 4.0, "unknown", "unknown"),
            DiarizationSegment(4.0, 5.0, "alice_jones", "Alice"),
        ]
        out = os.path.join(temp_dir, "out.json")
        write_json(out, audio_file="a.wav", duration_s=5.0, diarized_at="t", config={}, segments=segments)
        with open(out) as f:
            data = json.load(f)
        assert data["enrolled_users_matched"] == [
            {"id": "alice_smith", "name": "Alice"},
            {"id": "bob", "name": "Bob"},
            {"id": "alice_jones", "name": "Alice"},
        ]

    def test_empty_segments_writes_empty_list(self, temp_dir):
        out = os.path.join(temp_dir, "out.json")
        write_json(out, audio_file="a.wav", duration_s=10.0, diarized_at="t", config={}, segments=[])
        with open(out) as f:
            data = json.load(f)
        assert data["segments"] == []
        assert data["enrolled_users_matched"] == []

    def test_unknown_only_no_enrolled_users(self, temp_dir):
        segments = [
            DiarizationSegment(0.0, 1.0, "unknown", "unknown"),
            DiarizationSegment(1.0, 2.0, "unknown", "unknown"),
        ]
        out = os.path.join(temp_dir, "out.json")
        write_json(out, audio_file="a.wav", duration_s=2.0, diarized_at="t", config={}, segments=segments)
        with open(out) as f:
            data = json.load(f)
        assert data["enrolled_users_matched"] == []


class TestWriteRttm:
    def test_basic_rttm_uses_speaker_id(self, temp_dir, sample_segments):
        out = os.path.join(temp_dir, "out.rttm")
        write_rttm(out, audio_file_id="session", segments=sample_segments)
        with open(out) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 3
        parts0 = lines[0].split()
        assert parts0[0] == "SPEAKER"
        assert parts0[1] == "session"
        assert parts0[2] == "1"
        assert float(parts0[3]) == pytest.approx(0.42)
        assert float(parts0[4]) == pytest.approx(3.81 - 0.42)
        assert parts0[5] == "<NA>"
        assert parts0[6] == "<NA>"
        assert parts0[7] == "siddharth"   # speaker_id, not display name
        assert parts0[8] == "<NA>"
        assert parts0[9] == "<NA>"

    def test_rttm_unknown_label(self, temp_dir):
        out = os.path.join(temp_dir, "out.rttm")
        segs = [DiarizationSegment(1.0, 2.5, "unknown", "unknown")]
        write_rttm(out, audio_file_id="x", segments=segs)
        with open(out) as f:
            line = f.read().strip()
        parts = line.split()
        assert parts[7] == "unknown"
        assert float(parts[4]) == pytest.approx(1.5)

    def test_rttm_two_alices_keep_separate_ids(self, temp_dir):
        out = os.path.join(temp_dir, "out.rttm")
        segs = [
            DiarizationSegment(0.0, 1.0, "alice_smith", "Alice"),
            DiarizationSegment(1.0, 2.0, "alice_jones", "Alice"),
        ]
        write_rttm(out, audio_file_id="x", segments=segs)
        with open(out) as f:
            lines = f.read().strip().split("\n")
        assert lines[0].split()[7] == "alice_smith"
        assert lines[1].split()[7] == "alice_jones"

    def test_empty_segments_writes_empty_file(self, temp_dir):
        out = os.path.join(temp_dir, "out.rttm")
        write_rttm(out, audio_file_id="x", segments=[])
        with open(out) as f:
            assert f.read() == ""
