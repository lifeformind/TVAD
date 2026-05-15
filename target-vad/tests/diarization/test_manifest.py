"""Tests for the introductions-manifest loader."""

import json
import os
import tempfile

import pytest

from modes.diarization.manifest import ManifestEntry, load_manifest


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        yield f
    os.unlink(f.name)


def write_manifest(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestLoadManifest:
    def test_valid_manifest(self, tmp_path):
        path = str(tmp_path / "intros.json")
        write_manifest(path, [
            {"id": "siddharth", "name": "Siddharth Jain", "start": 0.0, "end": 28.5},
            {"id": "alice_smith", "name": "Alice", "start": 28.5, "end": 47.0},
        ])
        entries = load_manifest(path)
        assert len(entries) == 2
        assert entries[0] == ManifestEntry(id="siddharth", name="Siddharth Jain", start=0.0, end=28.5)
        assert entries[1] == ManifestEntry(id="alice_smith", name="Alice", start=28.5, end=47.0)

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(str(tmp_path / "nonexistent.json"))

    def test_malformed_json_raises_valueerror(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        with pytest.raises(ValueError, match="malformed JSON"):
            load_manifest(path)

    def test_top_level_not_array_raises(self, tmp_path):
        path = str(tmp_path / "obj.json")
        write_manifest(path, {"id": "x", "name": "x", "start": 0.0, "end": 1.0})
        with pytest.raises(ValueError, match="top-level value must be a JSON array"):
            load_manifest(path)

    def test_missing_key_raises_with_index(self, tmp_path):
        path = str(tmp_path / "missing.json")
        write_manifest(path, [
            {"id": "alice", "name": "Alice", "start": 0.0, "end": 5.0},
            {"id": "bob", "name": "Bob", "start": 5.0},  # missing "end"
        ])
        with pytest.raises(ValueError, match=r"entry 1.*missing required key 'end'"):
            load_manifest(path)

    def test_wrong_type_raises(self, tmp_path):
        path = str(tmp_path / "wrongtype.json")
        write_manifest(path, [{"id": "alice", "name": "Alice", "start": "0.0", "end": 5.0}])
        with pytest.raises(ValueError, match=r"entry 0.*'start' must be a number"):
            load_manifest(path)

    def test_end_le_start_raises(self, tmp_path):
        path = str(tmp_path / "badrange.json")
        write_manifest(path, [{"id": "alice", "name": "Alice", "start": 5.0, "end": 5.0}])
        with pytest.raises(ValueError, match=r"entry 0.*end must be > start"):
            load_manifest(path)

    def test_negative_start_raises(self, tmp_path):
        path = str(tmp_path / "negstart.json")
        write_manifest(path, [{"id": "alice", "name": "Alice", "start": -1.0, "end": 5.0}])
        with pytest.raises(ValueError, match=r"entry 0.*start must be >= 0"):
            load_manifest(path)

    def test_duplicate_id_raises(self, tmp_path):
        path = str(tmp_path / "dup.json")
        write_manifest(path, [
            {"id": "alice", "name": "Alice", "start": 0.0, "end": 5.0},
            {"id": "alice", "name": "Alice Two", "start": 5.0, "end": 10.0},
        ])
        with pytest.raises(ValueError, match=r"duplicate id 'alice'.*entries \[0, 1\]"):
            load_manifest(path)

    def test_empty_id_or_name_raises(self, tmp_path):
        path = str(tmp_path / "empty.json")
        write_manifest(path, [{"id": "", "name": "Alice", "start": 0.0, "end": 5.0}])
        with pytest.raises(ValueError, match=r"entry 0.*'id' must be a non-empty string"):
            load_manifest(path)
