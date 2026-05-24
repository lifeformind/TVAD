"""Tests for EventLogger — structured JSONL audit logging (F6)."""

import json
import os
import tempfile
from unittest.mock import patch

from core.logging.jsonl_logger import EventLogger


class TestEventLoggerPathTemplating:
    def test_date_and_session_id_interpolated(self):
        with tempfile.TemporaryDirectory() as d:
            logger = EventLogger(
                path_template=os.path.join(d, "{date}-{session_id}.jsonl"),
                session_id="abc123",
            )
            logger.log("test_event", {"key": "val"})
            files = os.listdir(d)
            assert len(files) == 1
            assert "abc123" in files[0]
            assert files[0].endswith(".jsonl")

    def test_subdirectories_created_automatically(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "dir", "{date}-{session_id}.jsonl")
            logger = EventLogger(path_template=path, session_id="s1")
            logger.log("evt", {})
            assert os.path.exists(os.path.join(d, "sub", "dir"))


class TestEventLoggerOutput:
    def test_log_writes_valid_json_line(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "{date}-{session_id}.jsonl")
            logger = EventLogger(path_template=path, session_id="s1")
            logger.log("wake_detected", {"phrase": "hey_jarvis", "score": 0.87})
            filepath = logger.current_path
            with open(filepath) as f:
                line = f.readline()
            record = json.loads(line)
            assert record["event"] == "wake_detected"
            assert record["payload"]["phrase"] == "hey_jarvis"
            assert record["payload"]["score"] == 0.87

    def test_ts_is_iso8601_utc(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "{date}-{session_id}.jsonl")
            logger = EventLogger(path_template=path, session_id="s1")
            logger.log("test", {})
            with open(logger.current_path) as f:
                record = json.loads(f.readline())
            ts = record["ts"]
            assert "T" in ts
            assert ts.endswith("Z") or "+" in ts

    def test_session_id_in_every_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "{date}-{session_id}.jsonl")
            logger = EventLogger(path_template=path, session_id="sess42")
            logger.log("a", {})
            logger.log("b", {"x": 1})
            with open(logger.current_path) as f:
                lines = f.readlines()
            for line in lines:
                assert json.loads(line)["session_id"] == "sess42"

    def test_multiple_logs_append_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "{date}-{session_id}.jsonl")
            logger = EventLogger(path_template=path, session_id="s1")
            logger.log("evt1", {"a": 1})
            logger.log("evt2", {"b": 2})
            logger.log("evt3", {"c": 3})
            with open(logger.current_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            assert json.loads(lines[0])["event"] == "evt1"
            assert json.loads(lines[2])["event"] == "evt3"


class TestEventLoggerNewSession:
    def test_new_session_writes_to_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "{date}-{session_id}.jsonl")
            logger = EventLogger(path_template=path, session_id="s1")
            logger.log("first", {})
            path1 = logger.current_path
            logger.start_session("s2")
            logger.log("second", {})
            path2 = logger.current_path
            assert path1 != path2
            assert os.path.exists(path1)
            assert os.path.exists(path2)
