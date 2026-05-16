"""Tests for metrics.py CLI orchestration."""

import json
import os
import shutil
import tempfile

import pytest

import metrics


def _seg(start, end, sid, name=None, text="", words_count=0, sentiment=None):
    return {
        "start": start, "end": end,
        "speaker_id": sid, "speaker": name or sid,
        "text": text,
        "words": [{"start": start + i * 0.1, "end": start + (i + 1) * 0.1,
                   "word": f"w{i}", "probability": 0.9}
                  for i in range(words_count)],
        "sentiment": sentiment,
    }


def _sent(pol_label, emo_label, pol_score=0.8, emo_score=0.6):
    polarities = {"positive", "neutral", "negative"}
    emotions = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
    pol_scores = {k: 0.05 for k in polarities}
    pol_scores[pol_label] = pol_score
    emo_scores = {k: 0.05 for k in emotions}
    emo_scores[emo_label] = emo_score
    return {
        "polarity": {"label": pol_label, "score": pol_score, "scores": pol_scores},
        "emotion":  {"label": emo_label, "score": emo_score, "scores": emo_scores},
    }


@pytest.fixture
def tmp_workspace():
    d = tempfile.mkdtemp()
    json_path = os.path.join(d, "session.diarization.json")
    config_path = os.path.join(d, "config.yaml")
    with open(config_path, "w") as f:
        f.write("metrics:\n  bucket_seconds: 300\n  top_k_highlights: 5\n  quote_max_chars: 100\n")
    data = {
        "audio_file": "irrelevant.wav",
        "duration_s": 30.0,
        "diarized_at": "2026-05-15T10:00:00Z",
        "config": {"pyannote_pipeline": "p"},
        "enrolled_users_matched": [{"id": "alice", "name": "Alice"}, {"id": "bob", "name": "Bob"}],
        "segments": [
            _seg(0, 10, "alice", "Alice", text="hello there friend", words_count=3,
                 sentiment=_sent("positive", "joy")),
            _seg(10, 20, "bob", "Bob", text="hi", words_count=1,
                 sentiment=_sent("neutral", "neutral")),
            _seg(20, 30, "alice", "Alice", text="ok", words_count=1,
                 sentiment=_sent("neutral", "neutral")),
        ],
        "passes_run": ["diarization", "transcription", "sentiment"],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)
    yield {"dir": d, "json": json_path, "config": config_path}
    shutil.rmtree(d, ignore_errors=True)


def _read(path):
    with open(path) as f:
        return json.load(f)


class TestMetricsOrchestration:
    def test_happy_path_writes_json_and_markdown(self, tmp_workspace):
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 0
        data = _read(tmp_workspace["json"])
        assert "contribution_metrics" in data
        assert "metrics_config" in data
        assert "metrics" in data["passes_run"]
        # Markdown is sibling file.
        md_path = os.path.join(tmp_workspace["dir"], "session.diarization.metrics.md")
        assert os.path.exists(md_path)
        with open(md_path, encoding="utf-8") as f:
            assert "# Session Metrics" in f.read()

    def test_out_path_leaves_input_unchanged(self, tmp_workspace):
        original = _read(tmp_workspace["json"])
        out_json = os.path.join(tmp_workspace["dir"], "out.json")
        rc = metrics.main([tmp_workspace["json"], "--out", out_json,
                           "--config", tmp_workspace["config"]])
        assert rc == 0
        assert _read(tmp_workspace["json"]) == original
        assert "contribution_metrics" in _read(out_json)

    def test_report_path_writes_markdown_to_specified_path(self, tmp_workspace):
        report_path = os.path.join(tmp_workspace["dir"], "custom-report.md")
        rc = metrics.main([tmp_workspace["json"], "--report", report_path,
                           "--config", tmp_workspace["config"]])
        assert rc == 0
        assert os.path.exists(report_path)

    def test_idempotent_rerun_overwrites_and_dedupes_passes_run(self, tmp_workspace):
        metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        data = _read(tmp_workspace["json"])
        assert data["passes_run"].count("metrics") == 1

    def test_missing_transcription_pass_exit_2(self, tmp_workspace):
        data = _read(tmp_workspace["json"])
        data["passes_run"] = ["diarization", "sentiment"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_missing_sentiment_pass_exit_2(self, tmp_workspace):
        data = _read(tmp_workspace["json"])
        data["passes_run"] = ["diarization", "transcription"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_segment_missing_text_exit_2(self, tmp_workspace):
        data = _read(tmp_workspace["json"])
        del data["segments"][1]["text"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_missing_metrics_config_block_exit_3(self, tmp_workspace):
        with open(tmp_workspace["config"], "w") as f:
            f.write("other:\n  key: value\n")
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 3
