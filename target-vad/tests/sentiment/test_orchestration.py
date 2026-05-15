"""Tests for sentiment.py orchestration with a stub SentimentClassifier."""

import json
import os
import shutil
import tempfile

import pytest

import sentiment


@pytest.fixture
def tmp_workspace():
    """A tmp dir with a diarization JSON that has post-transcription segments."""
    d = tempfile.mkdtemp()
    json_path = os.path.join(d, "session.diarization.json")
    json_data = {
        "audio_file": "irrelevant.wav",
        "duration_s": 90.0,
        "diarized_at": "2026-05-15T10:00:00Z",
        "transcribed_at": "2026-05-15T11:00:00Z",
        "config": {"pyannote_pipeline": "p", "identification_threshold": 0.55},
        "enrolled_users_matched": [],
        "segments": [
            {
                "start": 0.0, "end": 10.0,
                "speaker_id": "siddharth", "speaker": "Siddharth Jain",
                "text": "I am genuinely excited about this discussion",
                "words": [{"start": 0.0, "end": 0.5, "word": "I", "probability": 0.99}],
            },
            {
                "start": 10.0, "end": 25.0,
                "speaker_id": "unknown", "speaker": "unknown",
                "text": "Yes, agreed.",
                "words": [{"start": 10.0, "end": 10.3, "word": "Yes", "probability": 0.95}],
            },
        ],
        "passes_run": ["diarization", "transcription"],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    yield {"dir": d, "json": json_path}
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stub_classifier(monkeypatch):
    """Patch SentimentClassifier so no real models load. Returns the list of calls made to classify_batch."""
    calls = []

    class StubClassifier:
        def __init__(self, polarity_model, emotion_model, device):
            self.polarity_model = polarity_model
            self.emotion_model = emotion_model

        def load(self):
            pass

        def classify_batch(self, texts):
            calls.append(list(texts))
            results = []
            for i, _t in enumerate(texts):
                results.append({
                    "polarity": {
                        "label": "neutral",
                        "score": 0.72,
                        "scores": {"positive": 0.18, "neutral": 0.72, "negative": 0.10},
                    },
                    "emotion": {
                        "label": "joy" if i % 2 == 0 else "surprise",
                        "score": 0.51,
                        "scores": {
                            "joy": 0.85 if i % 2 == 0 else 0.05,
                            "surprise": 0.05 if i % 2 == 0 else 0.51,
                            "anger": 0.01, "fear": 0.01, "sadness": 0.02,
                            "disgust": 0.01, "neutral": 0.05,
                        },
                    },
                })
            return results

    monkeypatch.setattr(sentiment, "SentimentClassifier", StubClassifier)
    return calls


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class TestSentimentOrchestration:
    def test_fresh_classification_attaches_sentiment(self, tmp_workspace, stub_classifier):
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["sentiment"]["polarity"]["label"] == "neutral"
        assert data["segments"][0]["sentiment"]["emotion"]["label"] == "joy"
        assert data["segments"][1]["sentiment"]["emotion"]["label"] == "surprise"

    def test_passes_run_gains_sentiment(self, tmp_workspace, stub_classifier):
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["passes_run"] == ["diarization", "transcription", "sentiment"]

    def test_sentiment_config_block_written(self, tmp_workspace, stub_classifier):
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        cfg = data["sentiment_config"]
        assert cfg["polarity_model"] == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert cfg["emotion_model"] == "j-hartmann/emotion-english-distilroberta-base"
        assert cfg["device"] == "cpu"
        assert cfg["batch_size"] == 16
        assert "analyzed_at" in cfg

    def test_null_text_gets_null_sentiment(self, tmp_workspace, stub_classifier):
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["text"] = None
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["sentiment"] is None
        assert data["segments"][1]["sentiment"]["polarity"]["label"] == "neutral"
        # Stub was called once, with only segment 1's text
        assert stub_classifier == [["Yes, agreed."]]

    def test_empty_text_gets_null_sentiment(self, tmp_workspace, stub_classifier):
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["text"] = ""
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        assert data["segments"][0]["sentiment"] is None

    def test_skip_already_classified_segments(self, tmp_workspace, stub_classifier):
        """A segment with existing sentiment is skipped; only the un-classified one runs."""
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["sentiment"] = {
            "polarity": {"label": "positive", "score": 0.9, "scores": {"positive": 0.9, "neutral": 0.05, "negative": 0.05}},
            "emotion": {"label": "joy", "score": 0.8, "scores": {"joy": 0.8, "surprise": 0.05, "anger": 0.05, "fear": 0.05, "sadness": 0.02, "disgust": 0.03, "neutral": 0.0}},
        }
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        # Segment 0's sentiment was preserved (label still "positive")
        assert data["segments"][0]["sentiment"]["polarity"]["label"] == "positive"
        # Segment 1 was newly classified
        assert data["segments"][1]["sentiment"]["polarity"]["label"] == "neutral"
        # Classifier was called only once, with segment 1's text
        assert stub_classifier == [["Yes, agreed."]]

    def test_rerun_flag_reclassifies_all(self, tmp_workspace, stub_classifier):
        """--rerun discards existing sentiment and re-classifies everything."""
        data = _read_json(tmp_workspace["json"])
        data["segments"][0]["sentiment"] = {
            "polarity": {"label": "negative", "score": 0.99, "scores": {"positive": 0.0, "neutral": 0.01, "negative": 0.99}},
            "emotion": {"label": "anger", "score": 0.99, "scores": {"joy": 0.0, "surprise": 0.0, "anger": 0.99, "fear": 0.0, "sadness": 0.0, "disgust": 0.01, "neutral": 0.0}},
        }
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml", "--rerun"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        # Both segments freshly classified
        assert data["segments"][0]["sentiment"]["polarity"]["label"] == "neutral"
        assert data["segments"][1]["sentiment"]["polarity"]["label"] == "neutral"
        # Classifier was called once with both texts batched
        assert stub_classifier == [["I am genuinely excited about this discussion", "Yes, agreed."]]

    def test_missing_text_field_exits_2(self, tmp_workspace, stub_classifier):
        """Pre-flight: if any segment lacks a text field, exit 2 with a transcribe.py hint."""
        data = _read_json(tmp_workspace["json"])
        del data["segments"][1]["text"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_non_string_text_type_exits_2(self, tmp_workspace, stub_classifier):
        """Pre-flight: if any segment has text of wrong type (int, list, etc.), exit 2."""
        data = _read_json(tmp_workspace["json"])
        data["segments"][1]["text"] = 123  # not a string and not None
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_explicit_out_path_does_not_modify_input(self, tmp_workspace, stub_classifier):
        out_path = os.path.join(tmp_workspace["dir"], "out.json")
        rc = sentiment.main([
            tmp_workspace["json"], "--out", out_path, "--config", "config.yaml",
        ])
        assert rc == 0
        assert os.path.exists(out_path)
        original = _read_json(tmp_workspace["json"])
        assert "sentiment" not in original["segments"][0]
        modified = _read_json(out_path)
        assert "sentiment" in modified["segments"][0]

    def test_malformed_json_exits_2(self, tmp_workspace, stub_classifier):
        with open(tmp_workspace["json"], "w") as f:
            f.write("{not json")
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_missing_segments_field_exits_2(self, tmp_workspace, stub_classifier):
        bad = {"audio_file": "x.wav", "duration_s": 90.0}
        with open(tmp_workspace["json"], "w") as f:
            json.dump(bad, f)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 2

    def test_model_load_failure_exits_3(self, tmp_workspace, monkeypatch):
        """Model load failure surfaces as exit 3, not silent all-null."""
        class FailingLoadClassifier:
            def __init__(self, *a, **kw): pass

            def load(self):
                raise RuntimeError("simulated model download failure")

            def classify_batch(self, texts):
                raise AssertionError("classify_batch should never be called when load fails")

        monkeypatch.setattr(sentiment, "SentimentClassifier", FailingLoadClassifier)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 3
        # JSON should NOT have been modified
        data = _read_json(tmp_workspace["json"])
        assert "sentiment" not in data["segments"][0]

    def test_classify_batch_exception_marks_segments_null(self, tmp_workspace, monkeypatch):
        """If classify_batch raises mid-run, affected segments get sentiment: null, exit 0."""
        class PartialFailingClassifier:
            def __init__(self, *a, **kw): pass

            def load(self):
                pass

            def classify_batch(self, texts):
                raise RuntimeError("simulated batch crash")

        monkeypatch.setattr(sentiment, "SentimentClassifier", PartialFailingClassifier)
        rc = sentiment.main([tmp_workspace["json"], "--config", "config.yaml"])
        assert rc == 0
        data = _read_json(tmp_workspace["json"])
        # Both segments tried to classify but the batch crashed; both get null
        assert data["segments"][0]["sentiment"] is None
        assert data["segments"][1]["sentiment"] is None
