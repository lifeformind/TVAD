"""Tests for SentimentClassifier — mocked transformers.pipeline, no real model downloads."""

from unittest.mock import MagicMock, patch

import pytest

from modes.sentiment.classifier import SentimentClassifier


def _polarity_output(scores):
    """Build a mocked pipeline output for one text: list of {label, score} dicts.

    scores is a dict mapping canonical polarity labels to scores.
    """
    return [{"label": label, "score": score} for label, score in scores.items()]


def _emotion_output(scores):
    """Build a mocked pipeline output for one text: list of {label, score} dicts."""
    return [{"label": label, "score": score} for label, score in scores.items()]


class TestSentimentClassifier:
    def test_lazy_model_load(self):
        """SentimentClassifier does NOT construct pipelines until classify_batch() is called."""
        with patch("modes.sentiment.classifier.pipeline") as p:
            classifier = SentimentClassifier(
                polarity_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                emotion_model="j-hartmann/emotion-english-distilroberta-base",
                device="cpu",
            )
            p.assert_not_called()
            # Set up the mock to return per-text per-label outputs
            polarity_pipe = MagicMock()
            polarity_pipe.return_value = [
                _polarity_output({"positive": 0.18, "neutral": 0.72, "negative": 0.10}),
            ]
            emotion_pipe = MagicMock()
            emotion_pipe.return_value = [
                _emotion_output({
                    "joy": 0.12, "surprise": 0.51, "anger": 0.03, "fear": 0.04,
                    "sadness": 0.05, "disgust": 0.02, "neutral": 0.23,
                }),
            ]
            p.side_effect = [polarity_pipe, emotion_pipe]
            classifier.classify_batch(["hello"])
            assert p.call_count == 2

    def test_load_triggers_eager_pipeline_construction(self):
        """load() constructs both pipelines without running inference."""
        with patch("modes.sentiment.classifier.pipeline") as p:
            p.side_effect = [MagicMock(), MagicMock()]
            classifier = SentimentClassifier(
                polarity_model="m1", emotion_model="m2", device="cpu",
            )
            p.assert_not_called()
            classifier.load()
            assert p.call_count == 2
            # Second load() is a no-op
            classifier.load()
            assert p.call_count == 2

    def test_classify_batch_returns_nested_blocks(self):
        """classify_batch returns one nested {polarity, emotion} dict per input text in order."""
        polarity_pipe = MagicMock()
        polarity_pipe.return_value = [
            _polarity_output({"positive": 0.18, "neutral": 0.72, "negative": 0.10}),
            _polarity_output({"positive": 0.80, "neutral": 0.15, "negative": 0.05}),
        ]
        emotion_pipe = MagicMock()
        emotion_pipe.return_value = [
            _emotion_output({
                "joy": 0.12, "surprise": 0.51, "anger": 0.03, "fear": 0.04,
                "sadness": 0.05, "disgust": 0.02, "neutral": 0.23,
            }),
            _emotion_output({
                "joy": 0.85, "surprise": 0.05, "anger": 0.01, "fear": 0.01,
                "sadness": 0.02, "disgust": 0.01, "neutral": 0.05,
            }),
        ]
        with patch("modes.sentiment.classifier.pipeline", side_effect=[polarity_pipe, emotion_pipe]):
            classifier = SentimentClassifier(polarity_model="m1", emotion_model="m2", device="cpu")
            results = classifier.classify_batch(["text1", "text2"])

        assert len(results) == 2
        # First text: neutral polarity, surprise emotion (argmax)
        assert results[0]["polarity"]["label"] == "neutral"
        assert results[0]["polarity"]["score"] == pytest.approx(0.72)
        assert results[0]["polarity"]["scores"] == {
            "positive": pytest.approx(0.18),
            "neutral": pytest.approx(0.72),
            "negative": pytest.approx(0.10),
        }
        assert results[0]["emotion"]["label"] == "surprise"
        assert results[0]["emotion"]["score"] == pytest.approx(0.51)
        # Second text: positive polarity, joy emotion
        assert results[1]["polarity"]["label"] == "positive"
        assert results[1]["emotion"]["label"] == "joy"

    def test_empty_input_returns_empty_list(self):
        """classify_batch on [] returns [] without invoking pipelines."""
        with patch("modes.sentiment.classifier.pipeline") as p:
            p.side_effect = [MagicMock(), MagicMock()]
            classifier = SentimentClassifier(polarity_model="m1", emotion_model="m2", device="cpu")
            result = classifier.classify_batch([])
            assert result == []

    def test_non_canonical_polarity_label_raises(self):
        """A polarity model that emits a label outside {positive, neutral, negative} raises ValueError."""
        polarity_pipe = MagicMock()
        polarity_pipe.return_value = [
            [{"label": "WEIRD_LABEL", "score": 0.99}],
        ]
        emotion_pipe = MagicMock()
        emotion_pipe.return_value = [
            _emotion_output({
                "joy": 0.12, "surprise": 0.51, "anger": 0.03, "fear": 0.04,
                "sadness": 0.05, "disgust": 0.02, "neutral": 0.23,
            }),
        ]
        with patch("modes.sentiment.classifier.pipeline", side_effect=[polarity_pipe, emotion_pipe]):
            classifier = SentimentClassifier(polarity_model="m1", emotion_model="m2", device="cpu")
            with pytest.raises(ValueError, match=r"polarity model.*WEIRD_LABEL"):
                classifier.classify_batch(["text"])

    def test_non_canonical_emotion_label_raises(self):
        """An emotion model that emits a label outside the 7-class set raises ValueError."""
        polarity_pipe = MagicMock()
        polarity_pipe.return_value = [
            _polarity_output({"positive": 0.18, "neutral": 0.72, "negative": 0.10}),
        ]
        emotion_pipe = MagicMock()
        emotion_pipe.return_value = [
            [{"label": "boredom", "score": 0.99}],
        ]
        with patch("modes.sentiment.classifier.pipeline", side_effect=[polarity_pipe, emotion_pipe]):
            classifier = SentimentClassifier(polarity_model="m1", emotion_model="m2", device="cpu")
            with pytest.raises(ValueError, match=r"emotion model.*boredom"):
                classifier.classify_batch(["text"])

    def test_canonical_labels_preserved(self):
        """Both label sets are returned exactly as defined (no case changes, no mapping surprises)."""
        polarity_pipe = MagicMock()
        polarity_pipe.return_value = [
            _polarity_output({"negative": 0.6, "neutral": 0.3, "positive": 0.1}),
        ]
        emotion_pipe = MagicMock()
        emotion_pipe.return_value = [
            _emotion_output({
                "anger": 0.7, "joy": 0.05, "surprise": 0.05, "fear": 0.05,
                "sadness": 0.05, "disgust": 0.05, "neutral": 0.05,
            }),
        ]
        with patch("modes.sentiment.classifier.pipeline", side_effect=[polarity_pipe, emotion_pipe]):
            classifier = SentimentClassifier(polarity_model="m1", emotion_model="m2", device="cpu")
            results = classifier.classify_batch(["I am angry"])
        assert results[0]["polarity"]["label"] == "negative"
        assert results[0]["emotion"]["label"] == "anger"
