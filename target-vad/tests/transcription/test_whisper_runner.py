"""Tests for WhisperRunner — mocked WhisperModel, no real download."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modes.transcription.whisper_runner import WhisperRunner


SR = 16000


def _make_mock_model(segments_to_return):
    """Build a mock WhisperModel whose transcribe() returns the given iterable of segments + info."""
    model = MagicMock()
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    model.transcribe.return_value = (iter(segments_to_return), info)
    return model


def _mock_word(start, end, word, prob=0.9):
    w = MagicMock()
    w.start = start
    w.end = end
    w.word = word
    w.probability = prob
    return w


def _mock_segment(start, end, text, words):
    s = MagicMock()
    s.start = start
    s.end = end
    s.text = text
    s.words = words
    return s


class TestWhisperRunner:
    def test_lazy_model_load(self):
        """WhisperRunner does NOT construct the model until transcribe() is called."""
        with patch("modes.transcription.whisper_runner.WhisperModel") as Cls:
            runner = WhisperRunner(model="small", language="en", device="cpu", compute_type="int8")
            Cls.assert_not_called()  # not loaded yet
            Cls.return_value = _make_mock_model([])
            runner.transcribe(np.zeros(SR, dtype=np.float32), initial_prompt="")
            Cls.assert_called_once_with("small", device="cpu", compute_type="int8")

    def test_transcribe_returns_text_and_words(self):
        """Concatenates segment text; flattens word lists in order."""
        mock_model = _make_mock_model([
            _mock_segment(0.0, 1.0, " hello", [_mock_word(0.0, 0.5, "hello", 0.95)]),
            _mock_segment(1.0, 2.0, " world", [_mock_word(1.0, 1.5, "world", 0.92)]),
        ])
        with patch("modes.transcription.whisper_runner.WhisperModel", return_value=mock_model):
            runner = WhisperRunner(model="small", language="en", device="cpu", compute_type="int8")
            text, words = runner.transcribe(np.zeros(2 * SR, dtype=np.float32), initial_prompt="ctx")
        assert text == "hello world"
        assert words == [
            {"start": 0.0, "end": 0.5, "word": "hello", "probability": pytest.approx(0.95)},
            {"start": 1.0, "end": 1.5, "word": "world", "probability": pytest.approx(0.92)},
        ]

    def test_transcribe_passes_initial_prompt_to_model(self):
        mock_model = _make_mock_model([])
        with patch("modes.transcription.whisper_runner.WhisperModel", return_value=mock_model):
            runner = WhisperRunner(model="small", language="en", device="cpu", compute_type="int8")
            runner.transcribe(np.zeros(SR, dtype=np.float32), initial_prompt="prior text")
        kwargs = mock_model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == "prior text"
        assert kwargs["word_timestamps"] is True
        assert kwargs["language"] == "en"

    def test_transcribe_with_null_language_lets_whisper_auto_detect(self):
        """language=None means whisper auto-detects; we don't pass the kwarg."""
        mock_model = _make_mock_model([])
        with patch("modes.transcription.whisper_runner.WhisperModel", return_value=mock_model):
            runner = WhisperRunner(model="small", language=None, device="cpu", compute_type="int8")
            runner.transcribe(np.zeros(SR, dtype=np.float32), initial_prompt="")
        kwargs = mock_model.transcribe.call_args.kwargs
        assert "language" not in kwargs or kwargs["language"] is None

    def test_transcribe_empty_result_returns_empty_text_and_empty_words(self):
        mock_model = _make_mock_model([])  # no segments
        with patch("modes.transcription.whisper_runner.WhisperModel", return_value=mock_model):
            runner = WhisperRunner(model="small", language="en", device="cpu", compute_type="int8")
            text, words = runner.transcribe(np.zeros(SR, dtype=np.float32), initial_prompt="")
        assert text == ""
        assert words == []

    def test_model_preset_large_resolves_to_large_v3(self):
        """The 'large' preset maps to 'large-v3'; other strings pass through."""
        with patch("modes.transcription.whisper_runner.WhisperModel") as Cls:
            Cls.return_value = _make_mock_model([])
            runner = WhisperRunner(model="large", language="en", device="cpu", compute_type="int8")
            runner.transcribe(np.zeros(SR, dtype=np.float32), initial_prompt="")
            args, _ = Cls.call_args
            assert args[0] == "large-v3"

    def test_arbitrary_model_string_passes_through(self):
        """Strings other than the two presets are forwarded to faster-whisper as-is."""
        with patch("modes.transcription.whisper_runner.WhisperModel") as Cls:
            Cls.return_value = _make_mock_model([])
            runner = WhisperRunner(
                model="openai/whisper-large-v3-turbo",
                language="en", device="cpu", compute_type="int8",
            )
            runner.transcribe(np.zeros(SR, dtype=np.float32), initial_prompt="")
            args, _ = Cls.call_args
            assert args[0] == "openai/whisper-large-v3-turbo"

    def test_load_triggers_model_construction(self):
        """load() is a public eager-load entry point — useful to surface download errors early."""
        with patch("modes.transcription.whisper_runner.WhisperModel") as Cls:
            Cls.return_value = _make_mock_model([])
            runner = WhisperRunner(model="small", language="en", device="cpu", compute_type="int8")
            Cls.assert_not_called()
            runner.load()
            Cls.assert_called_once_with("small", device="cpu", compute_type="int8")
            # Second load() call should NOT trigger a second model construction
            runner.load()
            Cls.assert_called_once()
