"""Tests for the re-backed StreamingStt (openai-whisper / torch CUDA).

Pure-logic tests use a fake model object and always run. The real-CUDA
integration test lives in test_stt_cuda.py and is skipped without a GPU.
"""

import numpy as np
import pytest

from modes.talkback.stt import StreamingStt, TranscriptResult


def test_transcript_result_shape():
    r = TranscriptResult(text="hello world", mean_word_prob=0.87)
    assert r.text == "hello world"
    assert r.mean_word_prob == 0.87


def test_transcript_result_is_frozen():
    r = TranscriptResult(text="x", mean_word_prob=0.5)
    with pytest.raises(Exception):
        r.text = "y"  # frozen dataclass -> FrozenInstanceError


def test_transcript_result_is_the_canonical_director_type():
    # Single source of truth: stt.py RE-EXPORTS Plan 02's type, not a copy,
    # so Plan 02's wrap_transcript() isinstance check stays valid.
    from modes.director.transcript import TranscriptResult as Canonical
    assert TranscriptResult is Canonical


class _FakeWhisperModel:
    """Mimics whisper.load_model(...).transcribe() output shape."""

    def __init__(self, result):
        self._result = result

    def transcribe(self, audio, **kwargs):
        return self._result


def _make_stt_with_model(result):
    stt = StreamingStt.__new__(StreamingStt)
    stt._model = _FakeWhisperModel(result)
    stt._device = "cpu"   # _transcribe_sync reads this for the fp16 flag
    return stt


def test_mean_word_prob_averages_word_probabilities():
    result = {
        "text": " hello world ",
        "segments": [
            {"words": [
                {"word": "hello", "probability": 0.9},
                {"word": "world", "probability": 0.7},
            ]}
        ],
    }
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert isinstance(out, TranscriptResult)
    assert out.text == "hello world"
    assert out.mean_word_prob == pytest.approx(0.8)


def test_mean_word_prob_empty_text_is_zero():
    result = {"text": "  ", "segments": []}
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert out.text == ""
    assert out.mean_word_prob == 0.0


def test_mean_word_prob_multi_segment_concatenates_and_averages():
    result = {
        "text": " first second third ",
        "segments": [
            {"words": [
                {"word": "first", "probability": 1.0},
                {"word": "second", "probability": 0.5},
            ]},
            {"words": [
                {"word": "third", "probability": 0.6},
            ]},
        ],
    }
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert out.text == "first second third"
    assert out.mean_word_prob == pytest.approx((1.0 + 0.5 + 0.6) / 3.0)


def test_mean_word_prob_clamped_to_unit_interval():
    # whisper probabilities are already in [0,1]; defend against odd inputs.
    result = {
        "text": "x",
        "segments": [{"words": [{"word": "x", "probability": 1.4}]}],
    }
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert 0.0 <= out.mean_word_prob <= 1.0


@pytest.mark.asyncio
async def test_transcribe_segment_returns_transcript_result():
    result = {
        "text": " hi ",
        "segments": [{"words": [{"word": "hi", "probability": 0.95}]}],
    }
    stt = _make_stt_with_model(result)
    out = await stt.transcribe_segment(np.zeros(48000, dtype=np.float32))
    assert isinstance(out, TranscriptResult)
    assert out.text == "hi"
    assert out.mean_word_prob == pytest.approx(0.95)
