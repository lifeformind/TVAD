"""Tests for NemoStt (Parakeet-TDT backend, spec Appendix B).

Pure-logic tests use a fake NeMo model object (mirrors test_stt.py's
__new__ + fake-model bypass convention) so they run without nemo_toolkit
installed. NeMo import stays inside NemoStt._ensure_model(), never at
module import time (conftest does not stub nemo)."""

import numpy as np
import pytest

from modes.talkback.stt_nemo import NemoStt
from modes.director.transcript import TranscriptResult


class _Hyp:
    def __init__(self, text, word_confidence):
        self.text = text
        self.word_confidence = word_confidence

class _FakeNemoModel:
    def __init__(self, hyp):
        self._hyp = hyp
    def transcribe(self, audio, return_hypotheses=True, verbose=False):
        return [self._hyp]


def _make(hyp):
    stt = NemoStt.__new__(NemoStt)
    stt._model = _FakeNemoModel(hyp)
    stt._device = "cpu"
    return stt


@pytest.mark.asyncio
async def test_transcribe_returns_text_and_mean_confidence():
    stt = _make(_Hyp(" hello there ", [0.9, 0.7]))
    result = await stt.transcribe_segment(np.zeros(16000, dtype=np.float32))
    assert isinstance(result, TranscriptResult)
    assert result.text == "hello there"
    assert result.mean_word_prob == pytest.approx(0.8)

@pytest.mark.asyncio
async def test_missing_confidence_falls_back_to_one_for_nonempty_text():
    stt = _make(_Hyp("hi", None))
    result = await stt.transcribe_segment(np.zeros(16000, dtype=np.float32))
    assert result.mean_word_prob == 1.0

@pytest.mark.asyncio
async def test_empty_text_scores_zero():
    stt = _make(_Hyp("", None))
    result = await stt.transcribe_segment(np.zeros(16000, dtype=np.float32))
    assert result.text == "" and result.mean_word_prob == 0.0
