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
