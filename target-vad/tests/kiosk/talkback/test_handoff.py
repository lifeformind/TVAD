"""Tests for TalkbackHandoff and TalkbackResult dataclasses."""

from unittest.mock import MagicMock

import numpy as np

from modes.talkback.handoff import TalkbackHandoff, TalkbackResult


class TestTalkbackHandoff:
    def test_construction(self):
        mic = MagicMock()
        emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        seg = MagicMock()
        seg.duration_ms = 1000.0
        cfg = {"sample_rate_hz": 16000}

        h = TalkbackHandoff(
            mic=mic,
            primary_embedding=emb,
            first_segment=seg,
            config=cfg,
        )
        assert h.mic is mic
        assert h.primary_embedding is emb
        assert h.first_segment is seg
        assert h.config == {"sample_rate_hz": 16000}

    def test_embedding_is_192_dim(self):
        emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        h = TalkbackHandoff(
            mic=MagicMock(),
            primary_embedding=emb,
            first_segment=MagicMock(),
            config={},
        )
        assert h.primary_embedding.shape == (192,)


class TestTalkbackResult:
    def test_construction(self):
        r = TalkbackResult(
            reason="silence_timeout",
            turns=4,
            total_duration_s=47.3,
        )
        assert r.reason == "silence_timeout"
        assert r.turns == 4
        assert r.total_duration_s == 47.3

    def test_reason_values(self):
        for reason in ("silence_timeout", "hard_timeout", "stopped", "device_lost"):
            r = TalkbackResult(reason=reason, turns=0, total_duration_s=0.0)
            assert r.reason == reason
