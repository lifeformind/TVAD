"""Tests for bench/speaker_scores.py — kiosk speaker-score log parsing."""
import importlib.util
import os
import sys

import pytest

# Load bench/speaker_scores.py by path (bench/ is not an importable package).
# Register in sys.modules before exec so @dataclass can resolve __module__
# (required under `from __future__ import annotations`).
_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "bench", "speaker_scores.py"))
_spec = importlib.util.spec_from_file_location("speaker_scores", _MOD_PATH)
ss = importlib.util.module_from_spec(_spec)
sys.modules["speaker_scores"] = ss
_spec.loader.exec_module(ss)


def rec(event, payload, session_id="sess1", ts="2026-06-09T00:00:00Z"):
    return {"ts": ts, "session_id": session_id, "event": event, "payload": payload}


class TestExtractScores:
    def test_barge_in_rejected_is_reject(self):
        rows = ss.extract_scores([rec("barge_in_rejected", {"score": 0.247, "threshold": 0.75})])
        assert len(rows) == 1
        r = rows[0]
        assert r.source == "barge_in" and r.decision == "reject"
        assert r.score == pytest.approx(0.247) and r.threshold == pytest.approx(0.75)

    def test_turn_gate_accept_and_reject(self):
        rows = ss.extract_scores([
            rec("turn_gate", {"score": 0.71, "threshold": 0.5, "decision": "accept"}),
            rec("turn_gate", {"score": 0.04, "threshold": 0.5, "decision": "reject"}),
        ])
        assert [r.source for r in rows] == ["turn_gate", "turn_gate"]
        assert [r.decision for r in rows] == ["accept", "reject"]
        assert rows[0].score == pytest.approx(0.71)

    def test_barge_in_accepted_uses_primary_score(self):
        rows = ss.extract_scores([rec("barge_in", {"primary_score": 0.81, "cut_at_ms": 120})])
        assert len(rows) == 1 and rows[0].decision == "accept"
        assert rows[0].score == pytest.approx(0.81)

    def test_segment_scored_match_and_no_match(self):
        rows = ss.extract_scores([
            rec("segment_scored", {"score": 0.62, "decision": "match", "duration_ms": 900}),
            rec("segment_scored", {"score": 0.30, "decision": "no_match", "duration_ms": 800}),
        ])
        assert [r.source for r in rows] == ["primary_match", "primary_match"]
        assert [r.decision for r in rows] == ["accept", "reject"]

    def test_unrelated_events_ignored(self):
        rows = ss.extract_scores([
            rec("turn_started", {"turn_number": 1}),
            rec("session_ended", {"reason": "silence_timeout"}),
        ])
        assert rows == []


class TestStats:
    def test_summarize_basic(self):
        st = ss.summarize([0.2, 0.4, 0.6])
        assert st["n"] == 3
        assert st["mean"] == pytest.approx(0.4)
        assert st["median"] == pytest.approx(0.4)
        assert st["min"] == pytest.approx(0.2) and st["max"] == pytest.approx(0.6)

    def test_summarize_empty(self):
        assert ss.summarize([]) == {"n": 0}

    def test_accept_reject_split(self):
        rows = ss.extract_scores([
            rec("barge_in_rejected", {"score": 0.2, "threshold": 0.75}),
            rec("barge_in_rejected", {"score": 0.46, "threshold": 0.75}),
            rec("barge_in", {"primary_score": 0.80}),
        ])
        acc, rej = ss._accept_reject(rows, 0.75)
        assert (acc, rej) == (1, 2)


class TestSeparation:
    def test_separable_groups(self):
        bt = ss.best_threshold(self_scores=[0.6, 0.7, 0.8],
                               nonself_scores=[0.1, 0.2, 0.3])
        assert bt["separable"] is True
        assert bt["margin"] == pytest.approx(0.3)  # 0.6 - 0.3
        assert 0.3 < bt["threshold"] < 0.6
        assert bt["tpr"] == 1.0 and bt["tnr"] == 1.0
        assert bt["balanced_acc"] == 1.0

    def test_overlapping_groups(self):
        bt = ss.best_threshold(self_scores=[0.1, 0.2, 0.4],
                               nonself_scores=[0.05, 0.3, 0.5])
        assert bt["separable"] is False
        assert bt["margin"] < 0
        assert bt["balanced_acc"] < 1.0

    def test_empty_group_returns_empty(self):
        assert ss.best_threshold([], [0.1]) == {}
        assert ss.best_threshold([0.1], []) == {}

    def test_comparison_report_overlap_recommends_fix(self):
        self_rows = ss.extract_scores([
            rec("turn_gate", {"score": 0.05, "threshold": 0.5, "decision": "reject"}),
            rec("turn_gate", {"score": 0.39, "threshold": 0.5, "decision": "reject"}),
        ])
        nonself_rows = ss.extract_scores([
            rec("turn_gate", {"score": 0.08, "threshold": 0.5, "decision": "reject"}),
            rec("turn_gate", {"score": 0.20, "threshold": 0.5, "decision": "reject"}),
        ])
        text = ss.report_comparison(self_rows, nonself_rows, bins=10)
        assert "OVERLAPPING" in text
        assert "self (enrolled)" in text and "non-self (other)" in text

    def test_comparison_report_separable_shows_band(self):
        self_rows = ss.extract_scores([
            rec("turn_gate", {"score": 0.62, "threshold": 0.5, "decision": "accept"}),
            rec("turn_gate", {"score": 0.71, "threshold": 0.5, "decision": "accept"}),
        ])
        nonself_rows = ss.extract_scores([
            rec("turn_gate", {"score": 0.10, "threshold": 0.5, "decision": "reject"}),
            rec("turn_gate", {"score": 0.22, "threshold": 0.5, "decision": "reject"}),
        ])
        text = ss.report_comparison(self_rows, nonself_rows, bins=10)
        assert "CLEANLY SEPARABLE" in text
        assert "safe threshold band" in text


class TestReport:
    def test_report_mentions_threshold_and_counts(self):
        rows = ss.extract_scores([
            rec("barge_in_rejected", {"score": 0.2, "threshold": 0.75}),
            rec("barge_in", {"primary_score": 0.9}),
        ])
        text = ss.report(rows, threshold_override=None, bins=10, label="nonself")
        assert "nonself" in text
        assert "threshold 0.75" in text
        assert "accept(>=thr)=1" in text and "reject(<thr)=1" in text

    def test_report_empty_explains(self):
        text = ss.report([], threshold_override=None, bins=10, label=None)
        assert "No speaker scores" in text
