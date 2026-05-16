"""Golden-file and unit tests for modes.metrics.renderer."""

import os
from pathlib import Path

from modes.metrics import renderer

FIXTURES = Path(__file__).parent / "fixtures"


def _full_metrics_block():
    """Build the metrics block that should render to golden_report.md."""
    return {
        "session": {
            "duration_s": 90.0,
            "speech_duration_s": 78.4,
            "silence_duration_s": 11.6,
            "total_segments": 9,
            "total_words": 312,
            "unique_speakers": 2,
            "identified_speakers": 2,
            "unknown_segments": 0,
            "polarity_distribution": {"positive": 2, "neutral": 7, "negative": 0},
            "emotion_distribution": {"joy": 1, "neutral": 5, "surprise": 1,
                                     "disgust": 2, "anger": 0, "fear": 0, "sadness": 0},
        },
        "speakers": [
            {
                "speaker_id": "session_speaker_a", "speaker": "Speaker A",
                "participation": {
                    "talk_seconds": 52.1, "talk_percent": 66.5,
                    "segment_count": 5, "word_count": 198,
                    "words_per_minute": 228.0,
                    "mean_segment_seconds": 10.4, "median_segment_seconds": 8.7,
                    "max_segment_seconds": 42.6,
                },
                "sentiment": {
                    "polarity": {
                        "counts": {"positive": 1, "neutral": 4, "negative": 0},
                        "percent": {"positive": 20.0, "neutral": 80.0, "negative": 0.0},
                        "mean_top_confidence": 0.81,
                    },
                    "emotion": {
                        "counts": {"joy": 0, "neutral": 3, "surprise": 1, "disgust": 1,
                                   "anger": 0, "fear": 0, "sadness": 0},
                        "percent": {"joy": 0.0, "neutral": 60.0, "surprise": 20.0,
                                    "disgust": 20.0, "anger": 0.0, "fear": 0.0, "sadness": 0.0},
                        "mean_top_confidence": 0.62,
                    },
                },
                "turn_taking": {"turn_count": 4, "mean_gap_before_seconds": 1.42, "interruption_count": 1},
            },
            {
                "speaker_id": "session_speaker_b", "speaker": "Speaker B",
                "participation": {
                    "talk_seconds": 26.3, "talk_percent": 33.5,
                    "segment_count": 4, "word_count": 114,
                    "words_per_minute": 260.0,
                    "mean_segment_seconds": 6.6, "median_segment_seconds": 6.0,
                    "max_segment_seconds": 9.1,
                },
                "sentiment": {
                    "polarity": {
                        "counts": {"positive": 1, "neutral": 3, "negative": 0},
                        "percent": {"positive": 25.0, "neutral": 75.0, "negative": 0.0},
                        "mean_top_confidence": 0.79,
                    },
                    "emotion": {
                        "counts": {"joy": 1, "neutral": 2, "surprise": 0, "disgust": 1,
                                   "anger": 0, "fear": 0, "sadness": 0},
                        "percent": {"joy": 25.0, "neutral": 50.0, "surprise": 0.0,
                                    "disgust": 25.0, "anger": 0.0, "fear": 0.0, "sadness": 0.0},
                        "mean_top_confidence": 0.58,
                    },
                },
                "turn_taking": {"turn_count": 4, "mean_gap_before_seconds": 0.85, "interruption_count": 0},
            },
        ],
        "pairwise_followers": {"session_speaker_a": {"session_speaker_a": 0}},  # single-pair => omitted
        "timeline": [
            {"bucket_start_s": 0, "bucket_end_s": 300,
             "per_speaker_talk_s": {"session_speaker_a": 52.1, "session_speaker_b": 26.3},
             "per_speaker_polarity_mode": {"session_speaker_a": "neutral",
                                            "session_speaker_b": "neutral"},
             "per_speaker_emotion_mode": {"session_speaker_a": "neutral",
                                           "session_speaker_b": "neutral"}}
        ],
        "highlights": [
            {"kind": "longest_segment", "speaker_id": "session_speaker_a",
             "start": 0.0, "end": 42.4, "value_s": 42.4,
             "quote": "The whole idea is that both of them should..."},
            {"kind": "most_positive", "speaker_id": "session_speaker_b",
             "start": 65.0, "end": 71.0, "polarity_score": 0.94,
             "quote": "Yes, that's a great point about the radar pairing."},
        ],
    }


def _session_meta():
    return {
        "audio_file": "fixture.wav",
        "analyzed_at": "2026-05-16T18:42:01Z",
    }


def _name_lookup_from_speakers(speakers):
    return {s["speaker_id"]: s["speaker"] for s in speakers}


class TestRenderer:
    def test_full_report_matches_golden(self):
        metrics = _full_metrics_block()
        meta = _session_meta()
        out = renderer.render_markdown(metrics, meta)
        expected = (FIXTURES / "golden_report.md").read_text(encoding="utf-8")
        assert out == expected

    def test_empty_highlights_omits_notable_moments_section(self):
        metrics = _full_metrics_block()
        metrics["highlights"] = []
        out = renderer.render_markdown(metrics, _session_meta())
        assert "## Notable moments" not in out

    def test_single_speaker_omits_pairwise_section(self):
        metrics = _full_metrics_block()
        metrics["speakers"] = [metrics["speakers"][0]]
        out = renderer.render_markdown(metrics, _session_meta())
        assert "## Who follows whom" not in out

    def test_single_bucket_omits_activity_section(self):
        metrics = _full_metrics_block()
        # Already single-bucket. Just confirm.
        out = renderer.render_markdown(metrics, _session_meta())
        assert "## Activity over time" not in out

    def test_disgust_footnote_present_when_emotion_table_rendered(self):
        metrics = _full_metrics_block()
        out = renderer.render_markdown(metrics, _session_meta())
        assert "registered disagreement" in out
