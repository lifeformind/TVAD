"""Tests for the metrics aggregator functions."""

import pytest

from modes.metrics import aggregator


def _seg(start, end, sid, name=None, text="", words=None, sentiment=None):
    """Build a segment dict with sane defaults."""
    return {
        "start": start,
        "end": end,
        "speaker_id": sid,
        "speaker": name or sid,
        "text": text,
        "words": words if words is not None else [],
        "sentiment": sentiment,
    }


class TestAggregateParticipation:
    def test_two_speakers_basic(self):
        segments = [
            _seg(0.0, 10.0, "alice", text="hi there friend",
                 words=[{"start": 0, "end": 1, "word": "hi", "probability": 0.9},
                        {"start": 1, "end": 2, "word": "there", "probability": 0.9},
                        {"start": 2, "end": 3, "word": "friend", "probability": 0.9}]),
            _seg(10.0, 14.0, "bob", text="hey",
                 words=[{"start": 10, "end": 11, "word": "hey", "probability": 0.9}]),
            _seg(14.0, 20.0, "alice", text="ok cool",
                 words=[{"start": 14, "end": 15, "word": "ok", "probability": 0.9},
                        {"start": 15, "end": 16, "word": "cool", "probability": 0.9}]),
        ]

        result = aggregator.aggregate_participation(segments)

        assert set(result.keys()) == {"session", "per_speaker"}
        sess = result["session"]
        assert sess["speech_duration_s"] == pytest.approx(20.0)
        assert sess["total_segments"] == 3
        assert sess["total_words"] == 6
        assert sess["unique_speakers"] == 2
        assert sess["identified_speakers"] == 2
        assert sess["unknown_segments"] == 0

        alice = result["per_speaker"]["alice"]
        assert alice["talk_seconds"] == pytest.approx(16.0)
        assert alice["talk_percent"] == pytest.approx(80.0)
        assert alice["segment_count"] == 2
        assert alice["word_count"] == 5
        assert alice["words_per_minute"] == pytest.approx(18.8)
        assert alice["mean_segment_seconds"] == pytest.approx(8.0)
        assert alice["median_segment_seconds"] == pytest.approx(8.0)
        assert alice["max_segment_seconds"] == pytest.approx(10.0)

        bob = result["per_speaker"]["bob"]
        assert bob["talk_seconds"] == pytest.approx(4.0)
        assert bob["talk_percent"] == pytest.approx(20.0)
        assert bob["segment_count"] == 1
        assert bob["word_count"] == 1

    def test_unknown_counts_as_regular_bucket_but_separately_summarized(self):
        segments = [
            _seg(0.0, 5.0, "alice"),
            _seg(5.0, 10.0, "unknown"),
        ]
        result = aggregator.aggregate_participation(segments)
        assert result["session"]["unique_speakers"] == 2
        assert result["session"]["identified_speakers"] == 1
        assert result["session"]["unknown_segments"] == 1
        assert "unknown" in result["per_speaker"]
        assert result["per_speaker"]["unknown"]["talk_seconds"] == pytest.approx(5.0)

    def test_empty_segments(self):
        result = aggregator.aggregate_participation([])
        assert result["session"]["speech_duration_s"] == 0.0
        assert result["session"]["total_segments"] == 0
        assert result["session"]["total_words"] == 0
        assert result["session"]["unique_speakers"] == 0
        assert result["session"]["identified_speakers"] == 0
        assert result["session"]["unknown_segments"] == 0
        assert result["per_speaker"] == {}

    def test_zero_duration_segment_gives_none_wpm(self):
        segments = [_seg(0.0, 0.0, "alice", words=[{"start": 0, "end": 0, "word": "x", "probability": 0.9}])]
        result = aggregator.aggregate_participation(segments)
        assert result["per_speaker"]["alice"]["words_per_minute"] is None


def _sent(pol_label, emo_label, pol_score=0.8, emo_score=0.6):
    """Build a canonical sentiment dict for a segment."""
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


class TestAggregateSentiment:
    def test_per_speaker_counts_and_percents(self):
        segments = [
            _seg(0, 5, "alice", sentiment=_sent("positive", "joy")),
            _seg(5, 10, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(10, 15, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(15, 20, "alice", sentiment=_sent("neutral", "surprise")),
            _seg(20, 25, "bob", sentiment=_sent("positive", "joy")),
        ]
        result = aggregator.aggregate_sentiment(segments)

        assert set(result.keys()) == {"session", "per_speaker"}
        sess = result["session"]
        assert sess["polarity_distribution"] == {"positive": 2, "neutral": 3, "negative": 0}
        assert sess["emotion_distribution"]["joy"] == 2
        assert sess["emotion_distribution"]["neutral"] == 2
        assert sess["emotion_distribution"]["surprise"] == 1

        alice = result["per_speaker"]["alice"]
        assert alice["polarity"]["counts"] == {"positive": 1, "neutral": 3, "negative": 0}
        assert alice["polarity"]["percent"] == {"positive": 25.0, "neutral": 75.0, "negative": 0.0}
        assert alice["emotion"]["counts"]["joy"] == 1
        assert alice["emotion"]["counts"]["neutral"] == 2
        assert alice["emotion"]["counts"]["surprise"] == 1
        # Mean top confidence = avg of pol_score across alice's segments = 0.8
        assert alice["polarity"]["mean_top_confidence"] == pytest.approx(0.8)

    def test_null_sentiment_skipped_in_denominator(self):
        segments = [
            _seg(0, 5, "alice", sentiment=_sent("positive", "joy")),
            _seg(5, 10, "alice", sentiment=None),
            _seg(10, 15, "alice", sentiment=None),
        ]
        result = aggregator.aggregate_sentiment(segments)
        alice = result["per_speaker"]["alice"]
        # Only the 1 non-null segment counts. Percent uses denominator=1, not 3.
        assert alice["polarity"]["counts"] == {"positive": 1, "neutral": 0, "negative": 0}
        assert alice["polarity"]["percent"]["positive"] == 100.0

    def test_speaker_with_all_null_sentiment(self):
        segments = [
            _seg(0, 5, "alice", sentiment=None),
            _seg(5, 10, "alice", sentiment=None),
        ]
        result = aggregator.aggregate_sentiment(segments)
        alice = result["per_speaker"]["alice"]
        assert alice["polarity"]["counts"] == {"positive": 0, "neutral": 0, "negative": 0}
        assert alice["polarity"]["percent"] == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        assert alice["polarity"]["mean_top_confidence"] is None


class TestAggregateTurnTaking:
    def test_consecutive_same_speaker_collapses_to_one_turn(self):
        # Alice: 2 adjacent same-speaker segments -> 1 turn; Bob: 1 turn between.
        segments = [
            _seg(0.0, 5.0, "alice"),
            _seg(5.0, 10.0, "alice"),     # collapses with prior turn
            _seg(11.0, 15.0, "bob"),       # gap = 1.0
            _seg(16.0, 20.0, "alice"),     # gap = 1.0
        ]
        result = aggregator.aggregate_turn_taking(segments)
        assert result["per_speaker"]["alice"]["turn_count"] == 2
        assert result["per_speaker"]["bob"]["turn_count"] == 1

    def test_mean_gap_before_excludes_first_turn(self):
        segments = [
            _seg(0.0, 5.0, "alice"),       # first turn — no gap
            _seg(7.0, 10.0, "bob"),         # gap = 2.0
            _seg(13.0, 15.0, "alice"),      # gap = 3.0
            _seg(20.0, 25.0, "alice"),      # same-speaker, same turn (no new gap)
        ]
        result = aggregator.aggregate_turn_taking(segments)
        # Alice has 2 turns; only the second turn (start=13) counts a gap (3.0).
        assert result["per_speaker"]["alice"]["mean_gap_before_seconds"] == pytest.approx(3.0)
        # Bob's first and only turn (start=7) counts a gap of 2.0.
        assert result["per_speaker"]["bob"]["mean_gap_before_seconds"] == pytest.approx(2.0)

    def test_interruption_counted_when_overlap(self):
        segments = [
            _seg(0.0, 10.0, "alice"),
            _seg(8.0, 12.0, "bob"),         # starts before alice ends → interruption
        ]
        result = aggregator.aggregate_turn_taking(segments)
        assert result["per_speaker"]["bob"]["interruption_count"] == 1
        assert result["per_speaker"]["alice"]["interruption_count"] == 0

    def test_single_turn_speaker_has_no_gap(self):
        segments = [
            _seg(0.0, 5.0, "alice"),
        ]
        result = aggregator.aggregate_turn_taking(segments)
        assert result["per_speaker"]["alice"]["turn_count"] == 1
        # No prior turn → no gap. Use None to signal "not applicable".
        assert result["per_speaker"]["alice"]["mean_gap_before_seconds"] is None
        assert result["per_speaker"]["alice"]["interruption_count"] == 0

    def test_interrupting_speaker_does_not_contaminate_mean_gap(self):
        # Alice has two turns: first sets the session start, second interrupts Bob.
        # Bob has one turn with a clean gap.
        segments = [
            _seg(0.0, 5.0, "alice"),
            _seg(7.0, 15.0, "bob"),          # gap = 2.0 for bob
            _seg(10.0, 12.0, "alice"),       # interrupts bob (starts before 15.0)
        ]
        result = aggregator.aggregate_turn_taking(segments)
        # Alice has 2 turns. First turn: no gap. Second turn: interrupts.
        # → no gaps contributed, mean_gap stays None; interruption_count = 1.
        assert result["per_speaker"]["alice"]["turn_count"] == 2
        assert result["per_speaker"]["alice"]["mean_gap_before_seconds"] is None
        assert result["per_speaker"]["alice"]["interruption_count"] == 1
        # Bob: 1 turn, gap = 2.0, no interruption.
        assert result["per_speaker"]["bob"]["mean_gap_before_seconds"] == pytest.approx(2.0)
        assert result["per_speaker"]["bob"]["interruption_count"] == 0

    def test_empty_segments(self):
        result = aggregator.aggregate_turn_taking([])
        assert result == {"per_speaker": {}}


class TestAggregatePairwise:
    def test_who_follows_whom_with_collapsed_turns(self):
        segments = [
            _seg(0, 5, "alice"),
            _seg(5, 10, "alice"),     # same turn as previous
            _seg(10, 15, "bob"),       # alice → bob
            _seg(15, 20, "alice"),     # bob → alice
            _seg(20, 25, "alice"),     # same turn
            _seg(25, 30, "bob"),       # alice → bob
        ]
        result = aggregator.aggregate_pairwise(segments)
        # Two alice→bob transitions, one bob→alice. Self-loops excluded.
        assert result["alice"]["bob"] == 2
        assert result["bob"]["alice"] == 1
        assert result["alice"]["alice"] == 0
        assert result["bob"]["bob"] == 0

    def test_unknown_appears_as_row_and_column_when_present(self):
        segments = [
            _seg(0, 5, "alice"),
            _seg(5, 10, "unknown"),
            _seg(10, 15, "alice"),
        ]
        result = aggregator.aggregate_pairwise(segments)
        assert result["alice"]["unknown"] == 1
        assert result["unknown"]["alice"] == 1

    def test_single_speaker_matrix_empty_off_diagonal(self):
        segments = [
            _seg(0, 5, "alice"),
            _seg(5, 10, "alice"),
        ]
        result = aggregator.aggregate_pairwise(segments)
        assert result == {"alice": {"alice": 0}}
