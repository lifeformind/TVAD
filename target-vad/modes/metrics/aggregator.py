"""Pure aggregator functions over a list of diarization segments.

Each function takes a List[dict] of segments (the `segments` field of a Phase-2B
diarization JSON, with `text`, `words`, and `sentiment` fields present per the
plan's prerequisites) and returns a plain dict. No side effects, no model loads.
Determinism: identical input -> identical output.
"""

from statistics import mean, median
from typing import Dict, List


def aggregate_participation(segments: List[Dict]) -> Dict:
    """Compute participation stats (Tier 1).

    Returns:
        {
          "session": {speech_duration_s, total_segments, total_words,
                      unique_speakers, identified_speakers, unknown_segments},
          "per_speaker": {sid: {talk_seconds, talk_percent, segment_count,
                                 word_count, words_per_minute,
                                 mean_segment_seconds, median_segment_seconds,
                                 max_segment_seconds}}
        }

    Note: speech_duration_s is the raw sum of segment durations (overlap is
    double-counted). talk_percent for each speaker is computed against this
    sum, so per-speaker percentages always sum to 100%. silence is computed
    elsewhere via the merged interval union (so it can't go negative).
    """
    if not segments:
        return {
            "session": {
                "speech_duration_s": 0.0,
                "total_segments": 0,
                "total_words": 0,
                "unique_speakers": 0,
                "identified_speakers": 0,
                "unknown_segments": 0,
            },
            "per_speaker": {},
        }

    # Group segments by speaker_id, preserving first-appearance order.
    by_speaker: Dict[str, List[Dict]] = {}
    for seg in segments:
        sid = seg["speaker_id"]
        by_speaker.setdefault(sid, []).append(seg)

    speech_duration_s = sum(s["end"] - s["start"] for s in segments)
    total_words = sum(len(s.get("words") or []) for s in segments)
    unknown_segments = sum(1 for s in segments if s["speaker_id"] == "unknown")
    identified_speakers = sum(1 for sid in by_speaker if sid != "unknown")

    per_speaker: Dict[str, Dict] = {}
    for sid, segs in by_speaker.items():
        durations = [s["end"] - s["start"] for s in segs]
        talk = sum(durations)
        wc = sum(len(s.get("words") or []) for s in segs)
        per_speaker[sid] = {
            "talk_seconds": round(talk, 2),
            "talk_percent": round(100.0 * talk / speech_duration_s, 1) if speech_duration_s else 0.0,
            "segment_count": len(segs),
            "word_count": wc,
            "words_per_minute": round(60.0 * wc / talk, 1) if talk else None,
            "mean_segment_seconds": round(mean(durations), 2),
            "median_segment_seconds": round(median(durations), 2),
            "max_segment_seconds": round(max(durations), 2),
        }

    return {
        "session": {
            "speech_duration_s": round(speech_duration_s, 2),
            "total_segments": len(segments),
            "total_words": total_words,
            "unique_speakers": len(by_speaker),
            "identified_speakers": identified_speakers,
            "unknown_segments": unknown_segments,
        },
        "per_speaker": per_speaker,
    }


_POLARITY_LABELS = ("positive", "neutral", "negative")
_EMOTION_LABELS = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")


def aggregate_sentiment(segments: List[Dict]) -> Dict:
    """Compute polarity + emotion distributions, per speaker and session-wide.

    Segments with sentiment: null are skipped (no signal). Percentages use
    the speaker's classified-segment count as denominator. mean_top_confidence
    is None for speakers with zero classified segments.
    """
    by_speaker: Dict[str, List[Dict]] = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker_id"], []).append(seg)

    sess_pol = {k: 0 for k in _POLARITY_LABELS}
    sess_emo = {k: 0 for k in _EMOTION_LABELS}

    per_speaker: Dict[str, Dict] = {}
    for sid, segs in by_speaker.items():
        pol_counts = {k: 0 for k in _POLARITY_LABELS}
        emo_counts = {k: 0 for k in _EMOTION_LABELS}
        pol_top_scores: List[float] = []
        emo_top_scores: List[float] = []

        for s in segs:
            sent = s.get("sentiment")
            if sent is None:
                continue
            pol = sent["polarity"]
            emo = sent["emotion"]
            pol_counts[pol["label"]] += 1
            emo_counts[emo["label"]] += 1
            sess_pol[pol["label"]] += 1
            sess_emo[emo["label"]] += 1
            pol_top_scores.append(float(pol["score"]))
            emo_top_scores.append(float(emo["score"]))

        classified = sum(pol_counts.values())
        if classified:
            pol_percent = {k: round(100.0 * pol_counts[k] / classified, 1) for k in _POLARITY_LABELS}
            emo_total = sum(emo_counts.values())
            emo_percent = {k: round(100.0 * emo_counts[k] / emo_total, 1) for k in _EMOTION_LABELS}
            pol_mean = round(mean(pol_top_scores), 2)
            emo_mean = round(mean(emo_top_scores), 2)
        else:
            pol_percent = {k: 0.0 for k in _POLARITY_LABELS}
            emo_percent = {k: 0.0 for k in _EMOTION_LABELS}
            pol_mean = None
            emo_mean = None

        per_speaker[sid] = {
            "polarity": {
                "counts": pol_counts,
                "percent": pol_percent,
                "mean_top_confidence": pol_mean,
            },
            "emotion": {
                "counts": emo_counts,
                "percent": emo_percent,
                "mean_top_confidence": emo_mean,
            },
        }

    return {
        "session": {
            "polarity_distribution": sess_pol,
            "emotion_distribution": sess_emo,
        },
        "per_speaker": per_speaker,
    }
