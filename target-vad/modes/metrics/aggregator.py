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
