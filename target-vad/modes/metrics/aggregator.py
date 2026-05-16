"""Pure aggregator functions over a list of diarization segments.

Each function takes a List[dict] of segments (the `segments` field of a Phase-2B
diarization JSON, with `text`, `words`, and `sentiment` fields present per the
plan's prerequisites) and returns a plain dict. No side effects, no model loads.
Determinism: identical input -> identical output.
"""

import math
from collections import Counter
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


def _turns_from_segments(segments: List[Dict]) -> List[Dict]:
    """Collapse contiguous same-speaker segments into turns.

    A turn boundary occurs whenever the speaker_id changes. Returns a list of
    {speaker_id, start, end} dicts where start is the first segment's start and
    end is the last segment's end in that run.
    """
    turns: List[Dict] = []
    for seg in segments:
        if turns and turns[-1]["speaker_id"] == seg["speaker_id"]:
            turns[-1]["end"] = seg["end"]
        else:
            turns.append({
                "speaker_id": seg["speaker_id"],
                "start": seg["start"],
                "end": seg["end"],
            })
    return turns


def aggregate_turn_taking(segments: List[Dict]) -> Dict:
    """Compute per-speaker turn count, mean gap before turn, interruption count.

    A turn = a contiguous run of same-speaker segments. Gap and interruption are
    evaluated against the previous turn (any speaker). First turn in the session
    has no gap.

    mean_gap_before_seconds is computed only over non-interrupting turns — when a
    speaker interrupts (their turn starts before the previous turn ends), the
    event contributes to interruption_count, not the gap mean. This keeps the
    two metrics orthogonal: interruption_count tells you HOW OFTEN they jump in,
    mean_gap_before_seconds tells you their PACING on normal turn-take.
    """
    if not segments:
        return {"per_speaker": {}}
    turns = _turns_from_segments(segments)

    per_speaker: Dict[str, Dict] = {}
    for i, turn in enumerate(turns):
        sid = turn["speaker_id"]
        per_speaker.setdefault(sid, {"_gaps": [], "_interruptions": 0, "_turn_count": 0})
        per_speaker[sid]["_turn_count"] += 1
        if i > 0:
            prev = turns[i - 1]
            gap = turn["start"] - prev["end"]
            if turn["start"] < prev["end"]:
                per_speaker[sid]["_interruptions"] += 1
            else:
                per_speaker[sid]["_gaps"].append(gap)

    result: Dict[str, Dict] = {}
    for sid, accum in per_speaker.items():
        gaps = accum["_gaps"]
        result[sid] = {
            "turn_count": accum["_turn_count"],
            "mean_gap_before_seconds": round(mean(gaps), 2) if gaps else None,
            "interruption_count": accum["_interruptions"],
        }

    return {"per_speaker": result}


def aggregate_pairwise(segments: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compute the who-follows-whom transition matrix.

    Builds turns first (collapses contiguous same-speaker segments). For each
    adjacent pair of turns (prev, next), increments matrix[prev_sid][next_sid].
    The returned matrix is rectangular: every observed speaker is both a row
    and a column, with self-transitions always 0.
    """
    turns = _turns_from_segments(segments)
    speakers = []
    seen = set()
    for t in turns:
        if t["speaker_id"] not in seen:
            seen.add(t["speaker_id"])
            speakers.append(t["speaker_id"])

    matrix: Dict[str, Dict[str, int]] = {a: {b: 0 for b in speakers} for a in speakers}
    for i in range(1, len(turns)):
        prev_sid = turns[i - 1]["speaker_id"]
        next_sid = turns[i]["speaker_id"]
        if prev_sid != next_sid:
            matrix[prev_sid][next_sid] += 1
    return matrix


def aggregate_timeline(
    segments: List[Dict],
    duration_s: float,
    bucket_seconds: int,
) -> List[Dict]:
    """Bucket segments into fixed time windows.

    For each bucket:
      - per_speaker_talk_s: segment-overlap with bucket, apportioned proportionally
        for segments straddling boundaries.
      - per_speaker_polarity_mode / _emotion_mode: most-frequent label among the
        speaker's segments whose START falls within the bucket. Omitted if no
        such segments. Tie-broken by alphabetical label (stable on rerun).
        (Start-based, not overlap-based, to avoid double-counting a sentiment
        signal across bucket boundaries.)
    """
    n_buckets = max(1, math.ceil(duration_s / bucket_seconds)) if duration_s > 0 else 1
    buckets: List[Dict] = []
    for i in range(n_buckets):
        buckets.append({
            "bucket_start_s": i * bucket_seconds,
            "bucket_end_s": (i + 1) * bucket_seconds,
            "per_speaker_talk_s": {},
            "_pol_counters": {},
            "_emo_counters": {},
        })

    for seg in segments:
        sid = seg["speaker_id"]
        # Apportion talk seconds across overlapping buckets.
        for b in buckets:
            lo = max(seg["start"], b["bucket_start_s"])
            hi = min(seg["end"], b["bucket_end_s"])
            if hi > lo:
                b["per_speaker_talk_s"][sid] = round(
                    b["per_speaker_talk_s"].get(sid, 0.0) + (hi - lo), 2
                )

        # Credit polarity/emotion mode to the bucket containing the start.
        start_bucket_idx = min(int(seg["start"] // bucket_seconds), n_buckets - 1)
        b = buckets[start_bucket_idx]
        sent = seg.get("sentiment")
        if sent is not None:
            b["_pol_counters"].setdefault(sid, Counter())[sent["polarity"]["label"]] += 1
            b["_emo_counters"].setdefault(sid, Counter())[sent["emotion"]["label"]] += 1

    # Materialize modes from counters (stable tie-break: count desc, label asc).
    out: List[Dict] = []
    for b in buckets:
        pol_mode: Dict[str, str] = {}
        emo_mode: Dict[str, str] = {}
        for sid, ctr in b["_pol_counters"].items():
            top_label = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            pol_mode[sid] = top_label
        for sid, ctr in b["_emo_counters"].items():
            top_label = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            emo_mode[sid] = top_label
        out.append({
            "bucket_start_s": b["bucket_start_s"],
            "bucket_end_s": b["bucket_end_s"],
            "per_speaker_talk_s": b["per_speaker_talk_s"],
            "per_speaker_polarity_mode": pol_mode,
            "per_speaker_emotion_mode": emo_mode,
        })
    return out


def _truncate_quote(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def select_highlights(
    segments: List[Dict],
    timeline: List[Dict],
    top_k: int,
    quote_max_chars: int,
) -> List[Dict]:
    """Pick up to top_k highlights deterministically.

    Priority order: longest_segment, most_positive, most_negative,
    high_disgust_window, quietest_window, busiest_window, solo_dominator.
    Each kind's selection rule is documented in the spec. Ties are broken by
    earliest start time, then alphabetical speaker_id (stable on rerun).
    """
    highlights: List[Dict] = []

    def cap_reached() -> bool:
        return len(highlights) >= top_k

    # longest_segment
    if segments and not cap_reached():
        # Sort: longest duration desc, earliest start asc, alphabetical sid asc.
        winner = sorted(
            segments,
            key=lambda s: (-(s["end"] - s["start"]), s["start"], s["speaker_id"]),
        )[0]
        highlights.append({
            "kind": "longest_segment",
            "speaker_id": winner["speaker_id"],
            "start": round(winner["start"], 2),
            "end": round(winner["end"], 2),
            "value_s": round(winner["end"] - winner["start"], 2),
            "quote": _truncate_quote(winner.get("text", ""), quote_max_chars),
        })

    # most_positive — only consider segments whose label IS positive.
    pos_candidates = [s for s in segments
                      if (s.get("sentiment") is not None
                          and s["sentiment"]["polarity"]["label"] == "positive")]
    if pos_candidates and not cap_reached():
        winner = sorted(
            pos_candidates,
            key=lambda s: (-s["sentiment"]["polarity"]["scores"]["positive"],
                           s["start"], s["speaker_id"]),
        )[0]
        highlights.append({
            "kind": "most_positive",
            "speaker_id": winner["speaker_id"],
            "start": round(winner["start"], 2),
            "end": round(winner["end"], 2),
            "polarity_score": round(winner["sentiment"]["polarity"]["scores"]["positive"], 2),
            "quote": _truncate_quote(winner.get("text", ""), quote_max_chars),
        })

    # most_negative
    neg_candidates = [s for s in segments
                      if (s.get("sentiment") is not None
                          and s["sentiment"]["polarity"]["label"] == "negative")]
    if neg_candidates and not cap_reached():
        winner = sorted(
            neg_candidates,
            key=lambda s: (-s["sentiment"]["polarity"]["scores"]["negative"],
                           s["start"], s["speaker_id"]),
        )[0]
        highlights.append({
            "kind": "most_negative",
            "speaker_id": winner["speaker_id"],
            "start": round(winner["start"], 2),
            "end": round(winner["end"], 2),
            "polarity_score": round(winner["sentiment"]["polarity"]["scores"]["negative"], 2),
            "quote": _truncate_quote(winner.get("text", ""), quote_max_chars),
        })

    # high_disgust_window — bucket with max disgust-segment count.
    disgust_counts: List[Dict] = []
    for b in timeline:
        # Count disgust-labeled segment-starts in this bucket per speaker.
        total = 0
        per_sid: Dict[str, int] = {}
        for s in segments:
            if (b["bucket_start_s"] <= s["start"] < b["bucket_end_s"]
                    and s.get("sentiment") is not None
                    and s["sentiment"]["emotion"]["label"] == "disgust"):
                total += 1
                per_sid[s["speaker_id"]] = per_sid.get(s["speaker_id"], 0) + 1
        if total > 0:
            top_sid = sorted(per_sid.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            disgust_counts.append({"bucket": b, "count": total, "speaker_id": top_sid})
    if disgust_counts and not cap_reached():
        winner = sorted(disgust_counts, key=lambda d: (-d["count"], d["bucket"]["bucket_start_s"]))[0]
        highlights.append({
            "kind": "high_disgust_window",
            "bucket_start_s": winner["bucket"]["bucket_start_s"],
            "bucket_end_s": winner["bucket"]["bucket_end_s"],
            "speaker_id": winner["speaker_id"],
            "count": winner["count"],
        })

    # quietest_window / busiest_window / solo_dominator — only when >= 2 buckets.
    if len(timeline) >= 2:
        # Total talk per bucket = sum across all speakers.
        bucket_totals = [
            (b, round(sum(b["per_speaker_talk_s"].values()), 2)) for b in timeline
        ]

        if not cap_reached():
            winner = sorted(bucket_totals, key=lambda t: (t[1], t[0]["bucket_start_s"]))[0]
            highlights.append({
                "kind": "quietest_window",
                "bucket_start_s": winner[0]["bucket_start_s"],
                "bucket_end_s": winner[0]["bucket_end_s"],
                "total_talk_s": winner[1],
            })

        if not cap_reached():
            winner = sorted(bucket_totals, key=lambda t: (-t[1], t[0]["bucket_start_s"]))[0]
            highlights.append({
                "kind": "busiest_window",
                "bucket_start_s": winner[0]["bucket_start_s"],
                "bucket_end_s": winner[0]["bucket_end_s"],
                "total_talk_s": winner[1],
            })

        if not cap_reached():
            for b, total in bucket_totals:
                if total < 60.0:
                    continue
                # Find max-talk speaker in this bucket.
                # Sort by talk desc, alphabetical sid asc — consistent with other
                # tie-breaking in this function.
                top_sid, top_talk = sorted(
                    b["per_speaker_talk_s"].items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[0]
                if top_talk / total >= 0.8:
                    highlights.append({
                        "kind": "solo_dominator",
                        "bucket_start_s": b["bucket_start_s"],
                        "bucket_end_s": b["bucket_end_s"],
                        "speaker_id": top_sid,
                        "talk_s": round(top_talk, 2),
                        "total_talk_s": total,
                    })
                    break  # Only the first qualifying bucket.

    return highlights[:top_k]
