"""JSON and RTTM serializers for diarization output.

JSON schema is documented in
docs/superpowers/specs/2026-05-15-in-session-enrollment-design.md.
RTTM uses the standard NIST format with the speaker_id (not display name) in
the speaker column (field 8) — RTTM consumers expect stable tokens.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DiarizationSegment:
    """One timestamped, labeled speech segment.

    speaker_id is the stable storage key (id from EnrollmentStore or the literal
    'unknown'). speaker is the display name (free-form string from users.json or
    'unknown' when speaker_id == 'unknown').
    """
    start: float
    end: float
    speaker_id: str
    speaker: str


def _enrolled_users_in_first_appearance_order(
    segments: List[DiarizationSegment],
    enrolled_ids: set,
) -> List[Dict[str, str]]:
    """Return list of {id, name} objects deduped by id, in first-appearance order.

    Only emits speakers whose id is in `enrolled_ids` - filters out both the
    literal 'unknown' sentinel and pyannote-generated ids like 'SPEAKER_00'
    that don't correspond to any enrolled voiceprint.
    """
    seen = set()
    result = []
    for s in segments:
        if s.speaker_id not in enrolled_ids:
            continue
        if s.speaker_id not in seen:
            seen.add(s.speaker_id)
            result.append({"id": s.speaker_id, "name": s.speaker})
    return result


def _unknown_speakers_observed_in_first_appearance_order(
    segments: List[DiarizationSegment],
    enrolled_ids: set,
) -> List[Dict[str, Any]]:
    """Return list of {id, segment_count, talk_seconds} for recurring-unknown speakers.

    A 'recurring unknown' is a speaker_id that is neither the literal 'unknown'
    sentinel nor in the enrolled_ids set - i.e., a pyannote-generated id that
    was preserved by the threshold gate in ClusterIdentifier. Ordered by first
    appearance (earliest segment.start).
    """
    counts: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for s in segments:
        sid = s.speaker_id
        if sid == "unknown" or sid in enrolled_ids:
            continue
        if sid not in counts:
            counts[sid] = {"id": sid, "segment_count": 0, "talk_seconds": 0.0}
            order.append(sid)
        counts[sid]["segment_count"] += 1
        counts[sid]["talk_seconds"] += s.end - s.start
    # Round talk_seconds to 2 decimals for stable JSON output.
    for sid in counts:
        counts[sid]["talk_seconds"] = round(counts[sid]["talk_seconds"], 2)
    return [counts[sid] for sid in order]


def write_json(
    path: str,
    *,
    audio_file: str,
    duration_s: float,
    diarized_at: str,
    config: Dict[str, Any],
    segments: List[DiarizationSegment],
    enrolled_ids: set,
    passes_run: Optional[List[str]] = None,
) -> None:
    """Write the diarization timeline as JSON. Schema per spec.

    `enrolled_ids` is the set of ids that correspond to enrolled voiceprints
    (persistent or session-scoped). Speakers with ids outside this set and not
    equal to 'unknown' are treated as recurring unknowns (preserved pyannote
    cluster ids per the threshold gate in ClusterIdentifier).

    If `passes_run` is provided, it is emitted as a top-level field.
    """
    payload = {
        "audio_file": audio_file,
        "duration_s": duration_s,
        "diarized_at": diarized_at,
        "config": config,
        "enrolled_users_matched": _enrolled_users_in_first_appearance_order(segments, enrolled_ids),
        "unknown_speakers_observed": _unknown_speakers_observed_in_first_appearance_order(segments, enrolled_ids),
        "segments": [
            {"start": s.start, "end": s.end, "speaker_id": s.speaker_id, "speaker": s.speaker}
            for s in segments
        ],
    }
    if passes_run is not None:
        payload["passes_run"] = list(passes_run)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_rttm(
    path: str,
    *,
    audio_file_id: str,
    segments: List[DiarizationSegment],
) -> None:
    """Write segments as RTTM. The speaker column uses speaker_id (stable token)."""
    lines = []
    for s in segments:
        duration = s.end - s.start
        lines.append(
            f"SPEAKER {audio_file_id} 1 {s.start:.3f} {duration:.3f} <NA> <NA> {s.speaker_id} <NA> <NA>"
        )
    content = "\n".join(lines)
    if content:
        content += "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
