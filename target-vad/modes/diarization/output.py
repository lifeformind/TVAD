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


def _enrolled_users_in_first_appearance_order(segments: List[DiarizationSegment]) -> List[Dict[str, str]]:
    """Return list of {id, name} objects deduped by id, in first-appearance order."""
    seen = set()
    result = []
    for s in segments:
        if s.speaker_id == "unknown":
            continue
        if s.speaker_id not in seen:
            seen.add(s.speaker_id)
            result.append({"id": s.speaker_id, "name": s.speaker})
    return result


def write_json(
    path: str,
    *,
    audio_file: str,
    duration_s: float,
    diarized_at: str,
    config: Dict[str, Any],
    segments: List[DiarizationSegment],
    passes_run: Optional[List[str]] = None,
) -> None:
    """Write the diarization timeline as JSON. Schema per spec.

    If `passes_run` is provided, it is emitted as a top-level field. Future
    analysis passes (transcription, sentiment, ...) read and append to this list
    on subsequent enrichment runs. Omitted when None to preserve existing-caller
    behavior.
    """
    payload = {
        "audio_file": audio_file,
        "duration_s": duration_s,
        "diarized_at": diarized_at,
        "config": config,
        "enrolled_users_matched": _enrolled_users_in_first_appearance_order(segments),
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
