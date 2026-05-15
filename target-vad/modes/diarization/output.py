"""JSON and RTTM serializers for diarization output.

JSON schema is documented in docs/superpowers/specs/2026-05-14-classroom-diarization-design.md.
RTTM is the standard NIST speaker-diarization format:
    SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DiarizationSegment:
    """One timestamped, labeled speech segment."""
    start: float
    end: float
    speaker: str  # enrolled name or the literal "unknown"


def _enrolled_users_in_first_appearance_order(segments: List[DiarizationSegment]) -> List[str]:
    seen = set()
    result = []
    for s in segments:
        if s.speaker == "unknown":
            continue
        if s.speaker not in seen:
            seen.add(s.speaker)
            result.append(s.speaker)
    return result


def write_json(
    path: str,
    *,
    audio_file: str,
    duration_s: float,
    diarized_at: str,
    config: Dict[str, Any],
    segments: List[DiarizationSegment],
) -> None:
    """Write the diarization timeline as JSON. Schema per spec."""
    payload = {
        "audio_file": audio_file,
        "duration_s": duration_s,
        "diarized_at": diarized_at,
        "config": config,
        "enrolled_users_matched": _enrolled_users_in_first_appearance_order(segments),
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_rttm(
    path: str,
    *,
    audio_file_id: str,
    segments: List[DiarizationSegment],
) -> None:
    """Write segments as RTTM. Empty segments → empty file."""
    lines = []
    for s in segments:
        duration = s.end - s.start
        lines.append(
            f"SPEAKER {audio_file_id} 1 {s.start:.3f} {duration:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
        )
    content = "\n".join(lines)
    if content:
        content += "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
