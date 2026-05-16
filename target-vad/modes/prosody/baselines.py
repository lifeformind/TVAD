"""Pure per-speaker baseline aggregation over the prosody-enriched segment list.

For each speaker_id appearing in segments, computes:
  - pitch_hz_median:  median across the speaker's non-null pitch_hz_median values
  - pitch_hz_iqr:     75th - 25th percentile of those values (interquartile range)
  - energy_db_median: median across the speaker's non-null energy_db_mean values
  - energy_db_iqr:    75th - 25th percentile of those values
  - segment_count:    number of segments contributing at least one non-null
                      prosody field

Speakers whose every segment has prosody: null or all-null fields are omitted
from the result entirely (not emitted with null baselines). Robust to outliers
because median + IQR aren't pulled around by single shouts or whispers.
"""

from typing import Dict, List

import numpy as np


def compute_baselines(segments: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-speaker prosody baselines from a list of enriched segments."""
    by_speaker: Dict[str, Dict[str, List[float]]] = {}
    seg_counts: Dict[str, int] = {}

    for seg in segments:
        sid = seg["speaker_id"]
        prosody = seg.get("prosody")
        if prosody is None:
            continue

        pitch = prosody.get("pitch_hz_median")
        energy = prosody.get("energy_db_mean")
        # Count as a contributing segment if it has any non-null prosody field.
        any_non_null = any(prosody.get(k) is not None for k in (
            "pitch_hz_median", "pitch_hz_std", "pitch_range_hz",
            "energy_db_mean", "energy_db_range",
            "speech_rate_wps", "pause_ratio",
        ))
        if not any_non_null:
            continue

        bucket = by_speaker.setdefault(sid, {"pitch": [], "energy": []})
        seg_counts[sid] = seg_counts.get(sid, 0) + 1
        if pitch is not None:
            bucket["pitch"].append(float(pitch))
        if energy is not None:
            bucket["energy"].append(float(energy))

    result: Dict[str, Dict] = {}
    for sid, bucket in by_speaker.items():
        pitch_vals = bucket["pitch"]
        energy_vals = bucket["energy"]
        entry: Dict = {}
        if pitch_vals:
            entry["pitch_hz_median"] = round(float(np.median(pitch_vals)), 2)
            entry["pitch_hz_iqr"] = round(
                float(np.percentile(pitch_vals, 75) - np.percentile(pitch_vals, 25)), 2
            )
        else:
            entry["pitch_hz_median"] = None
            entry["pitch_hz_iqr"] = None
        if energy_vals:
            entry["energy_db_median"] = round(float(np.median(energy_vals)), 2)
            entry["energy_db_iqr"] = round(
                float(np.percentile(energy_vals, 75) - np.percentile(energy_vals, 25)), 2
            )
        else:
            entry["energy_db_median"] = None
            entry["energy_db_iqr"] = None
        entry["segment_count"] = seg_counts[sid]
        result[sid] = entry
    return result
