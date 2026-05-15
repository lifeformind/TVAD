"""Centroid sampling: pick evenly-spaced segments from a cluster up to a duration cap.

Rationale: ECAPA centroid quality saturates around 30 s of audio. For very long
clusters, sampling instead of concatenating everything reduces compute without
hurting embedding quality. We sample at segment granularity (no slicing within a
segment) so the caller can directly extract the chosen segments from the waveform.
"""

from typing import List, Tuple


def sample_cluster_segments(
    segments: List[Tuple[float, float]],
    max_seconds: float,
) -> List[Tuple[float, float]]:
    """Pick a subset of segments whose total duration is <= max_seconds.

    Selection is evenly-spaced across the input order. If a single segment is
    longer than max_seconds, it is still returned alone (segment granularity).

    Args:
        segments: list of (start_s, end_s) tuples, sorted by start.
        max_seconds: target maximum total duration of returned segments.

    Returns:
        Subset of input segments, in original order. Empty input → empty output.
    """
    if not segments:
        return []

    total = sum(end - start for start, end in segments)
    if total <= max_seconds:
        return list(segments)

    # Need to subsample. Walk evenly-spaced indices and accumulate until cap reached.
    n = len(segments)
    # Estimate target count by average segment length; clamp to [1, n].
    avg_len = total / n
    target_count = max(1, min(n, int(max_seconds / avg_len)))

    # Pick `target_count` evenly-spaced indices across [0, n-1].
    if target_count == 1:
        indices = [n // 2]
    else:
        step = (n - 1) / (target_count - 1)
        indices = [round(i * step) for i in range(target_count)]
        # Dedupe while preserving order in case rounding collapsed indices
        seen = set()
        indices = [i for i in indices if not (i in seen or seen.add(i))]

    # Greedily accept selected segments while total stays <= max_seconds.
    chosen: List[Tuple[float, float]] = []
    running = 0.0
    for idx in indices:
        start, end = segments[idx]
        dur = end - start
        if running + dur <= max_seconds or not chosen:
            chosen.append((start, end))
            running += dur
        if running >= max_seconds:
            break
    return chosen
