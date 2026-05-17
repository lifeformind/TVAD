"""In-session enrollment: turn manifest entries into voiceprints and overlay them.

enroll_from_intros() slices each manifest time range out of the full waveform,
embeds it via the injected ECAPA EmbeddingExtractor, and returns one
IntroVoiceprint per manifest entry. SessionEnrollmentView presents a merged
read-only view: intro voiceprints shadow persistent voiceprints with the same id.

Conflict-detection (cosine sanity check on overridden ids) lives in diarize.py
so the warning surfaces in the CLI output rather than buried in a logger.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rich.console import Console

from modes.diarization.manifest import ManifestEntry

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class IntroVoiceprint:
    id: str
    name: str
    embedding: np.ndarray


# 800 ms at 16 kHz — matches EmbeddingExtractor.MIN_DURATION_SAMPLES. Slices shorter
# than this are reflect-padded inside the embedder; the resulting embedding is
# unreliable, so we warn upstream where the manifest entry id is visible.
_MIN_INTRO_DURATION_S = 0.8


def enroll_from_intros(
    audio: np.ndarray,
    sample_rate: int,
    manifest: List[ManifestEntry],
    embedder,
) -> List[IntroVoiceprint]:
    """Embed each manifest entry's audio slice.

    End-of-audio is clamped silently (a warning is logged); slices that fall past
    the audio end or are shorter than ECAPA's MIN_DURATION (800 ms) produce a
    warning but still emit a (potentially low-quality) embedding so the manifest
    entry isn't silently dropped. No other validation happens here —
    load_manifest() already enforced ordering and types.
    """
    audio_len_s = len(audio) / sample_rate
    results: List[IntroVoiceprint] = []
    for entry in manifest:
        end_clamped = min(entry.end, audio_len_s)
        if end_clamped < entry.end:
            logger.warning(
                "intro '%s' end=%.2fs exceeds audio length %.2fs; clamping",
                entry.id, entry.end, audio_len_s,
            )
        if entry.start >= audio_len_s:
            logger.warning(
                "intro '%s' start=%.2fs is past audio end %.2fs; "
                "embedding will be silence-padded and unreliable",
                entry.id, entry.start, audio_len_s,
            )
        start_i = max(0, int(entry.start * sample_rate))
        end_i = max(start_i, int(end_clamped * sample_rate))
        slice_duration_s = (end_i - start_i) / sample_rate
        if 0 < slice_duration_s < _MIN_INTRO_DURATION_S:
            console.print(
                f"[yellow]warning:[/] intro entry for id={entry.id!r} has duration "
                f"{slice_duration_s:.2f}s (< {_MIN_INTRO_DURATION_S:.1f}s) — embedding may fail "
                f"or be unstable. Consider lengthening this manifest entry. Skipping."
            )
            continue
        slice_audio = audio[start_i:end_i]
        embedding = embedder.extract(slice_audio, sample_rate=sample_rate)
        results.append(IntroVoiceprint(id=entry.id, name=entry.name, embedding=embedding))
    return results


class SessionEnrollmentView:
    """Read-only overlay: intros shadow same-id entries in a persistent store.

    Exposes the methods consumed by ClusterIdentifier (get_all) and by
    diarize.py's flatten_clusters step (get_name).
    """

    def __init__(self, persistent_store, intros: List[IntroVoiceprint]):
        self._persistent = persistent_store
        # Map id -> intro for O(1) shadow lookup
        self._intros: Dict[str, IntroVoiceprint] = {ivp.id: ivp for ivp in intros}

    def get_all(self) -> Dict[str, np.ndarray]:
        merged: Dict[str, np.ndarray] = dict(self._persistent.get_all())
        for user_id, ivp in self._intros.items():
            merged[user_id] = ivp.embedding
        return merged

    def get_name(self, user_id: str) -> str:
        if user_id in self._intros:
            return self._intros[user_id].name
        return self._persistent.get_name(user_id)
