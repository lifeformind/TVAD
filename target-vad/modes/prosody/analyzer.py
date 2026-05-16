"""Per-segment prosody analyzer - pure function over an audio chunk + word timestamps.

Computes pitch (median / std / range), energy (mean / range in dB), and
rate (words-per-second / pause ratio). All seven fields may be null in
degenerate cases (no voiced frames, zero-length audio, empty word list).
No I/O, no model loads, no global state.
"""

from typing import Dict, List, Optional

import librosa
import numpy as np


def _round_or_none(x: Optional[float], decimals: int = 2) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), decimals)


def analyze_segment(
    audio_chunk: np.ndarray,
    sample_rate: int,
    words: List[Dict],
    segment_duration: float,
    cfg: Dict,
) -> Dict:
    """Compute prosody features for a single segment.

    Args:
        audio_chunk: mono float32 audio for this segment.
        sample_rate: e.g. 16000.
        words: list of {start, end, word, probability} from 2A. May be empty.
        segment_duration: duration_s from the segment's start/end. Must be >= 0.
        cfg: dict with keys pitch_min_hz, pitch_max_hz, frame_length_ms, hop_length_ms.

    Returns:
        Dict with seven keys (pitch_hz_median, pitch_hz_std, pitch_range_hz,
        energy_db_mean, energy_db_range, speech_rate_wps, pause_ratio). Each
        value is a float or None.
    """
    pitch_min = cfg["pitch_min_hz"]
    pitch_max = cfg["pitch_max_hz"]
    frame_length = int(sample_rate * cfg["frame_length_ms"] / 1000)
    hop_length = int(sample_rate * cfg["hop_length_ms"] / 1000)

    # Pitch via pyin.
    if len(audio_chunk) >= frame_length:
        try:
            f0, _voiced_flag, _voiced_prob = librosa.pyin(
                audio_chunk,
                fmin=pitch_min,
                fmax=pitch_max,
                sr=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:
            voiced_f0 = np.array([])
    else:
        voiced_f0 = np.array([])

    if len(voiced_f0) > 0:
        pitch_median = float(np.median(voiced_f0))
        pitch_std = float(np.std(voiced_f0))
        pitch_range = float(np.percentile(voiced_f0, 95) - np.percentile(voiced_f0, 5))
    else:
        pitch_median = None
        pitch_std = None
        pitch_range = None

    # Energy via RMS.
    if len(audio_chunk) > 0:
        try:
            rms = librosa.feature.rms(
                y=audio_chunk,
                frame_length=frame_length,
                hop_length=hop_length,
            )[0]
            db = librosa.amplitude_to_db(rms, ref=1.0)
            energy_db_mean = float(np.mean(db))
            energy_db_range = float(np.percentile(db, 95) - np.percentile(db, 5))
        except Exception:
            energy_db_mean = None
            energy_db_range = None
    else:
        energy_db_mean = None
        energy_db_range = None

    # Rate from word timestamps.
    if words and segment_duration > 0:
        word_total = sum(w["end"] - w["start"] for w in words)
        speech_rate_wps = len(words) / segment_duration
        pause_ratio = max(0.0, min(1.0, (segment_duration - word_total) / segment_duration))
    else:
        speech_rate_wps = None
        pause_ratio = None

    return {
        "pitch_hz_median": _round_or_none(pitch_median),
        "pitch_hz_std": _round_or_none(pitch_std),
        "pitch_range_hz": _round_or_none(pitch_range),
        "energy_db_mean": _round_or_none(energy_db_mean),
        "energy_db_range": _round_or_none(energy_db_range),
        "speech_rate_wps": _round_or_none(speech_rate_wps),
        "pause_ratio": _round_or_none(pause_ratio),
    }
