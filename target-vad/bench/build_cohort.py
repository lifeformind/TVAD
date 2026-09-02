#!/usr/bin/env python3
"""Build an AS-Norm imposter cohort from a directory of wav files.

Feed it NON-owner speech: podcast clips, LibriSpeech test-clean samples,
recordings of other people — ideally captured THROUGH the array (same
channel as live scoring). --augment adds a synthetic-reverb copy of each
clip to approximate the far-field channel for near-field source material.

Usage:
    python3 bench/build_cohort.py --wav-dir /path/to/cohort_wavs \
        --out voiceprints/cohort.npy [--augment]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _reverb(audio: np.ndarray, seed: int = 0, rt60_ms: int = 300, sr: int = 16000) -> np.ndarray:
    # Cheap synthetic room tail: exponentially-decaying noise impulse
    # response (no pyroomacoustics dependency).
    rng = np.random.default_rng(seed)
    n = int(sr * rt60_ms / 1000)
    ir = rng.normal(0, 1, n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    wet = np.convolve(audio, ir)[: len(audio)].astype(np.float32)
    peak = float(np.max(np.abs(wet))) or 1.0
    return wet / peak * float(np.max(np.abs(audio)))


def build_cohort(wav_paths, embedder, augment: bool = False, seed: int = 0) -> np.ndarray:
    rows = []
    for i, path in enumerate(wav_paths):
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        variants = [audio] + ([_reverb(audio, seed=seed + i)] if augment else [])
        for v in variants:
            emb = embedder.extract(v)
            rows.append(emb / (np.linalg.norm(emb) or 1.0))
    return np.stack(rows).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", default="voiceprints/cohort.npy")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    from core.speaker.embedder import EmbeddingExtractor
    paths = sorted(Path(args.wav_dir).glob("*.wav"))
    if len(paths) < 20:
        print(f"WARNING: only {len(paths)} wavs — AS-Norm wants >=100 cohort rows for stable stats")
    cohort = build_cohort(paths, EmbeddingExtractor(), augment=args.augment)
    np.save(args.out, cohort)
    print(f"wrote {cohort.shape} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
