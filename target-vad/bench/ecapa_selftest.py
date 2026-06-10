#!/usr/bin/env python3
"""Offline ECAPA self-similarity test on plain recordings (Batch 2c root-cause).

The live kiosk showed the *enrolled* speaker scoring 0.0-0.57 against their own
primary — bimodal, even on 1.2-1.7s segments. That could be ECAPA/mic reality,
an unrepresentative primary snapshot, or the talkback pipeline corrupting
segments. This tool removes the pipeline (no asyncio, TTS, AEC, or barge logic):
it loads a WAV, runs the SAME SileroVAD + ECAPA extractor the kiosk uses, and
reports how consistently one speaker embeds.

  Record ~25s of yourself (several clear sentences):
    arecord -f S16_LE -r 16000 -c 1 -d 25 self.wav
  Record a second person the same way:
    arecord -f S16_LE -r 16000 -c 1 -d 25 other.wav

  python3 bench/ecapa_selftest.py self.wav            # within-speaker consistency
  python3 bench/ecapa_selftest.py self.wav other.wav  # + self-vs-other separation

Interpretation:
  - within-self mean HIGH (~0.6-0.9): ECAPA+mic are fine; the low live scores are
    a PIPELINE bug (segments captured/processed wrong in talkback) — chase that.
  - within-self mean LOW (~0.3) or bimodal: ECAPA on this voice/mic is the limit;
    per-turn gating needs longer windows / M-of-N / better enrollment.
  - self.wav + other.wav: prints the achievable self-vs-other separation and a
    recommended threshold (or "OVERLAPPING").
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# bench/ is run from the target-vad/ root; make core.* importable.
sys.path.insert(0, ".")
from core.speaker.embedder import EmbeddingExtractor  # noqa: E402
from core.speaker.verifier import cosine_similarity   # noqa: E402
from core.vad.silero_vad import SileroVAD              # noqa: E402

VAD_CONFIG = {
    "sample_rate": 16000,
    "speech_threshold": 0.5,
    "min_speech_duration_ms": 300,
    "padding_ms": 200,
}


def load_wav(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load a WAV as float32 mono at sample_rate (resampling if needed)."""
    import librosa
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def segment_audio(audio: np.ndarray, vad: SileroVAD,
                  chunk: int = 512) -> list[np.ndarray]:
    """Run audio through SileroVAD and return the speech-segment waveforms."""
    vad.reset()
    segments = []
    for i in range(0, len(audio), chunk):
        for seg in vad.process_chunk(audio[i:i + chunk]):
            segments.append(seg.audio)
    # Flush a trailing in-progress segment with ~300ms of silence.
    for seg in vad.process_chunk(np.zeros(4800, dtype=np.float32)):
        segments.append(seg.audio)
    return segments


def embed(segments: list[np.ndarray],
          extractor: EmbeddingExtractor) -> list[np.ndarray]:
    return [extractor.extract(s) for s in segments]


def pairwise(embs: list[np.ndarray]) -> list[float]:
    """All off-diagonal pairwise cosine similarities (within one speaker)."""
    out = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            out.append(float(cosine_similarity(embs[i], embs[j])))
    return out


def cross(a: list[np.ndarray], b: list[np.ndarray]) -> list[float]:
    """All cross cosine similarities between two speakers' embeddings."""
    return [float(cosine_similarity(x, y)) for x in a for y in b]


def stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    s = sorted(scores)
    mean = sum(s) / len(s)
    var = sum((x - mean) ** 2 for x in s) / len(s)
    return {"n": len(s), "mean": mean, "min": s[0], "max": s[-1],
            "median": s[len(s) // 2], "stdev": var ** 0.5}


def _fmt(st: dict) -> str:
    if st.get("n", 0) == 0:
        return "(none)"
    return (f"n={st['n']}  mean={st['mean']:.3f}  median={st['median']:.3f}  "
            f"range=[{st['min']:.3f}, {st['max']:.3f}]  stdev={st['stdev']:.3f}")


def separation(within: list[float], crossed: list[float]) -> str:
    """Best self-vs-other threshold from clean recordings."""
    if not within or not crossed:
        return "  (need both recordings for a separation verdict)"
    margin = min(within) - max(crossed)
    if margin > 0:
        lo, hi = max(crossed), min(within)
        return (f"  margin = {margin:+.3f}  → CLEANLY SEPARABLE\n"
                f"  safe threshold band ({lo:.3f}, {hi:.3f}], midpoint "
                f"{((lo + hi) / 2):.3f}")
    return (f"  margin = {margin:+.3f}  → OVERLAPPING (no clean threshold)\n"
            "  per-turn ECAPA can't gate reliably on these segments")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("self_wav", help="recording of the enrolled speaker")
    ap.add_argument("other_wav", nargs="?", help="recording of a second person")
    args = ap.parse_args(argv)

    extractor = EmbeddingExtractor()
    vad = SileroVAD(VAD_CONFIG)

    self_audio = load_wav(args.self_wav)
    self_segs = segment_audio(self_audio, vad)
    self_embs = embed(self_segs, extractor)
    durs = [len(s) / 16000 * 1000 for s in self_segs]
    print(f"=== {args.self_wav}: {len(self_segs)} speech segments "
          f"(durations ms: {[round(d) for d in durs]}) ===")
    within = pairwise(self_embs)
    print(f"  within-speaker pairwise similarity: {_fmt(stats(within))}")
    if self_embs:
        to_first = [float(cosine_similarity(self_embs[0], e))
                    for e in self_embs[1:]]
        print(f"  vs first segment (the 'primary' snapshot): {_fmt(stats(to_first))}")

    if args.other_wav:
        vad2 = SileroVAD(VAD_CONFIG)
        other_segs = segment_audio(load_wav(args.other_wav), vad2)
        other_embs = embed(other_segs, extractor)
        print(f"\n=== {args.other_wav}: {len(other_segs)} speech segments ===")
        crossed = cross(self_embs, other_embs)
        print(f"  self-vs-other cross similarity: {_fmt(stats(crossed))}")
        print("\n--- separation ---")
        print(separation(within, crossed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
