"""Quick live mic test — records for 10s, runs VAD + optional embedding on detected speech."""

from core import compat  # noqa: F401
import sys
import time

import numpy as np
import yaml

from core.audio.mic_stream import MicrophoneStream
from vad.silero_vad import SileroVAD, SpeechSegment


def timed_stream(mic: MicrophoneStream, duration_s: float):
    """Yield mic chunks for at most duration_s seconds."""
    start = time.time()
    for chunk in mic.stream():
        yield chunk
        if time.time() - start > duration_s:
            return


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    duration = 10
    print(f"=== VAD Live Test ({duration}s) ===")
    print("Speak into your mic to test speech detection...")
    print(flush=True)

    vad = SileroVAD(config["vad"])
    mic = MicrophoneStream(config["audio"])
    segments: list[SpeechSegment] = []
    start = time.time()

    with mic:
        for segment in vad.process_stream(timed_stream(mic, duration)):
            segments.append(segment)
            print(
                f"  [{time.time() - start:.1f}s] Speech: {segment.duration_ms:.0f}ms | "
                f"peak={np.abs(segment.audio).max():.3f} | "
                f"rms={np.sqrt(np.mean(segment.audio**2)):.4f}"
            )
            sys.stdout.flush()

    print(f"\nDone. Detected {len(segments)} speech segment(s).")

    if segments:
        print("\nTesting speaker embeddings on all segments...")
        from speaker.embedder import EmbeddingExtractor

        embedder = EmbeddingExtractor()
        for i, seg in enumerate(segments):
            t0 = time.perf_counter()
            emb = embedder.extract(seg.audio)
            embed_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  Segment {i+1}: dim={len(emb)}, norm={np.linalg.norm(emb):.4f}, "
                f"time={embed_ms:.1f}ms {'(includes model load)' if i == 0 else ''}"
            )
    else:
        print("\nNo speech detected — try speaking louder or check your mic.")


if __name__ == "__main__":
    main()
