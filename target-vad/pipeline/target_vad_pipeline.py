"""Target VAD Pipeline — orchestrates VAD, speaker verification, and callback."""

import time
from typing import Callable, Optional

from rich.console import Console

from audio.mic_stream import MicrophoneStream
from speaker.embedder import EmbeddingExtractor
from speaker.enrollment_store import EnrollmentStore
from speaker.verifier import SpeakerVerifier, VerificationResult
from vad.silero_vad import SileroVAD, SpeechSegment

console = Console()


class TargetVADPipeline:
    """Main pipeline: mic -> VAD -> embed -> verify -> callback."""

    def __init__(self, config: dict):
        self.config = config
        self.vad = SileroVAD(config["vad"])
        self.embedder = EmbeddingExtractor()
        self.store = EnrollmentStore(config["paths"]["voiceprints_dir"])
        self.verifier = SpeakerVerifier(
            self.store, config["speaker"]["threshold"]
        )
        self.mic = MicrophoneStream(config["audio"])
        self.min_segment_ms = config["speaker"].get("min_segment_duration_ms", 800)
        self._running = False
        self.verbose = False

    def run(
        self,
        on_registered_speech: Callable[[SpeechSegment, VerificationResult, dict], None],
        on_unknown_speech: Optional[Callable[[SpeechSegment, VerificationResult, dict], None]] = None,
    ):
        """Main loop: stream mic -> VAD -> embed -> verify -> callback.

        Args:
            on_registered_speech: Called when a registered speaker is detected.
                Receives (segment, result, timing_info).
            on_unknown_speech: Optional callback for unregistered speech.
        """
        self._running = True
        self.vad.reset()

        console.print("[bold green]Pipeline started.[/] Listening...")

        try:
            with self.mic:
                for segment in self.vad.process_stream(self.mic.stream()):
                    if not self._running:
                        break

                    # Skip segments that are too short
                    if segment.duration_ms < self.min_segment_ms:
                        if self.verbose:
                            console.print(
                                f"[dim]Skipping short segment: {segment.duration_ms:.0f}ms[/]"
                            )
                        continue

                    # Measure embedding extraction time
                    t0 = time.perf_counter()
                    embedding = self.embedder.extract(segment.audio)
                    embed_ms = (time.perf_counter() - t0) * 1000

                    # Measure verification time
                    t1 = time.perf_counter()
                    result = self.verifier.verify(embedding)
                    verify_ms = (time.perf_counter() - t1) * 1000

                    timing = {
                        "embed_latency_ms": round(embed_ms, 2),
                        "verify_latency_ms": round(verify_ms, 2),
                        "total_overhead_ms": round(embed_ms + verify_ms, 2),
                        "segment_duration_ms": round(segment.duration_ms, 1),
                        "matched_user": result.matched_user,
                        "confidence": round(result.confidence, 4),
                    }

                    if result.is_registered:
                        on_registered_speech(segment, result, timing)
                    elif on_unknown_speech:
                        on_unknown_speech(segment, result, timing)

        except KeyboardInterrupt:
            console.print("\n[yellow]Pipeline stopped by user.[/]")
        finally:
            self._running = False

    def stop(self):
        """Stop the pipeline."""
        self._running = False
