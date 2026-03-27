"""Target VAD — main entry point."""

import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse

import yaml
from rich.console import Console
from rich.table import Table

from pipeline.target_vad_pipeline import TargetVADPipeline
from vad.silero_vad import SpeechSegment
from speaker.verifier import VerificationResult

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_demo_callbacks(verbose: bool):
    """Create demo-mode callbacks that print results to console."""

    def on_registered(segment: SpeechSegment, result: VerificationResult, timing: dict):
        console.print(
            f"[bold green][REGISTERED][/] user={result.matched_user} "
            f"| confidence={result.confidence:.2f} "
            f"| duration={timing['segment_duration_ms']:.1f}ms "
            f"| latency={timing['total_overhead_ms']:.0f}ms"
        )
        if verbose:
            _print_detail_table(result, timing)

    def on_unknown(segment: SpeechSegment, result: VerificationResult, timing: dict):
        best_user = max(result.all_scores, key=result.all_scores.get) if result.all_scores else "n/a"
        best_score = result.all_scores.get(best_user, 0.0) if result.all_scores else 0.0
        console.print(
            f"[dim][UNKNOWN]   [/] best_match={best_user} "
            f"| score={best_score:.2f} "
            f"| duration={timing['segment_duration_ms']:.1f}ms"
        )
        if verbose:
            _print_detail_table(result, timing)

    return on_registered, on_unknown


def _print_detail_table(result: VerificationResult, timing: dict):
    """Print a detailed table of scores and timing."""
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("User")
    table.add_column("Score", justify="right")
    for user, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        style = "green" if score >= result.confidence and result.is_registered else ""
        table.add_row(user, f"{score:.4f}", style=style)
    console.print(table)

    timing_table = Table(show_header=True, header_style="bold", box=None)
    timing_table.add_column("Metric")
    timing_table.add_column("Value", justify="right")
    timing_table.add_row("Embed latency", f"{timing['embed_latency_ms']:.1f}ms")
    timing_table.add_row("Verify latency", f"{timing['verify_latency_ms']:.1f}ms")
    timing_table.add_row("Total overhead", f"{timing['total_overhead_ms']:.1f}ms")
    timing_table.add_row("Segment duration", f"{timing['segment_duration_ms']:.1f}ms")
    console.print(timing_table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Target VAD — Speaker-gated voice activity detection")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--threshold", type=float, help="Override cosine similarity threshold")
    parser.add_argument("--verbose", action="store_true", help="Print per-segment detail tables")
    parser.add_argument("--demo", action="store_true", help="Demo mode: print results instead of TTS trigger")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.threshold is not None:
        config["speaker"]["threshold"] = args.threshold

    pipeline = TargetVADPipeline(config)
    pipeline.verbose = args.verbose

    if args.demo:
        on_reg, on_unk = make_demo_callbacks(args.verbose)
        pipeline.run(on_registered_speech=on_reg, on_unknown_speech=on_unk)
    else:
        # Non-demo mode: same as demo for now, placeholder for TTS integration
        console.print("[yellow]No TTS integration configured. Running in demo mode.[/]\n")
        on_reg, on_unk = make_demo_callbacks(args.verbose)
        pipeline.run(on_registered_speech=on_reg, on_unknown_speech=on_unk)


if __name__ == "__main__":
    main()
