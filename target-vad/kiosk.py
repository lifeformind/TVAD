"""Kiosk talkback entry point — wake-word activated speaker-locked session."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse

import numpy as np
import yaml
from rich.console import Console

from core.vad.silero_vad import SpeechSegment
from modes.kiosk.pipeline import KioskPipeline

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_dryrun_callbacks():
    """Print events to console; do not forward audio anywhere."""
    def on_primary_speech(segment: SpeechSegment, embedding: np.ndarray):
        console.print(
            f"[bold green][PRIMARY][/] {segment.duration_ms:.0f}ms "
            f"emb_norm={float(np.linalg.norm(embedding)):.3f}"
        )

    def on_session_started():
        console.print("[bold cyan][SESSION STARTED][/] Primary speaker locked")

    def on_session_ended(reason: str):
        console.print(f"[bold yellow][SESSION ENDED][/] reason={reason}\n")
        console.print('[dim][IDLE] Listening for wake phrase...[/]')

    return on_primary_speech, on_session_started, on_session_ended


def main():
    parser = argparse.ArgumentParser(description="Target VAD — Kiosk Talkback")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--wake-phrase",
        help="Override wake phrase (default from config). Bundled options: hey_jarvis, alexa, hey_mycroft.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print events instead of forwarding to a real downstream handler.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.wake_phrase:
        config["kiosk"]["wake_phrase"] = args.wake_phrase

    if not args.dry_run:
        # No real downstream handler is configured yet — fall back to dry-run
        # behavior with a warning.
        console.print(
            "[yellow]No downstream handler configured. Running in dry-run mode.[/]"
        )
    on_primary, on_started, on_ended = make_dryrun_callbacks()

    console.print(
        f"[bold][IDLE][/] Listening for [bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
    )
    pipeline = KioskPipeline(
        config=config,
        on_primary_speech=on_primary,
        on_session_started=on_started,
        on_session_ended=on_ended,
    )
    try:
        pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")


if __name__ == "__main__":
    main()
