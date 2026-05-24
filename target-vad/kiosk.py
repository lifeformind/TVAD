"""Kiosk talkback entry point — wake-word activated speaker-locked session."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse
import sys

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

    def on_event(event_type: str, payload: dict):
        if event_type == "wake_detected":
            console.print(
                f"[magenta][WAKE][/] phrase={payload['phrase']} "
                f"score={payload['score']:.3f}"
            )
        elif event_type == "segment_scored":
            color = "green" if payload["decision"] == "match" else "dim"
            tag = "MATCH" if payload["decision"] == "match" else "no_match"
            console.print(
                f"[{color}][SCORED][/] {payload['duration_ms']:.0f}ms "
                f"score={payload['score']:.3f} → {tag}"
            )

    return on_primary_speech, on_session_started, on_session_ended, on_event


def make_talkback_callbacks():
    """Print talkback events to console."""
    def on_session_started():
        console.print("[bold cyan][SESSION STARTED][/] Primary speaker locked")

    def on_session_ended(reason: str):
        console.print(f"[bold yellow][SESSION ENDED][/] reason={reason}\n")
        console.print('[dim][IDLE] Listening for wake phrase...[/]')

    def on_event(event_type: str, payload: dict):
        if event_type == "wake_detected":
            console.print(
                f"[magenta][WAKE][/] phrase={payload['phrase']} "
                f"score={payload['score']:.3f}"
            )
        elif event_type == "handoff_to_talkback":
            console.print("[bold cyan][HANDOFF][/] → TalkbackController")
        elif event_type == "user_turn_complete":
            console.print(f"[green][USER][/] \"{payload['text']}\"")
        elif event_type == "llm_response_started":
            console.print(f"[dim][LLM][/] first token in {payload['time_to_first_token_ms']:.0f}ms")
        elif event_type == "barge_in":
            console.print(
                f"[bold red][BARGE-IN][/] cut at {payload['cut_at_ms']:.0f}ms "
                f"(primary score={payload['primary_score']:.2f})"
            )
        elif event_type == "segment_scored":
            color = "green" if payload["decision"] == "match" else "dim"
            tag = "MATCH" if payload["decision"] == "match" else "no_match"
            console.print(
                f"[{color}][SCORED][/] {payload['duration_ms']:.0f}ms "
                f"score={payload['score']:.3f} → {tag}"
            )

    return on_session_started, on_session_ended, on_event


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
    parser.add_argument(
        "--talkback",
        action="store_true",
        help="Force talkback_enabled=true (full-duplex voice assistant mode).",
    )
    args = parser.parse_args()

    if args.dry_run and args.talkback:
        console.print(
            "[red]--dry-run and --talkback are incompatible.[/]\n"
            "[dim]Use --dry-run for event-only output, or --talkback for full voice assistant mode.[/]"
        )
        sys.exit(2)

    config = load_config(args.config)
    if args.wake_phrase:
        config["kiosk"]["wake_phrase"] = args.wake_phrase
    if args.talkback:
        config["kiosk"]["talkback_enabled"] = True

    talkback_enabled = config["kiosk"].get("talkback_enabled", False)

    if talkback_enabled:
        on_started, on_ended, on_event = make_talkback_callbacks()

        from core.logging.jsonl_logger import EventLogger
        from modes.talkback.controller import TalkbackController
        from modes.talkback.llm import LlmClient
        from modes.talkback.player import Player
        from modes.talkback.stt import StreamingStt
        from modes.talkback.tts import TtsEngine

        tb_cfg = config["kiosk"].get("talkback", {})
        logger = EventLogger(
            path_template=tb_cfg.get("logging", {}).get(
                "jsonl_path", "logs/kiosk-{date}-{session_id}.jsonl"
            ),
            session_id="pending",
        )

        stt_cfg = tb_cfg.get("stt", {})
        stt = StreamingStt(
            model=stt_cfg.get("model", "large-v3"),
            compute_type=stt_cfg.get("compute_type", "float16"),
            device=stt_cfg.get("device", "cuda"),
        )

        llm_cfg = tb_cfg.get("llm", {})
        llm = LlmClient(
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
            model=llm_cfg.get("model", "qwen2.5-7b-instruct-q5_k_m"),
            temperature=llm_cfg.get("temperature", 0.6),
            max_tokens=llm_cfg.get("max_tokens", 512),
        )

        tts_cfg = tb_cfg.get("tts", {})
        tts = TtsEngine(
            backend=tts_cfg.get("backend", "kokoro"),
            voice=tts_cfg.get("voice", "af_bella"),
            device=tts_cfg.get("device", "cuda"),
        )

        player = Player(sample_rate=tb_cfg.get("sample_rate_hz", 16000))

        controller = TalkbackController(
            stt=stt, llm=llm, tts=tts, player=player, logger=logger,
        )

        # Warm up backends before entering the mic loop so the first
        # handoff doesn't stall on model downloads / cold starts.
        import asyncio
        import numpy as np

        with console.status("[bold]Loading STT model (faster-whisper large-v3)..."):
            stt._ensure_model()
        console.print("[green]✓[/] STT loaded")

        with console.status("[bold]Loading TTS model (Kokoro)..."):
            tts._ensure_model()
        console.print("[green]✓[/] TTS loaded")

        with console.status("[bold]Checking LLM server..."):
            llm_ok = asyncio.run(llm.ping())
        if llm_ok:
            console.print("[green]✓[/] LLM server reachable")
        else:
            console.print("[red]✗[/] LLM server unreachable at " + llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"))
            console.print("[dim]Start llama-server and retry.[/]")
            sys.exit(3)

        console.print(
            f"[bold][TALKBACK][/] Listening for [bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
        )
        pipeline = KioskPipeline(
            config=config,
            on_primary_speech=lambda s, e: None,
            on_session_started=on_started,
            on_session_ended=on_ended,
            on_event=on_event,
            _talkback_controller=controller,
        )
    else:
        if not args.dry_run:
            console.print(
                "[yellow]No downstream handler configured. Running in dry-run mode.[/]"
            )
        on_primary, on_started, on_ended, on_event = make_dryrun_callbacks()

        console.print(
            f"[bold][IDLE][/] Listening for [bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
        )
        pipeline = KioskPipeline(
            config=config,
            on_primary_speech=on_primary,
            on_session_started=on_started,
            on_session_ended=on_ended,
            on_event=on_event,
        )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")


if __name__ == "__main__":
    main()
