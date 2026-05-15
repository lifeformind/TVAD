"""Transcription pass (Phase 2A) — see docs/superpowers/specs/2026-05-15-transcription-pass-design.md.

Reads a diarization JSON, locates its source WAV, runs faster-whisper per
segment with rolling-context initial_prompt, attaches text + words, and writes
the JSON back atomically. Idempotent — re-running skips segments that already
have non-null text; --retranscribe forces full re-run.
"""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim, must precede speechbrain imports

import argparse
import datetime as dt
import json
import logging
import os
import sys
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import yaml
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

from modes.transcription.rolling_context import RollingContext
from modes.transcription.whisper_runner import WhisperRunner

console = Console()
logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_MODEL_OR_IO = 3


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio_as_mono16k(path: str) -> np.ndarray:
    """Read a WAV file as mono float32 at 16 kHz. Reused contract from diarize.py."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON to a sibling .tmp file then atomic-rename to `path`."""
    dirname = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", suffix=".json", dir=dirname)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD - Transcription Pass (Phase 2A)")
    parser.add_argument("input", help="Path to a diarization JSON produced by diarize.py")
    parser.add_argument("--out", default=None, help="Output JSON path (default: in-place atomic write)")
    parser.add_argument("--audio", default=None, help="Override the WAV path embedded in the JSON")
    parser.add_argument("--model", default=None, help="Model preset (small|large) or any faster-whisper-compatible model identifier; overrides config")
    parser.add_argument("--retranscribe", action="store_true", help="Re-transcribe segments that already have text; default is to skip them")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)

    # Load and validate JSON
    if not os.path.exists(args.input):
        console.print(f"[red]Diarization JSON not found:[/] {args.input}")
        return EXIT_BAD_INPUT
    try:
        with open(args.input) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Diarization JSON is malformed:[/] {exc.msg} at offset {exc.pos}")
        return EXIT_BAD_INPUT
    if "segments" not in data or "audio_file" not in data:
        console.print("[red]Diarization JSON is missing required fields (segments, audio_file).[/]")
        return EXIT_BAD_INPUT

    # Load config + apply CLI overrides
    config = load_config(args.config)
    try:
        trans_cfg = dict(config["transcription"])
    except KeyError:
        console.print(
            f"[red]Config file {args.config!r} is missing the [bold]transcription:[/bold] block.[/]"
        )
        return EXIT_MODEL_OR_IO
    if args.model is not None:
        trans_cfg["model"] = args.model

    # Locate WAV
    audio_path = args.audio if args.audio else data["audio_file"]
    if not os.path.exists(audio_path):
        console.print(
            f"[red]Audio file not found:[/] {audio_path}\n"
            "[dim]Try passing [bold]--audio <path>[/] to point at a moved or re-encoded WAV.[/]"
        )
        return EXIT_BAD_INPUT
    try:
        console.print(f"[dim]Loading[/] {audio_path}")
        audio = load_audio_as_mono16k(audio_path)
    except Exception as exc:
        console.print(f"[red]Failed to read audio:[/] {exc}")
        return EXIT_BAD_INPUT

    sr = 16000
    actual_duration = len(audio) / sr
    expected_duration = float(data.get("duration_s", actual_duration))
    if abs(actual_duration - expected_duration) > 1.0:
        console.print(
            f"[yellow]Warning:[/] audio duration {actual_duration:.1f}s differs from JSON's "
            f"{expected_duration:.1f}s — file may have been re-encoded."
        )

    # If --retranscribe, clear existing transcripts so the loop processes every segment
    if args.retranscribe:
        for seg in data["segments"]:
            seg.pop("text", None)
            seg.pop("words", None)

    # Build runner
    runner = WhisperRunner(
        model=trans_cfg["model"],
        language=trans_cfg.get("language"),
        device=trans_cfg.get("device", "cpu"),
        compute_type=trans_cfg.get("compute_type", "int8"),
    )

    # Build rolling-context window
    initial_prompt_chars = int(trans_cfg.get("initial_prompt_chars", 200))
    ctx = RollingContext(max_chars=initial_prompt_chars)

    # Transcribe loop
    segments = data["segments"]
    total = len(segments)
    transcribed_count = 0
    skipped_count = 0
    failed_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Transcribing...", total=total)
        for seg in segments:
            existing_text = seg.get("text")
            if existing_text is not None:
                # Already transcribed — skip, but flow context through
                if existing_text:
                    ctx.append(existing_text)
                skipped_count += 1
                progress.advance(task)
                continue

            start_i = int(seg["start"] * sr)
            end_i = int(seg["end"] * sr)
            slice_audio = audio[start_i:end_i]
            try:
                text, words = runner.transcribe(slice_audio, initial_prompt=ctx.text())
                seg["text"] = text
                seg["words"] = words
                if text:
                    ctx.append(text)
                transcribed_count += 1
            except Exception as exc:
                logger.warning(
                    "transcription failed for segment [%.2f, %.2f]: %s",
                    seg["start"], seg["end"], exc,
                )
                seg["text"] = None
                seg["words"] = []
                failed_count += 1
            progress.advance(task)

    # Update top-level metadata
    passes = list(data.get("passes_run", ["diarization"]))
    if "transcription" not in passes:
        passes.append("transcription")
    data["passes_run"] = passes
    data["transcription_config"] = {
        "model": trans_cfg["model"],
        "language": trans_cfg.get("language"),
        "initial_prompt_chars": initial_prompt_chars,
        "compute_type": trans_cfg.get("compute_type", "int8"),
        "transcribed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Atomic write
    out_path = args.out or args.input
    try:
        _atomic_write_json(out_path, data)
    except Exception as exc:
        console.print(f"[red]Failed to write output:[/] {exc}")
        return EXIT_MODEL_OR_IO

    console.print(
        f"[green]Wrote[/] {out_path}\n"
        f"[dim]Transcribed {transcribed_count}, skipped {skipped_count}, failed {failed_count} "
        f"of {total} segments (model={trans_cfg['model']}, language={trans_cfg.get('language')})[/]"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
