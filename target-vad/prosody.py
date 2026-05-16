"""Prosody pass (Phase 4) - see docs/superpowers/specs/2026-05-16-prosody-pass-design.md.

Reads a post-2A diarization JSON + the audio WAV, computes per-segment pitch /
energy / rate features via librosa, attaches a 7-field `prosody` block per
segment, and emits a top-level prosody_baselines summary keyed by speaker_id.
Idempotent - rerunning skips segments that already have prosody; --rerun
forces full re-analysis.
"""

from core import compat  # noqa: F401 - torchaudio/speechbrain shim

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from typing import Dict, List

import yaml
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

from core.audio.load import load_audio_as_mono16k
from modes.prosody.analyzer import analyze_segment
from modes.prosody.baselines import compute_baselines

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CONFIG_OR_IO = 3

SR = 16000


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _atomic_write_json(path: str, data: dict) -> None:
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
    parser = argparse.ArgumentParser(description="Target VAD - Prosody Pass (Phase 4)")
    parser.add_argument("input", help="Path to a transcribed diarization JSON")
    parser.add_argument("--audio", default=None, help="Path to the WAV (default: from JSON's audio_file field)")
    parser.add_argument("--out", default=None, help="Output JSON path (default: in-place atomic write)")
    parser.add_argument("--rerun", action="store_true", help="Re-analyze segments that already have prosody")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)

    # Load JSON.
    if not os.path.exists(args.input):
        console.print(f"[red]Diarization JSON not found:[/] {args.input}")
        return EXIT_BAD_INPUT
    try:
        with open(args.input) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Diarization JSON is malformed:[/] {exc.msg} at offset {exc.pos}")
        return EXIT_BAD_INPUT

    if "segments" not in data:
        console.print("[red]Diarization JSON is missing the [bold]segments[/] field.[/]")
        return EXIT_BAD_INPUT

    passes_run = data.get("passes_run", [])
    if "transcription" not in passes_run:
        console.print(
            "[red]This JSON has not been transcribed yet.[/] "
            "[dim]Run [bold]transcribe.py[/] first.[/]"
        )
        return EXIT_BAD_INPUT

    for i, seg in enumerate(data["segments"]):
        for k in ("text", "words"):
            if k not in seg:
                console.print(
                    f"[red]Segment {i} is missing field [bold]{k}[/].[/] "
                    "[dim]This JSON is in a partial/inconsistent state - rerun transcribe.py.[/]"
                )
                return EXIT_BAD_INPUT

    # Resolve audio path.
    audio_path = args.audio or data.get("audio_file", "")
    if not audio_path or not os.path.exists(audio_path):
        console.print(f"[red]Audio file not found:[/] {audio_path or '(no path given)'}")
        console.print("[dim]Pass [bold]--audio[/] explicitly if the JSON's audio_file is wrong.[/]")
        return EXIT_BAD_INPUT

    # Load config.
    try:
        cfg_full = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/] {args.config}")
        return EXIT_CONFIG_OR_IO
    cfg = cfg_full.get("prosody")
    if not cfg:
        console.print("[red]Config is missing the [bold]prosody:[/] block.[/]")
        return EXIT_CONFIG_OR_IO

    # Load audio once.
    try:
        audio = load_audio_as_mono16k(audio_path)
    except Exception as exc:
        console.print(f"[red]Failed to read audio file:[/] {exc}")
        return EXIT_BAD_INPUT

    # Walk segments.
    segments = data["segments"]
    analyzed = 0
    skipped = 0
    failed = 0
    with Progress(
        TextColumn("[bold]Analyzing[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("prosody", total=len(segments))
        for seg in segments:
            if not args.rerun and seg.get("prosody") is not None:
                skipped += 1
                progress.advance(task)
                continue

            start_i = max(0, int(seg["start"] * SR))
            end_i = min(len(audio), int(seg["end"] * SR))
            chunk = audio[start_i:end_i]
            segment_duration = max(0.0, seg["end"] - seg["start"])
            words = seg.get("words") or []
            try:
                block = analyze_segment(chunk, SR, words, segment_duration, cfg)
            except Exception as exc:
                console.print(f"[yellow]warning:[/] analyzer crashed on segment {seg['start']:.2f}s: {exc}")
                seg["prosody"] = None
                failed += 1
                progress.advance(task)
                continue

            # Sentinel: prosody: null when ALL seven fields are None
            if all(v is None for v in block.values()):
                seg["prosody"] = None
            else:
                seg["prosody"] = block
            analyzed += 1
            progress.advance(task)

    # Baselines + top-level fields.
    data["prosody_baselines"] = compute_baselines(segments)
    data["prosody_config"] = {
        "pitch_min_hz": int(cfg["pitch_min_hz"]),
        "pitch_max_hz": int(cfg["pitch_max_hz"]),
        "frame_length_ms": int(cfg["frame_length_ms"]),
        "hop_length_ms": int(cfg["hop_length_ms"]),
        "analyzed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "passes_run" not in data:
        data["passes_run"] = []
    if "prosody" not in data["passes_run"]:
        data["passes_run"].append("prosody")

    out_path = args.out or args.input
    try:
        _atomic_write_json(out_path, data)
    except Exception as exc:
        console.print(f"[red]Failed to write prosody JSON:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    console.print(
        f"[green]Prosody written:[/] {analyzed} analyzed, {skipped} skipped (already had prosody), "
        f"{failed} failed (analyzer crash)."
    )
    console.print(f"  JSON -> {out_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
