"""Contribution metrics pass (Phase 3) — see docs/superpowers/specs/2026-05-16-contribution-metrics-design.md.

Reads a post-2A+2B diarization JSON, computes per-speaker + session-level
aggregates plus a bucketed activity timeline and deterministic narrative
highlights, writes the contribution_metrics block back atomically, and
renders a sibling Markdown report.
"""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from typing import Dict, List

import yaml
from rich.console import Console

from modes.metrics import aggregator, renderer

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CONFIG_OR_IO = 3


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


def _merged_speech_seconds(segments: List[Dict]) -> float:
    """Total wall-clock seconds where at least one speaker is talking."""
    if not segments:
        return 0.0
    intervals = sorted((s["start"], s["end"]) for s in segments)
    merged: List[List[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


def _build_metrics_block(data: Dict, cfg: Dict) -> Dict:
    segments = data["segments"]
    duration_s = float(data.get("duration_s", 0.0))
    bucket_seconds = int(cfg["bucket_seconds"])

    participation = aggregator.aggregate_participation(segments)
    sentiment = aggregator.aggregate_sentiment(segments)
    turn_taking = aggregator.aggregate_turn_taking(segments)
    pairwise = aggregator.aggregate_pairwise(segments)
    timeline = aggregator.aggregate_timeline(segments, duration_s, bucket_seconds)
    highlights = aggregator.select_highlights(
        segments, timeline, int(cfg["top_k_highlights"]), int(cfg["quote_max_chars"])
    )

    merged_speech = _merged_speech_seconds(segments)
    silence_s = round(max(0.0, duration_s - merged_speech), 2)

    session_block = {
        "duration_s": duration_s,
        "speech_duration_s": participation["session"]["speech_duration_s"],
        "silence_duration_s": silence_s,
        "total_segments": participation["session"]["total_segments"],
        "total_words": participation["session"]["total_words"],
        "unique_speakers": participation["session"]["unique_speakers"],
        "identified_speakers": participation["session"]["identified_speakers"],
        "unknown_segments": participation["session"]["unknown_segments"],
        "polarity_distribution": sentiment["session"]["polarity_distribution"],
        "emotion_distribution": sentiment["session"]["emotion_distribution"],
    }

    # Speakers list — order = first-appearance (matches enrolled_users_matched convention).
    seen = set()
    speakers_ordered: List[str] = []
    for s in segments:
        sid = s["speaker_id"]
        if sid not in seen:
            seen.add(sid)
            speakers_ordered.append(sid)
    name_lookup = {s["speaker_id"]: s["speaker"] for s in segments}

    speakers: List[Dict] = []
    for sid in speakers_ordered:
        speakers.append({
            "speaker_id": sid,
            "speaker": name_lookup[sid],
            "participation": participation["per_speaker"][sid],
            "sentiment": sentiment["per_speaker"][sid],
            "turn_taking": turn_taking["per_speaker"][sid],
        })

    return {
        "session": session_block,
        "speakers": speakers,
        "pairwise_followers": pairwise,
        "timeline": timeline,
        "highlights": highlights,
    }


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD — Contribution Metrics Pass (Phase 3)")
    parser.add_argument("input", help="Path to a transcribed + sentiment-classified diarization JSON")
    parser.add_argument("--out", default=None, help="Output JSON path (default: in-place atomic write)")
    parser.add_argument("--report", default=None, help="Markdown report path (default: <input-stem>.metrics.md)")
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
        console.print("[red]This JSON has not been transcribed yet.[/] "
                      "[dim]Run [bold]transcribe.py[/] first.[/]")
        return EXIT_BAD_INPUT
    if "sentiment" not in passes_run:
        console.print("[red]This JSON has not been sentiment-classified yet.[/] "
                      "[dim]Run [bold]sentiment.py[/] first.[/]")
        return EXIT_BAD_INPUT

    for i, seg in enumerate(data["segments"]):
        missing = [k for k in ("text", "words", "sentiment") if k not in seg]
        if missing:
            console.print(
                f"[red]Segment {i} is missing field(s) {missing!r}.[/] "
                "[dim]This JSON is in a partial/inconsistent state — rerun the prior pass.[/]"
            )
            return EXIT_BAD_INPUT

    # Load config.
    try:
        cfg_full = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/] {args.config}")
        return EXIT_CONFIG_OR_IO
    cfg = cfg_full.get("metrics")
    if not cfg:
        console.print(f"[red]Config is missing the [bold]metrics:[/] block.[/]")
        return EXIT_CONFIG_OR_IO

    # Build metrics + write.
    try:
        block = _build_metrics_block(data, cfg)
    except Exception as exc:
        console.print(f"[red]Failed to aggregate metrics:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    data["contribution_metrics"] = block
    data["metrics_config"] = {
        "bucket_seconds": int(cfg["bucket_seconds"]),
        "top_k_highlights": int(cfg["top_k_highlights"]),
        "quote_max_chars": int(cfg["quote_max_chars"]),
        "analyzed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "passes_run" not in data:
        data["passes_run"] = []
    if "metrics" not in data["passes_run"]:
        data["passes_run"].append("metrics")

    out_json = args.out or args.input
    try:
        _atomic_write_json(out_json, data)
    except Exception as exc:
        console.print(f"[red]Failed to write metrics JSON:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    # Markdown render + write.
    session_meta = {
        "audio_file": data.get("audio_file", ""),
        "analyzed_at": data["metrics_config"]["analyzed_at"],
    }
    md = renderer.render_markdown(block, session_meta)

    if args.report:
        report_path = args.report
    else:
        stem, _ext = os.path.splitext(args.input)
        report_path = stem + ".metrics.md"

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as exc:
        console.print(f"[red]Failed to write Markdown report:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    speakers_n = len(block["speakers"])
    segs_n = block["session"]["total_segments"]
    words_n = block["session"]["total_words"]
    hl_n = len(block["highlights"])
    console.print(
        f"[green]Metrics written:[/] {speakers_n} speakers, {segs_n} segments, "
        f"{words_n} words, {hl_n} highlights."
    )
    console.print(f"  JSON     -> {out_json}")
    console.print(f"  Markdown -> {report_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
