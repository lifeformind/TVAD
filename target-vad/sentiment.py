"""Sentiment pass (Phase 2B) — see docs/superpowers/specs/2026-05-15-sentiment-pass-design.md.

Reads a transcribed diarization JSON, classifies each segment's text with both
a polarity model and an emotion model, attaches a nested sentiment block per
segment, and writes the JSON back atomically. Idempotent — re-running skips
segments that already have sentiment; --rerun forces full re-classification.
"""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim, must precede speechbrain imports

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from typing import List

import yaml
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

from modes.sentiment.classifier import SentimentClassifier

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_MODEL_OR_IO = 3


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
    parser = argparse.ArgumentParser(description="Target VAD - Sentiment Pass (Phase 2B)")
    parser.add_argument("input", help="Path to a transcribed diarization JSON")
    parser.add_argument("--out", default=None, help="Output JSON path (default: in-place atomic write)")
    parser.add_argument("--rerun", action="store_true", help="Re-classify segments that already have sentiment; default is to skip them")
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
    if "segments" not in data:
        console.print("[red]Diarization JSON is missing the [bold]segments[/bold] field.[/]")
        return EXIT_BAD_INPUT

    # Pre-flight: every segment must have a `text` field (transcription pass must have run)
    for i, seg in enumerate(data["segments"]):
        if "text" not in seg:
            console.print(
                f"[red]Segment {i} has no [bold]text[/bold] field — transcription pass hasn't run yet.[/]\n"
                "[dim]Run [bold]transcribe.py[/] first to populate text on every segment.[/]"
            )
            return EXIT_BAD_INPUT

    # Load config + sentiment block
    config = load_config(args.config)
    try:
        sent_cfg = dict(config["sentiment"])
    except KeyError:
        console.print(
            f"[red]Config file {args.config!r} is missing the [bold]sentiment:[/bold] block.[/]"
        )
        return EXIT_MODEL_OR_IO

    # --rerun: clear existing sentiment so the loop processes every segment with text
    if args.rerun:
        for seg in data["segments"]:
            seg.pop("sentiment", None)

    # Build classifier; eagerly load models so download/config failures surface here
    # rather than getting masked by the per-batch exception handler later.
    classifier = SentimentClassifier(
        polarity_model=sent_cfg["polarity_model"],
        emotion_model=sent_cfg["emotion_model"],
        device=sent_cfg.get("device", "cpu"),
    )
    try:
        classifier.load()
    except Exception as exc:
        console.print(
            f"[red]Failed to load sentiment models:[/] {exc}\n"
            "[dim]Check your network, HuggingFace cache, and config.sentiment.*.[/]"
        )
        return EXIT_MODEL_OR_IO

    batch_size = int(sent_cfg.get("batch_size", 16))

    # Walk segments and split into three buckets: skip (already classified), null-out (no text),
    # and queue (text present, needs classification).
    segments = data["segments"]
    total = len(segments)
    skipped_count = 0
    null_count = 0
    queue_indices: List[int] = []  # indices of segments to classify
    for i, seg in enumerate(segments):
        text = seg.get("text")
        existing = seg.get("sentiment")
        if existing is not None:
            # Already classified; preserve
            skipped_count += 1
            continue
        if text is None or text == "":
            seg["sentiment"] = None
            null_count += 1
            continue
        queue_indices.append(i)

    # Classify in batches with a progress bar
    classified_count = 0
    failed_count = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Classifying...", total=len(queue_indices))
        for start in range(0, len(queue_indices), batch_size):
            batch_idxs = queue_indices[start:start + batch_size]
            batch_texts = [segments[i]["text"] for i in batch_idxs]
            try:
                batch_results = classifier.classify_batch(batch_texts)
                for idx, result in zip(batch_idxs, batch_results):
                    segments[idx]["sentiment"] = result
                    classified_count += 1
            except Exception as exc:
                console.print(
                    f"[yellow]warning:[/] classification failed for batch of {len(batch_idxs)} "
                    f"segments (indices {batch_idxs[0]}..{batch_idxs[-1]}): {exc}"
                )
                for idx in batch_idxs:
                    segments[idx]["sentiment"] = None
                    failed_count += 1
            progress.advance(task, advance=len(batch_idxs))

    # Update top-level metadata
    passes = list(data.get("passes_run", ["diarization", "transcription"]))
    if "sentiment" not in passes:
        passes.append("sentiment")
    data["passes_run"] = passes
    data["sentiment_config"] = {
        "polarity_model": sent_cfg["polarity_model"],
        "emotion_model": sent_cfg["emotion_model"],
        "device": sent_cfg.get("device", "cpu"),
        "batch_size": batch_size,
        "analyzed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        f"[dim]Classified {classified_count}, skipped {skipped_count} (already had sentiment), "
        f"nulled {null_count} (null/empty text), failed {failed_count} of {total} segments "
        f"(polarity={sent_cfg['polarity_model']}, emotion={sent_cfg['emotion_model']})[/]"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
