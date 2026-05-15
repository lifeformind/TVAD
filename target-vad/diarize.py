"""Classroom diarization entry point — see docs/superpowers/specs/2026-05-14-classroom-diarization-design.md."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim, must precede speechbrain imports

import argparse
import datetime as dt
import os
import sys
from typing import List

import numpy as np
import soundfile as sf
import yaml
from rich.console import Console

from core.speaker.embedder import EmbeddingExtractor
from core.speaker.enrollment_store import EnrollmentStore
from modes.diarization.diarizer import Diarizer
from modes.diarization.identifier import ClusterIdentifier
from modes.diarization.output import DiarizationSegment, write_json, write_rttm

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CONFIG_OR_MODEL = 3


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio_as_mono16k(path: str) -> np.ndarray:
    """Read a WAV file and return mono float32 at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mix down
    if sr != 16000:
        from scipy.signal import resample_poly
        # Use rational resampling for exact rate change
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def flatten_clusters(
    clusters: dict, cluster_labels: dict
) -> List[DiarizationSegment]:
    """Convert {cluster_id: [(start,end),...]} + {cluster_id: label} → sorted segment list."""
    segments: List[DiarizationSegment] = []
    for cid, time_ranges in clusters.items():
        label = cluster_labels.get(cid, "unknown")
        for start, end in time_ranges:
            segments.append(DiarizationSegment(start=start, end=end, speaker=label))
    segments.sort(key=lambda s: s.start)
    return segments


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD — Classroom Diarization (S1)")
    parser.add_argument("input", help="Path to a WAV file")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input>.diarization.json)")
    parser.add_argument("--rttm", action="store_true", help="Also write an RTTM file alongside the JSON")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", action="store_true", help="Reserved — JSON-lines event log (not yet wired)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        console.print(f"[red]Input file not found:[/] {args.input}")
        return EXIT_BAD_INPUT

    config = load_config(args.config)
    diar_cfg = config["diarization"]
    hf_token_var = diar_cfg["hf_token_env_var"]
    hf_token = os.environ.get(hf_token_var, "")
    if not hf_token:
        console.print(
            f"[red]HuggingFace token not set.[/] Get one at https://hf.co/settings/tokens, "
            f"accept the gated model at https://hf.co/{diar_cfg['pyannote_pipeline']}, "
            f"then set [bold]{hf_token_var}[/] in your environment."
        )
        return EXIT_CONFIG_OR_MODEL

    # Load audio
    try:
        console.print(f"[dim]Loading[/] {args.input}")
        audio = load_audio_as_mono16k(args.input)
    except Exception as exc:
        console.print(f"[red]Failed to read audio file:[/] {exc}")
        console.print("[dim]If this is an unusual codec, try converting to PCM WAV first (e.g. with ffmpeg).[/]")
        return EXIT_BAD_INPUT

    duration_s = float(len(audio) / 16000)
    console.print(f"[dim]Loaded[/] {duration_s:.1f}s of audio @ 16 kHz mono")

    # Diarize
    diarizer = Diarizer(pipeline_name=diar_cfg["pyannote_pipeline"], hf_token=hf_token)
    try:
        with console.status("[bold]Diarizing...[/]", spinner="dots"):
            clusters = diarizer.diarize(audio, sample_rate=16000)
    except Exception as exc:
        console.print(f"[red]Diarization failed:[/] {exc}")
        console.print(
            "[dim]If this looks like a model download issue, check your HF token has access "
            f"to {diar_cfg['pyannote_pipeline']} (the model is gated and requires accepting its license).[/]"
        )
        return EXIT_CONFIG_OR_MODEL

    if not clusters:
        console.print("[yellow]No speech detected — writing empty timeline.[/]")
        labels: dict = {}
    else:
        console.print(f"[green]{len(clusters)} cluster(s) found.[/] Identifying...")

        # Identify clusters
        embedder = EmbeddingExtractor()
        store = EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])
        if not store.list_users():
            console.print("[yellow]No enrolled voiceprints — all clusters will be labeled 'unknown'.[/]")
        identifier = ClusterIdentifier(
            embedder=embedder,
            enrollment_store=store,
            threshold=diar_cfg["identification_threshold"],
            max_sample_seconds=diar_cfg["centroid_max_sample_seconds"],
        )
        labels = identifier.label_clusters(audio, sample_rate=16000, clusters=clusters)
        for cid, label in labels.items():
            console.print(f"  [dim]{cid}[/] → [bold]{label}[/]")

    # Build output
    segments = flatten_clusters(clusters, labels)
    out_path = args.out or (args.input + ".diarization.json")
    write_json(
        out_path,
        audio_file=os.path.abspath(args.input),
        duration_s=duration_s,
        diarized_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config={
            "pyannote_pipeline": diar_cfg["pyannote_pipeline"],
            "identification_threshold": diar_cfg["identification_threshold"],
        },
        segments=segments,
    )
    console.print(f"[green]Wrote[/] {out_path}")

    if args.rttm:
        rttm_path = os.path.splitext(out_path)[0] + ".rttm"
        audio_id = os.path.splitext(os.path.basename(args.input))[0]
        write_rttm(rttm_path, audio_file_id=audio_id, segments=segments)
        console.print(f"[green]Wrote[/] {rttm_path}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
