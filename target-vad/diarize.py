"""Classroom diarization entry point — see docs/superpowers/specs/2026-05-15-in-session-enrollment-design.md."""

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
from core.speaker.verifier import cosine_similarity
from modes.diarization.diarizer import Diarizer
from modes.diarization.identifier import ClusterIdentifier
from modes.diarization.intro_enrollment import (
    SessionEnrollmentView,
    enroll_from_intros,
)
from modes.diarization.manifest import load_manifest
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
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def flatten_clusters(
    clusters: dict, cluster_labels: dict, view: SessionEnrollmentView
) -> List[DiarizationSegment]:
    """Convert {cluster_id: [(start,end),...]} + {cluster_id: matched_id} → sorted segment list.

    Display name comes from view.get_name(matched_id) when matched_id != 'unknown'.
    """
    segments: List[DiarizationSegment] = []
    for cid, time_ranges in clusters.items():
        matched_id = cluster_labels.get(cid, "unknown")
        if matched_id == "unknown":
            display = "unknown"
        else:
            try:
                display = view.get_name(matched_id)
            except KeyError:
                # Cluster matched an id that isn't registered — shouldn't happen
                # because get_all() only exposes registered ids, but be defensive.
                display = matched_id
        for start, end in time_ranges:
            segments.append(DiarizationSegment(
                start=start, end=end, speaker_id=matched_id, speaker=display,
            ))
    segments.sort(key=lambda s: s.start)
    return segments


def _log_intro_conflicts(intros, persistent_store, warn_threshold: float) -> None:
    """For each intro that shadows a persistent id, log the cosine and warn if suspicious."""
    for ivp in intros:
        persistent_vp = persistent_store.get(ivp.id)
        if persistent_vp is None:
            continue
        cos = cosine_similarity(ivp.embedding, persistent_vp)
        if cos >= warn_threshold:
            console.print(
                f"[dim][INTRO OVERRIDE][/] id={ivp.id} cosine={cos:.3f} (intro replaces persistent voiceprint)"
            )
        else:
            console.print(
                f"[yellow][INTRO SUSPICIOUS][/] id={ivp.id} cosine={cos:.3f} "
                f"(below warn threshold {warn_threshold}; same id intended?)"
            )


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD - Classroom Diarization (S1)")
    parser.add_argument("input", help="Path to a WAV file")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input>.diarization.json)")
    parser.add_argument("--rttm", action="store_true", help="Also write an RTTM file alongside the JSON")
    parser.add_argument("--introductions", default=None, help="Path to a JSON manifest of intro time-ranges + ids + names")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        console.print(f"[red]Input file not found:[/] {args.input}")
        return EXIT_BAD_INPUT

    config = load_config(args.config)
    try:
        diar_cfg = config["diarization"]
    except KeyError:
        console.print(
            f"[red]Config file {args.config!r} is missing the [bold]diarization:[/bold] block.[/] "
            "See [dim]config.yaml.example[/] or the project README for the expected shape."
        )
        return EXIT_CONFIG_OR_MODEL

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

    # Build persistent store + optional intros
    persistent_store = EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])
    embedder = EmbeddingExtractor()
    intros = []
    if args.introductions:
        try:
            manifest = load_manifest(args.introductions)
        except FileNotFoundError as exc:
            console.print(f"[red]Introductions manifest not found:[/] {exc}")
            return EXIT_BAD_INPUT
        except ValueError as exc:
            console.print(f"[red]Introductions manifest invalid:[/] {exc}")
            return EXIT_BAD_INPUT
        console.print(f"[dim]Loaded[/] {len(manifest)} intro entries")
        intros = enroll_from_intros(audio, 16000, manifest, embedder)
        _log_intro_conflicts(intros, persistent_store, diar_cfg["intro_override_warn_threshold"])

    view = SessionEnrollmentView(persistent_store, intros)

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
        console.print("[yellow]No speech detected - writing empty timeline.[/]")
        labels: dict = {}
    else:
        console.print(f"[green]{len(clusters)} cluster(s) found.[/] Identifying...")
        if not view.get_all():
            console.print("[yellow]No enrolled voiceprints (persistent or session) - all clusters will be labeled 'unknown'.[/]")
        identifier = ClusterIdentifier(
            embedder=embedder,
            enrollment_store=view,
            threshold=diar_cfg["identification_threshold"],
            max_sample_seconds=diar_cfg["centroid_max_sample_seconds"],
            unknown_min_segments=diar_cfg.get("unknown_min_segments", 2),
            unknown_min_seconds=diar_cfg.get("unknown_min_seconds", 10.0),
        )
        labels = identifier.label_clusters(audio, sample_rate=16000, clusters=clusters)
        for cid, matched_id in labels.items():
            if matched_id == "unknown":
                console.print(f"  [dim]{cid}[/] -> [bold]unknown[/]")
            elif matched_id == cid:
                # Threshold passed but no enrollment match - pyannote id preserved.
                console.print(f"  [dim]{cid}[/] -> [bold]{matched_id}[/] [dim](recurring unknown)[/]")
            else:
                display = view.get_name(matched_id)
                console.print(f"  [dim]{cid}[/] -> [bold]{matched_id}[/] ([dim]{display}[/])")

    # Build output
    segments = flatten_clusters(clusters, labels, view)
    enrolled_ids = set(view.get_all().keys())
    out_path = args.out or (args.input + ".diarization.json")
    write_json(
        out_path,
        audio_file=os.path.abspath(args.input),
        duration_s=duration_s,
        diarized_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config={
            "pyannote_pipeline": diar_cfg["pyannote_pipeline"],
            "identification_threshold": diar_cfg["identification_threshold"],
            "intro_override_warn_threshold": diar_cfg["intro_override_warn_threshold"],
            "unknown_min_segments": diar_cfg.get("unknown_min_segments", 2),
            "unknown_min_seconds": diar_cfg.get("unknown_min_seconds", 10.0),
            "introductions_manifest": args.introductions,
        },
        segments=segments,
        enrolled_ids=enrolled_ids,
        passes_run=["diarization"],
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
