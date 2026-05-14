"""Enrollment CLI — register speaker voiceprints from microphone."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse
import os
import sys
import time

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.audio.mic_stream import MicrophoneStream
from speaker.embedder import EmbeddingExtractor
from speaker.enrollment_store import EnrollmentStore
from vad.silero_vad import SileroVAD

console = Console()

# Phonetically diverse pangrams + tongue-twisters. Cycled if more than 7 utterances requested.
ENROLLMENT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz, judge my vow.",
    "The five boxing wizards jump quickly.",
    "She sells seashells down by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_enroll(args):
    """Record N utterances and enroll a new speaker."""
    config = load_config(args.config)
    utterances = args.utterances or config["speaker"]["enrollment_utterances"]
    min_self_sim = config["speaker"].get("enrollment_min_self_similarity", 0.6)
    max_retries = config["speaker"].get("enrollment_max_retries", 3)

    console.print(Panel(
        f"Enrolling [bold cyan]{args.user}[/] with {utterances} utterances\n"
        f"Self-similarity gate: ≥ {min_self_sim:.2f} (max {max_retries} retries per utterance)",
        title="Speaker Enrollment",
    ))

    vad = SileroVAD(config["vad"])
    embedder = EmbeddingExtractor()
    store = EnrollmentStore(config["paths"]["voiceprints_dir"])

    # Clear any previous partial enrollment
    utt_path = store._utterances_path(args.user)
    if os.path.exists(utt_path):
        os.remove(utt_path)

    mic = MicrophoneStream(config["audio"])
    accepted_embeddings = []  # in-memory; running mean computed for self-similarity gate

    for i in range(utterances):
        sentence = ENROLLMENT_SENTENCES[i % len(ENROLLMENT_SENTENCES)]
        accepted = False

        for attempt in range(1, max_retries + 1):
            retry_tag = f"  [dim](attempt {attempt}/{max_retries})[/]" if attempt > 1 else ""
            console.print(f"\n[bold yellow]Utterance {i + 1}/{utterances}[/]{retry_tag}")
            console.print(f"  Read aloud: [bold cyan]\"{sentence}\"[/]")
            console.print("  [dim]Speak now...[/]")
            vad.reset()

            with mic:
                segment = None
                for seg in vad.process_stream(mic.stream()):
                    segment = seg
                    break  # Take the first speech segment

            if segment is None:
                console.print("  [red]No speech detected.[/]")
                continue

            console.print(f"  Captured {segment.duration_ms:.0f}ms of speech")
            embedding = embedder.extract(segment.audio)

            if accepted_embeddings:
                running_mean = np.mean(accepted_embeddings, axis=0)
                rm_norm = np.linalg.norm(running_mean)
                if rm_norm > 0:
                    running_mean = running_mean / rm_norm
                sim = float(np.dot(embedding, running_mean))
                if sim < min_self_sim:
                    console.print(
                        f"  [red]Rejected[/] — similarity to your prior utterances = {sim:.3f} "
                        f"(need ≥ {min_self_sim:.2f}). Re-reading the same sentence."
                    )
                    continue
                console.print(f"  [green]Accepted[/] (similarity = {sim:.3f})")
            else:
                console.print("  [green]Accepted[/] (seed utterance)")

            accepted_embeddings.append(embedding)
            store.enroll(args.user, embedding)
            accepted = True
            break

        if not accepted:
            console.print(
                f"  [red]Giving up on utterance {i + 1} after {max_retries} attempts. "
                f"Continuing with remaining utterances.[/]"
            )

        if i < utterances - 1:
            time.sleep(0.5)

    # Finalize
    count = store.utterance_count(args.user)
    if count > 0:
        store.finalize_enrollment(args.user)
        vp = store.get(args.user)
        console.print(Panel(
            f"[bold green]Enrollment complete![/]\n"
            f"User: {args.user}\n"
            f"Utterances accepted: {count}/{utterances}\n"
            f"Embedding norm: {np.linalg.norm(vp):.4f}\n"
            f"Embedding mean: {np.mean(vp):.6f}",
            title="Success",
        ))
    else:
        console.print("[red]No utterances captured. Enrollment failed.[/]")


def cmd_list(args):
    """List all enrolled users."""
    config = load_config(args.config)
    store = EnrollmentStore(config["paths"]["voiceprints_dir"])
    users = store.list_users()
    if not users:
        console.print("[yellow]No enrolled users.[/]")
    else:
        console.print("[bold]Enrolled users:[/]")
        for u in users:
            vp = store.get(u)
            console.print(f"  - {u} (dim={len(vp)}, norm={np.linalg.norm(vp):.4f})")


def cmd_delete(args):
    """Delete a user's voiceprint."""
    config = load_config(args.config)
    store = EnrollmentStore(config["paths"]["voiceprints_dir"])
    if store.get(args.user) is None:
        console.print(f"[red]User '{args.user}' not found.[/]")
        return
    store.delete(args.user)
    console.print(f"[green]Deleted voiceprint for '{args.user}'.[/]")


def cmd_test(args):
    """Run a live verification test (shows scores without enrolling)."""
    config = load_config(args.config)
    vad = SileroVAD(config["vad"])
    embedder = EmbeddingExtractor()
    store = EnrollmentStore(config["paths"]["voiceprints_dir"])

    from speaker.verifier import SpeakerVerifier
    verifier = SpeakerVerifier(store, config["speaker"]["threshold"])

    users = store.list_users()
    if not users:
        console.print("[yellow]No enrolled users to test against.[/]")
        return

    console.print("[bold]Live verification test[/] — speak to see scores. Ctrl+C to stop.\n")
    mic = MicrophoneStream(config["audio"])

    try:
        with mic:
            vad.reset()
            for segment in vad.process_stream(mic.stream()):
                embedding = embedder.extract(segment.audio)
                result = verifier.verify(embedding)

                if result.is_registered:
                    console.print(
                        f"[bold green][MATCH][/] user={result.matched_user} "
                        f"| confidence={result.confidence:.3f} "
                        f"| duration={segment.duration_ms:.0f}ms"
                    )
                else:
                    console.print(
                        f"[dim][NO MATCH][/] best_score={result.confidence:.3f} "
                        f"| duration={segment.duration_ms:.0f}ms"
                    )
                for user, score in result.all_scores.items():
                    console.print(f"    {user}: {score:.4f}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Test stopped.[/]")


def main():
    parser = argparse.ArgumentParser(description="Target VAD Speaker Enrollment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a new speaker")
    p_enroll.add_argument("--user", required=True, help="Username to enroll")
    p_enroll.add_argument("--utterances", type=int, help="Number of utterances")

    # list
    sub.add_parser("list", help="List enrolled users")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a user's voiceprint")
    p_delete.add_argument("--user", required=True, help="Username to delete")

    # test
    sub.add_parser("test", help="Live verification test")

    args = parser.parse_args()
    commands = {
        "enroll": cmd_enroll,
        "list": cmd_list,
        "delete": cmd_delete,
        "test": cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
