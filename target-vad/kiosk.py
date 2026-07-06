# kiosk.py
"""Kiosk talkback entry point — wake-word activated, Director-owned session."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse
import sys
from typing import Any, Optional

import yaml
from rich.console import Console

from modes.director.wakegate import WakeGate

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_event_printer(console: Console):
    """ONE owner for all console event prints (spec section 4a — single owner).

    Emits [WAKE]/[SESSION STARTED]/[SESSION ENDED]/[IDLE]. There is no separate
    [HANDOFF] tag any more: the WakeGate's session_started IS the handoff, so a
    second component can no longer double-print it."""
    def on_event(event_type: str, payload: dict) -> None:
        if event_type == "wake_detected":
            console.print(
                f"[magenta][WAKE][/] phrase={payload['phrase']} "
                f"score={payload['score']:.3f}"
            )
        elif event_type == "session_started":
            console.print("[bold cyan][SESSION STARTED][/] Primary speaker locked")
        elif event_type == "session_ended":
            console.print(f"[bold yellow][SESSION ENDED][/] reason={payload['reason']}\n")
            console.print("[dim][IDLE] Listening for wake phrase...[/]")
        elif event_type == "awaiting_speech_timeout":
            # Pre-session abort (no session ever started); fall back to IDLE.
            console.print("[dim][IDLE] No speech after wake; listening again...[/]")
    return on_event


def build_wakegate(
    config: dict,
    console: Console,
    runtime: Any,
    _mic: Optional[Any] = None,
    _vad: Optional[Any] = None,
    _embedder: Optional[Any] = None,
    _wake_detector: Optional[Any] = None,
) -> WakeGate:
    """Construct the WakeGate around a DirectorRuntime. Underscore kwargs inject
    fakes in tests; production passes none and the WakeGate builds real I/O."""
    return WakeGate(
        config=config,
        runtime=runtime,
        on_event=_make_event_printer(console),
        _mic=_mic, _vad=_vad, _embedder=_embedder, _wake_detector=_wake_detector,
    )


def _assert_array_startup(config: dict, console: Console) -> None:
    """Director-10 startup asserts, run once before the wake loop.

    (1) Pinned TTS output resolves -> exit(4) if not (fail loud: TTS off the
        array means no hardware AEC and Bug A returns invisibly).
    (2) ReSpeaker AGC off (AGCONOFF=0; volatile, reset on power cycle) ->
        warn-only on failure: AGC-on degrades proximity-floor stability, not
        correctness."""
    tb_cfg = config["kiosk"].get("talkback", {})
    spec = tb_cfg.get("output_device")
    if spec is not None:
        from modes.director.assembly import resolve_output_device
        import sounddevice as sd
        try:
            devices = sd.query_devices()
            idx = resolve_output_device(spec, devices)
            name = devices[idx]["name"]
        except (RuntimeError, IndexError) as e:
            console.print(f"[red]✗[/] {e}")
            sys.exit(4)
        console.print(f"[green]✓[/] TTS output pinned: {name}")
    try:
        from core.audio import respeaker
        dev = respeaker.find()
        if dev is None:
            raise RuntimeError("ReSpeaker not found on USB (2886:0018)")
        respeaker.write_param(dev, "AGCONOFF", 0)
        console.print("[green]✓[/] ReSpeaker AGC off")
    except Exception as e:
        console.print(
            f"[yellow]![/] ReSpeaker AGC assert failed ({e}); "
            "continuing with AGC on — proximity floors will be less stable")


class _LazyDirectorRuntime:
    """The object the WakeGate calls as runtime.run(handoff).

    Plan-02's DirectorRuntime needs the per-session handoff (mic/vad/embedder/
    embeddings) to build its workers, but those only exist once the WakeGate
    snapshots the first segment. So the heavy backends (STT/TTS/LLM/Player) are
    warmed ONCE here at startup, and run(handoff) constructs the real per-session
    DirectorRuntime via build_director_runtime(handoff, ...) and drives it to a
    DirectorResult. One blocking call per session, exactly as the WakeGate
    expects (spec section 4a.2)."""

    def __init__(self, config: dict, stt, llm, tts, player, logger):
        self._config = config
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._player = player
        self._logger = logger

    def run(self, handoff):
        from modes.director.assembly import build_director_runtime
        tb_cfg = self._config["kiosk"].get("talkback", {})
        watchdog_tick_s = tb_cfg.get("watchdog", {}).get("tick_ms", 500) / 1000.0
        factory = build_director_runtime(
            handoff, stt=self._stt, llm=self._llm, tts=self._tts,
            player=self._player, logger=self._logger,
            _watchdog_tick_s=watchdog_tick_s,
        )
        return factory.run(handoff)


def _build_runtime(config: dict) -> _LazyDirectorRuntime:
    """Warm STT/TTS/LLM, then wrap them in the lazy runtime that the WakeGate
    drives once per session. Imports are local so a bare `import kiosk` never
    loads GPU backends."""
    import asyncio

    from core.logging.jsonl_logger import EventLogger
    from modes.talkback.llm import LlmClient
    from modes.talkback.player import Player
    from modes.talkback.stt import StreamingStt
    from modes.talkback.tts import TtsEngine

    tb_cfg = config["kiosk"].get("talkback", {})
    logger = EventLogger(
        path_template=tb_cfg.get("logging", {}).get(
            "jsonl_path", "logs/kiosk-{date}-{session_id}.jsonl"),
        session_id="pending",
    )

    stt_cfg = tb_cfg.get("stt", {})
    stt = StreamingStt(
        model=stt_cfg.get("model", "base.en"),
        device=stt_cfg.get("device", "cuda"),
    )
    llm_cfg = tb_cfg.get("llm", {})
    llm = LlmClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        model=llm_cfg.get("model", "gemma-3-4b-it"),
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

    with console.status("[bold]Loading STT model..."):
        stt._ensure_model()
    console.print("[green]✓[/] STT loaded")
    with console.status("[bold]Loading TTS model (Kokoro)..."):
        tts._ensure_model()
    console.print("[green]✓[/] TTS loaded")
    with console.status("[bold]Checking LLM server..."):
        llm_ok = asyncio.run(llm.ping())
    if not llm_ok:
        console.print("[red]✗[/] LLM server unreachable at "
                      + llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"))
        console.print("[dim]Start llama-server and retry.[/]")
        sys.exit(3)
    console.print("[green]✓[/] LLM server reachable")

    return _LazyDirectorRuntime(config, stt, llm, tts, player, logger)


def main():
    parser = argparse.ArgumentParser(description="Target VAD — Kiosk Talkback")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--wake-phrase",
        help="Override wake phrase (default from config). Bundled options: "
             "hey_jarvis, alexa, hey_mycroft.",
    )
    parser.add_argument(
        "--talkback", action="store_true",
        help="Force talkback_enabled=true (full-duplex voice assistant mode).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.wake_phrase:
        config["kiosk"]["wake_phrase"] = args.wake_phrase
    if args.talkback:
        config["kiosk"]["talkback_enabled"] = True

    if not config["kiosk"].get("talkback_enabled", False):
        console.print(
            "[yellow]Director kiosk requires talkback. Re-run with --talkback "
            "(or set kiosk.talkback_enabled: true).[/]"
        )
        sys.exit(2)

    runtime = _build_runtime(config)
    _assert_array_startup(config, console)
    console.print(
        f"[bold][TALKBACK][/] Listening for "
        f"[bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
    )
    gate = build_wakegate(config, console, runtime=runtime)

    try:
        gate.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")


if __name__ == "__main__":
    main()
