"""python3 -m tune [--port 8765] [--host 127.0.0.1] [--config config.yaml]
[--start-kiosk]"""

import argparse
import os
import signal

from tune.kiosk_proc import KioskProcError, KioskProcess
from tune.server import TuningServer


def _sigterm(signum, frame):
    """SIGTERM must run the same cleanup as Ctrl-C — never orphan a kiosk."""
    raise SystemExit(0)


LOCAL_LIB = os.path.expanduser("~/.local/lib")


def ensure_local_lib_path(environ=os.environ):
    """Mirror kiosk-stack.sh's LD_LIBRARY_PATH export: PortAudio lives in
    ~/.local/lib on this box, so a kiosk spawned from a bare `python3 -m tune`
    can't import sounddevice without it (live 2026-07-13)."""
    parts = environ.get("LD_LIBRARY_PATH", "")
    if LOCAL_LIB not in parts.split(":"):
        environ["LD_LIBRARY_PATH"] = f"{LOCAL_LIB}:{parts}" if parts else LOCAL_LIB


def start_kiosk_if_requested(kproc, requested, out=print):
    """Best-effort auto-start (kiosk-stack.sh tune / --start-kiosk): a refusal
    (foreign kiosk already running) must not kill the console — the browser's
    Start button remains the retry path."""
    if not requested:
        return
    try:
        kproc.start(diag=True)
        out("[tune] kiosk auto-started (DIAG on)")
    except KioskProcError as e:
        out(f"[tune] kiosk auto-start refused: {e} — "
            "use the browser's Start button once resolved")


def main():
    ap = argparse.ArgumentParser(description="Kiosk tuning console")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to tune from another machine on the LAN")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--start-kiosk", action="store_true",
                    help="start the kiosk (DIAG on) as soon as the console is up")
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    ensure_local_lib_path()   # the kiosk child inherits our environment
    kproc = KioskProcess(cwd=os.path.dirname(config_path))
    server = TuningServer(config_path=config_path, kproc=kproc,
                          host=args.host, port=args.port)
    print(f"[tune] console at http://{args.host}:{server.port}/  "
          f"(config: {config_path})")
    signal.signal(signal.SIGTERM, _sigterm)
    start_kiosk_if_requested(kproc, args.start_kiosk)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        kproc.stop()       # never orphan a kiosk
        server.shutdown()
        print("[tune] stopped.")


if __name__ == "__main__":
    main()
