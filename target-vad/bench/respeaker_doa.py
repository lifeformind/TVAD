#!/usr/bin/env python3
"""ReSpeaker Mic Array v2.0 (XVF-3000) DOA/VAD probe.

Thin CLI over core.audio.respeaker — the kiosk's array-control module owns
the register protocol (one implementation, no drift).

Usage:
  python3 bench/respeaker_doa.py            # single reading of every param
  python3 bench/respeaker_doa.py watch [seconds]   # live sampling loop (default 30s)

GO/NO-GO for the bystander-rejection DOA leg: in `watch` mode, speak while
moving left -> center -> right of the array; DOAANGLE should track you.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.audio.respeaker import PARAMS, find, read_param  # noqa: E402


def main():
    dev = find()
    if dev is None:
        sys.exit("ReSpeaker Mic Array v2.0 (2886:0018) not found on USB")
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        print(f"watching DOA for {duration:.0f}s — speak while moving around the array")
        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < duration:
            doa = read_param(dev, "DOAANGLE")
            vad = read_param(dev, "VOICEACTIVITY")
            speech = read_param(dev, "SPEECHDETECTED")
            cur = (doa, vad, speech)
            if cur != last:
                print(f"t={time.monotonic() - t0:6.2f}s  doa={doa:3d}°  vad={vad}  speech={speech}")
                last = cur
            time.sleep(0.1)
    else:
        for name in PARAMS:
            print(f"{name} = {read_param(dev, name)}")


if __name__ == "__main__":
    main()
