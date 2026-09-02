#!/usr/bin/env python3
"""Throwaway spike: can PROXIMITY (near-field ATR) + CAMERA reject bystanders?

Spatial direction-finding is dead on this rig (C10 is hardware dual-mono; the two
USB mics can't be clock-synced -- see memory c10-dual-mono-no-stereo). So this probe
now measures the NON-spatial levers that can still deliver "never answer a stranger":

  - ATR2500x RMS (dBFS)         near-field cardioid: owner ~10" is LOUD, bystander far is quiet
  - C10/ATR level ratio (dB)    on-axis-near (owner) reads lower ratio than off/far sources
  - camera face present + x      owner is the face in frame (Director-07 proven)

The core question: is there an ATR level (+ camera-present) threshold that ADMITS the
owner and REJECTS even a LOUD bystander at realistic distance, with margin?

Solo is fine for the level gap (owner-at-mic vs you-at-bystander-distance, recorded
sequentially). The camera frame is grabbed DURING the talking so the owner is framed.

Run where the kiosk's audio works (PortAudio on the lib path):
    LD_LIBRARY_PATH=$HOME/.local/lib python3 bench/spatial_voice_probe.py run
Subcommands:
    devices            list input devices + resolved C10/ATR indices
    meter              live ATR/C10 RMS meters (level/aim setup)
    collect LABEL      record one labelled position, print the row, append CSV
    run                walk the proximity positions with prompts, print verdict

NOT production code. No camera CAP_PROP_FPS (switches the C10 UVC mode; YuNet blinds).
"""
import argparse
import sys
import threading
import time

import numpy as np

CSV_PATH = "/tmp/claude-1000/spatial_voice_probe.csv"
C10_SR = 16000
ATR_SR = 44100
DUR_S = 3.0
SETTLE_S = 0.6
# (label, spoken hint) — solo-friendly proximity protocol
POSITIONS = [
    ("OWNER", "stand at the SERVED-USER spot, ~10in from the ATR, face the camera, talk normally"),
    ("BYST-3FT", "step to a BYSTANDER spot ~3ft away/to the side, talk normally"),
    ("BYST-6FT", "BYSTANDER ~6ft+ away, talk normally"),
    ("BYST-LOUD", "BYSTANDER ~3-6ft away talking LOUDLY (the hard case)"),
    ("ROOM", "step away and stay SILENT — ambient/noise-floor baseline"),
]


def _sd():
    try:
        import sounddevice as sd
        return sd
    except OSError as exc:
        print(f"[probe] sounddevice/PortAudio unavailable: {exc}\n"
              f"        retry with: LD_LIBRARY_PATH=$HOME/.local/lib python3 {sys.argv[0]} ...",
              file=sys.stderr)
        raise SystemExit(2)


def resolve_devices(sd):
    c10 = atr = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        name = d["name"].lower()
        if "c10" in name and c10 is None:
            c10 = i
        if "atr2500" in name and atr is None:
            atr = i
    return c10, atr


class _Recorder(threading.Thread):
    """Capture `dur`s from one device on its own thread (mics open the same window)."""
    def __init__(self, sd, index, channels, sr, dur):
        super().__init__(daemon=True)
        self._sd, self._index, self._ch, self._sr, self._dur = sd, index, channels, sr, dur
        self.audio = None
        self.error = None

    def run(self):
        try:
            buf = self._sd.rec(int(self._sr * self._dur), samplerate=self._sr,
                               channels=self._ch, dtype="float32", device=self._index)
            self._sd.wait()
            self.audio = buf
        except Exception as exc:  # noqa: BLE001
            self.error = exc


def rms_db(x):
    if x is None or x.size == 0:
        return float("nan")
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-9)


# ---- camera (persistent; grabbed DURING the audio window) -------------------
def open_camera():
    try:
        from modes.director.vision.opencv_backend import OpenCvBackend, cv2_available
    except Exception:
        return None
    if not cv2_available():
        return None
    be = OpenCvBackend(0, 640, 360, 0.40, 0.015)
    return be if be.open() else None


def face_from(be):
    """(center_x in [-1,1], area_frac, present_bool). present = face & area>=min."""
    if be is None:
        return float("nan"), float("nan"), False
    for _ in range(4):          # small drain of stale UVC frames
        be.grab()
    frame = be.grab()
    if frame is None:
        return float("nan"), float("nan"), False
    h, w = frame.shape[:2]
    f = be._largest_face(frame)
    if f is None:
        return float("nan"), float("nan"), False
    cx = ((float(f[0]) + float(f[2]) / 2.0) / w - 0.5) * 2.0
    area = (float(f[2]) * float(f[3])) / (w * h)
    return cx, area, bool(area >= 0.015)


def measure(sd, c10_idx, atr_idx, cam):
    rc = _Recorder(sd, c10_idx, 2, C10_SR, DUR_S) if c10_idx is not None else None
    ra = _Recorder(sd, atr_idx, 1, ATR_SR, DUR_S) if atr_idx is not None else None
    for r in (rc, ra):
        if r:
            r.start()
    time.sleep(min(1.3, DUR_S / 2))         # grab camera mid-window (owner framed)
    face_x, face_area, present = face_from(cam)
    for r in (rc, ra):
        if r:
            r.join()
    c10 = rc.audio[int(C10_SR * SETTLE_S):, 0] if rc and rc.error is None else None
    atr = ra.audio[int(ATR_SR * SETTLE_S):, 0] if ra and ra.error is None else None
    c10_db, atr_db = rms_db(c10), rms_db(atr)
    return {
        "atr_db": atr_db,
        "c10_db": c10_db,
        "ratio_db": c10_db - atr_db,
        "face_x": face_x,
        "face_area": face_area,
        "present": 1.0 if present else 0.0,
    }


_HDR = ("label", "atr_db", "c10_db", "ratio_db", "face_x", "face_area", "present")


def _fmt_row(label, m):
    pres = "FACE" if m["present"] >= 0.5 else "----"
    return (f"{label:<10} ATR={m['atr_db']:6.1f}dB  C10={m['c10_db']:6.1f}dB  "
            f"C10/ATR={m['ratio_db']:+5.1f}dB  cam[{pres} x={m['face_x']:+.2f} "
            f"area={m['face_area']:.3f}]")


def _append_csv(label, m):
    import os
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a") as fh:
        if new:
            fh.write(",".join(_HDR) + "\n")
        fh.write(",".join([label] + [f"{m[k]:.5f}" for k in _HDR[1:]]) + "\n")


def _verdict(rows):
    by = {lbl: m for lbl, m in rows}
    print("\n================= SUMMARY =================")
    for lbl, m in rows:
        print(_fmt_row(lbl, m))
    print("------------------------------------------")
    owner = by.get("OWNER")
    if not owner:
        print("no OWNER baseline; rerun.", file=sys.stderr)
        return
    bys = {k: by[k] for k in ("BYST-3FT", "BYST-6FT", "BYST-LOUD") if k in by}
    room = by.get("ROOM")
    print("\nPROXIMITY (ATR level gap, owner vs bystander — want owner >> bystander):")
    loudest = None
    for k, m in bys.items():
        gap = owner["atr_db"] - m["atr_db"]
        print(f"   OWNER - {k:<9} = {gap:+5.1f}dB")
        if loudest is None or m["atr_db"] > by[loudest]["atr_db"]:
            loudest = k
    if room:
        print(f"   ROOM floor (ATR)     = {room['atr_db']:6.1f}dB   "
              f"(owner {owner['atr_db'] - room['atr_db']:+.1f}dB above floor)")
    if loudest:
        worst_gap = owner["atr_db"] - by[loudest]["atr_db"]
        thr = (owner["atr_db"] + by[loudest]["atr_db"]) / 2.0
        verdict = "GO" if worst_gap >= 6.0 else ("MARGINAL" if worst_gap >= 3.0 else "NO-GO")
        print(f"\n   worst case = {loudest}: owner is {worst_gap:+.1f}dB above it -> {verdict}")
        print(f"   a level gate at ~{thr:.1f}dBFS would admit owner, reject {loudest} "
              f"(margins {owner['atr_db']-thr:+.1f} / {thr-by[loudest]['atr_db']:+.1f}dB)")
    print("\nC10/ATR ratio (owner on-axis-near should read LOWER than off/far sources):")
    for lbl, m in rows:
        if lbl != "ROOM":
            d = m["ratio_db"] - owner["ratio_db"]
            print(f"   {lbl:<9} ratio={m['ratio_db']:+5.1f}dB  (vs owner {d:+5.1f}dB)")
    print("\nCAMERA presence (owner should be FACE; side/far bystanders often ----):")
    for lbl, m in rows:
        print(f"   {lbl:<9} {'FACE' if m['present']>=0.5 else '----'}  x={m['face_x']:+.2f}")
    print("\n'Never answer a stranger' -> we want OWNER clearly above BYST-LOUD on ATR")
    print("level AND owner = the FACE in frame. Read the gaps above with that lens.")


def cmd_devices(args):
    sd = _sd()
    c10, atr = resolve_devices(sd)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            tag = "  <- C10" if i == c10 else ("  <- ATR" if i == atr else "")
            print(f"{i:>2} in_ch={d['max_input_channels']} "
                  f"sr={int(d['default_samplerate'])} {d['name']}{tag}")
    print(f"\nresolved: C10={c10}  ATR={atr}")
    if atr is None:
        print("WARNING: ATR not found -> proximity lever can't be measured.", file=sys.stderr)


def cmd_meter(args):
    sd = _sd()
    c10, atr = resolve_devices(sd)
    print("live RMS (Ctrl-C to stop). Set ATR gain/aim here; don't change gain mid-test.")
    try:
        while True:
            ra = _Recorder(sd, atr, 1, ATR_SR, 0.4)
            rc = _Recorder(sd, c10, 2, C10_SR, 0.4)
            ra.start(); rc.start(); ra.join(); rc.join()
            a = rms_db(ra.audio[:, 0]) if ra.audio is not None else float("nan")
            c = rms_db(rc.audio[:, 0]) if rc.audio is not None else float("nan")
            print(f"\rATR={a:6.1f}dB  C10={c:6.1f}dB  C10/ATR={c-a:+5.1f}dB     ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print()


def cmd_collect(args):
    sd = _sd()
    c10, atr = resolve_devices(sd)
    cam = open_camera()
    try:
        m = measure(sd, c10, atr, cam)
    finally:
        if cam:
            cam.close()
    print(_fmt_row(args.label, m))
    _append_csv(args.label, m)


def cmd_cue(args):
    """Realistic noisy-room test: capture the bystander floor, then cue the owner to
    speak OVER it and report the level gap. Re-run for more samples."""
    sd = _sd()
    c10, atr = resolve_devices(sd)
    cam = open_camera()
    print("camera:", "open" if cam else "UNAVAILABLE")
    try:
        input("\nLet the bystanders talk; you STAY SILENT. Press Enter to grab the ROOM...")
        for n in (3, 2, 1):
            print(f"  {n}...", end="", flush=True)
            time.sleep(0.4)
        print(" capturing AMBIENT")
        amb = measure(sd, c10, atr, cam)
        print("  " + _fmt_row("AMBIENT", amb))

        input("\nNow get ready to speak AT THE MIC (owner spot). Press Enter, then watch for the cue...")
        for n in (3, 2, 1):
            print(f"  {n}...", end="", flush=True)
            time.sleep(0.7)
        print("\n\n   ████████  SPEAK NOW — talk at the mic  ████████\n", flush=True)
        own = measure(sd, c10, atr, cam)
        print("   " + _fmt_row("OWNER", own))

        gap = own["atr_db"] - amb["atr_db"]
        verdict = "GO" if gap >= 6.0 else ("MARGINAL" if gap >= 3.0 else "NO-GO")
        print(f"\n   OWNER over BYSTANDER-CHATTER (ATR) = {gap:+.1f}dB  -> {verdict}")
        print(f"   camera owner: {'FACE' if own['present'] >= 0.5 else '----'}  "
              f"x={own['face_x']:+.2f}")
        print(f"   C10/ATR ratio: owner {own['ratio_db']:+.1f}dB vs ambient "
              f"{amb['ratio_db']:+.1f}dB")
        _append_csv("CUE-AMBIENT", amb)
        _append_csv("CUE-OWNER", own)
        print(f"\n   rows appended to {CSV_PATH}")
    finally:
        if cam:
            cam.close()


def cmd_run(args):
    sd = _sd()
    c10, atr = resolve_devices(sd)
    print(f"devices: C10={c10} ATR={atr}  (dur={DUR_S}s/position)")
    if atr is None:
        raise SystemExit("ATR not found; cannot measure the proximity lever.")
    cam = open_camera()
    print("camera:", "open" if cam else "UNAVAILABLE (presence column will be ----)")
    rows = []
    try:
        for pos, hint in POSITIONS:
            input(f"\n[{pos}] {hint}\n  press Enter, then talk for {DUR_S:.0f}s"
                  f"{' (stay quiet)' if pos == 'ROOM' else ''}...")
            for n in (3, 2, 1):
                print(f"  {n}...", end="", flush=True)
                time.sleep(0.4)
            print(" capturing")
            m = measure(sd, c10, atr, cam)
            print("  " + _fmt_row(pos, m))
            _append_csv(pos, m)
            rows.append((pos, m))
    finally:
        if cam:
            cam.close()
    _verdict(rows)
    print(f"\nrows appended to {CSV_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices").set_defaults(fn=cmd_devices)
    sub.add_parser("meter").set_defaults(fn=cmd_meter)
    c = sub.add_parser("collect")
    c.add_argument("label")
    c.set_defaults(fn=cmd_collect)
    sub.add_parser("cue").set_defaults(fn=cmd_cue)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
