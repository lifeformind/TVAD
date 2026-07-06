"""ReSpeaker 4 Mic Array v2.0 (XVF-3000) USB vendor-control access.

Protocol re-implemented from Seeed's usb_4_mic_array tuning.py (verified
2026-07-06 against the reference and a live register read) — no external
dependency beyond pyusb. DSP params are VOLATILE (reset on power cycle), so
the kiosk asserts what it needs at every startup (Director-10: AGCONOFF=0).
Director-11 reads DOAANGLE/SPEECHDETECTED through this same module.

Operational prerequisite: /etc/udev/rules.d/60-respeaker.rules granting
MODE 0666 for 2886:0018 (replug to apply), else reads fail with Errno 13.
"""

import struct

import usb.core
import usb.util

VID, PID = 0x2886, 0x0018
TIMEOUT_MS = 5000

# name -> (param_id, offset, is_int, writable)
PARAMS = {
    "AGCONOFF":       (19, 0, True, True),    # 0 = off, 1 = on
    "SPEECHDETECTED": (19, 22, True, False),  # stable through word gaps
    "VOICEACTIVITY":  (19, 32, True, False),  # flickers ~150ms
    "DOAANGLE":       (21, 0, True, False),   # 0..359 degrees
}


def find():
    """The array's pyusb device, or None if not on the bus."""
    return usb.core.find(idVendor=VID, idProduct=PID)


def read_param(dev, name):
    param_id, offset, is_int, _ = PARAMS[name]
    cmd = 0x80 | offset
    if is_int:
        cmd |= 0x40
    resp = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, param_id, 8, TIMEOUT_MS)
    lo, hi = struct.unpack("<ii", bytes(resp))
    return lo if is_int else lo * (2.0 ** hi)


def write_param(dev, name, value):
    param_id, offset, is_int, writable = PARAMS[name]
    if not writable:
        raise ValueError(f"{name} is read-only")
    if is_int:
        payload = struct.pack("<iii", offset, int(value), 1)
    else:
        payload = struct.pack("<ifi", offset, float(value), 0)
    dev.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, 0, param_id, payload, TIMEOUT_MS)
