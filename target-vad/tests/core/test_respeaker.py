# tests/core/test_respeaker.py
"""core/audio/respeaker.py — XVF-3000 USB vendor-control protocol.

Everything runs against a mocked pyusb device: the encoding (wValue/wIndex/
payload) is the part worth pinning, verified 2026-07-06 against Seeed's
tuning.py and a live register read (AGCONOFF=1). No hardware in CI."""

import struct
from unittest.mock import MagicMock

import pytest
import usb.util

from core.audio.respeaker import PARAMS, find, read_param, write_param

BM_IN = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
BM_OUT = usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE


def _dev(read_value=1):
    dev = MagicMock()
    dev.ctrl_transfer = MagicMock(
        return_value=bytes(struct.pack("<ii", read_value, 0)))
    return dev


def test_read_agconoff_encoding_and_value():
    dev = _dev(read_value=1)
    assert read_param(dev, "AGCONOFF") == 1
    args = dev.ctrl_transfer.call_args[0]
    # (bmRequestType, bRequest, wValue, wIndex, length, timeout)
    assert args[0] == BM_IN
    assert args[1] == 0
    assert args[2] == 0x80 | 0x40 | 0        # offset 0, int flag
    assert args[3] == 19                     # AGCONOFF param id
    assert args[4] == 8


def test_read_doaangle_targets_param_21():
    dev = _dev(read_value=97)
    assert read_param(dev, "DOAANGLE") == 97
    args = dev.ctrl_transfer.call_args[0]
    assert args[2] == 0x80 | 0x40 | 0 and args[3] == 21


def test_write_agconoff_zero_payload():
    dev = _dev()
    write_param(dev, "AGCONOFF", 0)
    args = dev.ctrl_transfer.call_args[0]
    # (bmRequestType, bRequest, wValue, wIndex, payload, timeout)
    assert args[0] == BM_OUT
    assert args[1] == 0 and args[2] == 0
    assert args[3] == 19
    assert bytes(args[4]) == struct.pack("<iii", 0, 0, 1)   # offset, value, int-type


def test_write_readonly_param_raises():
    dev = _dev()
    with pytest.raises(ValueError):
        write_param(dev, "DOAANGLE", 5)
    dev.ctrl_transfer.assert_not_called()


def test_find_returns_none_when_absent(monkeypatch):
    import usb.core
    monkeypatch.setattr(usb.core, "find", lambda **kw: None)
    assert find() is None


def test_params_table_covers_directors_registers():
    # D10 needs AGCONOFF (rw); D11 will need the ro trio.
    assert PARAMS["AGCONOFF"][3] is True                     # writable
    for name in ("DOAANGLE", "SPEECHDETECTED", "VOICEACTIVITY"):
        assert PARAMS[name][3] is False                      # read-only
