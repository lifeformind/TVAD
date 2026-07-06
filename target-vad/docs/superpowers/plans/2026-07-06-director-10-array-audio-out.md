# Director-10: Array Audio-Out Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route all kiosk TTS through the ReSpeaker's playback device (fail loud if it can't), so the XVF-3000's hardware AEC kills Bug A; assert AGC off at startup; put the software AEC dormant via config.

**Architecture:** Three edges, zero Director state-machine changes: (1) a new `core/audio/respeaker.py` USB-control module (register protocol verified live 2026-07-06); (2) fail-loud output-device resolution in `modes/director/assembly.py`; (3) startup asserts in `kiosk.py`. Config truth pass ships `output_device: "ReSpeaker"`, `aec.enabled: false`, and deletes the dead `input_device` key.

**Tech Stack:** Python 3.12, pytest (+pytest-asyncio already configured), pyusb (new requirement), sounddevice/PortAudio.

**Spec:** `docs/superpowers/specs/2026-07-06-director-10-array-audio-out-design.md` — read it first.

## Global Constraints

- Every commit message ends with the trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Run tests as `python3 -m pytest` from the repo root `target-vad/` (no bare `python` on this box).
- No hardware in unit tests — everything USB/audio is mocked; the array itself is only touched in live validation.
- `bench/spatial_voice_probe.py` stays untracked. Stage explicit paths only; never `git add -A`.
- Fail-loud invariant (spec §3.1): when `output_device` is set, routing failures RAISE — never fall back to another device (a silent drift would resurrect Bug A invisibly). When it is `null`, keep today's best-effort behavior.
- PortAudio device name for the array is `'ReSpeaker 4 Mic Array (UAC1.0): USB Audio'` — the pin substring is `"ReSpeaker"`, NOT the ALSA card id `ArrayUAC10`.

---

### Task 1: `core/audio/respeaker.py` — array USB-control module

**Files:**
- Create: `core/audio/respeaker.py`
- Create: `tests/core/test_respeaker.py`
- Modify: `requirements.txt` (add `pyusb`)
- Modify: `bench/respeaker_doa.py` (rewrite as thin CLI over the module; then `git add` it — it becomes tracked in this task)

**Interfaces:**
- Produces: `find() -> usb.core.Device | None`; `read_param(dev, name: str) -> int | float`; `write_param(dev, name: str, value) -> None` (raises `ValueError` for read-only params); `PARAMS: dict[str, tuple[int, int, bool, bool]]` mapping name → (param_id, offset, is_int, writable). Task 3 calls `find()` + `write_param(dev, "AGCONOFF", 0)`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_respeaker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/core/test_respeaker.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'core.audio.respeaker'`

- [ ] **Step 3: Write the module**

Create `core/audio/respeaker.py`:

```python
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
```

Append to `requirements.txt` (keep existing lines untouched):

```
pyusb>=1.2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/core/test_respeaker.py -q`
Expected: `6 passed`

- [ ] **Step 5: Rewrite the bench probe as a thin CLI over the module**

Replace the entire contents of `bench/respeaker_doa.py` with:

```python
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
```

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all pass (698 + 6 new = 704), 2 skipped

- [ ] **Step 7: Commit** (the probe becomes tracked here — explicit paths, never `git add -A`)

```bash
git add core/audio/respeaker.py tests/core/test_respeaker.py requirements.txt bench/respeaker_doa.py
git commit -m "feat(director-10): core/audio/respeaker.py — XVF-3000 USB control module

Register protocol (read + write) promoted from the bench probe, verified
against Seeed tuning.py and a live AGCONOFF read. The probe becomes a thin
tracked CLI over the module. pyusb added to requirements.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fail-loud output-device resolution in the assembly

**Files:**
- Modify: `modes/director/assembly.py` (add `resolve_output_device`; rework `_open_output_stream`, currently at the bottom of the file, `def _open_output_stream(tb_cfg: dict)`)
- Create: `tests/director/test_output_device.py`

**Interfaces:**
- Produces: `resolve_output_device(spec, devices) -> int | None` — pure, importable as `from modes.director.assembly import resolve_output_device`; raises `RuntimeError` (message contains the spec string) when a string spec matches no output-capable device. Task 3's startup assert imports exactly this.
- Consumes: nothing from Task 1.

- [ ] **Step 1: Write the failing tests**

Create `tests/director/test_output_device.py`:

```python
# tests/director/test_output_device.py
"""resolve_output_device (Director-10): pin TTS to the array, fail loud.

Pure-function tests over a fake sd.query_devices() table — no PortAudio."""

import pytest

from modes.director.assembly import resolve_output_device

DEVICES = [
    {"name": "NVIDIA: LG SDQHD (hw:0,3)", "max_output_channels": 2},
    {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)",
     "max_output_channels": 2},
    {"name": "pipewire", "max_output_channels": 64},
]


def test_none_passes_through():
    assert resolve_output_device(None, DEVICES) is None


def test_int_passes_through():
    assert resolve_output_device(5, DEVICES) == 5


def test_substring_match_is_case_insensitive():
    assert resolve_output_device("respeaker", DEVICES) == 1


def test_input_only_device_with_matching_name_is_skipped():
    devices = [
        {"name": "ReSpeaker 4 Mic Array (capture only)", "max_output_channels": 0},
        {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio", "max_output_channels": 2},
    ]
    assert resolve_output_device("ReSpeaker", devices) == 1


def test_first_output_capable_match_wins():
    devices = [
        {"name": "ReSpeaker A", "max_output_channels": 2},
        {"name": "ReSpeaker B", "max_output_channels": 2},
    ]
    assert resolve_output_device("ReSpeaker", devices) == 0


def test_no_match_raises_actionable_runtimeerror():
    with pytest.raises(RuntimeError) as exc:
        resolve_output_device("ReSpeaker", DEVICES[:1])
    assert "ReSpeaker" in str(exc.value)
    assert "output_device" in str(exc.value)


def test_build_aec_disabled_returns_none():
    # Spec s3.3: aec.enabled false -> assembly passes aec=None to ingestion
    # (whose _apply_aec no-ops on None — existing tested path). Pin the
    # config edge here since the shipped config now relies on it.
    from modes.director.assembly import _build_aec
    assert _build_aec({"aec": {"enabled": False}}) is None
    assert _build_aec({}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_output_device.py -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_output_device'`

- [ ] **Step 3: Implement**

In `modes/director/assembly.py`, add above `_open_output_stream`:

```python
def resolve_output_device(spec, devices):
    """Resolve kiosk.talkback.output_device to a PortAudio device index.

    None -> None (system default; legacy best-effort path). int -> passthrough
    (today's escape hatch). str -> case-insensitive substring match over
    OUTPUT-capable devices; no match -> RuntimeError. Fail loud is the point
    (Director-10): TTS must verifiably reach the array — its onboard AEC
    cancels the kiosk's own voice (Bug A); silently landing on another device
    would resurrect the bug invisibly. NB PortAudio's name for the array is
    'ReSpeaker 4 Mic Array (UAC1.0): USB Audio', not the ALSA id ArrayUAC10.
    """
    if spec is None or isinstance(spec, int):
        return spec
    needle = str(spec).lower()
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0 and needle in d.get("name", "").lower():
            return i
    raise RuntimeError(
        f"output_device {spec!r} not found among PortAudio output devices. "
        f"Check the ReSpeaker's USB connection (lsusb should list 2886:0018) "
        f"and the powered speaker on its 3.5mm jack, or set "
        f"kiosk.talkback.output_device to null for the system default.")
```

Replace the body of `_open_output_stream` with:

```python
def _open_output_stream(tb_cfg: dict):  # pragma: no cover - needs real audio device
    """Open a persistent sounddevice OutputStream so playback frames can be
    written + recorded as the AEC reference (controller.py:316-323).

    output_device set (Director-10 array pin): failures RAISE — never fall
    back to another device. output_device null: legacy best-effort — system
    default, None (no audible output) on any failure."""
    if os.environ.get("TVAD_NO_OUTPUT"):
        # Isolation switch: skip opening the OutputStream entirely. If the mic
        # then works (turns transcribed), opening the output stream was resetting
        # the (USB) audio device and killing the mic.
        if _DIAG:
            print("[DIAG assembly] TVAD_NO_OUTPUT set: NOT opening OutputStream",
                  file=sys.stderr, flush=True)
        return None
    device_spec = tb_cfg.get("output_device")
    if device_spec is None:
        try:
            import sounddevice as sd
            stream = sd.OutputStream(
                samplerate=tb_cfg.get("sample_rate_hz", 16000), channels=1,
                dtype="float32", device=None,
            )
            stream.start()
            return stream
        except Exception:
            return None
    import sounddevice as sd
    device = resolve_output_device(device_spec, sd.query_devices())
    stream = sd.OutputStream(
        samplerate=tb_cfg.get("sample_rate_hz", 16000), channels=1,
        dtype="float32", device=device,
    )
    stream.start()
    return stream
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_output_device.py tests/director/test_assembly.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add modes/director/assembly.py tests/director/test_output_device.py
git commit -m "feat(director-10): fail-loud output-device resolution (array pin)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: kiosk.py startup asserts (output pin + AGC off)

**Files:**
- Modify: `kiosk.py` (new `_assert_array_startup(config, console)`; called from `main()` after `_build_runtime`, before `build_wakegate`)
- Modify: `tests/director/test_kiosk_entrypoint.py` (append tests)

**Interfaces:**
- Consumes: `resolve_output_device` from Task 2 (`from modes.director.assembly import resolve_output_device`); `find` / `write_param` from Task 1 (`from core.audio import respeaker`).
- Produces: `kiosk._assert_array_startup(config: dict, console) -> None` — exits with code 4 when the pinned output device is missing; warns and continues on any AGC-assert failure.

- [ ] **Step 1: Write the failing tests**

Append to `tests/director/test_kiosk_entrypoint.py`:

```python
class _FakeSd:
    """Stand-in sounddevice module for _assert_array_startup."""
    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return self._devices


_ARRAY_DEVICES = [
    {"name": "NVIDIA: HDMI 1", "max_output_channels": 8},
    {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio", "max_output_channels": 2},
]


def _cfg_with_output(output_device="ReSpeaker"):
    cfg = _config()
    cfg["kiosk"]["talkback"]["output_device"] = output_device
    return cfg


def test_startup_assert_happy_path_pins_output_and_kills_agc(monkeypatch):
    import sys as _sys
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice", _FakeSd(_ARRAY_DEVICES))
    fake_dev = object()
    calls = []
    monkeypatch.setattr("core.audio.respeaker.find", lambda: fake_dev)
    monkeypatch.setattr("core.audio.respeaker.write_param",
                        lambda dev, name, value: calls.append((dev, name, value)))
    console = MagicMock()
    kiosk._assert_array_startup(_cfg_with_output(), console)
    assert calls == [(fake_dev, "AGCONOFF", 0)]


def test_startup_assert_missing_output_device_exits_4(monkeypatch):
    import sys as _sys
    import pytest
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice",
                        _FakeSd([{"name": "NVIDIA: HDMI 1", "max_output_channels": 8}]))
    console = MagicMock()
    with pytest.raises(SystemExit) as exc:
        kiosk._assert_array_startup(_cfg_with_output(), console)
    assert exc.value.code == 4


def test_startup_assert_agc_failure_is_nonfatal(monkeypatch):
    import sys as _sys
    import kiosk

    monkeypatch.setitem(_sys.modules, "sounddevice", _FakeSd(_ARRAY_DEVICES))
    monkeypatch.setattr("core.audio.respeaker.find", lambda: None)  # array USB absent
    console = MagicMock()
    kiosk._assert_array_startup(_cfg_with_output(), console)        # must not raise
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "AGC" in printed                                          # loud warning


def test_startup_assert_null_output_device_skips_pin_check(monkeypatch):
    import kiosk

    # No sounddevice module injected: with output_device null the pin check
    # must not even import it. AGC assert still runs (and here fails softly).
    monkeypatch.setattr("core.audio.respeaker.find", lambda: None)
    console = MagicMock()
    kiosk._assert_array_startup(_cfg_with_output(output_device=None), console)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_kiosk_entrypoint.py -q`
Expected: new tests FAIL with `AttributeError: module 'kiosk' has no attribute '_assert_array_startup'` (the pre-existing test still passes)

- [ ] **Step 3: Implement**

In `kiosk.py`, add after `build_wakegate` (module level):

```python
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
            idx = resolve_output_device(spec, sd.query_devices())
        except RuntimeError as e:
            console.print(f"[red]✗[/] {e}")
            sys.exit(4)
        name = sd.query_devices()[idx]["name"] if isinstance(idx, int) else str(idx)
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
```

In `main()`, insert the call between `_build_runtime` and `build_wakegate`:

```python
    runtime = _build_runtime(config)
    _assert_array_startup(config, console)
    console.print(
        f"[bold][TALKBACK][/] Listening for "
        f"[bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
    )
    gate = build_wakegate(config, console, runtime=runtime)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_kiosk_entrypoint.py -q`
Expected: all pass (1 pre-existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add kiosk.py tests/director/test_kiosk_entrypoint.py
git commit -m "feat(director-10): startup asserts — output pin (exit 4) + AGC off (warn)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Config truth pass + ledger + full suite

**Files:**
- Modify: `config.yaml` (talkback block: `output_device`, `aec.enabled`, delete `input_device`)
- Modify: `/home/ldrgx10/FullDuplexVoice/TVAD/.superpowers/sdd/progress.md` (append build status; note it lives OUTSIDE the repo — do not `git add` it)

**Interfaces:**
- Consumes: the runtime behavior wired in Tasks 2–3 (`output_device` string now resolves fail-loud; `aec.enabled: false` → `_build_aec` returns `None` → ingestion `aec=None`, an existing tested path).
- Produces: the shipped config the live validation runs on.

- [ ] **Step 1: Verify the dead key really is dead**

Run: `grep -rn "input_device" modes/ core/ tests/ --include=*.py | grep -v __pycache__`
Expected: no output (zero consumers — verified at plan time too)

- [ ] **Step 2: Edit config.yaml**

In the `kiosk.talkback` block, replace:

```yaml
    sample_rate_hz: 16000
    frame_ms: 10
    output_device: null
    input_device: null
```

with:

```yaml
    sample_rate_hz: 16000
    frame_ms: 10
    # Director-10: TTS plays through the ReSpeaker's playback path so the
    # XVF-3000's onboard AEC cancels the kiosk's own voice against its ch5
    # reference IN HARDWARE (kills Bug A: TTS bleed served / self-replies).
    # The powered speaker hangs off the array's 3.5mm jack. Name substring,
    # resolved against PortAudio OUTPUT devices at startup; resolution
    # failure refuses to start (exit 4) — never fall back silently, a
    # routing drift would resurrect Bug A invisibly. NB PortAudio's device
    # name does not contain the ALSA card id "ArrayUAC10". null = system
    # default (legacy best-effort). (input_device key deleted: never read.)
    output_device: "ReSpeaker"
```

and replace:

```yaml
    aec:
      enabled: true
```

with:

```yaml
    aec:
      # Director-10: OFF — TTS plays through the array, whose onboard AEC
      # cancels it in hardware before ch0 reaches our capture path. The
      # software-AEC code path is kept intact and dormant; flip true only
      # if TTS ever plays through a non-array output again.
      enabled: false
```

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all pass (715 total: 698 base + 6 Task 1 + 7 Task 2 + 4 Task 3), 2 skipped

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "feat(director-10): config truth pass — TTS pinned to array, software AEC dormant

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Append build status to the SDD ledger** (at `/home/ldrgx10/FullDuplexVoice/TVAD/.superpowers/sdd/progress.md`, outside the repo — no git add)

Append:

```markdown
## 2026-07-06 — DIRECTOR-10 BUILT (branch feat/director-10-array-audio-out), awaiting live val

Tasks 1-4 done: core/audio/respeaker.py (USB control, AGCONOFF verified live);
fail-loud resolve_output_device + _open_output_stream rework; kiosk startup
asserts (output pin exit-4 / AGC warn-only); config truth pass (output_device
"ReSpeaker", aec.enabled false, input_device deleted). Suite 715 pass / 2 skip.
MERGE GATE (spec s7, live): (1) Bug A dead — no self-interjections/replies
during TTS, safety windows unpolluted; (2) barge-in survives + re-measure
barge_in.speaker_threshold from DIAG; (3) floor recalibrated AGC-off, margins
wider than D09's 0.0002; (4) re-check turn_gate.speaker_threshold (0.15 was
tuned around bleed) — raise toward 0.25-0.30 if the live gap supports it.
Operational pre-checks: powered speaker on the array jack powered on + audible
(play a tone), udev rule active (bench/respeaker_doa.py reads without sudo).
Then verdict note docs/notes/2026-07-06-director-10-live.md + finishing skill.
```

---

## Live validation (after Task 4 — human + kiosk, not subagent work)

Run `TVAD_DIAG=1 ./kiosk-stack.sh start` and walk the merge gate (spec §7). Startup must print `✓ TTS output pinned: ReSpeaker …` and `✓ ReSpeaker AGC off`. Config retunes from checks 2–4 are committed with rationale comments (D09 pattern); results go in `docs/notes/2026-07-06-director-10-live.md`; then finishing-a-development-branch.
