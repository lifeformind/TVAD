# Director-10: Array Audio-Out Migration — Design

**Date:** 2026-07-06
**Status:** approved design, pre-implementation
**Depends on:** Director-08 + Director-09 (merged to master 2026-07-06); ReSpeaker
4 Mic Array v2.0 validated same day (see `docs/notes/2026-07-06-director-09-live.md`
and `bench/respeaker_doa.py`)
**Successor:** Director-11 (DOA cone vote) builds on this substrate — deliberately
split out; tuning direction-finding before the speaker physically moves to the
array would have to be redone.

## 1. Problem

Bug A (D09 live validation, KNOWN ISSUE): the kiosk's own TTS bleeds past the
software AEC, passes the interjection gate, gets transcribed AND served — the
kiosk answers its own voice, and bleed pollutes the safety net's accumulated
windows (mixed windows scored 0.133–0.207, straddling any threshold; that is
what made the live WARN streak unreachable while the kiosk answered a hijacker).

Root cause is the capture path, not the Director: our software AEC cancels
against the Player's reference ring, but the acoustic coupling between the kiosk
speaker and the far-field array beats it. The ReSpeaker's XVF-3000 solves this
in hardware: audio played through the array's own playback device is looped into
its ch5 AEC reference and cancelled from the processed ch0 capture *before our
code ever sees the mic signal*. A powered speaker is attached to the array's
3.5mm jack (line-out; the jack carries whatever plays through the array's USB
playback path).

## 2. Goals / non-goals

Goals:
1. All kiosk TTS (replies, nudges) plays through the ReSpeaker's playback
   device — verifiably, or the kiosk refuses to start (fail loud; approach
   chosen over guarded fallback and OS-default-sink, mirroring the mic-swap
   lesson: a silent routing drift would resurrect Bug A invisibly).
2. Software AEC goes dormant via config (`aec.enabled: false`); the code path
   is kept intact and re-enableable.
3. Volatile array DSP params are asserted at startup — initially `AGCONOFF=0`
   (stable levels for proximity floors) — via a project-owned control module
   that also becomes Director-11's DOA foundation.
4. Live re-measurement on the new substrate: proximity floor, barge-in speaker
   threshold (long-standing debt from the AEC no-op era), and
   `turn_gate.speaker_threshold` (0.15 was tuned around bleed-polluted windows).

Non-goals:
- No Director state-machine, reducer, or worker-protocol changes.
- No software-AEC removal (dormant, not deleted).
- No mid-session routing watchdog: startup verification covers the real threat
  (misconfiguration); mid-session USB loss already degrades safely.
- No DOA usage (that is Director-11).

## 3. Design

### 3.1 Output routing + verification (fail loud)

`build_director_runtime` currently opens
`sd.OutputStream(device=tb_cfg.get("output_device"))` with `output_device:
null` (system default). Change:

- `kiosk.talkback.output_device: "ReSpeaker"` — a name substring, resolved at
  startup by a small helper: case-insensitive substring match over
  `sd.query_devices()`, output-capable devices only (`max_output_channels >
  0`), first match wins (deterministic). Integer values pass through unchanged
  (today's escape hatch stays). NB the PortAudio device name is
  `'ReSpeaker 4 Mic Array (UAC1.0): USB Audio'` — it does NOT contain the ALSA
  card ID `ArrayUAC10`, hence `"ReSpeaker"` as the pin value.
- Resolution failure raises `RuntimeError` with an actionable message
  ("ReSpeaker playback device not found — check USB connection / lsusb for
  2886:0018") **before any session starts**.
- If the named device resolves and the stream opens, routing is verified by
  construction — the stream is on the array; there is no default-sink election
  to drift.

**Mechanics REVISED during live validation (2026-07-06).** The pre-plan
spike showed a direct PortAudio open of the array's playback end working —
but only because the sink was idle: the first live kiosk run crashed with
`Device unavailable (-9985)`, and instrumented reproduction confirmed
PipeWire holds the array's ALSA playback PCM whenever its sink node is
active (plus its suspend-timeout tail). With the array as default sink,
the direct open is a race. Production mechanics are therefore:

1. Resolve the config substring against **live PipeWire sinks** (`pw-dump`,
   `media.class == Audio/Sink`, match on node.name or node.description) —
   `resolve_pipewire_sink` raises the actionable RuntimeError on no match.
2. Open PortAudio's `'pipewire'` device with **`PIPEWIRE_NODE`=<resolved
   node.name>** scoped around the open — measured to hard-pin the stream's
   links to that sink regardless of default-sink elections.
3. The resolve step is what carries the fail-loud guarantee: a
   `PIPEWIRE_NODE` naming a nonexistent node falls back to the default sink
   *silently* (measured), so we only ever set it to a name pw-dump just
   confirmed. An unreadable pw-dump is treated as fatal at startup
   (unverifiable routing = untrusted routing). Int specs remain a raw
   PortAudio-index escape hatch (bounds-checked at startup).

Protocol registers verified the same day: AGCONOFF read live (= 1, AGC on —
the startup assert is real work) and the write payload confirmed against
Seeed's tuning.py (`struct.pack('<iii', offset, value, 1)`, ctrl_transfer
OUT, wValue=0, wIndex=param_id). Invariant unchanged: **TTS verifiably
reaches the array or the kiosk refuses to start.**

### 3.2 Array control module — `core/audio/respeaker.py`

Promoted from `bench/respeaker_doa.py`'s register protocol (itself
re-implemented from Seeed's tuning.py — no external dependency):

- `find()` → pyusb device for 2886:0018, or `None`.
- `read_param(dev, name)` / `write_param(dev, name, value)` — the XVF-3000
  USB vendor control protocol (ctrl_transfer; read: wValue = 0x80 | offset,
  | 0x40 for int params; wIndex = param id).
- Param table carried in the module (AGCONOFF now; DOAANGLE / SPEECHDETECTED /
  VOICEACTIVITY already known for Director-11).

The kiosk entrypoint calls `write_param(dev, "AGCONOFF", 0)` at startup inside
try/except: on any failure (device absent, udev permissions), a **loud warning
and continue** — AGC-on degrades floor stability but not correctness. XVF-3000
params are volatile, so this runs at every stack start. The bench probe is
rewritten to import this module (one protocol implementation, no drift).

Operational prerequisite (documented, already in place on this box):
`/etc/udev/rules.d/60-respeaker.rules` granting MODE 0666 for 2886:0018;
requires replug to apply.

### 3.3 Software AEC dormant

`kiosk.talkback.aec.enabled: false` shipped in config with a rationale
comment. The assembly already passes `aec=None` when disabled and ingestion's
`_apply_aec` no-ops on `None` — an existing, tested path. The PlaybackWorker
still records post-gain frames into the Player's reference ring under the same
write lock: harmless while software AEC is off, keeps the race-fixed teardown
invariants byte-identical, and makes re-enabling a one-line config flip if TTS
ever plays through a non-array output again.

### 3.4 Runtime behavior (unchanged by design)

Duck/Restore still apply gain in software before each frame write — ducking
behavior is identical through the new device. Barge-in during SPEAKING now
sees ch0 with TTS already hardware-cancelled; your voice passes through, so
interjection audio is dramatically cleaner. That is a measurement change, not
a code change — hence the re-measure items in the merge gate.

## 4. Config summary

| Key | Change | Why |
|---|---|---|
| `kiosk.talkback.output_device` | `null` → `"ReSpeaker"` (+ comment) | pin TTS to the array (PipeWire-sink resolve + PIPEWIRE_NODE; direct ALSA open races PipeWire's card reservation) |
| `kiosk.talkback.aec.enabled` | `true` → `false` (+ comment) | hardware AEC does the job; flip true only if TTS leaves the array |
| `kiosk.talkback.input_device` | **deleted** | never read by any code (config truth) |
| `barge_in.speaker_threshold` | re-measured live | first-ever measurement with real echo cancellation |
| `barge_in.proximity.rms_factor` / floor semantics | re-measured live, AGC off | D09 saw 0.0002-thin margins on the array |
| `turn_gate.speaker_threshold` | re-checked live (0.15 → toward 0.25–0.30 if the gap supports it) | 0.15 was tuned around bleed-polluted windows |

## 5. Error handling

| Failure | Behavior |
|---|---|
| Array sink absent from PipeWire at startup (or pw-dump unreadable) | Refuse to start (exit 4); actionable message |
| USB control unreachable (udev/permissions) | Loud warning; kiosk continues (AGC stays on) |
| Powered speaker unplugged from the jack | Undetectable in software (line-out has no sensing); documented operational check |
| Mid-session USB device loss | Existing behavior: stream write fails, `_play_audio`'s `except` breaks the write loop; session ends via existing timeout machinery |

## 6. Testing

Unit (TDD, no hardware in CI):
- Device-resolution helper: substring match → index; no match → RuntimeError
  with actionable message; multiple matches → first output-capable,
  deterministic; integer passthrough. Faked `sd.query_devices` table.
- `core/audio/respeaker.py`: read/write against a mocked pyusb device —
  pin the wValue/wIndex encoding for AGCONOFF (the offset/id table is the
  part worth locking down); `find()` absent → `None`.
- Assembly/config mapping: `output_device` flows into the OutputStream open;
  `aec.enabled: false` → `aec=None` into ingestion.
- Entrypoint startup path: AGC-assert failure is non-fatal (mocked USB error
  → warning, startup completes).
- Bench probe imports the new module (no duplicated protocol).
- No reducer/worker behavior tests change — that is the point of the design.

## 7. Live validation — merge gate

| # | Check | Pass looks like |
|---|---|---|
| 1 | Bug A dead | Kiosk mid-reply: no interjection events from its own TTS, no self-replies, safety-net windows during TTS unpolluted |
| 2 | Barge-in survives | Owner interrupts mid-reply reliably; `barge_in.speaker_threshold` re-measured from DIAG scores |
| 3 | Floor recalibrated | AGC off; owner accepted; distant podcast `too_quiet`-rejected with visibly wider margins than D09's 0.0002 |
| 4 | `turn_gate.speaker_threshold` re-check | Owner windows re-measured bleed-free; threshold raised if the live gap supports it |

Config retunes are committed with rationale comments (D09 pattern); results go
in a verdict note `docs/notes/YYYY-MM-DD-director-10-live.md`.

## 8. Risks

- ~~PortAudio may not open the array playback directly~~ — **RESOLVED**: spike
  ran 2026-07-06, direct mono 16 kHz open works (§3.1).
- **Hardware AEC quality unknown under duck/barge conditions** — if residual
  echo still passes the interjection gate, the software AEC can be re-enabled
  on top (config flip) and measured; worst case Bug A mitigation stays at
  D09's threshold-based interim.
- **16 kHz playback ceiling** (array USB audio is 16 kHz S16_LE): TTS is
  already played at `sample_rate_hz: 16000`, so no change — noted so nobody
  "upgrades" playback quality onto this device later without revisiting.
