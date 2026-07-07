# Director-11 — DOA cone vote (design)

**Date:** 2026-07-07
**Status:** approved for planning
**Depends on:** Director-10 (merged `d1f1acd`) — TTS through the ReSpeaker,
capture on processed ch0, `core/audio/respeaker.py` USB control module.

## 1. Problem

Identity and loudness gates cannot tell *where* speech comes from. Live
2026-07-06/07: a podcast louder than the proximity floor was served as user
turns; mixed ECAPA windows straddle any threshold; background audio repeatedly
ducks story playback to 0.35. The ReSpeaker's onboard DOA was spike-validated
2026-07-06: ~1° resolution, ~150ms updates, owner steady at ~97°, podcast
pinned ~100° off-axis, **DOA holds the owner during simultaneous owner+podcast
speech**, and a ±20° cone around the owner's bearing scored 100% on the spike
trace.

Direction becomes a fourth gate alongside proximity, camera presence, and the
ECAPA safety net: a segment is only owner-speech if it comes from the owner's
direction.

## 2. Scope

Gate all three audio decision points (user choice):

1. **New turns** (LISTENING → `SegmentEndpointed`) — reject out-of-cone turns.
2. **Interjections** (EVALUATING → `InterjectionSegment`) — out-of-cone rung
   in the reject ladder (Restore, never Cut).
3. **Duck-at-onset** (SPEAKING → `NearFieldOnset`) — out-of-cone onset does
   not duck.

**Non-goals:** simultaneous-overlap transcription (the louder source still
wins STT — the cone correctly passes an overlapped segment because the owner
IS speaking; fixing the words needs beam-steering/source separation, a future
project). Head-down false ABSENT (D07 gap) untouched.

## 3. Constraint that shapes the architecture

`DOAANGLE` is the bearing of the **current** dominant sound. Reading it after
a segment ends returns whatever is making noise *now*. Therefore DOA must be
sampled continuously and segments scored over their own time span — a
query-at-decision-time design cannot work.

## 4. Components

### 4.1 `core/audio/doa_tracker.py` — DoaTracker (NEW)

A daemon-thread sampler over the existing `core.audio.respeaker` module.

- Every `poll_ms` (default 150): read `DOAANGLE` and `SPEECHDETECTED`
  (`respeaker.read_param`), append `(time.monotonic(), angle_deg, speech_flag)`
  to a bounded deque (maxlen 600 ≈ 90 s).
- **Reads (thread-safe):**
  - `latest() -> (t, angle, speech) | None` — newest sample.
  - `median_between(t0, t1) -> float | None` — circular median of angles with
    `speech_flag == 1` in `[t0, t1]`; `None` if no qualifying samples.
- **Circularity:** angles are 0–359°. Circular distance
  (`min(|a-b|, 360-|a-b|)`) and circular median (minimize summed circular
  distance over candidate sample angles — O(n²) on ≤ a few hundred samples is
  fine) so the 359°/1° wraparound is 2°, not 358°.
- **Failure latch:** any USB exception → log once, mark unavailable, all
  subsequent reads return `None`. No retry loop (a dead control path stays
  dead for the session; next session's startup assert reports it).
- Lifecycle: `start()` / `stop()` (joins the thread), owned by assembly,
  stopped at session teardown. `doa.enabled: false` → tracker is never
  constructed.

### 4.2 Event plumbing (`events.py`, `workers/ingestion.py`)

New optional field on three frozen dataclasses, default `None` (existing
tests and the no-array path untouched):

- `SegmentEndpointed.doa_angle: float | None` — ingestion stamps
  `median_between(segment_start, segment_end)` (it already tracks the span).
- `InterjectionSegment.doa_angle: float | None` — same.
- `NearFieldOnset.doa_angle: float | None` — `latest()`'s angle if that
  sample has `speech_flag == 1`, else `None`.

Ingestion receives the tracker (or `None`) at construction; `None` tracker →
all stamps are `None`.

### 4.3 Cone vote (pure, `reducer.py`)

```python
def in_owner_cone(ctx, doa_angle) -> bool | None:
    # None (abstain) if doa_angle is None or ctx.owner_bearing is None
    # True  if circular_distance(doa_angle, ctx.owner_bearing) <= ctx.cfg.doa_cone_deg
    # False otherwise
```

Abstain means the cone expresses no opinion and the remaining gates decide —
the same fail-safe semantics as camera `UNAVAILABLE`.

**Wiring:**

- `classify_new_turn`: new branch **after** the proximity check —
  `in_owner_cone(...) is False` → new verdict
  `TurnVerdict.REJECT_OUT_OF_CONE = "out_of_cone"` (added to
  `gate_diag_reason`'s reject set; DIAG prints `REJECT=out_of_cone`).
- `_on_interjection_segment`: new rung after the proximity pre-gate —
  out-of-cone → `_restore_speaking(ctx)` (Restore, never Cut).
- Onset handler: `NearFieldOnset` with `in_owner_cone(...) is False` → no
  Duck, stay SPEAKING. Abstain → duck as today (wrong duck costs a second of
  volume; missed owner barge-in costs the interaction).

### 4.4 Owner bearing — seed + tracked (`context.py`, `assembly.py`, `reducer.py`)

- `ctx.owner_bearing: float | None` (new context field, default `None`).
- **Calibrate:** assembly computes `median_between` over the enrollment/seed
  segment's span and sets `owner_bearing`. Unavailable at seed time →
  `None` → cone abstains all session (pure D10 behavior).
- **Track:** on an ACCEPTED (served) turn whose `doa_angle` is not `None` and
  in-cone, update `owner_bearing = circular_ema(owner_bearing, doa_angle,
  alpha=cfg.doa_bearing_ema_alpha)` (EMA on the shortest-arc delta). Rejected
  and abstained segments never move the bearing — a bystander cannot drag the
  cone.
- DIAG: bearing at session start (`[DIAG assembly] owner bearing: 97°`) and
  on update (`[DIAG runtime] owner bearing 97.0° -> 99.1°`).

### 4.5 Startup assert (`kiosk.py`, extends D10's `_assert_array_startup`)

One `read_param("DOAANGLE")` round-trip when `doa.enabled`:

- Success → `✓ DOA control readable (bearing now 143°)`.
- Failure → `✗ DOA unavailable — cone gate will abstain` — **warn-only**
  (fail-open, user choice), NOT exit 4. AGC assert unchanged.

## 5. Config (`config.yaml`, under `turn_gate:`)

```yaml
turn_gate:
  doa:
    enabled: true          # false = no tracker, all doa_angle fields None
    cone_deg: 20           # cone half-width, degrees; spike-validated
    poll_ms: 150           # matches firmware update cadence
    bearing_ema_alpha: 0.3 # served-turn bearing tracking rate
```

Mapped into `DirectorConfig` as `doa_enabled`, `doa_cone_deg`, `doa_poll_ms`,
`doa_bearing_ema_alpha` (strict types, following the D09 config-truth
pattern: every key read, no dead keys).

## 6. Failure policy (user choice: fail open, loud DIAG)

Any absence of signal — tracker unavailable, `doa.enabled: false`, no
speech-flagged samples in a segment's span, no calibrated bearing — produces
`None` somewhere in the chain, and `in_owner_cone` abstains. The kiosk
degrades to exact D10 behavior; the other three gates still stand. The
startup assert and DIAG lines make the degraded mode visible.

## 7. Testing

- `tests/core/test_doa_tracker.py`: circular median incl. wraparound
  (e.g. samples 358°, 2° → median distance-2° result, not 180°);
  speech-flag filtering; empty window → `None`; failure latch (read_param
  raises once → all reads `None` forever); `respeaker.read_param` mocked.
- Reducer: cone reject/abstain/accept at all three decision points; onset
  abstain still ducks; interjection out-of-cone restores (never cuts);
  bearing EMA updates only on served in-cone turns; wraparound EMA;
  `doa_enabled: false` → behavior byte-identical to D10 (no-regression).
- Ingestion: events carry `median_between` over the correct span; `None`
  tracker → `None` stamps (tracker stubbed).
- Entrypoint: DOA probe success prints ✓; failure prints ✗ and does NOT
  exit (contrast with the sink-resolve exit-4 tests).

## 8. Live merge gate (4 checks)

1. **Loud podcast, owner quiet:** podcast segments show `REJECT=out_of_cone`;
   zero served podcast turns; story playback does NOT duck while the podcast
   plays off-axis (gain stays 1.0).
2. **Owner normal:** turns served; bearing tracks in DIAG; zero out_of_cone
   rejects of the owner.
3. **Owner barge-in with podcast running:** owner interjection still cuts the
   story (spike: DOA holds the owner during simultaneous speech).
4. **Fail-open:** break the control path (e.g. revoke udev perms) → startup
   prints the ✗ warn, session behaves exactly like D10.

Then verdict note + finishing-a-development-branch.
