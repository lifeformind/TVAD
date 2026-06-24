# Director-07 Vision Floor Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make camera presence + enrolled identity the Director's floor-control authority — keep serving the owner, free the kiosk fast when they physically leave, treat a stranger as the owner leaving — while audio stays content-only.

**Architecture:** A separate `VisionWorker` thread (CPU-only YuNet detection + SFace identity at ~3 fps) self-enrolls the owner's face at session start, then emits a single `OwnerPresenceEvent(status)` onto the existing async EventBus on debounced status changes. The reducer records presence in Context and adds **one** owner-absent end-condition inside the existing `_on_tick` — so the watchdog stays the sole timeout authority. When vision is disabled, the camera is absent, or cv2 is missing, the worker is `None` and the runtime is byte-for-byte today's Director (no-regression guarantee).

**Tech Stack:** Python 3.12, OpenCV (`cv2.FaceDetectorYN` YuNet + `cv2.FaceRecognizerSF` SFace, both pure-OpenCV ONNX, CPU), numpy, asyncio. Reuses the validated models/logic from `bench/vision_presence_probe.py` (spike) and the OpenCV-Zoo model URLs.

## Global Constraints

- **Every git commit MUST end with the trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **CPU-only, aarch64 (GB10).** Detection + identity are pure-OpenCV ONNX. NO insightface (no aarch64 wheel). Keep the GPU for the conversation stack.
- **cv2 is an OPTIONAL import.** All OpenCV use is lazy, inside functions, wrapped so a missing/broken cv2 yields a logged fallback (`VisionWorker = None`), never an import-time crash. Mirror the existing `try/except` optional-backend pattern in `modes/director/assembly.py:45-53`.
- **No-regression guarantee.** With `vision.enabled: false`, no camera, no cv2, or no captured face reference, the assembled runtime equals today's Director. This is an asserted test.
- **Single mutator.** Only `reduce()` mutates State/Context. `OwnerPresenceEvent` is pure state-recording — it NEVER causes a transition; the owner-absent decision happens only on a `Tick`.
- **Sole timeout authority preserved.** The owner-absent end is a *condition inside* `_on_tick`, not a new timer. The §5 silence timeout (30 s) and nudge (25 s) are UNCHANGED (decision 1: presence is an add-on).
- **Do NOT set `CAP_PROP_FPS`** on the camera — on this UVC cam it switches the capture mode and YuNet then detects nothing (spike finding). Use `width=640, height=360`; pace with timed `read()`/`grab()`.
- **Config defaults (verbatim, spec §8):** `identity_threshold: 0.40`, `min_area_frac: 0.015`, `fps: 3`, `width: 640`, `height: 360`, `present_after_s: 1.0`, `absent_after_s: 2.0`, `owner_absent_grace_s: 3.0`, `active_talk_guard_s: 3.0`, `enroll_frames: 8`, `camera_index: 0`.
- **Spec:** `docs/superpowers/specs/2026-06-24-director-floor-control-design.md`.

---

## File Structure

- `modes/director/events.py` (modify) — add `PresenceStatus` enum + `OwnerPresenceEvent`.
- `modes/director/context.py` (modify) — add `presence_status` + `presence_since`.
- `modes/director/config.py` (modify) — add `owner_absent_grace_s`, `active_talk_guard_s`.
- `modes/director/reducer.py` (modify) — record `OwnerPresenceEvent`; owner-absent end in `_on_tick`.
- `modes/director/assembly.py` (modify) — map `vision.*` config; `_build_vision`; pass into runtime.
- `modes/director/runtime.py` (modify) — start/stop the `VisionWorker`.
- `modes/director/vision/__init__.py` (create) — package.
- `modes/director/vision/classify.py` (create) — pure `classify_presence` + `PresenceDebouncer`.
- `modes/director/vision/monitor.py` (create) — `PresenceMonitor.observe` (classify+debounce+emit-on-change).
- `modes/director/vision/enroll.py` (create) — `enroll_reference` (frames → mean embedding).
- `modes/director/vision/opencv_backend.py` (create) — live cv2 adapter (detect+embed, frame source).
- `modes/director/workers/vision.py` (create) — `VisionWorker` thread (self-enroll → monitor → emit).
- `config.yaml` (modify) — `kiosk.talkback.vision` block.
- Tests under `tests/director/` and `tests/director/vision/`.

---

## Task 1: Presence data types (event, context, config fields)

**Files:**
- Modify: `modes/director/events.py`
- Modify: `modes/director/context.py:10-33`
- Modify: `modes/director/config.py:7-19`
- Test: `tests/director/test_presence_types.py`

**Interfaces:**
- Produces: `PresenceStatus` enum `{PRESENT, ABSENT, UNAVAILABLE}` and `OwnerPresenceEvent(status: PresenceStatus, now: float)` in `events.py`; `Context.presence_status: PresenceStatus` (init `UNAVAILABLE`), `Context.presence_since: float` (init = session `now`); `DirectorConfig.owner_absent_grace_s: float = 3.0`, `DirectorConfig.active_talk_guard_s: float = 3.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_presence_types.py
from modes.director import events as E
from modes.director.events import PresenceStatus
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.talkback.conversation import ConversationManager


def test_owner_presence_event_carries_status_and_time():
    ev = E.OwnerPresenceEvent(status=PresenceStatus.PRESENT, now=12.5)
    assert ev.status is PresenceStatus.PRESENT
    assert ev.now == 12.5


def test_presence_status_members():
    assert {s.name for s in PresenceStatus} == {"PRESENT", "ABSENT", "UNAVAILABLE"}


def test_config_floor_control_defaults():
    cfg = DirectorConfig()
    assert cfg.owner_absent_grace_s == 3.0
    assert cfg.active_talk_guard_s == 3.0


def test_context_starts_unavailable_at_session_now():
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="x"),
                      now=7.0, proximity_rms=0.0)
    assert ctx.presence_status is PresenceStatus.UNAVAILABLE
    assert ctx.presence_since == 7.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_presence_types.py -v`
Expected: FAIL (`ImportError: cannot import name 'PresenceStatus'`).

- [ ] **Step 3: Add the event + enum**

In `modes/director/events.py`, add at the top after the existing `from dataclasses import dataclass`:

```python
import enum


class PresenceStatus(enum.Enum):
    PRESENT = "PRESENT"          # the enrolled owner's face is in frame
    ABSENT = "ABSENT"            # no owner: empty frame OR a stranger (identity fail)
    UNAVAILABLE = "UNAVAILABLE"  # camera can't judge (glitch / not yet enrolled)
```

And append at the end of the file:

```python
@dataclass(frozen=True)
class OwnerPresenceEvent:
    """Camera floor-control signal (Director-07). Emitted by VisionWorker on a
    DEBOUNCED status change only. Pure state-recording in the reducer — never a
    transition; the owner-absent decision happens on a Tick."""
    status: PresenceStatus
    now: float
```

- [ ] **Step 4: Add the config fields**

In `modes/director/config.py`, add to `DirectorConfig` after the `duck_level` line:

```python
    # Camera floor control (Director-07, spec §8). Presence is an ADD-ON: these
    # never touch the silence timeout above; they only add an owner-absent end.
    owner_absent_grace_s: float = 3.0   # sustained ABSENT this long => free the kiosk
    active_talk_guard_s: float = 3.0    # never owner-absent-end within this of owner speech
```

- [ ] **Step 5: Add the context fields**

In `modes/director/context.py`, add the import at the top:

```python
from modes.director.events import PresenceStatus
```

Add these fields to the `Context` dataclass (after `interrupted_stack`):

```python
    presence_status: PresenceStatus = PresenceStatus.UNAVAILABLE  # camera floor control
    presence_since: float = 0.0          # monotonic time of the last presence change
```

In `new_context`, set `presence_since` to the session clock by adding it to the constructor call:

```python
    return Context(
        cfg=cfg, conversation=conversation, proximity_rms=proximity_rms,
        now=now, started_at=now, last_speech_at=now,
        presence_since=now,
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/test_presence_types.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add modes/director/events.py modes/director/context.py modes/director/config.py tests/director/test_presence_types.py
git commit -m "feat(director-07): presence data types — OwnerPresenceEvent + context/config fields

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Reducer — record presence + owner-absent end-condition

**Files:**
- Modify: `modes/director/reducer.py:20-61` (dispatch) and `:64-75` (`_on_tick`)
- Test: `tests/director/test_reducer_presence.py`

**Interfaces:**
- Consumes: `PresenceStatus`, `OwnerPresenceEvent` (Task 1); `Context.presence_status/presence_since`, `DirectorConfig.owner_absent_grace_s/active_talk_guard_s` (Task 1); `commands.EndSession(reason)` (existing).
- Produces: reducer behavior — `OwnerPresenceEvent` records status+since with no transition; `_on_tick` returns `(IDLE, [EndSession("owner_absent")])` when ABSENT is sustained ≥ grace AND no owner speech within the guard.

- [ ] **Step 1: Write the failing tests**

```python
# tests/director/test_reducer_presence.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    cfg = DirectorConfig(owner_absent_grace_s=3.0, active_talk_guard_s=3.0,
                         silence_timeout_s=30.0, hard_timeout_s=300.0)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=0.0)
    return ctx


def test_presence_event_records_without_transition():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.OwnerPresenceEvent(PresenceStatus.PRESENT, now=2.0))
    assert state is State.SPEAKING and cmds == []
    assert ctx.presence_status is PresenceStatus.PRESENT
    assert ctx.presence_since == 2.0


def test_absent_sustained_past_grace_ends_session():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0                     # no recent speech (guard satisfied at t>=3)
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    # tick at 10 + grace; last_speech_at far in the past
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=13.0))
    assert state is State.IDLE
    assert cmds == [C.EndSession("owner_absent")]


def test_absent_within_grace_does_not_end():
    ctx = _ctx()
    ctx.last_speech_at = 0.0
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=12.0))   # only 2s < 3s grace
    assert state is State.LISTENING and cmds == []


def test_active_talk_guard_suppresses_owner_absent():
    ctx = _ctx()
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    ctx.last_speech_at = 12.5                    # owner spoke recently
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=14.0))  # absent 4s>grace, but spoke 1.5s ago<guard
    assert state is State.LISTENING and cmds == []


def test_unavailable_falls_back_to_silence_timeout():
    ctx = _ctx()
    ctx.last_speech_at = 0.0
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, now=1.0))
    # well under silence_timeout, camera unavailable => no owner-absent end
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=5.0))
    assert state is State.LISTENING and cmds == []
    # but the unchanged silence timeout still fires at 30s
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=31.0))
    assert state is State.IDLE and cmds == [C.EndSession("silence_timeout")]


def test_present_does_not_extend_silence_timeout():
    ctx = _ctx()
    ctx.last_speech_at = 0.0
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.PRESENT, now=1.0))
    # decision 1 (add-on): a present-but-silent owner STILL times out at 30s
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=31.0))
    assert state is State.IDLE and cmds == [C.EndSession("silence_timeout")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_reducer_presence.py -v`
Expected: FAIL (`OwnerPresenceEvent` not handled — `test_presence_event_records_without_transition` sees no recording; absent tests don't end).

- [ ] **Step 3: Add the dispatch branch**

In `modes/director/reducer.py`, add the import near the top (with the other `events`/`state` imports):

```python
from modes.director.events import PresenceStatus
```

In `reduce()`, add this branch before the final `return state, []`:

```python
    if isinstance(event, E.OwnerPresenceEvent):
        ctx.presence_status = event.status      # pure state-recording: NO transition
        ctx.presence_since = event.now          # the owner-absent decision is on Tick
        return state, []
```

- [ ] **Step 4: Add the owner-absent end-condition to `_on_tick`**

In `_on_tick`, add this block immediately before the final `return state, []`:

```python
    # Camera floor control (Director-07): owner physically gone -> free the kiosk
    # fast. Add-on only — runs AFTER the unchanged hard/silence/nudge checks, and
    # the watchdog is still the sole timeout authority (a condition, not a timer).
    if (ctx.presence_status is PresenceStatus.ABSENT
            and ctx.now - ctx.presence_since >= ctx.cfg.owner_absent_grace_s
            and ctx.now - ctx.last_speech_at >= ctx.cfg.active_talk_guard_s):
        return State.IDLE, [C.EndSession("owner_absent")]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/test_reducer_presence.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Run the full director reducer suite (no regression)**

Run: `python3 -m pytest tests/director/ -q`
Expected: PASS (all existing director tests still green).

- [ ] **Step 7: Commit**

```bash
git add modes/director/reducer.py tests/director/test_reducer_presence.py
git commit -m "feat(director-07): reducer records presence + owner-absent end-condition in _on_tick

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Config wiring (`vision.*` → DirectorConfig + config.yaml)

**Files:**
- Modify: `modes/director/assembly.py:89-101` (`_director_config_from`)
- Modify: `config.yaml` (add `kiosk.talkback.vision`)
- Test: `tests/director/test_config_vision_mapping.py`

**Interfaces:**
- Consumes: `DirectorConfig.owner_absent_grace_s/active_talk_guard_s` (Task 1).
- Produces: `_director_config_from(tb_cfg)` maps `tb_cfg["vision"]["owner_absent_grace_s"]` / `["active_talk_guard_s"]` onto `DirectorConfig`, defaulting to 3.0 when absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_config_vision_mapping.py
from modes.director.assembly import _director_config_from


def test_vision_floor_control_mapped():
    cfg = _director_config_from({"vision": {"owner_absent_grace_s": 4.0,
                                            "active_talk_guard_s": 2.0}})
    assert cfg.owner_absent_grace_s == 4.0
    assert cfg.active_talk_guard_s == 2.0


def test_vision_floor_control_defaults_when_absent():
    cfg = _director_config_from({})
    assert cfg.owner_absent_grace_s == 3.0
    assert cfg.active_talk_guard_s == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_config_vision_mapping.py -v`
Expected: FAIL (mapped values fall back to defaults; first test fails on 4.0 != 3.0).

- [ ] **Step 3: Map the keys**

In `modes/director/assembly.py`, in `_director_config_from`, add a `vision` lookup and two fields to the returned `DirectorConfig(...)`:

```python
    barge = tb_cfg.get("barge_in", {})
    vision = tb_cfg.get("vision", {})
    return DirectorConfig(
        silence_timeout_s=tb_cfg.get("silence_timeout_s", 30.0),
        hard_timeout_s=tb_cfg.get("hard_timeout_s", 300.0),
        endpoint_threshold=tb_cfg.get("turn_gate", {}).get("endpoint_threshold", 0.5),
        min_speech_ms=barge.get("min_speech_ms", 120.0),
        verify_window_ms=barge.get("verify_window_ms", 700.0),
        speaker_threshold=barge.get("speaker_threshold", 0.20),
        duck_level=barge.get("duck_level", 0.35),
        owner_absent_grace_s=vision.get("owner_absent_grace_s", 3.0),
        active_talk_guard_s=vision.get("active_talk_guard_s", 3.0),
    )
```

- [ ] **Step 4: Add the config.yaml block**

In `config.yaml`, under `kiosk.talkback:` (e.g. after the `crowd_focus:` block), add:

```yaml
    # Camera floor control (Director-07). Presence is the floor authority: keep
    # serving the owner, free the kiosk fast when they physically leave, treat a
    # stranger as the owner leaving. ADD-ON: the silence_timeout above is unchanged.
    # CPU-only YuNet+SFace (no insightface); see docs/notes/2026-06-23-vision-presence.md.
    vision:
      enabled: true
      camera_index: 0
      width: 640                # spike-validated mode; do NOT set CAP_PROP_FPS
      height: 360
      fps: 3                    # low-rate dedicated capture (detection ~2% of a core)
      identity_threshold: 0.40  # spike Tier-2 GO: self >=0.79 vs stranger <=0.06
      min_area_frac: 0.015      # zone/size gate at 640x360
      present_after_s: 1.0      # debounce hysteresis -> PRESENT
      absent_after_s: 2.0       # debounce hysteresis -> ABSENT
      owner_absent_grace_s: 3.0 # sustained ABSENT this long => free the kiosk
      active_talk_guard_s: 3.0  # never owner-absent-end within this of owner speech
      enroll_frames: 8          # owner face-reference frames captured at session start
      # Reserved: the deferred audio speaker-check seam (spec §9). NOT wired in this
      # plan — revisit after live data (Q3/Q5 deferred). No consumer reads this yet.
      audio_safety_net:
        enabled: false
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/test_config_vision_mapping.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add modes/director/assembly.py config.yaml tests/director/test_config_vision_mapping.py
git commit -m "feat(director-07): map vision.* config + add kiosk.talkback.vision block

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Pure presence classification + debouncer

**Files:**
- Create: `modes/director/vision/__init__.py` (empty)
- Create: `modes/director/vision/classify.py`
- Test: `tests/director/vision/test_classify.py`

**Interfaces:**
- Consumes: `PresenceStatus` (Task 1).
- Produces:
  - `classify_presence(face_embedding, box, frame_w, frame_h, reference, *, identity_threshold, min_area_frac, zone=(0.2,0.0,0.6,1.0)) -> PresenceStatus` — returns `PRESENT` only when the largest central face is big enough AND `cosine(face_embedding, reference) >= identity_threshold`; else `ABSENT`. `face_embedding=None`/`box=None` (no face) => `ABSENT`.
  - `cosine(a, b) -> float`.
  - `PresenceDebouncer(present_after_s, absent_after_s)` with `update(detected: bool, now: float) -> str` returning `"present"`/`"absent"` (ported from the spike).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/vision/test_classify.py
import numpy as np
from modes.director.events import PresenceStatus
from modes.director.vision.classify import classify_presence, cosine, PresenceDebouncer

REF = np.array([1.0, 0.0, 0.0], dtype=np.float32)
BIG_CENTER = (256, 96, 128, 168)   # ~0.093 area-frac, centered in 640x360


def test_owner_present():
    s = classify_presence(np.array([0.9, 0.1, 0.0]), BIG_CENTER, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.PRESENT


def test_stranger_absent():
    s = classify_presence(np.array([0.0, 1.0, 0.0]), BIG_CENTER, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_no_face_absent():
    s = classify_presence(None, None, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_owner_too_small_absent():
    tiny = (310, 170, 20, 26)      # ~0.0023 area-frac < 0.015
    s = classify_presence(np.array([1.0, 0.0, 0.0]), tiny, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_owner_offcenter_absent():
    edge = (0, 0, 128, 168)        # center x-frac ~0.1 < zone start 0.2
    s = classify_presence(np.array([1.0, 0.0, 0.0]), edge, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_cosine_basic():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_debouncer_hysteresis():
    deb = PresenceDebouncer(present_after_s=1.0, absent_after_s=2.0)
    assert deb.update(True, 0.0) == "absent"     # starts absent
    assert deb.update(True, 0.5) == "absent"     # < present_after
    assert deb.update(True, 1.1) == "present"    # >= present_after
    assert deb.update(False, 2.0) == "present"   # < absent_after
    assert deb.update(False, 3.2) == "absent"    # >= absent_after
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/vision/test_classify.py -v`
Expected: FAIL (`ModuleNotFoundError: modes.director.vision.classify`).

- [ ] **Step 3: Create the package + module**

Create empty `modes/director/vision/__init__.py`.

Create `modes/director/vision/classify.py`:

```python
"""Pure presence classification + debounce (no cv2, no I/O). The live cv2 adapter
(opencv_backend.py) supplies the face embedding + box; this decides PRESENT/ABSENT.
Logic ported from the validated spike bench/vision_presence_probe.py."""
import numpy as np

from modes.director.events import PresenceStatus


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _box_in_zone(box, frame_w, frame_h, zone, min_area_frac) -> bool:
    x, y, w, h = box
    cx, cy = (x + w / 2) / frame_w, (y + h / 2) / frame_h
    zx, zy, zw, zh = zone
    in_zone = (zx <= cx <= zx + zw) and (zy <= cy <= zy + zh)
    big_enough = (w * h) / (frame_w * frame_h) >= min_area_frac
    return bool(in_zone and big_enough)


def classify_presence(face_embedding, box, frame_w, frame_h, reference, *,
                      identity_threshold, min_area_frac,
                      zone=(0.2, 0.0, 0.6, 1.0)) -> PresenceStatus:
    """PRESENT only when the central, large-enough face matches the owner reference
    (cosine >= identity_threshold). No face, off-center, too small, or a stranger
    (low cosine) all read ABSENT. UNAVAILABLE is the caller's job (no reference /
    detector error), NOT here."""
    if face_embedding is None or box is None:
        return PresenceStatus.ABSENT
    if not _box_in_zone(box, frame_w, frame_h, zone, min_area_frac):
        return PresenceStatus.ABSENT
    if cosine(face_embedding, reference) >= identity_threshold:
        return PresenceStatus.PRESENT
    return PresenceStatus.ABSENT


class PresenceDebouncer:
    """Hysteresis over raw per-frame detections. 'present' after present_after_s of
    continuous detection, 'absent' after absent_after_s of continuous non-detection.
    Starts 'absent'. (Ported verbatim from the spike.)"""

    def __init__(self, present_after_s=1.0, absent_after_s=2.0):
        self._pa = present_after_s
        self._aa = absent_after_s
        self._state = "absent"
        self._since = None

    def update(self, detected: bool, now: float) -> str:
        if self._since is None or self._since[0] != detected:
            self._since = (detected, now)
        run = now - self._since[1]
        if self._state == "absent" and detected and run >= self._pa:
            self._state = "present"
        elif self._state == "present" and not detected and run >= self._aa:
            self._state = "absent"
        return self._state
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/vision/test_classify.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add modes/director/vision/__init__.py modes/director/vision/classify.py tests/director/vision/test_classify.py
git commit -m "feat(director-07): pure presence classification + debouncer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: PresenceMonitor — classify + debounce + emit-on-change

**Files:**
- Create: `modes/director/vision/monitor.py`
- Test: `tests/director/vision/test_monitor.py`

**Interfaces:**
- Consumes: `PresenceStatus` (Task 1); `classify_presence`, `PresenceDebouncer` (Task 4).
- Produces: `PresenceMonitor(classify_fn, debouncer)` where `classify_fn(frame) -> PresenceStatus` (PRESENT/ABSENT) or raises; `observe(frame, now) -> Optional[PresenceStatus]` returns a status **only when the emitted status changes** (else `None`). `frame is None` or `classify_fn` raising → `UNAVAILABLE`. Recovery from `UNAVAILABLE` resets the debouncer so present/absent re-accrue cleanly.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/vision/test_monitor.py
from modes.director.events import PresenceStatus as PS
from modes.director.vision.classify import PresenceDebouncer
from modes.director.vision.monitor import PresenceMonitor


def _monitor(seq):
    """classify_fn pops from seq; an Exception instance is raised."""
    box = list(seq)

    def classify_fn(frame):
        v = box.pop(0)
        if isinstance(v, Exception):
            raise v
        return v
    return PresenceMonitor(classify_fn, PresenceDebouncer(present_after_s=1.0,
                                                          absent_after_s=2.0))


def test_emits_present_once_on_debounced_change():
    m = _monitor([PS.PRESENT, PS.PRESENT, PS.PRESENT])
    assert m.observe("f", 0.0) is None        # absent->absent (no change from init)
    assert m.observe("f", 0.6) is None        # still debouncing
    assert m.observe("f", 1.2) is PS.PRESENT  # debounced present: emit once
    # no re-emit on subsequent presents (next observe would need another frame)


def test_classify_error_is_unavailable():
    m = _monitor([RuntimeError("yunet boom")])
    assert m.observe("f", 0.0) is PS.UNAVAILABLE


def test_none_frame_is_unavailable():
    m = PresenceMonitor(lambda f: PS.PRESENT, PresenceDebouncer())
    assert m.observe(None, 0.0) is PS.UNAVAILABLE


def test_recovery_from_unavailable_re_debounces():
    m = _monitor([RuntimeError("x"), PS.PRESENT, PS.PRESENT])
    assert m.observe("f", 0.0) is PS.UNAVAILABLE
    assert m.observe("f", 0.1) is None        # present_after not yet met after reset
    assert m.observe("f", 1.2) is PS.PRESENT  # re-accrued from the recovery instant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/vision/test_monitor.py -v`
Expected: FAIL (`ModuleNotFoundError: modes.director.vision.monitor`).

- [ ] **Step 3: Implement the monitor**

Create `modes/director/vision/monitor.py`:

```python
"""PresenceMonitor — the pure per-frame core of the VisionWorker. Wraps a
classify_fn (frame -> PRESENT/ABSENT, may raise) and a PresenceDebouncer, and
returns a status ONLY when the emitted status changes (edge), so the worker emits
one OwnerPresenceEvent per change. A None frame or a classify error is UNAVAILABLE
(fail-safe: the reducer then leans on the audio silence timeout)."""
from typing import Optional

from modes.director.events import PresenceStatus


class PresenceMonitor:
    def __init__(self, classify_fn, debouncer):
        self._classify = classify_fn
        self._deb = debouncer
        self._emitted: Optional[PresenceStatus] = None  # last status we returned as a change
        self._unavailable = False

    def observe(self, frame, now: float) -> Optional[PresenceStatus]:
        if frame is None:
            return self._go_unavailable()
        try:
            raw = self._classify(frame)           # PRESENT or ABSENT
        except Exception:                         # noqa: BLE001 — detector glitch
            return self._go_unavailable()
        if self._unavailable:
            # Recover: reset the debouncer so present/absent re-accrue from now.
            self._unavailable = False
            self._deb.__init__(self._deb._pa, self._deb._aa)
        detected = raw is PresenceStatus.PRESENT
        stable = (PresenceStatus.PRESENT if self._deb.update(detected, now) == "present"
                  else PresenceStatus.ABSENT)
        if stable is not self._emitted:
            self._emitted = stable
            return stable
        return None

    def _go_unavailable(self) -> Optional[PresenceStatus]:
        self._unavailable = True
        if self._emitted is not PresenceStatus.UNAVAILABLE:
            self._emitted = PresenceStatus.UNAVAILABLE
            return PresenceStatus.UNAVAILABLE
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/vision/test_monitor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add modes/director/vision/monitor.py tests/director/vision/test_monitor.py
git commit -m "feat(director-07): PresenceMonitor — debounced emit-on-change + UNAVAILABLE

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Owner face enrollment helper

**Files:**
- Create: `modes/director/vision/enroll.py`
- Test: `tests/director/vision/test_enroll.py`

**Interfaces:**
- Produces: `enroll_reference(grab_fn, embed_fn, *, n_frames, max_attempts) -> Optional[np.ndarray]` — calls `grab_fn() -> frame_or_None` and `embed_fn(frame) -> embedding_or_None`, collects up to `n_frames` non-None embeddings within `max_attempts` grabs, returns their L2-normalized mean, or `None` if it never got a face (caller then stays UNAVAILABLE — NEVER ABSENT, so a failed enroll can't falsely free the kiosk).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/vision/test_enroll.py
import numpy as np
from modes.director.vision.enroll import enroll_reference


def test_enroll_means_embeddings():
    frames = iter(["a", "b", "c"])
    embs = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    ref = enroll_reference(lambda: next(frames, None),
                           lambda f: embs[f], n_frames=3, max_attempts=10)
    assert ref is not None
    assert np.allclose(ref, [1.0, 0.0])


def test_enroll_skips_no_face_frames():
    seq = iter(["x", None, "x"])
    ref = enroll_reference(lambda: next(seq, None),
                           lambda f: np.array([0.0, 1.0]) if f else None,
                           n_frames=2, max_attempts=10)
    assert ref is not None
    assert np.allclose(ref / np.linalg.norm(ref), [0.0, 1.0])


def test_enroll_returns_none_when_no_face_ever():
    ref = enroll_reference(lambda: "frame", lambda f: None,
                           n_frames=3, max_attempts=5)
    assert ref is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/vision/test_enroll.py -v`
Expected: FAIL (`ModuleNotFoundError: modes.director.vision.enroll`).

- [ ] **Step 3: Implement the helper**

Create `modes/director/vision/enroll.py`:

```python
"""Owner face enrollment — capture a stable reference embedding at session start.
A failed enroll returns None; the VisionWorker then reports UNAVAILABLE (never
ABSENT), so the camera can't falsely free the kiosk when it simply can't see a face."""
from typing import Optional

import numpy as np


def enroll_reference(grab_fn, embed_fn, *, n_frames, max_attempts) -> Optional[np.ndarray]:
    embs = []
    attempts = 0
    while len(embs) < n_frames and attempts < max_attempts:
        attempts += 1
        frame = grab_fn()
        if frame is None:
            continue
        e = embed_fn(frame)
        if e is not None:
            embs.append(np.asarray(e, dtype=np.float64).ravel())
    if not embs:
        return None
    mean = np.mean(np.stack(embs), axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm else mean
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/vision/test_enroll.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add modes/director/vision/enroll.py tests/director/vision/test_enroll.py
git commit -m "feat(director-07): owner face enrollment helper (None on failure)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: OpenCV backend (live detect + embed adapter)

**Files:**
- Create: `modes/director/vision/opencv_backend.py`
- Test: `tests/director/vision/test_opencv_backend.py`

**Interfaces:**
- Consumes: `classify_presence` (Task 4); the OpenCV-Zoo model URLs + `ensure_model` pattern from `bench/vision_presence_probe.py`.
- Produces: `OpenCvBackend(camera_index, width, height, identity_threshold, min_area_frac)` with:
  - `open() -> bool` (open camera + load YuNet/SFace; False on any failure),
  - `grab() -> Optional[frame]` (one decoded frame or None),
  - `embed(frame) -> Optional[np.ndarray]` (largest face SFace embedding or None),
  - `make_classify_fn(reference) -> (frame -> PresenceStatus)` binding `classify_presence` to the live detect+embed of the largest central face,
  - `close()`.
  - Module-level `cv2_available() -> bool`.

  **CI note:** cv2/camera are not in CI. This task's test only asserts the import is cv2-free and `cv2_available()` degrades gracefully; the detect/embed paths are validated live via the spike harness numbers (`docs/notes/2026-06-23-vision-presence.md`). All cv2 use is lazy inside methods.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/vision/test_opencv_backend.py
import importlib


def test_module_imports_without_cv2():
    # Importing the backend must NOT import cv2 at module load (lazy inside methods).
    mod = importlib.import_module("modes.director.vision.opencv_backend")
    assert hasattr(mod, "OpenCvBackend")
    assert hasattr(mod, "cv2_available")


def test_cv2_available_is_bool():
    from modes.director.vision.opencv_backend import cv2_available
    assert isinstance(cv2_available(), bool)


def test_open_returns_false_on_bad_index(monkeypatch):
    # With no usable camera, open() must return False (never raise).
    from modes.director.vision.opencv_backend import OpenCvBackend, cv2_available
    if not cv2_available():
        import pytest
        pytest.skip("cv2 not installed in this environment")
    b = OpenCvBackend(camera_index=999, width=640, height=360,
                      identity_threshold=0.40, min_area_frac=0.015)
    assert b.open() is False
    b.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/vision/test_opencv_backend.py -v`
Expected: FAIL (`ModuleNotFoundError: modes.director.vision.opencv_backend`).

- [ ] **Step 3: Implement the backend**

Create `modes/director/vision/opencv_backend.py`:

```python
"""Live OpenCV adapter for camera presence/identity (YuNet detect + SFace embed).
ALL cv2 use is lazy inside methods so importing this module never needs cv2 — a
missing/broken cv2 degrades to VisionWorker=None upstream. Models + logic are the
validated spike (bench/vision_presence_probe.py)."""
import pathlib
import sys
import urllib.request
from typing import Optional

import numpy as np

from modes.director.events import PresenceStatus
from modes.director.vision.classify import classify_presence

CACHE = pathlib.Path.home() / ".cache" / "target-vad" / "vision"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")


def cv2_available() -> bool:
    try:
        import cv2  # noqa: F401
        return hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
    except Exception:
        return False


def _ensure_model(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


class OpenCvBackend:
    def __init__(self, camera_index, width, height, identity_threshold, min_area_frac):
        self._index = camera_index
        self._w = width
        self._h = height
        self._thr = identity_threshold
        self._min_area = min_area_frac
        self._cap = None
        self._det = None
        self._rec = None

    def open(self) -> bool:
        try:
            import cv2
            self._det = cv2.FaceDetectorYN.create(
                str(_ensure_model(YUNET_URL, CACHE / "yunet.onnx")), "",
                (self._w, self._h), 0.7, 0.3, 50)
            self._rec = cv2.FaceRecognizerSF.create(
                str(_ensure_model(SFACE_URL, CACHE / "sface.onnx")), "")
            cap = cv2.VideoCapture(self._index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
            # NB: do NOT set CAP_PROP_FPS — it switches this UVC camera's mode.
            if not cap.isOpened():
                cap.release()
                return False
            ok, _ = cap.read()
            if not ok:
                cap.release()
                return False
            self._cap = cap
            return True
        except Exception as exc:   # noqa: BLE001
            print(f"[vision] OpenCvBackend.open failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return False

    def grab(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def _largest_face(self, frame):
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: float(f[2]) * float(f[3]))

    def embed(self, frame) -> Optional[np.ndarray]:
        f = self._largest_face(frame)
        if f is None:
            return None
        aligned = self._rec.alignCrop(frame, f)
        return self._rec.feature(aligned).ravel().copy()

    def make_classify_fn(self, reference):
        """Bind classify_presence to live detect+embed of the largest central face."""
        def classify_fn(frame) -> PresenceStatus:
            h, w = frame.shape[:2]
            f = self._largest_face(frame)
            if f is None:
                return classify_presence(None, None, w, h, reference,
                                         identity_threshold=self._thr,
                                         min_area_frac=self._min_area)
            box = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
            aligned = self._rec.alignCrop(frame, f)
            emb = self._rec.feature(aligned).ravel()
            return classify_presence(emb, box, w, h, reference,
                                     identity_threshold=self._thr,
                                     min_area_frac=self._min_area)
        return classify_fn

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/vision/test_opencv_backend.py -v`
Expected: PASS (3 passed; the third may SKIP if cv2 is absent).

- [ ] **Step 5: Commit**

```bash
git add modes/director/vision/opencv_backend.py tests/director/vision/test_opencv_backend.py
git commit -m "feat(director-07): live OpenCV backend (YuNet detect + SFace embed), lazy cv2

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: VisionWorker (thread: self-enroll → monitor → emit)

**Files:**
- Create: `modes/director/workers/vision.py`
- Test: `tests/director/test_vision_worker.py`

**Interfaces:**
- Consumes: `OwnerPresenceEvent`, `PresenceStatus` (Task 1); `PresenceMonitor` (Task 5); `enroll_reference` (Task 6); `PresenceDebouncer`/`classify` (Task 4); `OpenCvBackend` (Task 7); the `EventBus` (`bus.emit` is async).
- Produces: `VisionWorker(backend, bus, *, fps, present_after_s, absent_after_s, enroll_frames)` with `start(loop)` (spawn the capture thread, capturing the running asyncio loop for cross-thread emit), `stop()` (signal + join), and a synchronous, thread-free `_run_once(now) -> Optional[OwnerPresenceEvent]` that does one enroll-or-monitor step (the unit-tested core). Emits via `asyncio.run_coroutine_threadsafe(bus.emit(ev), loop)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_vision_worker.py
from modes.director.events import OwnerPresenceEvent, PresenceStatus as PS
from modes.director.workers.vision import VisionWorker


class FakeBackend:
    """grab/embed/make_classify_fn driven by scripted sequences."""
    def __init__(self, grabs, embeds, classifies):
        self._grabs = list(grabs)
        self._embeds = list(embeds)
        self._classifies = list(classifies)
        self.opened = False
        self.closed = False

    def open(self): self.opened = True; return True
    def grab(self): return self._grabs.pop(0) if self._grabs else None
    def embed(self, frame): return self._embeds.pop(0) if self._embeds else None

    def make_classify_fn(self, reference):
        seq = self._classifies

        def fn(frame):
            return seq.pop(0)
        return fn

    def close(self): self.closed = True


def _worker(backend):
    return VisionWorker(backend, bus=None, fps=10.0, present_after_s=1.0,
                        absent_after_s=2.0, enroll_frames=2)


def test_enrolls_then_monitors_and_emits_present():
    import numpy as np
    be = FakeBackend(grabs=["f", "f", "f", "f"],
                     embeds=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
                     classifies=[PS.PRESENT, PS.PRESENT, PS.PRESENT])
    w = _worker(be)
    # First _run_once enrolls (consumes enroll grabs+embeds), returns no event yet.
    assert w._run_once(0.0) is None
    assert w._run_once(0.1) is None     # monitor: still debouncing
    ev = w._run_once(1.2)               # debounced present
    assert isinstance(ev, OwnerPresenceEvent) and ev.status is PS.PRESENT


def test_failed_enroll_reports_unavailable_not_absent():
    be = FakeBackend(grabs=["f", "f", "f", "f", "f"], embeds=[None, None, None, None, None],
                     classifies=[])
    w = _worker(be)
    ev = w._run_once(0.0)
    assert isinstance(ev, OwnerPresenceEvent) and ev.status is PS.UNAVAILABLE
    # stays unavailable; never emits ABSENT off a failed enroll
    assert w._run_once(0.1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_vision_worker.py -v`
Expected: FAIL (`ModuleNotFoundError: modes.director.workers.vision`).

- [ ] **Step 3: Implement the worker**

Create `modes/director/workers/vision.py`:

```python
"""VisionWorker — the only camera-touching component. A background thread that
(1) opens the backend, (2) self-enrolls the owner's face at session start, then
(3) monitors presence at ~fps and emits OwnerPresenceEvent on debounced changes.
Cross-thread emit uses run_coroutine_threadsafe onto the runtime's loop. Any
failure degrades to UNAVAILABLE (fail-safe); it never raises into the session.

Camera ownership lives entirely here — the WakeGate stays camera-free."""
import asyncio
import sys
import threading
import time
from typing import Optional

from modes.director.events import OwnerPresenceEvent, PresenceStatus
from modes.director.vision.classify import PresenceDebouncer
from modes.director.vision.enroll import enroll_reference
from modes.director.vision.monitor import PresenceMonitor


class VisionWorker:
    def __init__(self, backend, bus, *, fps, present_after_s, absent_after_s,
                 enroll_frames, clock=time.monotonic):
        self._backend = backend
        self._bus = bus
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._present_after_s = present_after_s
        self._absent_after_s = absent_after_s
        self._enroll_frames = enroll_frames
        self._clock = clock
        self._monitor: Optional[PresenceMonitor] = None   # None until enrolled
        self._enrolled = False
        self._unavailable_emitted = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop = None

    def _run_once(self, now: float) -> Optional[OwnerPresenceEvent]:
        """One synchronous step (testable). Enroll on first call; then monitor."""
        if not self._enrolled:
            ref = enroll_reference(self._backend.grab, self._backend.embed,
                                   n_frames=self._enroll_frames,
                                   max_attempts=self._enroll_frames * 5)
            self._enrolled = True
            if ref is None:
                # Can't see the owner -> UNAVAILABLE (never ABSENT off a bad enroll).
                if not self._unavailable_emitted:
                    self._unavailable_emitted = True
                    return OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, now)
                return None
            self._monitor = PresenceMonitor(
                self._backend.make_classify_fn(ref),
                PresenceDebouncer(self._present_after_s, self._absent_after_s))
            return None
        if self._monitor is None:
            return None                          # enroll failed; stay UNAVAILABLE
        frame = self._backend.grab()
        status = self._monitor.observe(frame, now)
        return OwnerPresenceEvent(status, now) if status is not None else None

    def _emit(self, ev: OwnerPresenceEvent) -> None:
        if self._bus is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._bus.emit(ev), self._loop)
        except Exception:                        # noqa: BLE001 — loop closing
            pass

    def _loop_body(self) -> None:
        try:
            if not self._backend.open():
                self._emit(OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, self._clock()))
                return
            while self._running:
                t0 = self._clock()
                ev = self._run_once(t0)
                if ev is not None:
                    self._emit(ev)
                dt = self._period - (self._clock() - t0)
                if dt > 0:
                    time.sleep(dt)
        except Exception as exc:                 # noqa: BLE001 — never crash the session
            print(f"[vision] worker thread died: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            self._emit(OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, self._clock()))
        finally:
            self._backend.close()

    def start(self, loop) -> None:
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._loop_body, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/test_vision_worker.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add modes/director/workers/vision.py tests/director/test_vision_worker.py
git commit -m "feat(director-07): VisionWorker thread — self-enroll, monitor, emit-on-change

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Assembly `_build_vision` + runtime start/stop (no-regression)

**Files:**
- Modify: `modes/director/assembly.py` (add `_build_vision`; pass into `DirectorRuntime`)
- Modify: `modes/director/runtime.py:37-51` (`__init__`), `:63-83` (`run_async`), `:100-120` (`_teardown`)
- Test: `tests/director/test_build_vision.py`, `tests/director/test_runtime_vision_lifecycle.py`

**Interfaces:**
- Consumes: `VisionWorker` (Task 8); `OpenCvBackend`, `cv2_available` (Task 7); `tb_cfg["vision"]` config.
- Produces: `_build_vision(tb_cfg) -> Optional[VisionWorker]` (None when `enabled` is false OR cv2 unavailable); `DirectorRuntime.__init__(..., vision=None)`; `run_async` calls `vision.start(get_running_loop())`; `_teardown` calls `vision.stop()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/director/test_build_vision.py
from modes.director.assembly import _build_vision


def test_disabled_returns_none():
    assert _build_vision({"vision": {"enabled": False}}, bus=object()) is None


def test_missing_vision_block_returns_none():
    assert _build_vision({}, bus=object()) is None


def test_enabled_but_no_cv2_returns_none(monkeypatch):
    import modes.director.assembly as A
    monkeypatch.setattr(A, "_cv2_available", lambda: False)
    assert _build_vision({"vision": {"enabled": True}}, bus=object()) is None
```

```python
# tests/director/test_runtime_vision_lifecycle.py
import asyncio
from modes.director.runtime import DirectorRuntime


class _Spy:
    def __init__(self): self.started = None; self.stopped = False
    def start(self, loop): self.started = loop
    def stop(self): self.stopped = True


class _NoopIngestion:
    async def run(self):
        await asyncio.sleep(3600)
    def stop(self): pass


class _NoopPlayback:
    async def drain(self): pass
    def close(self): pass


class _NoopGen:
    async def aclose(self): pass


class _ImmediateWatchdog:
    def start(self): pass
    def request_stop(self, reason): pass
    async def stop(self): pass


class _Director:
    class _Ctx:
        class conversation: turn_count = 0
        conversation = conversation()
    ctx = _Ctx()
    def dispatch(self, event): return []


def test_runtime_starts_and_stops_vision():
    from modes.director import commands as C
    vision = _Spy()
    bus_events = [C.EndSession("test")]   # one event then end

    class _Bus:
        async def get(self):
            return bus_events.pop(0)

    rt = DirectorRuntime(director=_Director(), bus=_Bus(),
                         watchdog=_ImmediateWatchdog(), ingestion=_NoopIngestion(),
                         stt_worker=None, generation=_NoopGen(),
                         playback=_NoopPlayback(), clock=lambda: 0.0, vision=vision)
    # EndSession routes to result_reason; patch _route minimally:
    async def _drive():
        await rt.run_async()
    # Make dispatch return the EndSession so the loop exits:
    rt._director.dispatch = lambda e: [e]
    asyncio.new_event_loop().run_until_complete(_drive())
    assert vision.started is not None      # started with the running loop
    assert vision.stopped is True          # stopped in teardown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_build_vision.py tests/director/test_runtime_vision_lifecycle.py -v`
Expected: FAIL (`_build_vision` missing; `DirectorRuntime.__init__` has no `vision`).

- [ ] **Step 3: Add `_build_vision` to assembly**

In `modes/director/assembly.py`, add this import near the top (module level is fine — it's cv2-free):

```python
from modes.director.vision.opencv_backend import OpenCvBackend, cv2_available as _cv2_available
from modes.director.workers.vision import VisionWorker
```

Add the builder (next to `_build_pvad`):

```python
def _build_vision(tb_cfg: dict, *, bus):
    """Build the Director-07 camera floor-control VisionWorker, or None for the
    no-vision path (today's Director). None when vision is disabled or cv2 is
    unavailable — the no-regression guarantee. The worker self-enrolls the owner
    at session start and reports UNAVAILABLE if it can't (never falsely ABSENT)."""
    v = tb_cfg.get("vision", {})
    if not v.get("enabled", False):
        return None
    if not _cv2_available():
        print("[director] vision enabled but cv2/FaceDetectorYN unavailable -> "
              "floor control via audio timeout only", file=sys.stderr, flush=True)
        return None
    backend = OpenCvBackend(
        camera_index=v.get("camera_index", 0), width=v.get("width", 640),
        height=v.get("height", 360), identity_threshold=v.get("identity_threshold", 0.40),
        min_area_frac=v.get("min_area_frac", 0.015))
    return VisionWorker(
        backend, bus, fps=v.get("fps", 3.0),
        present_after_s=v.get("present_after_s", 1.0),
        absent_after_s=v.get("absent_after_s", 2.0),
        enroll_frames=v.get("enroll_frames", 8))
```

In `build_director_runtime`, after the `ingestion = IngestionWorker(...)` block, build vision and pass it to the runtime:

```python
    vision = _build_vision(tb_cfg, bus=bus)
    ...
    runtime = DirectorRuntime(
        director=director, bus=bus, watchdog=watchdog, ingestion=ingestion,
        stt_worker=stt_worker, generation=generation, playback=playback,
        clock=clock, vision=vision,
    )
```

- [ ] **Step 4: Wire the runtime lifecycle**

In `modes/director/runtime.py`, add `vision=None` to `__init__` and store it:

```python
    def __init__(self, director, bus, watchdog, ingestion, stt_worker,
                 generation, playback, clock, vision=None):
        ...
        self._playback = playback
        self._vision = vision
        self._clock = clock
```

In `run_async`, after `self._watchdog.start()`, start vision with the running loop:

```python
        self._watchdog.start()
        if self._vision is not None:
            self._vision.start(asyncio.get_running_loop())
```

In `_teardown`, stop vision first (before cancelling ingestion):

```python
    async def _teardown(self, ingestion_task) -> None:
        if self._vision is not None:
            self._vision.stop()
        self._ingestion.stop()
        ...
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/director/test_build_vision.py tests/director/test_runtime_vision_lifecycle.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite (no-regression gate)**

Run: `python3 -m pytest -q`
Expected: PASS (all prior tests green; vision off by injection/CI → runtime unchanged).

- [ ] **Step 7: Commit**

```bash
git add modes/director/assembly.py modes/director/runtime.py tests/director/test_build_vision.py tests/director/test_runtime_vision_lifecycle.py
git commit -m "feat(director-07): wire VisionWorker into assembly + runtime lifecycle (None=no-regression)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Live validation + verdict note

**Files:**
- Create: `docs/notes/2026-06-24-director-07-live.md`
- (No code; this task is the live gate.)

**Interfaces:** none (manual run on the GB10 with the camera).

- [ ] **Step 1: Run the kiosk with vision enabled and observe owner-absent**

Run: `TVAD_DIAG=1 ./kiosk-stack.sh start`
Stand at the kiosk, wake it, speak a turn, then **step out of frame** and stay out. In the DIAG log, confirm: `event=OwnerPresenceEvent -> ... ` lines appear, and within ~`owner_absent_grace_s` of leaving you see `EndSession("owner_absent")` (NOT waiting the full 30 s silence timeout).

- [ ] **Step 2: Confirm no-regression with vision disabled**

Set `kiosk.talkback.vision.enabled: false` in `config.yaml`, run again, and confirm the session behaves exactly as before (silence timeout at 30 s, nudge at 25 s; no presence events).

- [ ] **Step 3: Confirm fail-safe (camera unplugged)**

With `vision.enabled: true` but the camera unplugged, confirm startup logs the `cv2/...unavailable` or an `UNAVAILABLE` event and the session falls back to the 30 s audio timeout (never a spurious `owner_absent`).

- [ ] **Step 4: Write the verdict note**

Record the three observations (owner-absent latency, no-regression, fail-safe) with the measured owner-absent end latency in `docs/notes/2026-06-24-director-07-live.md`. Note any threshold tuning needed (e.g. `owner_absent_grace_s` feel, `identity_threshold` at the real mount).

- [ ] **Step 5: Commit**

```bash
git add docs/notes/2026-06-24-director-07-live.md config.yaml
git commit -m "docs(director-07): live validation verdict — owner-absent, no-regression, fail-safe

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Refinement vs spec §7 (intentional)

The spec described capturing the owner's face embedding **at wake** and threading it
through `DirectorHandoff.primary_face_embedding`. This plan instead has the
**`VisionWorker` self-enroll at session start** (Task 8): it owns the camera entirely,
so its first action is to enroll the owner (who is standing there, having just woken
the kiosk), then it monitors. This keeps the WakeGate **camera-free** — it has strict
single-ownership grep tests (`tests/director/test_wakegate_single_ownership.py`) and no
camera dependency — and avoids modifying `handoff.py`/`wakegate.py`. Same instant
(session start immediately follows wake), smaller blast radius, identical behavior. A
failed self-enroll reports `UNAVAILABLE` (never `ABSENT`), so it can't falsely free the
kiosk.

## Deferred (NOT in this plan)

- **Audio speaker-check seam (SafetyNet + Lockout).** Spec §9 / decisions Q3+Q5: the
  bystander-beside-owner content leak is explicitly accepted for V1. The
  `vision.audio_safety_net.enabled` config key is added (Task 3) as a reserved,
  consumer-less flag. Wiring `SafetyNet.accumulate`/`maybe_verify` into `IngestionWorker`
  with `Lockout` (WARN→EJECT→IDLE) as the action arm is a separate increment, taken up
  only after live data shows the leak is a real problem in the space.
- **Master architecture doc** was already revised alongside the spec (commit on this
  branch); no doc work remains in this plan.
