# Bystander Gate v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject non-owner NEW turns in the Director using proximity (RMS) + camera presence, reject-by-default, hardware-independent.

**Architecture:** A single pure-reducer change in `_on_user_segment`, gated by a new `reject_bystanders` flag (default off = byte-for-byte today). A `classify_new_turn` verdict helper is the single source of truth, reused by the reducer (decide) and the runtime (DIAG observability). Reuses the already-calibrated `ctx.proximity_rms` and the Director-07 `ctx.presence_status`. No new worker, event, or hardware.

**Tech Stack:** Python 3.12, pytest, the existing Director FSM (`modes/director/`).

**Spec:** `docs/superpowers/specs/2026-06-25-bystander-gate-v1-design.md`

## Global Constraints

- Every git commit MUST end with the trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- The reducer is PURE: no I/O, no await, no clock reads. `now` only via Tick/event fields. Only `reduce()` mutates `State`/`Context`.
- No-regression: with `reject_bystanders` false, the new-turn path must behave byte-for-byte as today (legacy branch).
- Fail-safe: the presence gate fires ONLY on `PresenceStatus.ABSENT`. `PRESENT` and `UNAVAILABLE` both ALLOW (never block the owner on camera uncertainty).
- Config flag is read as a STRICT bool: only literal boolean `True` enables (mirrors the Director-07 `enabled: flase` fail-open fix).
- Run the full suite with `python3 -m pytest -q` (the binary is `python3`, not `python`).

---

### Task 1: Config flag `reject_bystanders`

**Files:**
- Modify: `modes/director/config.py` (add field)
- Modify: `modes/director/assembly.py:127-137` (`_director_config_from` mapping)
- Modify: `config.yaml` (under `kiosk.talkback.turn_gate`)
- Test: `tests/director/test_config_reject_bystanders.py`

**Interfaces:**
- Produces: `DirectorConfig.reject_bystanders: bool` (default `False`); `_director_config_from(tb_cfg)` maps `turn_gate.reject_bystanders` to it as a strict bool.

- [ ] **Step 1: Write the failing test**

Create `tests/director/test_config_reject_bystanders.py`:

```python
from modes.director.assembly import _director_config_from
from modes.director.config import DirectorConfig


def test_default_is_false():
    assert DirectorConfig().reject_bystanders is False


def test_mapping_true_enables():
    cfg = _director_config_from({"turn_gate": {"reject_bystanders": True}})
    assert cfg.reject_bystanders is True


def test_mapping_absent_key_is_false():
    cfg = _director_config_from({"turn_gate": {}})
    assert cfg.reject_bystanders is False


def test_mapping_malformed_value_is_false():
    # the 'flase' lesson: a non-bool (typo/string/int) must NOT enable
    for bad in ("flase", "true", 1, "yes"):
        cfg = _director_config_from({"turn_gate": {"reject_bystanders": bad}})
        assert cfg.reject_bystanders is False, bad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_config_reject_bystanders.py -q`
Expected: FAIL (`DirectorConfig` has no `reject_bystanders`; `_director_config_from` doesn't set it).

- [ ] **Step 3: Add the config field**

In `modes/director/config.py`, add after `active_talk_guard_s` (the last field):

```python
    active_talk_guard_s: float = 3.0    # never owner-absent-end within this of owner speech
    # Bystander gate v1 (Director-08, spec 2026-06-25). Reject NON-owner new turns by
    # proximity + camera presence. Default off => byte-for-byte today's reducer.
    reject_bystanders: bool = False
```

- [ ] **Step 4: Map it in assembly (strict bool)**

In `modes/director/assembly.py`, inside `_director_config_from`, add to the `DirectorConfig(...)` call (after `active_talk_guard_s=...`):

```python
        active_talk_guard_s=vision.get("active_talk_guard_s", 3.0),
        reject_bystanders=tb_cfg.get("turn_gate", {}).get("reject_bystanders", False) is True,
    )
```

(The `is True` makes only a literal boolean `True` enable — a malformed string/int is treated as off.)

- [ ] **Step 5: Add the config.yaml key**

In `config.yaml`, under `kiosk.talkback.turn_gate`, after the `verify_window_ms: 2000` line, add:

```yaml
      # Bystander gate v1 (Director-08): reject NON-owner NEW turns by proximity +
      # camera presence (reject-by-default, "never answer a stranger"). Fail-safe:
      # rejects only when too quiet/distant OR owner ABSENT from frame; PRESENT and
      # UNAVAILABLE both allow. Off in code by default (no-regression); shipped on
      # here. Only a real boolean true enables (the 'flase' lesson).
      reject_bystanders: true
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_config_reject_bystanders.py -q`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add modes/director/config.py modes/director/assembly.py config.yaml tests/director/test_config_reject_bystanders.py
git commit -m "$(cat <<'EOF'
feat(director-08): reject_bystanders config flag (strict-bool, default off)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: New-turn gate in the reducer

**Files:**
- Modify: `modes/director/reducer.py` (add `TurnVerdict` + `classify_new_turn`; rewrite `_on_user_segment`)
- Test: `tests/director/test_reducer_reject_bystanders.py`

**Interfaces:**
- Consumes: `DirectorConfig.reject_bystanders` (Task 1); `ctx.proximity_rms`, `ctx.presence_status` (existing); `E.SegmentEndpointed(duration_ms, rms, is_target, endpoint_prob)`.
- Produces: `TurnVerdict` (enum: `ACCEPT`, `ACCUMULATE`, `REJECT_NOT_TARGET`, `REJECT_TOO_QUIET`, `REJECT_OWNER_ABSENT`); `classify_new_turn(ctx, ev) -> TurnVerdict` (pure). Used by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/director/test_reducer_reject_bystanders.py`:

```python
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


def _ctx(reject, proximity_rms=0.5, presence=PresenceStatus.UNAVAILABLE, now=5.0):
    cfg = DirectorConfig(reject_bystanders=reject, endpoint_threshold=0.5)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=proximity_rms)
    ctx.presence_status = presence
    ctx.last_speech_at = 0.0          # distinct from now=5.0 so we can see resets
    return ctx


def _seg(rms=1.0, is_target=True, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=500.0, rms=rms,
                               is_target=is_target, endpoint_prob=endpoint)


# ---- flag OFF: no-regression (proximity/presence ignored; clock always resets) ----

def test_off_complete_target_accepts_even_if_quiet_and_absent():
    ctx = _ctx(reject=False, proximity_rms=0.9, presence=PresenceStatus.ABSENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=0.0001, endpoint=0.9))
    assert state is State.LISTENING and cmds == [C.TranscribeUserTurn()]
    assert ctx.last_speech_at == 5.0          # reset (legacy)


def test_off_nontarget_no_transcribe_but_resets():
    ctx = _ctx(reject=False)
    state, cmds = reduce(State.LISTENING, ctx, _seg(is_target=False))
    assert cmds == []
    assert ctx.last_speech_at == 5.0          # legacy resets even for non-target


def test_off_incomplete_accumulates_and_resets():
    ctx = _ctx(reject=False)
    state, cmds = reduce(State.LISTENING, ctx, _seg(endpoint=0.1))
    assert cmds == []
    assert ctx.last_speech_at == 5.0


# ---- flag ON: reject-by-default ----

def test_on_quiet_rejected_no_reset():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=0.1, endpoint=0.9))
    assert cmds == []
    assert ctx.last_speech_at == 0.0          # NOT reset


def test_on_owner_absent_rejected_no_reset():
    ctx = _ctx(reject=True, proximity_rms=0.0, presence=PresenceStatus.ABSENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert cmds == []
    assert ctx.last_speech_at == 0.0


def test_on_present_proximate_accepts_and_resets():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert cmds == [C.TranscribeUserTurn()]
    assert ctx.last_speech_at == 5.0


def test_on_unavailable_proximate_accepts_failsafe():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.UNAVAILABLE)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert cmds == [C.TranscribeUserTurn()]   # camera can't judge -> allow


def test_on_nontarget_rejected_no_reset():
    ctx = _ctx(reject=True, proximity_rms=0.0, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(is_target=False, rms=1.0))
    assert cmds == []
    assert ctx.last_speech_at == 0.0


def test_on_incomplete_owner_accumulates_and_resets():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.1))
    assert cmds == []                          # not complete yet
    assert ctx.last_speech_at == 5.0           # but plausibly-owner -> reset


def test_on_rejected_chatter_does_not_block_owner_absent_end():
    # owner present, then leaves; bystander chatter (rejected) must NOT keep the
    # kiosk alive / block the Director-07 owner-absent end.
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT, now=0.0)
    # advance clock to t=11 and mark owner ABSENT at t=10
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    reduce(State.LISTENING, ctx, E.Tick(now=11.0))         # ctx.now -> 11; 11-10<grace, no end
    # loud bystander chatter while owner ABSENT -> rejected, last_speech_at stays 0.0
    reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert ctx.last_speech_at == 0.0
    # owner-absent end now fires (grace + guard satisfied) instead of being blocked
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=13.5))
    assert state is State.IDLE and cmds == [C.EndSession("owner_absent")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_reducer_reject_bystanders.py -q`
Expected: FAIL (several — current `_on_user_segment` resets the clock unconditionally and applies no proximity/presence gate, so the flag-on tests fail).

- [ ] **Step 3: Add the verdict helper and rewrite `_on_user_segment`**

In `modes/director/reducer.py`, add `import enum` at the top (with the other imports), then add this enum + helper above `_on_user_segment`:

```python
class TurnVerdict(enum.Enum):
    ACCEPT = "accept"                       # complete owner turn -> transcribe
    ACCUMULATE = "accumulate"               # plausibly-owner, endpoint not yet met
    REJECT_NOT_TARGET = "not_target"        # pVAD bystander
    REJECT_TOO_QUIET = "too_quiet"          # below proximity floor (distant bystander)
    REJECT_OWNER_ABSENT = "owner_absent_frame"  # owner not in camera frame


def classify_new_turn(ctx: Context, ev: E.SegmentEndpointed) -> TurnVerdict:
    """Pure verdict for a LISTENING SegmentEndpointed. Single source of truth for the
    reducer (decide) and the runtime (DIAG). reject_bystanders off -> legacy verdict."""
    complete = ev.endpoint_prob >= ctx.cfg.endpoint_threshold
    if not ctx.cfg.reject_bystanders:
        if not ev.is_target:
            return TurnVerdict.REJECT_NOT_TARGET
        return TurnVerdict.ACCEPT if complete else TurnVerdict.ACCUMULATE
    if not ev.is_target:
        return TurnVerdict.REJECT_NOT_TARGET
    if ev.rms < ctx.proximity_rms:
        return TurnVerdict.REJECT_TOO_QUIET
    if ctx.presence_status is PresenceStatus.ABSENT:
        return TurnVerdict.REJECT_OWNER_ABSENT
    return TurnVerdict.ACCEPT if complete else TurnVerdict.ACCUMULATE
```

Then replace the body of `_on_user_segment` (currently `reducer.py:97-103`) with:

```python
def _on_user_segment(ctx: Context, ev: E.SegmentEndpointed) -> tuple:
    v = classify_new_turn(ctx, ev)
    if not ctx.cfg.reject_bystanders:
        ctx.last_speech_at = ctx.now             # legacy: any voiced segment resets
        if v is TurnVerdict.ACCEPT:
            return State.LISTENING, [C.TranscribeUserTurn()]
        return State.LISTENING, []
    # reject-by-default: only plausibly-owner speech resets the silence/presence clock
    if v in (TurnVerdict.ACCEPT, TurnVerdict.ACCUMULATE):
        ctx.last_speech_at = ctx.now
        if v is TurnVerdict.ACCEPT:
            return State.LISTENING, [C.TranscribeUserTurn()]
        return State.LISTENING, []
    return State.LISTENING, []                    # rejected: no clock reset
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_reducer_reject_bystanders.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Run the existing reducer suite for no-regression**

Run: `python3 -m pytest tests/director/test_reducer_presence.py tests/director/ -q`
Expected: PASS (all existing director tests still green — the legacy branch is unchanged behavior).

- [ ] **Step 6: Commit**

```bash
git add modes/director/reducer.py tests/director/test_reducer_reject_bystanders.py
git commit -m "$(cat <<'EOF'
feat(director-08): new-turn bystander gate in reducer (proximity + presence)

classify_new_turn verdict helper + reject-by-default _on_user_segment. Flag off =
legacy (byte-for-byte). Flag on: reject too-quiet / owner-absent / non-target, and
reset the silence/owner-absent clock only on accepted turns so rejected chatter no
longer keeps the kiosk alive. UNAVAILABLE/PRESENT allow (fail-safe).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Runtime DIAG observability

**Files:**
- Modify: `modes/director/reducer.py` (add `gate_diag_reason`)
- Modify: `modes/director/runtime.py:73-78` (emit the DIAG line)
- Test: `tests/director/test_gate_diag_reason.py`

**Interfaces:**
- Consumes: `classify_new_turn`, `TurnVerdict` (Task 2).
- Produces: `gate_diag_reason(ctx, ev) -> Optional[str]` — the reject reason string for a new-turn segment, or `None` if accepted/accumulating.

- [ ] **Step 1: Write the failing test**

Create `tests/director/test_gate_diag_reason.py`:

```python
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director.events import PresenceStatus
from modes.director.reducer import gate_diag_reason
from modes.talkback.conversation import ConversationManager


def _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT):
    cfg = DirectorConfig(reject_bystanders=reject, endpoint_threshold=0.5)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=5.0, proximity_rms=proximity_rms)
    ctx.presence_status = presence
    return ctx


def _seg(rms=1.0, is_target=True, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=500.0, rms=rms,
                               is_target=is_target, endpoint_prob=endpoint)


def test_reason_for_too_quiet():
    ctx = _ctx(proximity_rms=0.5)
    assert gate_diag_reason(ctx, _seg(rms=0.1)) == "too_quiet"


def test_reason_for_owner_absent():
    ctx = _ctx(presence=PresenceStatus.ABSENT, proximity_rms=0.0)
    assert gate_diag_reason(ctx, _seg(rms=1.0)) == "owner_absent_frame"


def test_reason_none_when_accepted():
    ctx = _ctx(presence=PresenceStatus.PRESENT, proximity_rms=0.5)
    assert gate_diag_reason(ctx, _seg(rms=1.0, endpoint=0.9)) is None


def test_reason_none_while_accumulating():
    ctx = _ctx(presence=PresenceStatus.PRESENT, proximity_rms=0.5)
    assert gate_diag_reason(ctx, _seg(rms=1.0, endpoint=0.1)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_gate_diag_reason.py -q`
Expected: FAIL (`gate_diag_reason` not defined).

- [ ] **Step 3: Add `gate_diag_reason`**

In `modes/director/reducer.py`, after `classify_new_turn`, add:

```python
def gate_diag_reason(ctx: Context, ev: E.SegmentEndpointed):
    """DIAG-only: the reject reason for a new-turn segment, or None if accepted /
    still accumulating. None for non-reject verdicts keeps the log quiet."""
    v = classify_new_turn(ctx, ev)
    if v in (TurnVerdict.REJECT_NOT_TARGET, TurnVerdict.REJECT_TOO_QUIET,
             TurnVerdict.REJECT_OWNER_ABSENT):
        return v.value
    return None
```

Note `Optional` is not imported in reducer.py; the bare annotation-free return is fine (no type hint needed on the return). Do NOT add an unused import.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_gate_diag_reason.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire it into the runtime DIAG**

In `modes/director/runtime.py`, add to the imports near the top (with the other `modes.director` imports):

```python
from modes.director.reducer import gate_diag_reason
from modes.director import events as E
```

(If `events as E` is already imported, don't duplicate it — check the existing import block first.)

Then in `run_async`, inside the `while` loop, right AFTER the existing `_diag(...)` block (the one printing `event=...`), add:

```python
                if _DIAG and isinstance(event, E.SegmentEndpointed):
                    reason = gate_diag_reason(self._director.ctx, event)
                    if reason is not None:
                        _diag(f"new-turn REJECT={reason} rms={event.rms:.4f} "
                              f"prox={self._director.ctx.proximity_rms:.4f} "
                              f"presence={self._director.ctx.presence_status.name}")
```

(`classify_new_turn` reads only `proximity_rms`/`presence_status`/event fields — none mutated by `dispatch()` for a SegmentEndpointed — so calling it after dispatch yields the same verdict.)

- [ ] **Step 6: Run the full director suite**

Run: `python3 -m pytest tests/director/ -q`
Expected: PASS (all director tests, including the new ones).

- [ ] **Step 7: Commit**

```bash
git add modes/director/reducer.py modes/director/runtime.py tests/director/test_gate_diag_reason.py
git commit -m "$(cat <<'EOF'
feat(director-08): runtime DIAG line for rejected new-turns (gate_diag_reason)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Full-suite gate + live validation

**Files:**
- None (verification only); then a verdict note `docs/notes/2026-06-25-director-08-live.md`.

- [ ] **Step 1: Run the entire test suite**

Run: `python3 -m pytest -q`
Expected: PASS (all tests; ~640+). No failures. If anything fails, STOP and fix before proceeding.

- [ ] **Step 2: Live validation (human at the kiosk)**

With `config.yaml` `kiosk.talkback.turn_gate.reject_bystanders: true`, run:
`TVAD_DIAG=1 ./kiosk-stack.sh start`

Confirm, reading the DIAG log:
- **Bystander rejected:** while the owner is served, a bystander speaking from across the room produces a `new-turn REJECT=too_quiet ...` (or `owner_absent_frame`) line and NO following `UserTurnTranscribed` — the kiosk does not answer it.
- **Owner served normally:** the owner's turns still produce `UserTurnTranscribed` → normal replies (proximate + PRESENT).
- **Chatter doesn't hold the kiosk:** step away while bystanders keep talking → the session still ends (`owner_absent` if camera on, else `silence_timeout`), not held open by the rejected chatter.

- [ ] **Step 3: No-regression live check**

Set `reject_bystanders: false`, run again. Confirm NO `new-turn REJECT=` lines appear and behavior matches today's kiosk (turns accepted as before).

- [ ] **Step 4: Write the verdict note + restore config**

Restore `reject_bystanders: true` (production). Write `docs/notes/2026-06-25-director-08-live.md` with the three observations above (bystander rejected, owner served, chatter doesn't hold), then commit:

```bash
git add config.yaml docs/notes/2026-06-25-director-08-live.md
git commit -m "$(cat <<'EOF'
docs(director-08): live validation verdict — bystander gate v1

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Finish the branch**

Use the superpowers:finishing-a-development-branch skill to merge `feat/director-08-bystander-gate` to master (local), after the full suite passes and live validation is green.

---

## Notes for the implementer

- The `reject_bystanders` flag ships ON in `config.yaml` but defaults OFF in `DirectorConfig` — same pattern as Director-07's `vision.enabled`. The default-off is the asserted no-regression contract.
- `classify_new_turn` is the single source of truth: the reducer decides with it and the runtime logs with it. Do not duplicate the gate logic anywhere else.
- The ReSpeaker DOA-cone vote (future) will slot into `classify_new_turn` as one more reject branch (`if bearing not in owner_cone: return TurnVerdict.REJECT_OFF_AXIS`) — keep the verdict structure clean for that.
- Do NOT wire `SafetyNet`/`Lockout` here — accumulated-ECAPA is explicitly deferred.
