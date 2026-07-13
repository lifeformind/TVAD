# Kiosk Tuning Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One browser tab holding the whole kiosk tuning loop: tabbed knob editor over `config.yaml` (targeted line edits, comments byte-identical) + kiosk run as the server's child with its DIAG stdout streamed live to a log pane.

**Architecture:** New top-level `tune/` package. A frozen-dataclass knob registry is the single source of truth; a pure-function line editor rewrites only scalar values in `config.yaml` with an all-or-nothing round-trip guard; a child-process manager owns `kiosk.py --talkback` and fans its stdout out to SSE clients; a stdlib `ThreadingHTTPServer` ties them together behind a single static HTML page.

**Tech Stack:** Python stdlib only (`http.server`, `subprocess`, `threading`, `queue`, `urllib`) + PyYAML (already a dependency) + one vanilla-JS HTML file. **No new dependencies. No build step.**

**Spec:** `docs/superpowers/specs/2026-07-13-tuning-console-design.md` — read it before starting any task.

## Global Constraints

- Repo root for all paths below: `target-vad/` (the git repo root is one level up; `git add` paths therefore work from `target-vad/`).
- Run tests as `python3 -m pytest ...` — never bare `pytest` or `python`.
- Stage explicit paths only. NEVER `git add -A` (`bench/spatial_voice_probe.py` is deliberately untracked).
- Every commit message ends with the trailer line: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Branch: `feat/tuning-ui` (already created, stacked on `feat/director-11-doa-cone-vote`).
- Default bind `127.0.0.1:8765`. The LLM owns 8080 — never bind it.
- Save is all-or-nothing: any locate/render/round-trip failure ⇒ `ConfigEditError`, nothing written.
- Strict-bool config keys (`turn_gate.require_speaker_match`, `turn_gate.reject_bystanders`, `turn_gate.doa.enabled`) must only ever be written as literal `true`/`false`.
- The tuning server must never orphan a kiosk child (stop on shutdown) and must never start when a foreign `kiosk.py --talkback` runs.

---

### Task 1: `tune/config_edit.py` — targeted line editor

**Files:**
- Create: `tune/__init__.py` (empty)
- Create: `tune/config_edit.py`
- Test: `tests/tune/test_config_edit.py`

**Interfaces:**
- Produces: `set_values(text: str, changes: dict[str, object]) -> str` (pure; raises `ConfigEditError`), `ConfigEditError(Exception)`, `get_path(data: dict, path: str) -> object` (raises `ConfigEditError` on unknown path). Task 4's server calls `set_values` and `get_path`; Task 2's registry test calls `get_path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tune/test_config_edit.py`:

```python
"""Targeted line edits to config.yaml: values change, everything else is
byte-identical. All-or-nothing — any failure leaves the text untouched
(the function is pure; the caller only writes the returned string)."""

from pathlib import Path

import pytest
import yaml

from tune.config_edit import ConfigEditError, get_path, set_values

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture()
def text():
    return REAL_CONFIG.read_text()


def _reload(edited, path):
    return get_path(yaml.safe_load(edited), path)


# ---- happy paths, one per scalar kind ----

def test_float_edit_touches_exactly_one_line(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0})
    assert _reload(edited, "kiosk.talkback.turn_gate.doa.cone_deg") == 25.0
    diff = [(a, b) for a, b in zip(text.split("\n"), edited.split("\n")) if a != b]
    assert len(diff) == 1
    old_line, new_line = diff[0]
    assert "cone_deg" in old_line
    # the inline comment survives verbatim
    assert old_line.split("#", 1)[1] == new_line.split("#", 1)[1]


def test_int_edit(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.doa.min_in_cone_samples": 4})
    assert _reload(edited, "kiosk.talkback.turn_gate.doa.min_in_cone_samples") == 4


def test_strict_bool_renders_literal_true_false(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.reject_bystanders": False})
    line = [l for l in edited.split("\n") if "reject_bystanders" in l][0]
    assert "reject_bystanders: false" in line


def test_none_renders_null_and_null_becomes_number(text):
    # rms_threshold is null in the file today; set it, then set it back
    key = "kiosk.talkback.barge_in.proximity.rms_threshold"
    edited = set_values(text, {key: 0.04})
    assert _reload(edited, key) == 0.04
    back = set_values(edited, {key: None})
    assert _reload(back, key) is None
    assert "rms_threshold: null" in back


def test_string_edit_is_quoted(text):
    edited = set_values(text, {"kiosk.wake_phrase": "hey_jarvis"})
    assert _reload(edited, "kiosk.wake_phrase") == "hey_jarvis"
    line = [l for l in edited.split("\n") if l.strip().startswith("wake_phrase")][0]
    assert '"hey_jarvis"' in line


def test_block_scalar_edit_keeps_header_and_indents(text):
    key = "kiosk.talkback.llm.system_prompt"
    edited = set_values(text, {key: "Line one.\nLine two."})
    assert _reload(edited, key).rstrip("\n") == "Line one.\nLine two."
    lines = edited.split("\n")
    hdr = [i for i, l in enumerate(lines) if l.strip().startswith("system_prompt")][0]
    assert lines[hdr].rstrip().endswith("|")
    assert lines[hdr + 1] == "        Line one."          # key at 6 -> body at 8
    # the section after the block is intact
    assert _reload(edited, "kiosk.talkback.tts.voice") == "af_bella"


def test_multi_change_in_one_call(text):
    edited = set_values(text, {
        "kiosk.talkback.turn_gate.speaker_threshold": 0.18,
        "kiosk.talkback.barge_in.duck_level": 0.5,
    })
    assert _reload(edited, "kiosk.talkback.turn_gate.speaker_threshold") == 0.18
    assert _reload(edited, "kiosk.talkback.barge_in.duck_level") == 0.5


def test_same_key_name_different_sections_are_independent(text):
    # speaker_threshold exists under BOTH turn_gate and barge_in
    edited = set_values(text, {"kiosk.talkback.barge_in.speaker_threshold": 0.25})
    assert _reload(edited, "kiosk.talkback.barge_in.speaker_threshold") == 0.25
    assert _reload(edited, "kiosk.talkback.turn_gate.speaker_threshold") == 0.15


def test_comments_and_untouched_lines_are_byte_identical(text):
    edited = set_values(text, {"kiosk.talkback.silence_timeout_s": 45})
    kept = [l for l in text.split("\n") if "silence_timeout_s" not in l]
    kept_after = [l for l in edited.split("\n") if "silence_timeout_s" not in l]
    assert kept == kept_after


# ---- refusals (all-or-nothing) ----

def test_unknown_path_refused(text):
    with pytest.raises(ConfigEditError, match="no_such"):
        set_values(text, {"kiosk.no_such_key": 1})


def test_section_path_refused(text):
    with pytest.raises(ConfigEditError):
        set_values(text, {"kiosk.talkback.turn_gate.doa": 1})


def test_one_bad_path_aborts_the_whole_save(text):
    with pytest.raises(ConfigEditError):
        set_values(text, {
            "kiosk.talkback.turn_gate.doa.cone_deg": 25.0,
            "kiosk.bogus": 1,
        })
    # pure function: caller's text is untouched by construction


def test_duplicate_line_ambiguity_refused():
    dup = "a:\n  x: 1\n  x: 2\n"
    with pytest.raises(ConfigEditError, match="multiple"):
        set_values(dup, {"a.x": 3})


def test_unsupported_value_type_refused(text):
    with pytest.raises(ConfigEditError, match="unsupported"):
        set_values(text, {"kiosk.wake_threshold": [1, 2]})


def test_edit_never_disturbs_other_leaves(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0})
    before, after = yaml.safe_load(text), yaml.safe_load(edited)
    before["kiosk"]["talkback"]["turn_gate"]["doa"]["cone_deg"] = 25.0
    assert before == after
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/tune/test_config_edit.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tune'`.

- [ ] **Step 3: Implement**

Create empty `tune/__init__.py`, then `tune/config_edit.py`:

```python
"""Targeted line edits to config.yaml.

Only the scalar value on a known key's line changes; every comment and all
formatting stay byte-identical. Pure function, all-or-nothing: any locate,
render, or round-trip failure raises ConfigEditError and nothing is returned.
The caller (tune.server) writes the returned text atomically."""

from __future__ import annotations

import json
import re

import yaml


class ConfigEditError(Exception):
    pass


_KEY_RE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z_][\w-]*):(?P<after>.*)$")
_BLOCK_HEADERS = ("|", "|-", "|+", ">", ">-", ">+")


def get_path(data: dict, path: str):
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigEditError(f"unknown config path: {path}")
        node = node[part]
    return node


def set_values(text: str, changes: dict[str, object]) -> str:
    if not changes:
        return text
    original = yaml.safe_load(text)
    for path, value in changes.items():
        if isinstance(get_path(original, path), dict):
            raise ConfigEditError(f"{path}: is a section, not a value")
        _render(value, path)  # type-check before touching anything
    lines = text.split("\n")
    found: dict[str, tuple[int, int, str]] = {}
    for idx, path, indent, after in _iter_key_lines(lines):
        if path in changes:
            if path in found:
                raise ConfigEditError(f"{path}: multiple lines match")
            found[path] = (idx, indent, after)
    missing = sorted(set(changes) - set(found))
    if missing:
        raise ConfigEditError("could not locate line(s) for: " + ", ".join(missing))
    # bottom-up so a block-scalar replacement can't shift later indices
    for path in sorted(found, key=lambda p: found[p][0], reverse=True):
        idx, indent, after = found[path]
        if after.strip() in _BLOCK_HEADERS:
            lines = _replace_block(lines, idx, indent, changes[path], path)
        else:
            lines[idx] = _replace_scalar(lines[idx], changes[path], path)
    edited = "\n".join(lines)
    _verify(original, edited, changes)
    return edited


def _iter_key_lines(lines):
    """Yield (index, dotted-path, indent, value-part) for every mapping key,
    tracking nesting by indentation and skipping block-scalar bodies."""
    stack: list[tuple[int, str]] = []
    block_key_indent = None  # indent of an open block scalar's key, else None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if block_key_indent is not None:
            if indent > block_key_indent:
                continue  # inside the block scalar body
            block_key_indent = None
        m = _KEY_RE.match(line)
        if not m:
            continue  # flow continuation / list item — never a registry target
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m["key"]))
        after = m["after"]
        if after.strip() in _BLOCK_HEADERS:
            block_key_indent = indent
        yield i, ".".join(k for _, k in stack), indent, after


def _render(value, path: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # double-quoted, matches the file's style
    raise ConfigEditError(
        f"{path}: unsupported value type {type(value).__name__}")


def _replace_scalar(line: str, value, path: str) -> str:
    m = re.match(r"^(?P<pre>[ ]*[A-Za-z_][\w-]*:[ \t]*)(?P<rest>.*)$", line)
    rest = m["rest"]
    if not rest or rest.lstrip().startswith("#"):
        raise ConfigEditError(f"{path}: no inline value on its line")
    if rest[0] in "\"'":
        end = rest.find(rest[0], 1)
        if end == -1:
            raise ConfigEditError(f"{path}: unterminated quote")
        tail = rest[end + 1:]
    else:
        cm = re.search(r"[ \t]+#", rest)
        end = cm.start() if cm else len(rest.rstrip())
        tail = rest[end:]
    return m["pre"] + _render(value, path) + tail


def _replace_block(lines, idx, key_indent, value, path):
    if not isinstance(value, str):
        raise ConfigEditError(f"{path}: block scalar needs a string")
    end = idx + 1
    while end < len(lines) and (
            not lines[end].strip()
            or len(lines[end]) - len(lines[end].lstrip(" ")) > key_indent):
        end += 1
    while end > idx + 1 and not lines[end - 1].strip():
        end -= 1  # trailing blank lines belong to the document, not the block
    pad = " " * (key_indent + 2)
    body = [pad + l if l else "" for l in value.rstrip("\n").split("\n")]
    return lines[:idx + 1] + body + lines[end:]


def _verify(original: dict, edited_text: str, changes: dict) -> None:
    try:
        after = yaml.safe_load(edited_text)
    except yaml.YAMLError as e:
        raise ConfigEditError(f"edit produced unparseable YAML: {e}") from e
    for path, want in changes.items():
        got = get_path(after, path)
        if _norm(got) != _norm(want):
            raise ConfigEditError(
                f"{path}: round-trip mismatch ({got!r} != {want!r})")
    for path, was in _leaves(original):
        if path in changes:
            continue
        if get_path(after, path) != was:
            raise ConfigEditError(
                f"{path}: unintended change ({was!r} -> {get_path(after, path)!r})")


def _norm(v):
    return v.rstrip("\n") if isinstance(v, str) else v


def _leaves(data: dict, prefix: str = ""):
    for k, v in data.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _leaves(v, p)
        else:
            yield p, v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/tune/test_config_edit.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite (regression guard)**

Run: `python3 -m pytest -q`
Expected: everything passes (799+ passed, 2 skipped, plus the new tests).

- [ ] **Step 6: Commit**

```bash
git add tune/__init__.py tune/config_edit.py tests/tune/test_config_edit.py
git commit -m "feat(tune): targeted config.yaml line editor (all-or-nothing)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `tune/knobs.py` — the knob registry

**Files:**
- Create: `tune/knobs.py`
- Test: `tests/tune/test_knobs.py`

**Interfaces:**
- Consumes: `tune.config_edit.get_path` (Task 1) in the meta-test.
- Produces: `Knob` frozen dataclass (fields: `path, tab, label, kind, doc, why, min, max, step, choices, nullable, strict_bool, danger`), `KNOBS: tuple[Knob, ...]`, `TABS: tuple[str, ...]`, `BY_PATH: dict[str, Knob]`, `as_json() -> list[dict]`. Task 4's server uses `BY_PATH` for validation and `as_json()` for `/api/state`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tune/test_knobs.py`:

```python
"""Registry meta-tests: the registry must stay true to the real config.yaml
(catches drift when config evolves) and internally sane."""

from pathlib import Path

import yaml

from tune.config_edit import get_path
from tune.knobs import BY_PATH, KNOBS, TABS, Knob, as_json

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
CONFIG = yaml.safe_load(REAL_CONFIG.read_text())

KIND_TYPES = {
    "float": (int, float), "int": (int,), "bool": (bool,),
    "select": (str,), "text": (str,), "textarea": (str,),
}


def test_every_knob_path_exists_in_real_config():
    for k in KNOBS:
        get_path(CONFIG, k.path)  # raises on drift


def test_every_knob_value_matches_its_kind():
    for k in KNOBS:
        v = get_path(CONFIG, k.path)
        if v is None:
            assert k.nullable, f"{k.path} is null but not nullable"
            continue
        assert isinstance(v, KIND_TYPES[k.kind]), f"{k.path}: {v!r} not {k.kind}"
        if k.kind != "bool":
            assert not isinstance(v, bool), f"{k.path}: bool in a {k.kind} knob"


def test_selects_include_the_current_value():
    for k in KNOBS:
        if k.kind == "select":
            assert get_path(CONFIG, k.path) in k.choices, k.path


def test_numeric_knobs_have_ranges_containing_current_value():
    for k in KNOBS:
        if k.kind in ("float", "int"):
            assert k.min is not None and k.max is not None and k.min < k.max, k.path
            v = get_path(CONFIG, k.path)
            if v is not None:
                assert k.min <= v <= k.max, f"{k.path}: {v} outside [{k.min},{k.max}]"


def test_tabs_are_the_declared_set_and_nonempty():
    assert set(k.tab for k in KNOBS) == set(TABS)
    for t in TABS:
        assert any(k.tab == t for k in KNOBS)


def test_strict_bools_are_exactly_the_documented_keys():
    strict = {k.path for k in KNOBS if k.strict_bool}
    assert strict == {
        "kiosk.talkback.turn_gate.require_speaker_match",
        "kiosk.talkback.turn_gate.reject_bystanders",
        "kiosk.talkback.turn_gate.doa.enabled",
    }


def test_paths_unique_and_by_path_complete():
    assert len({k.path for k in KNOBS}) == len(KNOBS)
    assert BY_PATH == {k.path: k for k in KNOBS}


def test_as_json_is_serializable_and_ordered_like_knobs():
    rows = as_json()
    assert [r["path"] for r in rows] == [k.path for k in KNOBS]
    assert all(isinstance(r["choices"], list) for r in rows)


def test_excluded_keys_are_not_registered():
    for banned in ("kiosk.talkback.output_device", "core.audio.device_index",
                   "kiosk.talkback.stt.backend", "kiosk.talkback.llm.base_url",
                   "kiosk.talkback.aec.enabled", "kiosk.talkback.crowd_focus.enabled",
                   "core.audio.sample_rate", "core.vad.sample_rate",
                   "kiosk.talkback.sample_rate_hz"):
        assert banned not in BY_PATH, banned
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/tune/test_knobs.py -v`
Expected: `ModuleNotFoundError: No module named 'tune.knobs'`.

- [ ] **Step 3: Implement the registry**

Create `tune/knobs.py` (this is the complete registry — spec section 4b; `doc` = what it does, `why` = the live-tuning rationale carried forward from config.yaml's comments):

```python
"""The knob registry — single source of truth for the tuning console.

Data, not behavior: the server serializes it for the page and validates saves
against it; config_edit consumes only `path`. Excluded on purpose (spec 4b):
sample rates (coupled), device pins, backend selectors, dormant paths
(aec, crowd_focus), paths/logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Knob:
    path: str
    tab: str
    label: str
    kind: str            # float | int | bool | select | text | textarea
    doc: str
    why: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple = ()
    nullable: bool = False
    strict_bool: bool = False
    danger: bool = False


TB = "kiosk.talkback."
TG = TB + "turn_gate."

WAKE = "Wake & Enrollment"
GATE = "Turn Gate"
DOA = "DOA Cone"
BARGE = "Barge-in & Duck"
CAM = "Presence (Camera)"
TIME = "Session & Timing"
VOICE = "Voice Pipeline"
AUDIO = "Audio & VAD ⚠"

TABS = (WAKE, GATE, DOA, BARGE, CAM, TIME, VOICE, AUDIO)

KNOBS: tuple[Knob, ...] = (
    # ---- Wake & Enrollment ----
    Knob("kiosk.wake_phrase", WAKE, "Wake phrase", "select",
         "openWakeWord phrase the kiosk arms on.",
         choices=("hey_mycroft", "hey_jarvis", "alexa")),
    Knob("kiosk.wake_threshold", WAKE, "Wake threshold", "float",
         "Wake detector score required to trigger.",
         min=0.1, max=0.95, step=0.05),
    Knob("kiosk.awaiting_speech_timeout_s", WAKE, "Await-speech timeout (s)", "int",
         "After a wake, seconds to wait for first speech before re-arming.",
         "Backstops the D11 seed-retry loop: refused seeds retry inside this window.",
         min=3, max=60, step=1),
    Knob(TB + "verify_before_serve_threshold", WAKE, "Seed self-check", "float",
         "Split-half self-similarity the first segment must clear to start a session.",
         "Below this the seed is refused and retried within the same wake (D09/D11).",
         min=0.5, max=0.95, step=0.05),
    Knob("core.speaker.threshold", WAKE, "Speaker cosine threshold", "float",
         "ECAPA cosine used by the enrollment tooling for target-vs-other.",
         min=0.3, max=0.95, step=0.05),
    Knob("core.speaker.min_segment_duration_ms", WAKE, "Min segment (ms)", "int",
         "Shortest segment the speaker pipeline will embed.",
         min=200, max=3000, step=100),
    Knob("core.speaker.enrollment_utterances", WAKE, "Enrollment utterances", "int",
         "Utterances collected by the offline enroll.py flow.",
         min=1, max=10, step=1),
    Knob("core.speaker.enrollment_min_self_similarity", WAKE, "Enrollment self-sim", "float",
         "Minimum self-similarity for an enrollment to be accepted.",
         "0.6 admitted drifty enrollments that later false-rejected the real user; "
         "0.80 matches the ~2% EER point on >=5s cumulative audio.",
         min=0.5, max=0.95, step=0.05),
    Knob("core.speaker.enrollment_max_retries", WAKE, "Enrollment retries", "int",
         "Re-enrollment attempts before giving up.", min=0, max=10, step=1),

    # ---- Turn Gate ----
    Knob(TG + "require_speaker_match", GATE, "Speaker safety net", "bool",
         "Master enable for ECAPA verification of served turn audio.",
         strict_bool=True),
    Knob(TG + "speaker_threshold", GATE, "Verify threshold", "float",
         "ECAPA score a 2s served-audio window must reach to count as the owner.",
         "Live-tuned 2026-07-06: owner windows 0.230-0.466, stranger 0.069; 0.30 sat "
         "inside the owner band -> two false ejects; 0.15 = midpoint of the live gap.",
         min=0.0, max=0.6, step=0.01),
    Knob(TG + "verify_window_ms", GATE, "Verify window (ms)", "int",
         "Served turn audio accumulates to this length before each ECAPA verify.",
         "Short single turns are never judged alone (ECAPA needs 2-3s).",
         min=500, max=5000, step=100),
    Knob(TG + "endpoint_threshold", GATE, "Endpoint threshold", "float",
         "Smart Turn endpoint_prob >= this means the turn is complete.",
         min=0.1, max=0.9, step=0.05),
    Knob(TG + "reject_bystanders", GATE, "Reject bystanders", "bool",
         "Reject non-owner NEW turns by proximity + camera (never answer a stranger).",
         strict_bool=True),
    Knob(TG + "lockout.enabled", GATE, "Eject authority", "bool",
         "false = shadow mode: verdicts + WARN DIAG only, no session ends."),
    Knob(TG + "lockout.window_size", GATE, "Smoother window (N)", "int",
         "M-of-N smoother over verify windows.", min=1, max=10, step=1),
    Knob(TG + "lockout.min_matches", GATE, "Smoother matches (M)", "int",
         "Windows that must pass within the smoother; 1-of-3 fails only after "
         "~6s of served non-matching speech.", min=0, max=10, step=1),
    Knob(TB + "lockout_idle_after_s", GATE, "Post-eject quiet hold (s)", "int",
         "After a speaker_mismatch eject, the near field must be quiet this long "
         "before a fresh wake is accepted (never a permanent lockout).",
         min=0, max=60, step=1),

    # ---- DOA Cone ----
    Knob(TG + "doa.enabled", DOA, "DOA cone gate", "bool",
         "Direction as a fourth gate; any missing signal -> abstain (fail open, "
         "exact Director-10 behavior).", strict_bool=True),
    Knob(TG + "doa.cone_deg", DOA, "Cone half-width (deg)", "float",
         "Half-width around the owner bearing.",
         "+/-20 scored 100% on the 2026-07-06 spike.",
         min=5, max=90, step=5),
    Knob(TG + "doa.poll_ms", DOA, "Poll cadence (ms)", "int",
         "DOAANGLE sampling period; the XVF-3000 updates ~150ms, faster buys nothing.",
         min=50, max=1000, step=50),
    Knob(TG + "doa.bearing_ema_alpha", DOA, "Bearing tracking rate", "float",
         "EMA rate the owner bearing moves toward served in-cone medians.",
         min=0.0, max=1.0, step=0.05),
    Knob(TG + "doa.min_in_cone_fraction", DOA, "Min in-cone fraction", "float",
         "Share of a segment's speech samples that must point at the owner.",
         "Vote = 'did the owner speak during this segment', not 'who spoke most' — "
         "a duration-majority median voted the podcast on merged segments "
         "(live 2026-07-07).",
         min=0.05, max=1.0, step=0.05),
    Knob(TG + "doa.min_in_cone_samples", DOA, "Min in-cone samples", "int",
         "AND at least this many samples (~450ms at 150ms polls). Fewer TOTAL "
         "samples than this = abstain, never reject.",
         min=1, max=20, step=1),

    # ---- Barge-in & Duck ----
    Knob(TB + "barge_in.enabled", BARGE, "Barge-in", "bool",
         "The session primary can interrupt the AI mid-reply."),
    Knob(TB + "barge_in.min_speech_ms", BARGE, "Onset min speech (ms)", "int",
         "Near-field speech must run this long to trigger the duck.",
         min=40, max=500, step=10),
    Knob(TB + "barge_in.speaker_threshold", BARGE, "Interjection threshold", "float",
         "ECAPA score for CUT (owner) vs RESTORE (bystander/noise).",
         "Provisional: live barge-ins with AEC on scored ~0.12-0.41; re-measure via "
         "bench/speaker_scores.py --source barge_in.",
         min=0.0, max=0.6, step=0.01),
    Knob(TB + "barge_in.conf_floor", BARGE, "STT confidence floor", "float",
         "Interjection mean_word_prob below this -> RESTORE, not CUT.",
         min=0.0, max=1.0, step=0.05),
    Knob(TB + "barge_in.verify_window_ms", BARGE, "Verify window (ms)", "int",
         "Clean audio captured during the duck before verifying.",
         "1200 rejected most real barge-ins as too_short; proximity guards bystanders.",
         min=200, max=2000, step=50),
    Knob(TB + "barge_in.duck_level", BARGE, "Duck level", "float",
         "TTS gain while capturing an interruption.",
         "Partial (not near-silent) so a rejected interjection doesn't lose the "
         "reply's tail; louder duck = more self-voice for the AEC to cancel.",
         min=0.0, max=1.0, step=0.05),
    Knob(TB + "barge_in.duck_ramp_ms", BARGE, "Duck ramp (ms)", "int",
         "Gain ramp time into/out of the duck.", min=0, max=500, step=10),
    Knob(TB + "barge_in.proximity.enabled", BARGE, "Proximity gate", "bool",
         "Ignore speech too quiet to be someone at the kiosk."),
    Knob(TB + "barge_in.proximity.rms_threshold", BARGE, "Proximity floor (RMS)", "float",
         "Absolute floor; null = auto-calibrate from the seed's RMS.",
         min=0.0, max=0.2, step=0.005, nullable=True),
    Knob(TB + "barge_in.proximity.rms_factor", BARGE, "Floor factor", "float",
         "Auto-calibrated floor = seed RMS x this.", min=0.1, max=1.0, step=0.05),
    Knob(TB + "barge_in.proximity.max_floor", BARGE, "Floor cap", "float",
         "Cap on the calibrated floor.",
         "Wake seeds ran 0.085-0.21 across sessions; a shouted wake priced the "
         "owner's normal 0.04-0.10 voice out of its own session (2026-07-07).",
         min=0.01, max=0.2, step=0.005),

    # ---- Presence (Camera) ----
    Knob(TB + "vision.enabled", CAM, "Camera presence", "bool",
         "Presence is the floor authority: free the kiosk fast when the owner leaves."),
    Knob(TB + "vision.identity_threshold", CAM, "Face identity threshold", "float",
         "SFace cosine for owner-vs-stranger.",
         "Spike Tier-2 GO: self >=0.79 vs stranger <=0.06.",
         min=0.1, max=0.9, step=0.05),
    Knob(TB + "vision.min_area_frac", CAM, "Min face area", "float",
         "Face box share of the 640x360 frame required to count.",
         min=0.005, max=0.2, step=0.005),
    Knob(TB + "vision.present_after_s", CAM, "PRESENT debounce (s)", "float",
         "Sustained detection before flipping to PRESENT.",
         min=0.0, max=5.0, step=0.25),
    Knob(TB + "vision.absent_after_s", CAM, "ABSENT debounce (s)", "float",
         "Sustained non-detection before flipping to ABSENT.",
         min=0.0, max=10.0, step=0.25),
    Knob(TB + "vision.owner_absent_grace_s", CAM, "Absent grace (s)", "float",
         "Sustained ABSENT this long frees the kiosk.", min=0.0, max=15.0, step=0.5),
    Knob(TB + "vision.active_talk_guard_s", CAM, "Talk guard (s)", "float",
         "Never owner-absent-end within this of owner speech.",
         min=0.0, max=10.0, step=0.5),
    Knob(TB + "vision.enroll_frames", CAM, "Enroll frames", "int",
         "Owner face-reference frames captured at session start.",
         min=1, max=30, step=1),
    Knob(TB + "vision.fps", CAM, "Camera FPS", "int",
         "Low-rate dedicated capture (detection ~2% of a core at 3).",
         min=1, max=10, step=1),

    # ---- Session & Timing ----
    Knob(TB + "silence_timeout_s", TIME, "Silence timeout (s)", "int",
         "Pause between your turns that ends the session.", min=5, max=120, step=5),
    Knob(TB + "hard_timeout_s", TIME, "Hard timeout (s)", "int",
         "Absolute session cap.", min=30, max=1800, step=30),
    Knob(TB + "nudge_lead_s", TIME, "Nudge lead (s)", "float",
         "'Are you still there?' fires this many seconds BEFORE the silence timeout.",
         min=0.0, max=30.0, step=1),
    Knob(TB + "watchdog.tick_ms", TIME, "Watchdog tick (ms)", "int",
         "The Director's single clock source.", min=100, max=2000, step=50),

    # ---- Voice Pipeline ----
    Knob(TB + "stt.model", VOICE, "STT model", "select",
         "Whisper model for turn transcription.",
         "GB10 p95: tiny 67 / base.en 84 / small.en 189 / medium.en 415 ms.",
         choices=("tiny", "base.en", "small.en", "medium.en")),
    Knob(TB + "stt.end_of_utterance_tail_ms", VOICE, "Utterance tail (ms)", "int",
         "Audio kept after the endpoint before transcribing.",
         min=0, max=2000, step=50),
    Knob(TB + "llm.temperature", VOICE, "LLM temperature", "float",
         "Sampling temperature.", min=0.0, max=2.0, step=0.05),
    Knob(TB + "llm.max_tokens", VOICE, "LLM max tokens", "int",
         "Reply cap.", min=32, max=2048, step=32),
    Knob(TB + "llm.system_prompt", VOICE, "System prompt", "textarea",
         "Voice-assistant persona; keep replies short and markdown-free "
         "(markdown leaks into TTS)."),
    Knob(TB + "tts.voice", VOICE, "TTS voice", "text",
         "Kokoro voice id (e.g. af_bella)."),
    Knob(TB + "chunker.max_chunk_chars", VOICE, "TTS chunk chars", "int",
         "Sentence-chunker cap per synthesized chunk.", min=40, max=400, step=10),

    # ---- Audio & VAD (structural danger) ----
    Knob("core.vad.speech_threshold", AUDIO, "VAD speech threshold", "float",
         "Silero speech probability per chunk.", min=0.1, max=0.9, step=0.05),
    Knob("core.vad.min_speech_duration_ms", AUDIO, "VAD min speech (ms)", "int",
         "Shortest run kept as speech.", min=0, max=2000, step=50),
    Knob("core.vad.padding_ms", AUDIO, "VAD padding (ms)", "int",
         "Silence padding kept around segments.", min=0, max=1000, step=50),
    Knob("core.audio.channels", AUDIO, "Capture channels", "int",
         "STRUCTURAL. 6 = all array channels, keep only column 0 (XVF-3000 "
         "processed output).",
         "channels: 1 makes PipeWire DOWNMIX all six — raw capsules AND the ch5 "
         "playback reference — more than doubling own-TTS bleed (measured 2026-07-06).",
         min=1, max=8, step=1, danger=True),
    Knob("core.audio.use_channel", AUDIO, "Capture column", "int",
         "STRUCTURAL. Which captured column the kiosk consumes; 0 = beamformed + "
         "hardware AEC + NS.", min=0, max=7, step=1, danger=True),
    Knob("core.audio.chunk_size", AUDIO, "Chunk size (frames)", "int",
         "STRUCTURAL. Capture chunk; 480 = 30ms at 16kHz.",
         min=160, max=1920, step=160, danger=True),
)

BY_PATH: dict[str, Knob] = {k.path: k for k in KNOBS}


def as_json() -> list[dict]:
    rows = []
    for k in KNOBS:
        d = asdict(k)
        d["choices"] = list(k.choices)
        rows.append(d)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/tune/test_knobs.py -v`
Expected: all PASS. If a range/kind assertion fails, fix the REGISTRY (the config file is the truth).

- [ ] **Step 5: Commit**

```bash
git add tune/knobs.py tests/tune/test_knobs.py
git commit -m "feat(tune): knob registry with drift meta-tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `tune/kiosk_proc.py` — child-process manager

**Files:**
- Create: `tune/kiosk_proc.py`
- Test: `tests/tune/test_kiosk_proc.py`

**Interfaces:**
- Produces: `KioskProcess(cmd=None, cwd=None, term_grace_s=5.0, ring_size=2000, foreign_pids=None)` with `start(diag: bool)`, `stop()`, `restart()`, `status() -> dict`, `attach() -> tuple[list[str], queue.Queue]`, `detach(q)`; `KioskProcError(Exception)`. Task 4's server holds one instance. `foreign_pids` is a callable injection point for tests (production default scans pgrep).

- [ ] **Step 1: Write the failing tests**

Create `tests/tune/test_kiosk_proc.py`:

```python
"""KioskProcess lifecycle against fake children (bash one-liners): start/stop/
restart, TERM-then-KILL, foreign-process refusal, ring buffer + SSE fan-out,
exit announcement. Nothing here touches the real kiosk."""

import queue
import time

import pytest

from tune.kiosk_proc import KioskProcError, KioskProcess


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _proc(script, **kw):
    kw.setdefault("foreign_pids", lambda: [])
    return KioskProcess(cmd=["bash", "-c", script], **kw)


def test_start_stop_lifecycle():
    p = _proc("sleep 30")
    p.start(diag=False)
    st = p.status()
    assert st["running"] is True and isinstance(st["pid"], int)
    p.stop()
    assert p.status() == {"running": False, "pid": None, "diag": False}


def test_double_start_refused():
    p = _proc("sleep 30")
    p.start(diag=False)
    try:
        with pytest.raises(KioskProcError, match="already running"):
            p.start(diag=False)
    finally:
        p.stop()


def test_foreign_kiosk_refused():
    p = KioskProcess(cmd=["bash", "-c", "sleep 30"], foreign_pids=lambda: [4242])
    with pytest.raises(KioskProcError, match="4242"):
        p.start(diag=False)
    assert p.status()["running"] is False


def test_stop_escalates_to_kill_when_term_ignored():
    p = _proc("trap '' TERM; echo up; sleep 60", term_grace_s=0.3)
    p.start(diag=False)
    snapshot, q = p.attach()
    assert _wait(lambda: "up" in "\n".join(p.attach()[0]))
    t0 = time.monotonic()
    p.stop()
    assert time.monotonic() - t0 < 5.0          # grace 0.3 + KILL, not 60s
    assert p.status()["running"] is False


def test_output_lands_in_ring_ansi_stripped():
    p = _proc(r"printf '\033[1mhello\033[0m world\n'; sleep 30")
    p.start(diag=False)
    try:
        assert _wait(lambda: any("hello world" == l for l in p.attach()[0]))
    finally:
        p.stop()


def test_exit_is_announced_with_code():
    p = _proc("echo bye; exit 3")
    p.start(diag=False)
    assert _wait(lambda: any("exited (code 3)" in l for l in p.attach()[0]))
    assert p.status()["running"] is False


def test_attach_replays_ring_then_streams_live():
    p = _proc("echo first; sleep 0.3; echo second; sleep 30")
    p.start(diag=False)
    try:
        assert _wait(lambda: any("first" in l for l in p.attach()[0]))
        snapshot, q = p.attach()
        assert any("first" in l for l in snapshot)
        line = q.get(timeout=5.0)
        assert "second" in line
        p.detach(q)
    finally:
        p.stop()


def test_diag_env_reaches_the_child():
    p = _proc('echo "diag=${TVAD_DIAG:-unset}"; sleep 30')
    p.start(diag=True)
    try:
        assert _wait(lambda: any("diag=1" in l for l in p.attach()[0]))
        assert p.status()["diag"] is True
    finally:
        p.stop()
    p2 = _proc('echo "diag=${TVAD_DIAG:-unset}"; sleep 30')
    p2.start(diag=False)
    try:
        assert _wait(lambda: any("diag=unset" in l for l in p2.attach()[0]))
    finally:
        p2.stop()


def test_restart_reuses_last_diag_flag():
    p = _proc('echo "diag=${TVAD_DIAG:-unset}"; sleep 30')
    p.start(diag=True)
    p.restart()
    try:
        assert p.status() == {"running": True, "pid": p.status()["pid"], "diag": True}
        assert _wait(lambda: "\n".join(p.attach()[0]).count("diag=1") >= 2)
    finally:
        p.stop()


def test_ring_is_bounded():
    p = _proc("for i in $(seq 1 50); do echo line$i; done; sleep 30", ring_size=10)
    p.start(diag=False)
    try:
        assert _wait(lambda: any("line50" in l for l in p.attach()[0]))
        snapshot, q = p.attach()
        p.detach(q)
        assert len(snapshot) <= 10
        assert not any("line1" == l for l in snapshot)
    finally:
        p.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/tune/test_kiosk_proc.py -v`
Expected: `ModuleNotFoundError: No module named 'tune.kiosk_proc'`.

- [ ] **Step 3: Implement**

Create `tune/kiosk_proc.py`:

```python
"""Child-process manager for kiosk.py.

The tuning server owns exactly one kiosk child: Start/Stop/Restart with
TERM-then-KILL semantics (mirrors kiosk-stack.sh term_then_kill), stdout+stderr
pumped by a reader thread into a bounded ring and fanned out to SSE
subscribers. Never starts over a foreign `kiosk.py --talkback` (same
pgrep + /proc/<pid>/comm guard as the stack script); never orphans the
child — the server calls stop() on shutdown."""

from __future__ import annotations

import collections
import os
import queue
import re
import subprocess
import sys
import threading

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class KioskProcError(Exception):
    pass


def default_foreign_pids() -> list[int]:
    """PIDs of python kiosk.py --talkback processes we did not start."""
    out = subprocess.run(["pgrep", "-f", r"kiosk\.py --talkback"],
                         capture_output=True, text=True)
    pids = []
    for tok in out.stdout.split():
        try:
            pid = int(tok)
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
        except (ValueError, OSError):
            continue
        if comm.startswith("python"):
            pids.append(pid)
    return pids


class KioskProcess:
    def __init__(self, cmd=None, cwd=None, term_grace_s=5.0, ring_size=2000,
                 foreign_pids=None):
        self._cmd = list(cmd) if cmd else [sys.executable, "kiosk.py", "--talkback"]
        self._cwd = cwd
        self._grace = term_grace_s
        self._foreign_pids = foreign_pids or default_foreign_pids
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._ring: collections.deque[str] = collections.deque(maxlen=ring_size)
        self._subs: list[queue.Queue] = []
        self._diag = False

    # ---- lifecycle ----

    def start(self, diag: bool) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise KioskProcError(
                    f"kiosk already running (pid {self._proc.pid})")
            own = self._proc.pid if self._proc is not None else None
            foreign = [p for p in self._foreign_pids() if p != own]
            if foreign:
                raise KioskProcError(
                    "a kiosk.py --talkback this server did not start is running "
                    f"(pid {foreign[0]}); stop it first (kiosk-stack.sh stop)")
            env = dict(os.environ,
                       PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
            env.pop("TVAD_DIAG", None)
            if diag:
                env["TVAD_DIAG"] = "1"
            self._proc = subprocess.Popen(
                self._cmd, cwd=self._cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1)
            self._diag = diag
            self._reader = threading.Thread(
                target=self._pump, args=(self._proc,),
                name="kiosk-pump", daemon=True)
            self._reader.start()

    def stop(self) -> None:
        with self._lock:
            proc, reader = self._proc, self._reader
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._proc = None
            return
        proc.terminate()
        try:
            proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if reader is not None:
            reader.join(timeout=5.0)
        with self._lock:
            self._proc = None

    def restart(self) -> None:
        diag = self._diag
        self.stop()
        self.start(diag=diag)

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {"running": running,
                    "pid": self._proc.pid if running else None,
                    "diag": self._diag if running else False}

    # ---- log fan-out ----

    def attach(self) -> tuple[list[str], queue.Queue]:
        """Snapshot of the ring + a live queue, atomically (no gap/dup)."""
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            snapshot = list(self._ring)
            self._subs.append(q)
        return snapshot, q

    def detach(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _emit(self, line: str) -> None:
        with self._lock:
            self._ring.append(line)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass  # slow client: it still has the ring on reconnect

    def _pump(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            self._emit(_ANSI_RE.sub("", raw.rstrip("\n")))
        code = proc.wait()
        self._emit(f"[tune] kiosk exited (code {code})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/tune/test_kiosk_proc.py -v`
Expected: all PASS (the KILL-escalation test takes ~0.5s, none should hang).

- [ ] **Step 5: Commit**

```bash
git add tune/kiosk_proc.py tests/tune/test_kiosk_proc.py
git commit -m "feat(tune): kiosk child-process manager with SSE fan-out ring

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `tune/server.py` + `tune/__main__.py` — HTTP API

**Files:**
- Create: `tune/server.py`
- Create: `tune/__main__.py`
- Test: `tests/tune/test_server_api.py`

**Interfaces:**
- Consumes: `tune.knobs.BY_PATH`, `tune.knobs.as_json` (Task 2); `tune.config_edit.set_values`, `get_path`, `ConfigEditError` (Task 1); `tune.kiosk_proc.KioskProcess`, `KioskProcError` (Task 3).
- Produces: `TuningServer(config_path, kproc, host="127.0.0.1", port=8765, llm_url="http://127.0.0.1:8080/v1/models")` with `.httpd` (a `ThreadingHTTPServer`), `.serve_forever()`, `.shutdown()`, `.port` (resolved — pass `port=0` in tests for an ephemeral port). Task 5's page consumes the JSON API exactly as specified here.

**API contract (spec 4e):** JSON responses; errors `{"error": msg}` with 4xx.
- `GET /` → `tune/static/index.html` (404 until Task 5 creates it — the test only asserts non-500).
- `GET /api/state` → `{"knobs": [ {...knob fields, "value": current} ], "kiosk": {"running","pid","diag"}, "llm": {"reachable"}, "config_path": str}`; values read fresh from disk each call.
- `POST /api/save` body `{"changes": {path: value}}` → 200 `{"saved": [paths]}` | 400 unknown path / bad value | 409 `ConfigEditError`.
- `POST /api/kiosk/start` body `{"diag": bool}` → 200 status dict | 409 `KioskProcError`.
- `POST /api/kiosk/stop` → 200 status dict (idempotent).
- `POST /api/kiosk/restart` → 200 status dict.
- `GET /api/logs` → SSE: every line as `data: <line>\n\n`, ring replayed first, 15s keep-alive comments (`: ping`).

- [ ] **Step 1: Write the failing tests**

Create `tests/tune/test_server_api.py`:

```python
"""API tests against a real TuningServer on an ephemeral port, with a temp
copy of the real config.yaml and a fake kiosk child. The UI is not the trust
boundary: hand-crafted bad POSTs must be rejected server-side."""

import http.client
import json
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

from tune.kiosk_proc import KioskProcess
from tune.server import TuningServer

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture()
def srv(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(REAL_CONFIG, cfg)
    kproc = KioskProcess(cmd=["bash", "-c", "echo kiosk-up; sleep 30"],
                         foreign_pids=lambda: [])
    server = TuningServer(config_path=str(cfg), kproc=kproc, port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server, cfg
    kproc.stop()
    server.shutdown()


def _req(server, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data) if data else None


def test_state_has_knobs_with_fresh_values(srv):
    server, cfg = srv
    status, state = _req(server, "GET", "/api/state")
    assert status == 200
    by_path = {k["path"]: k for k in state["knobs"]}
    assert by_path["kiosk.talkback.turn_gate.doa.cone_deg"]["value"] == 20
    assert state["kiosk"]["running"] is False
    assert "reachable" in state["llm"]
    # values are read fresh: hand-edit the file, state reflects it
    cfg.write_text(cfg.read_text().replace("cone_deg: 20", "cone_deg: 30"))
    _, state2 = _req(server, "GET", "/api/state")
    by_path2 = {k["path"]: k for k in state2["knobs"]}
    assert by_path2["kiosk.talkback.turn_gate.doa.cone_deg"]["value"] == 30


def test_save_writes_file_and_preserves_comments(srv):
    server, cfg = srv
    before = cfg.read_text()
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0}})
    assert status == 200 and out == {"saved": ["kiosk.talkback.turn_gate.doa.cone_deg"]}
    after = cfg.read_text()
    assert yaml.safe_load(after)["kiosk"]["talkback"]["turn_gate"]["doa"]["cone_deg"] == 25.0
    diff = [1 for a, b in zip(before.split("\n"), after.split("\n")) if a != b]
    assert len(diff) == 1


def test_save_rejects_unregistered_path(srv):
    server, cfg = srv
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.output_device": "hax"}})
    assert status == 400 and "output_device" in out["error"]
    assert "hax" not in cfg.read_text()


def test_save_rejects_out_of_range_and_wrong_kind(srv):
    server, cfg = srv
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": 500}})
    assert status == 400
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.reject_bystanders": "yes"}})
    assert status == 400
    status, out = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.wake_phrase": "hey_hacker"}})
    assert status == 400


def test_save_nullable_accepts_null_others_do_not(srv):
    server, cfg = srv
    status, _ = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.barge_in.proximity.rms_threshold": None}})
    assert status == 200
    status, _ = _req(server, "POST", "/api/save", {
        "changes": {"kiosk.talkback.turn_gate.doa.cone_deg": None}})
    assert status == 400


def test_save_all_or_nothing(srv):
    server, cfg = srv
    before = cfg.read_text()
    status, _ = _req(server, "POST", "/api/save", {"changes": {
        "kiosk.talkback.turn_gate.doa.cone_deg": 25.0,
        "kiosk.talkback.output_device": "hax"}})
    assert status == 400
    assert cfg.read_text() == before


def test_kiosk_start_stop_restart_roundtrip(srv):
    server, cfg = srv
    status, st = _req(server, "POST", "/api/kiosk/start", {"diag": True})
    assert status == 200 and st["running"] is True and st["diag"] is True
    status, _ = _req(server, "POST", "/api/kiosk/start", {"diag": True})
    assert status == 409
    status, st = _req(server, "POST", "/api/kiosk/restart", None)
    assert status == 200 and st["running"] is True
    status, st = _req(server, "POST", "/api/kiosk/stop", None)
    assert status == 200 and st["running"] is False
    status, st = _req(server, "POST", "/api/kiosk/stop", None)  # idempotent
    assert status == 200


def test_logs_sse_replays_ring(srv):
    server, cfg = srv
    _req(server, "POST", "/api/kiosk/start", {"diag": False})
    time.sleep(0.5)  # let the echo land in the ring
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as s:
        s.sendall(b"GET /api/logs HTTP/1.1\r\nHost: x\r\n\r\n")
        buf = b""
        deadline = time.monotonic() + 5
        while b"kiosk-up" not in buf and time.monotonic() < deadline:
            buf += s.recv(4096)
    assert b"text/event-stream" in buf
    assert b"data: kiosk-up" in buf


def test_unknown_route_404s_with_json(srv):
    server, cfg = srv
    status, out = _req(server, "GET", "/api/nope")
    assert status == 404 and "error" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/tune/test_server_api.py -v`
Expected: `ModuleNotFoundError: No module named 'tune.server'`.

- [ ] **Step 3: Implement the server**

Create `tune/server.py`:

```python
"""Stdlib HTTP server for the tuning console.

Server-side validation mirrors the knob registry (kind, range, choices,
nullable, strict-bool) — the browser page is NOT the trust boundary for the
file the kiosk boots from. Saves go through config_edit.set_values and land
atomically (temp file + os.replace, same directory)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tune import config_edit, knobs
from tune.config_edit import ConfigEditError
from tune.kiosk_proc import KioskProcError

import yaml

_STATIC = Path(__file__).parent / "static"
_LLM_CACHE_S = 3.0
_SSE_PING_S = 15.0


def validate(knob: knobs.Knob, value) -> str | None:
    """Return an error message, or None if the value is acceptable."""
    if value is None:
        return None if knob.nullable else f"{knob.path}: null not allowed"
    if knob.kind == "bool":
        return None if isinstance(value, bool) else f"{knob.path}: expected true/false"
    if knob.kind in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{knob.path}: expected a number"
        if knob.kind == "int" and not float(value).is_integer():
            return f"{knob.path}: expected an integer"
        if not (knob.min <= value <= knob.max):
            return f"{knob.path}: {value} outside [{knob.min}, {knob.max}]"
        return None
    if knob.kind == "select":
        return None if value in knob.choices else \
            f"{knob.path}: {value!r} not one of {list(knob.choices)}"
    if knob.kind in ("text", "textarea"):
        return None if isinstance(value, str) else f"{knob.path}: expected a string"
    return f"{knob.path}: unknown kind {knob.kind}"


def _coerce(knob: knobs.Knob, value):
    """JSON numbers arrive as int OR float; land them as the knob's kind."""
    if value is None or knob.kind not in ("float", "int"):
        return value
    return int(value) if knob.kind == "int" else float(value)


class TuningServer:
    def __init__(self, config_path: str, kproc, host: str = "127.0.0.1",
                 port: int = 8765,
                 llm_url: str = "http://127.0.0.1:8080/v1/models"):
        self.config_path = os.path.abspath(config_path)
        self.kproc = kproc
        self.llm_url = llm_url
        self._llm_cache = (0.0, False)
        self._save_lock = threading.Lock()
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet: the log pane is the product
                pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    return self._index()
                if self.path == "/api/state":
                    return self._json(200, server_ref.state())
                if self.path == "/api/logs":
                    return self._sse()
                self._json(404, {"error": f"no route: {self.path}"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"null")
                except json.JSONDecodeError as e:
                    return self._json(400, {"error": f"bad JSON: {e}"})
                if self.path == "/api/save":
                    status, payload = server_ref.save(body)
                    return self._json(status, payload)
                if self.path == "/api/kiosk/start":
                    return self._kiosk(lambda: server_ref.kproc.start(
                        diag=bool((body or {}).get("diag", True))))
                if self.path == "/api/kiosk/stop":
                    return self._kiosk(server_ref.kproc.stop)
                if self.path == "/api/kiosk/restart":
                    return self._kiosk(server_ref.kproc.restart)
                self._json(404, {"error": f"no route: {self.path}"})

            def _kiosk(self, action):
                try:
                    action()
                except KioskProcError as e:
                    return self._json(409, {"error": str(e)})
                self._json(200, server_ref.kproc.status())

            def _index(self):
                page = _STATIC / "index.html"
                if not page.exists():
                    return self._json(404, {"error": "index.html not built yet"})
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse(self):
                snapshot, q = server_ref.kproc.attach()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for line in snapshot:
                        self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            line = q.get(timeout=_SSE_PING_S)
                            self.wfile.write(f"data: {line}\n\n".encode())
                        except Exception:  # queue.Empty -> keep-alive
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server_ref.kproc.detach(q)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    # ---- app logic (handler-independent, easy to test) ----

    def state(self) -> dict:
        data = yaml.safe_load(Path(self.config_path).read_text())
        rows = knobs.as_json()
        for row in rows:
            row["value"] = config_edit.get_path(data, row["path"])
        return {"knobs": rows, "kiosk": self.kproc.status(),
                "llm": {"reachable": self._llm_reachable()},
                "config_path": self.config_path}

    def save(self, body) -> tuple[int, dict]:
        changes = (body or {}).get("changes")
        if not isinstance(changes, dict) or not changes:
            return 400, {"error": "body must be {\"changes\": {path: value}}"}
        coerced = {}
        for path, value in changes.items():
            knob = knobs.BY_PATH.get(path)
            if knob is None:
                return 400, {"error": f"not a registered knob: {path}"}
            err = validate(knob, value)   # validate the RAW value first —
            if err:                       # coercing 2.5 -> 2 would hide the error
                return 400, {"error": err}
            coerced[path] = _coerce(knob, value)
        with self._save_lock:
            text = Path(self.config_path).read_text()
            try:
                edited = config_edit.set_values(text, coerced)
            except ConfigEditError as e:
                return 409, {"error": str(e)}
            self._write_atomic(edited)
        return 200, {"saved": sorted(coerced)}

    def _write_atomic(self, text: str) -> None:
        d = os.path.dirname(self.config_path)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".config-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, self.config_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _llm_reachable(self) -> bool:
        ts, val = self._llm_cache
        if time.monotonic() - ts < _LLM_CACHE_S:
            return val
        try:
            with urllib.request.urlopen(self.llm_url, timeout=0.5):
                val = True
        except Exception:
            val = False
        self._llm_cache = (time.monotonic(), val)
        return val

    def serve_forever(self):
        self.httpd.serve_forever()

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
```

Create `tune/__main__.py`:

```python
"""python3 -m tune [--port 8765] [--host 127.0.0.1] [--config config.yaml]"""

import argparse
import os

from tune.kiosk_proc import KioskProcess
from tune.server import TuningServer


def main():
    ap = argparse.ArgumentParser(description="Kiosk tuning console")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to tune from another machine on the LAN")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    kproc = KioskProcess(cwd=os.path.dirname(config_path))
    server = TuningServer(config_path=config_path, kproc=kproc,
                          host=args.host, port=args.port)
    print(f"[tune] console at http://{args.host}:{server.port}/  "
          f"(config: {config_path})")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/tune/test_server_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole tune suite**

Run: `python3 -m pytest tests/tune -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tune/server.py tune/__main__.py tests/tune/test_server_api.py
git commit -m "feat(tune): HTTP API — state/save/kiosk lifecycle/SSE logs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `tune/static/index.html` — the page

**Files:**
- Create: `tune/static/index.html`
- Test: extend `tests/tune/test_server_api.py` (one test)

**Interfaces:**
- Consumes: the exact API contract from Task 4 (`/api/state`, `/api/save`, `/api/kiosk/*`, `/api/logs` SSE).
- Produces: the served page. No JS framework, no build step, inline CSS/JS, dark theme.

**Behavior checklist (spec 4f)** — the implementation below covers all of it:
- Tab bar from the knobs' `tab` fields, in registry order.
- Knob row: label · control · dirty dot · `doc`; `why` behind a ⓘ expander; `danger` knobs get an amber left border.
- Numerics = range slider + exact number input, kept in sync (sliders alone can't hit 0.15).
- Bools = checkbox toggles; selects = dropdowns; `textarea` for the system prompt; nullable numerics get an "auto (null)" checkbox that disables the inputs.
- Save bar appears when dirty; lists `path: old → new`; POSTs only changed paths; 4xx shows the server's message and keeps edits; "Revert edits" reloads saved values (page-local).
- Kiosk strip: ● running (pid) / ○ stopped, LLM dot, DIAG checkbox, Start/Stop/Restart buttons; after a save while running shows "config changed — restart to apply".
- Log pane: monospace, auto-scroll with pause-on-scroll-up, Clear button, highlight for `[DIAG`, `REJECT=`, `[WAKE]`, `[SESSION STARTED]`, `[SESSION ENDED]`, `Traceback`.
- State poll every 3s updates kiosk/LLM status and non-dirty values only.

- [ ] **Step 1: Write the failing test**

Append to `tests/tune/test_server_api.py`:

```python
def test_index_served_with_expected_ui_hooks(srv):
    server, cfg = srv
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    assert resp.status == 200
    for hook in ("id=\"tabs\"", "id=\"panes\"", "id=\"savebar\"",
                 "id=\"logpane\"", "/api/state", "/api/save", "/api/logs",
                 "/api/kiosk/"):
        assert hook in body, hook
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tune/test_server_api.py::test_index_served_with_expected_ui_hooks -v`
Expected: FAIL — 404 `index.html not built yet`.

- [ ] **Step 3: Create the page**

Create `tune/static/index.html` with exactly this content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kiosk Tuning Console</title>
<style>
  :root {
    --bg:#101418; --panel:#171d24; --line:#2a333d; --fg:#d7dde3;
    --dim:#8a95a1; --accent:#4fb6ff; --ok:#41c98d; --warn:#e0a93e;
    --bad:#e05c5c; --mono:ui-monospace,'Cascadia Code',Menlo,monospace;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 system-ui,sans-serif; display:flex;
         flex-direction:column; height:100vh; }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:15px; margin:0; }
  header .path { color:var(--dim); font-family:var(--mono); font-size:12px; }
  #tabs { display:flex; gap:2px; padding:0 12px; border-bottom:1px solid var(--line); }
  #tabs button { background:none; border:none; border-bottom:2px solid transparent;
    color:var(--dim); padding:8px 12px; cursor:pointer; font:inherit; }
  #tabs button.active { color:var(--fg); border-bottom-color:var(--accent); }
  #panes { flex:1 1 45%; overflow-y:auto; padding:12px 16px; }
  .knob { display:grid; grid-template-columns:230px 1fr 14px; gap:4px 14px;
          padding:9px 8px; border-bottom:1px solid var(--line); align-items:center; }
  .knob.danger { border-left:3px solid var(--warn); padding-left:10px; }
  .knob label.name { font-weight:600; }
  .knob .doc { grid-column:1 / -1; color:var(--dim); font-size:12.5px; }
  .knob .why { grid-column:1 / -1; color:var(--warn); font-size:12.5px;
               display:none; white-space:pre-wrap; }
  .knob.openwhy .why { display:block; }
  .ctrl { display:flex; gap:10px; align-items:center; }
  .ctrl input[type=range] { flex:1; accent-color:var(--accent); }
  .ctrl input[type=number] { width:90px; }
  input, select, textarea { background:var(--panel); color:var(--fg);
    border:1px solid var(--line); border-radius:4px; padding:4px 6px; font:inherit; }
  textarea { width:100%; min-height:70px; font-family:var(--mono); font-size:12.5px; }
  .dirty { width:8px; height:8px; border-radius:50%; background:transparent; }
  .knob.isdirty .dirty { background:var(--accent); }
  .whybtn { background:none; border:none; color:var(--dim); cursor:pointer; }
  #savebar { display:none; border-top:1px solid var(--line);
    background:var(--panel); padding:8px 16px; }
  #savebar.show { display:block; }
  #savebar .changes { font-family:var(--mono); font-size:12px; color:var(--dim);
    max-height:70px; overflow-y:auto; margin-bottom:6px; }
  #savebar .err { color:var(--bad); font-size:12.5px; margin-left:10px; }
  button.primary { background:var(--accent); color:#04212f; border:none;
    border-radius:4px; padding:5px 14px; font:inherit; font-weight:600; cursor:pointer; }
  button.ghost { background:none; border:1px solid var(--line); color:var(--fg);
    border-radius:4px; padding:5px 12px; font:inherit; cursor:pointer; }
  #kioskstrip { display:flex; align-items:center; gap:12px; padding:8px 16px;
    border-top:1px solid var(--line); background:var(--panel); }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
  .dot.on { background:var(--ok); } .dot.off { background:var(--dim); }
  .dot.bad { background:var(--bad); }
  #restarthint { color:var(--warn); display:none; }
  #logpane { flex:1 1 35%; overflow-y:auto; background:#0b0e12;
    font-family:var(--mono); font-size:12px; padding:8px 12px;
    border-top:1px solid var(--line); white-space:pre-wrap; }
  .l-diag { color:#7fb3d5; } .l-reject { color:var(--bad); }
  .l-wake { color:#c39be0; } .l-session { color:var(--ok); }
  .l-trace { color:var(--warn); }
</style>
</head>
<body>
<header>
  <h1>Kiosk Tuning Console</h1>
  <span class="path" id="cfgpath"></span>
</header>
<nav id="tabs"></nav>
<main id="panes"></main>
<div id="savebar">
  <div class="changes" id="changelist"></div>
  <button class="primary" id="savebtn">Save</button>
  <button class="ghost" id="revertbtn">Revert edits</button>
  <span class="err" id="saveerr"></span>
</div>
<div id="kioskstrip">
  <span><span class="dot off" id="kioskdot"></span> <span id="kiosklabel">kiosk stopped</span></span>
  <span><span class="dot off" id="llmdot"></span> LLM</span>
  <label><input type="checkbox" id="diagbox" checked> DIAG</label>
  <button class="ghost" id="startbtn">Start</button>
  <button class="ghost" id="stopbtn">Stop</button>
  <button class="primary" id="restartbtn">Restart</button>
  <span id="restarthint">config changed — restart to apply</span>
  <span style="flex:1"></span>
  <button class="ghost" id="clearlog">Clear log</button>
</div>
<pre id="logpane"></pre>
<script>
"use strict";
const $ = id => document.getElementById(id);
let KNOBS = [], SAVED = {}, EDITS = {}, activeTab = null, autoscroll = true;

const norm = v => (typeof v === "number" ? Math.round(v * 1e9) / 1e9 : v);
const isDirty = p => p in EDITS && norm(EDITS[p]) !== norm(SAVED[p]);
const dirtyPaths = () => Object.keys(EDITS).filter(isDirty);

async function api(method, path, body) {
  const r = await fetch(path, {method, headers: {"Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.status);
  return data;
}

function buildTabs() {
  const tabs = [...new Set(KNOBS.map(k => k.tab))];
  $("tabs").innerHTML = "";
  tabs.forEach(t => {
    const b = document.createElement("button");
    b.textContent = t;
    b.onclick = () => { activeTab = t; render(); };
    b.className = t === activeTab ? "active" : "";
    $("tabs").appendChild(b);
  });
  if (!activeTab) activeTab = tabs[0];
}

function control(k, cur) {
  const wrap = document.createElement("div");
  wrap.className = "ctrl";
  const set = v => { EDITS[k.path] = v; refreshDirty(k.path); };
  if (k.kind === "float" || k.kind === "int") {
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = k.min; slider.max = k.max; slider.step = k.step;
    const num = document.createElement("input");
    num.type = "number"; num.min = k.min; num.max = k.max; num.step = k.step;
    const sync = v => { slider.value = v; num.value = v; };
    slider.oninput = () => { sync(slider.value); set(Number(slider.value)); };
    num.oninput = () => { sync(num.value); set(Number(num.value)); };
    let auto = null;
    if (k.nullable) {
      auto = document.createElement("label");
      const cb = document.createElement("input"); cb.type = "checkbox";
      cb.checked = cur === null;
      cb.onchange = () => {
        slider.disabled = num.disabled = cb.checked;
        set(cb.checked ? null : Number(num.value || k.min));
      };
      auto.append(cb, " auto (null)");
      slider.disabled = num.disabled = cur === null;
    }
    sync(cur === null ? k.min : cur);
    wrap.append(slider, num); if (auto) wrap.append(auto);
  } else if (k.kind === "bool") {
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = cur === true;
    cb.onchange = () => set(cb.checked);
    wrap.append(cb);
  } else if (k.kind === "select") {
    const sel = document.createElement("select");
    k.choices.forEach(c => {
      const o = document.createElement("option");
      o.value = c; o.textContent = c; o.selected = c === cur;
      sel.appendChild(o);
    });
    sel.onchange = () => set(sel.value);
    wrap.append(sel);
  } else if (k.kind === "textarea") {
    const ta = document.createElement("textarea");
    ta.value = cur ?? ""; ta.oninput = () => set(ta.value);
    wrap.append(ta);
  } else {
    const inp = document.createElement("input");
    inp.type = "text"; inp.value = cur ?? ""; inp.oninput = () => set(inp.value);
    wrap.append(inp);
  }
  return wrap;
}

function render() {
  buildTabs();
  const pane = $("panes"); pane.innerHTML = "";
  KNOBS.filter(k => k.tab === activeTab).forEach(k => {
    const row = document.createElement("div");
    row.className = "knob" + (k.danger ? " danger" : "");
    row.dataset.path = k.path;
    const name = document.createElement("label");
    name.className = "name"; name.textContent = k.label;
    if (k.why) {
      const b = document.createElement("button");
      b.className = "whybtn"; b.textContent = " ⓘ";
      b.onclick = () => row.classList.toggle("openwhy");
      name.appendChild(b);
    }
    const cur = k.path in EDITS ? EDITS[k.path] : SAVED[k.path];
    const dot = document.createElement("span"); dot.className = "dirty";
    row.append(name, control(k, cur), dot);
    const doc = document.createElement("div");
    doc.className = "doc"; doc.textContent = k.doc; row.append(doc);
    if (k.why) {
      const why = document.createElement("div");
      why.className = "why"; why.textContent = k.why; row.append(why);
    }
    pane.append(row);
    refreshDirty(k.path);
  });
}

function refreshDirty(path) {
  const row = document.querySelector(`.knob[data-path="${path}"]`);
  if (row) row.classList.toggle("isdirty", isDirty(path));
  const dirty = dirtyPaths();
  $("savebar").classList.toggle("show", dirty.length > 0);
  $("changelist").textContent = dirty.map(p =>
    `${p}: ${JSON.stringify(SAVED[p])} → ${JSON.stringify(EDITS[p])}`).join("\n");
}

$("savebtn").onclick = async () => {
  const changes = {};
  dirtyPaths().forEach(p => changes[p] = EDITS[p]);
  $("saveerr").textContent = "";
  try {
    await api("POST", "/api/save", {changes});
    Object.assign(SAVED, changes);
    EDITS = {};
    render();
    if (lastKiosk.running) $("restarthint").style.display = "inline";
  } catch (e) { $("saveerr").textContent = e.message; }
};
$("revertbtn").onclick = () => { EDITS = {}; $("saveerr").textContent = ""; render(); };

let lastKiosk = {running: false};
function applyState(st) {
  $("cfgpath").textContent = st.config_path;
  st.knobs.forEach(k => { SAVED[k.path] = k.value; });
  KNOBS = st.knobs;
  lastKiosk = st.kiosk;
  $("kioskdot").className = "dot " + (st.kiosk.running ? "on" : "off");
  $("kiosklabel").textContent = st.kiosk.running
    ? `kiosk running (pid ${st.kiosk.pid}${st.kiosk.diag ? ", DIAG" : ""})`
    : "kiosk stopped";
  $("llmdot").className = "dot " + (st.llm.reachable ? "on" : "bad");
  if (!st.kiosk.running) $("restarthint").style.display = "none";
}

async function poll(rerender) {
  try {
    const st = await api("GET", "/api/state");
    applyState(st);
    if (rerender) render();
  } catch (e) { /* server briefly away; next poll retries */ }
}

async function kiosk(action) {
  try {
    await api("POST", `/api/kiosk/${action}`,
      action === "start" ? {diag: $("diagbox").checked} : {});
    $("restarthint").style.display = "none";
    await poll(false);
  } catch (e) { logLine(`[tune] ${e.message}`); }
}
$("startbtn").onclick = () => kiosk("start");
$("stopbtn").onclick = () => kiosk("stop");
$("restartbtn").onclick = () => kiosk("restart");
$("clearlog").onclick = () => { $("logpane").innerHTML = ""; };

const HILITE = [
  [/REJECT=|\[EJECT/, "l-reject"], [/\[DIAG/, "l-diag"], [/\[WAKE\]/, "l-wake"],
  [/\[SESSION (STARTED|ENDED)\]/, "l-session"], [/Traceback/, "l-trace"],
];
function logLine(text) {
  const div = document.createElement("div");
  for (const [re, cls] of HILITE) if (re.test(text)) { div.className = cls; break; }
  div.textContent = text;
  const pane = $("logpane");
  pane.appendChild(div);
  while (pane.childNodes.length > 3000) pane.removeChild(pane.firstChild);
  if (autoscroll) pane.scrollTop = pane.scrollHeight;
}
$("logpane").onscroll = () => {
  const p = $("logpane");
  autoscroll = p.scrollTop + p.clientHeight >= p.scrollHeight - 8;
};
new EventSource("/api/logs").onmessage = ev => logLine(ev.data);

poll(true).then(() => setInterval(() => poll(false), 3000));
</script>
</body>
</html>
```

Note the 3s poll deliberately calls `poll(false)` — it refreshes kiosk/LLM status and `SAVED` values but never re-renders the panes, so a dirty control is never clobbered mid-drag (spec section 5).

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/tune/test_server_api.py -v`
Expected: all PASS, including `test_index_served_with_expected_ui_hooks`.

- [ ] **Step 5: Full suite**

Run: `python3 -m pytest -q`
Expected: everything passes.

- [ ] **Step 6: Live smoke (spec section 8 — needs the box, ~3 minutes)**

```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m tune --port 8765
```

Then in a browser at `http://127.0.0.1:8765/`:
1. Change `Cone half-width` 20 → 25, Save → `git diff config.yaml` shows exactly one changed line, comments intact; change it back and Save.
2. If the LLM is up (`./kiosk-stack.sh status`): Start → DIAG lines stream into the log pane. Stop → `pgrep -f 'kiosk.py --talkback'` is empty.
3. Ctrl-C the tune server while the kiosk runs → kiosk is stopped too.

Record the results in the commit message (what was verified, what wasn't — e.g. if the LLM was down, say the kiosk-start check is pending).

- [ ] **Step 7: Commit**

```bash
git add tune/static/index.html tests/tune/test_server_api.py
git commit -m "feat(tune): tuning console page — tabs, save bar, kiosk strip, live log

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done Criteria

- `python3 -m pytest -q` fully green.
- `python3 -m tune` serves the console; the five-tab-plus-three loop works live per Task 5 Step 6.
- `git diff config.yaml` after any UI save shows only value spans changed.
- No new pip dependencies (`git diff requirements.txt` empty).
