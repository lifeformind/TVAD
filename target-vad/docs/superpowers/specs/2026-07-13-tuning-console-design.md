# Kiosk Tuning Console — Design

**Date:** 2026-07-13
**Branch:** `feat/tuning-ui` (stacked on `feat/director-11-doa-cone-vote` — the DOA
knobs only exist there; merges after D11 lands)
**Status:** approved design, pre-plan

## 1. Problem

Tuning the kiosk today means: edit `config.yaml` by hand in an editor, re-run
`TVAD_DIAG=1 ./kiosk-stack.sh start` in a terminal, watch the DIAG stream, repeat.
The knob surface is large (~50 keys across wake, enrollment, turn gate, DOA cone,
barge-in, presence, timing, voice pipeline) and the rationale for each value lives
in comments scattered through the file. The 2026-07-07 D11 live-gate evening was
five runs of this loop; more evenings like it are coming.

## 2. Goal

One browser tab that holds the whole tuning loop:

> tweak knobs (organized in tabs by kind of tweak) → **Save** (writes
> `config.yaml`, comments untouched) → **Restart kiosk** → watch the live DIAG
> stream in a log pane → repeat.

## 3. Decisions (user-confirmed)

| Question | Decision |
|---|---|
| Apply model | Edit `config.yaml` + restart. No hot-apply — values are snapshotted into `DirectorConfig` at session build anyway. |
| Scope | Kiosk UX only: `core.vad`, `core.speaker`, `core.audio`, and the whole `kiosk` / `kiosk.talkback` tree. The offline pipeline sections (`diarization`, `transcription`, `sentiment`, `metrics`, `prosody`) are out. |
| Form | Local web UI. Stdlib Python server + one static HTML page, vanilla JS. **No new dependencies.** |
| Save strategy | Targeted line edits — only the scalar on each known key's line changes; every comment and all formatting stay byte-identical. Ambiguity ⇒ refuse the save. |
| Kiosk control | The tuning server owns the kiosk as a child process: Start / Stop / Restart buttons, stdout streamed to a live log pane in the page. LLM stays `kiosk-stack.sh`'s job. |
| Extras declined | Diff-before-save, snapshots/revert, preset profiles. |

## 4. Architecture

New top-level package `tune/`:

```
tune/
  __init__.py
  __main__.py       # python3 -m tune [--port 8765] [--config config.yaml]
  knobs.py          # the knob registry (single source of truth)
  config_edit.py    # targeted line editor for config.yaml
  kiosk_proc.py     # child-process manager for kiosk.py
  server.py         # stdlib ThreadingHTTPServer + handlers
  static/index.html # the page (inline CSS + JS, no build step)
```

The server binds `127.0.0.1:8765` by default (`--host 0.0.0.0` opt-in for
LAN use; LLM owns 8080). No auth — localhost/LAN-trust, explicitly out of scope.

### 4a. `knobs.py` — knob registry

```python
@dataclass(frozen=True)
class Knob:
    path: str          # dotted config path, e.g. "kiosk.talkback.turn_gate.doa.cone_deg"
    tab: str           # tab title
    label: str         # short human name
    kind: str          # "float" | "int" | "bool" | "select" | "text" | "textarea"
    doc: str           # one-line what-it-does
    why: str = ""      # tuning history / rationale (distilled from config.yaml comments)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple = ()        # for kind="select"
    nullable: bool = False     # e.g. proximity.rms_threshold (null = auto-calibrate)
    strict_bool: bool = False  # writes only real true/false (the 'flase' lesson)
    danger: bool = False       # structural — styled with a warning
KNOBS: tuple[Knob, ...] = (...)
```

- The registry is data, not behavior: the server serializes it to JSON for the
  page; `config_edit` consumes only `path`.
- `strict_bool` knobs render as toggles and serialize to literal `true`/`false`.
- `nullable` numeric knobs render with an "auto" checkbox that writes `null`.
- `why` carries the live-tuned history forward (e.g. speaker_threshold: "0.30 sat
  inside the owner band → two false ejects; 0.15 = midpoint of the live gap").

### 4b. Tabs and knob inventory

Prefix `tb.` = `kiosk.talkback.`, `tg.` = `kiosk.talkback.turn_gate.`

| Tab | Knobs |
|---|---|
| **Wake & Enrollment** | `kiosk.wake_phrase` (select: hey_mycroft / hey_jarvis / alexa), `kiosk.wake_threshold`, `kiosk.awaiting_speech_timeout_s`, `tb.verify_before_serve_threshold`, `core.speaker.threshold`, `core.speaker.min_segment_duration_ms`, `core.speaker.enrollment_utterances`, `core.speaker.enrollment_min_self_similarity`, `core.speaker.enrollment_max_retries` |
| **Turn Gate** | `tg.require_speaker_match` (strict), `tg.speaker_threshold`, `tg.verify_window_ms`, `tg.endpoint_threshold`, `tg.reject_bystanders` (strict), `tg.lockout.enabled`, `tg.lockout.window_size`, `tg.lockout.min_matches`, `tb.lockout_idle_after_s` |
| **DOA Cone** | `tg.doa.enabled` (strict), `tg.doa.cone_deg`, `tg.doa.poll_ms`, `tg.doa.bearing_ema_alpha`, `tg.doa.min_in_cone_fraction`, `tg.doa.min_in_cone_samples` |
| **Barge-in & Duck** | `tb.barge_in.enabled`, `.min_speech_ms`, `.speaker_threshold`, `.conf_floor`, `.verify_window_ms`, `.duck_level`, `.duck_ramp_ms`, `.proximity.enabled`, `.proximity.rms_threshold` (nullable), `.proximity.rms_factor`, `.proximity.max_floor` |
| **Presence (Camera)** | `tb.vision.enabled`, `.identity_threshold`, `.min_area_frac`, `.present_after_s`, `.absent_after_s`, `.owner_absent_grace_s`, `.active_talk_guard_s`, `.enroll_frames`, `.fps` |
| **Session & Timing** | `tb.silence_timeout_s`, `tb.hard_timeout_s`, `tb.nudge_lead_s`, `tb.watchdog.tick_ms` |
| **Voice Pipeline** | `tb.stt.model` (select: tiny/base.en/small.en/medium.en), `tb.stt.end_of_utterance_tail_ms`, `tb.llm.temperature`, `tb.llm.max_tokens`, `tb.llm.system_prompt` (textarea), `tb.tts.voice`, `tb.chunker.max_chunk_chars` |
| **Audio & VAD ⚠** | `core.vad.speech_threshold`, `.min_speech_duration_ms`, `.padding_ms`, `core.audio.channels` (danger), `core.audio.use_channel` (danger), `core.audio.chunk_size` (danger) — editable, styled as structural danger with the load-bearing comments surfaced (`channels: 6` / `use_channel: 0` = XVF-3000 processed output) |

Not exposed: sample rates (three coupled keys; changing one alone breaks capture),
device pins (`output_device`, `device_index`, `camera_index` — setup-time, not
tuning), backend selectors (`stt.backend`, `tts.backend`, `llm.base_url/model` —
change with code, not per-run), `aec.enabled` and `crowd_focus.*` (dormant paths),
`paths.*`, `logging.*`, `frame_ms`, `chunker.sentence_terminators`.

### 4c. `config_edit.py` — targeted line editor

`set_values(text: str, changes: dict[str, object]) -> str` — pure function,
all-or-nothing:

1. **Locate**: walk `text` line by line, tracking the current key path via
   indentation (2-space YAML, comment/blank lines skipped for path tracking).
   The target is the unique line whose accumulated path equals the knob path.
   Zero or multiple matches ⇒ `ConfigEditError` (save aborted, nothing written).
2. **Replace**: on that line, substitute only the span between `key:` and the
   inline ` #` comment (if any), preserving surrounding whitespace. Values are
   rendered as YAML scalars: bools → `true`/`false`, `None` → `null`, strings
   quoted exactly when needed. Block scalars (`llm.system_prompt`'s `|`) get a
   dedicated path: replace the indented block's lines, keep the `|` header.
3. **Verify round-trip**: `yaml.safe_load` the edited text; every changed path
   must read back equal to the requested value AND every unchanged scalar in the
   file must equal its pre-edit value. Any mismatch ⇒ `ConfigEditError`.

The server wraps it: read `config.yaml` → `set_values` → write atomically
(temp file + `os.replace` in the same directory).

### 4d. `kiosk_proc.py` — child-process manager

- `start(diag: bool)`: refuses if a child is already running, and refuses if a
  **foreign** `kiosk.py --talkback` is running (same `pgrep -f` +
  `/proc/<pid>/comm == python*` guard as `kiosk-stack.sh`; the error names the
  PID). Spawns `python3 kiosk.py --talkback` with `PYTHONFAULTHANDLER=1`,
  `PYTHONUNBUFFERED=1`, plus `TVAD_DIAG=1` when the UI checkbox is on;
  stdout+stderr merged into a pipe.
- A reader thread pumps the pipe into a bounded in-memory ring (last 2000 lines,
  ANSI escapes stripped) and fans out to any connected SSE clients. Child exit
  is announced as a synthetic line: `[tune] kiosk exited (code N)`.
- `stop()`: SIGTERM → wait 5 s → SIGKILL (mirrors `term_then_kill`). `restart()`
  = `stop()` then `start()`.
- The LLM is not managed. The UI shows a reachability dot from a cached probe of
  `http://127.0.0.1:8080/v1/models`; if it's down, kiosk.py's own exit-3 message
  lands in the log pane.
- Server shutdown (Ctrl-C) stops a running child — never orphan a kiosk.

### 4e. `server.py` — HTTP API

Stdlib `ThreadingHTTPServer`. JSON in/out; errors are
`{"error": "<message>"}` with 4xx/5xx.

| Route | Method | Behavior |
|---|---|---|
| `/` | GET | `static/index.html` |
| `/api/state` | GET | `{knobs: [{...Knob, value}], kiosk: {running, pid, diag}, llm: {reachable}, config_path}` — values read fresh from `config.yaml` each call |
| `/api/save` | POST | body `{changes: {path: value}}` → validate each path is a registered knob and value is in-range/right-kind → `config_edit.set_values` → atomic write. Returns `{saved: [paths]}` or 409 with the `ConfigEditError` message |
| `/api/kiosk/start` | POST | body `{diag: bool}`; 409 if running/foreign |
| `/api/kiosk/stop` | POST | idempotent |
| `/api/kiosk/restart` | POST | stop + start with same diag flag |
| `/api/logs` | GET | SSE (`text/event-stream`): replays the ring, then live lines |

Server-side validation mirrors the registry (kind, min/max, choices, strict-bool)
so a hand-crafted POST can't write garbage — the UI is not the trust boundary
for the file the kiosk boots from.

### 4f. `static/index.html` — the page

Single file, inline CSS + JS, no framework, no build step, dark theme.

- **Tab bar** across the top; one pane per tab from section 4b.
- **Knob row**: label · control · current-vs-saved indicator · `doc` line;
  `why` behind a ⓘ expander. Numerics = slider **plus** exact number input
  (sliders alone can't hit 0.15 reliably); bools = toggles; selects = dropdowns;
  `system_prompt` = textarea.
- **Save bar**: appears when dirty; lists changed knobs (`old → new`); Save
  POSTs only the changed paths; on 409 shows the server's message and keeps
  edits. A "Revert edits" link reloads saved values (page-local, not a file
  revert).
- **Kiosk strip**: status (● running pid / ○ stopped), LLM dot, DIAG checkbox,
  Start/Stop/Restart. After a save with the kiosk running, the strip prompts
  "config changed — restart to apply".
- **Log pane**: bottom half; monospace, auto-scroll with pause-on-scroll-up,
  Clear button; regex highlighting for `[DIAG`, `REJECT=`, `[WAKE]`,
  `[SESSION STARTED]`, `[SESSION ENDED]`, `Traceback`.

## 5. Error handling

- **Save**: all-or-nothing; any locate/render/round-trip failure aborts with a
  message naming the path. The file is never left half-edited (atomic replace).
- **Concurrent edits**: if `config.yaml` changed on disk since the page loaded
  (hand-edit in parallel), the round-trip guard still protects correctness;
  values shown refresh on every `/api/state` poll (page polls every 3 s when
  idle, never overwriting a dirty control).
- **Kiosk death**: exit code line in the log pane; strip flips to stopped.
- **SSE client drop**: reader thread discards dead clients silently.

## 6. Testing

`python3 -m pytest`, tests under `tests/tune/`:

- **`test_config_edit.py`** — against a copy of the real `config.yaml`: set each
  kind (float/int/bool/null/string/block scalar); comments byte-identical
  outside the edited value span; unknown path refused; duplicate-key ambiguity
  refused; round-trip mismatch refused; multi-change atomicity (one bad path ⇒
  no change applied).
- **`test_knobs.py`** — meta-test: every registered knob path resolves in the
  real `config.yaml` (catches drift when config evolves); kinds/ranges sane;
  strict-bool knobs cover exactly the strict-bool config keys.
- **`test_kiosk_proc.py`** — with a fake child script (`sleep`/`echo`): start/
  stop/restart lifecycle, TERM-then-KILL, foreign-process refusal, ring buffer,
  exit announcement.
- **`test_server_api.py`** — real server on an ephemeral port via
  `http.client`: state/save/kiosk routes, validation rejections, SSE replay.

## 7. Out of scope

Hot-apply / live dashboard, preset profiles, snapshots/revert,
diff-before-save (beyond the save bar's change list), auth/TLS, offline
pipeline sections, editing keys not in the registry, managing the LLM server.

## 8. Acceptance (live)

1. Open the page, change `doa.cone_deg` 20 → 25, Save → `git diff config.yaml`
   shows exactly one changed line, comments intact.
2. Restart from the page → DIAG stream appears in the log pane; wake the kiosk
   and watch `[DIAG wakegate]` lines live.
3. Stop from the page → kiosk exits cleanly; `pgrep -f kiosk.py` empty.
4. Kill the tuning server with a kiosk running → kiosk is stopped too.
