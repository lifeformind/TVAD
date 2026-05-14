# Shared Speaker Stack Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `target-vad/` so shared primitives live in `core/` and per-mode orchestration has a `modes/` home, with the existing 23-test suite continuing to pass at every step. This is a pure-refactor prerequisite for the Classroom Diarization (S1) and Kiosk Talkback (S2) builds.

**Architecture:** Mechanical move of `audio/`, `speaker/`, `vad/`, `compat.py` into `core/`. Add empty `modes/{kiosk,diarization}/` skeleton. Re-namespace `config.yaml` so existing top-level keys nest under `core:`. Update import paths and config-key accesses in every consumer. No behavior changes; the existing tests are the regression net.

**Tech Stack:** Python 3.14 (invoke as `py -3.14`), pytest, PyYAML. Existing project layout under `c:\repos\TVAD\target-vad\`.

**Reference spec:** [`docs/superpowers/specs/2026-05-14-shared-speaker-stack.md`](../specs/2026-05-14-shared-speaker-stack.md)

**Working directory:** All commands assume `cwd = c:\repos\TVAD\target-vad\` unless noted otherwise. The `target-vad/` subdirectory is the project root; the repo root is `c:\repos\TVAD\`.

---

## Pre-flight notes

- This is a refactor with **no new behavior**. The only "test" is "all 23 existing tests still pass at every commit."
- **Commit after every task** so any breakage is bisectable to a single small change.
- **Never use `git add -A`**: stage explicit paths only. `__pycache__` and any transient `_tmp_*.py` files must NOT be staged.
- **Windows note:** after a directory move, stale `__pycache__` may exist at the old location. Deleting it is safe; pytest will rebuild as needed.
- The default `python` on this machine is 3.12 and lacks the project's deps. **Always use `py -3.14`** for pytest and any Python invocation.

---

## Task 1: Baseline + directory skeleton

**Files:**
- Create: `target-vad/core/__init__.py`
- Create: `target-vad/modes/__init__.py`
- Create: `target-vad/modes/kiosk/__init__.py`
- Create: `target-vad/modes/diarization/__init__.py`

- [ ] **Step 1: Confirm the baseline test suite passes**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed in <N>s`. If anything is failing before we start, STOP and resolve before proceeding.

- [ ] **Step 2: Create `core/__init__.py`**

Write file `target-vad/core/__init__.py` with content:

```python
"""Shared primitives consumed by all TVAD modes (kiosk, diarization, legacy pipeline)."""
```

- [ ] **Step 3: Create `modes/__init__.py`**

Write file `target-vad/modes/__init__.py` with content:

```python
"""Per-mode orchestration: kiosk, diarization."""
```

- [ ] **Step 4: Create `modes/kiosk/__init__.py` (empty placeholder)**

Write file `target-vad/modes/kiosk/__init__.py` with content:

```python
"""Wake-word kiosk talkback pipeline. See docs/superpowers/specs/2026-05-14-kiosk-talkback-design.md."""
```

- [ ] **Step 5: Create `modes/diarization/__init__.py` (empty placeholder)**

Write file `target-vad/modes/diarization/__init__.py` with content:

```python
"""Classroom diarization & identification pipeline. See docs/superpowers/specs/2026-05-14-classroom-diarization-design.md."""
```

- [ ] **Step 6: Verify directory structure**

Run: `ls target-vad/core target-vad/modes target-vad/modes/kiosk target-vad/modes/diarization`
Expected: each lists at least an `__init__.py`.

- [ ] **Step 7: Run tests — should still pass (no behavior change)**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`.

- [ ] **Step 8: Commit**

```bash
git add target-vad/core/__init__.py target-vad/modes/__init__.py target-vad/modes/kiosk/__init__.py target-vad/modes/diarization/__init__.py
git commit -m "$(cat <<'EOF'
chore: add core/ and modes/ directory skeleton

Prerequisite for the kiosk and diarization mode builds. Empty
__init__.py modules only — no behavior change. The existing 23-test
suite continues to pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Move `compat.py` into `core/`

**Files:**
- Move: `target-vad/compat.py` → `target-vad/core/compat.py`
- Modify: `target-vad/enroll.py:3`
- Modify: `target-vad/main.py:3`
- Modify: `target-vad/live_test.py:3`

- [ ] **Step 1: Move the file**

Run: `git mv target-vad/compat.py target-vad/core/compat.py`
Expected: no stdout. `target-vad/compat.py` no longer exists; `target-vad/core/compat.py` does.

- [ ] **Step 2: Update import in `enroll.py`**

In `target-vad/enroll.py` line 3, change:

```python
import compat  # noqa: F401 — torchaudio/speechbrain shim
```

to:

```python
from core import compat  # noqa: F401 — torchaudio/speechbrain shim
```

- [ ] **Step 3: Update import in `main.py`**

In `target-vad/main.py` line 3, change:

```python
import compat  # noqa: F401 — torchaudio/speechbrain shim
```

to:

```python
from core import compat  # noqa: F401 — torchaudio/speechbrain shim
```

- [ ] **Step 4: Update import in `live_test.py`**

In `target-vad/live_test.py` line 3, change:

```python
import compat  # noqa: F401
```

to:

```python
from core import compat  # noqa: F401
```

- [ ] **Step 5: Run tests**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`. (Note: tests don't import `compat` themselves; they exercise the libraries it patches. Pass means our shim is still being applied wherever it matters.)

- [ ] **Step 6: Commit**

```bash
git add target-vad/core/compat.py target-vad/enroll.py target-vad/main.py target-vad/live_test.py
git commit -m "$(cat <<'EOF'
refactor: move compat.py into core/

Update entry-point imports. No behavior change; 23/23 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Move `audio/` into `core/`

**Files:**
- Move: `target-vad/audio/` → `target-vad/core/audio/`
- Modify: `target-vad/enroll.py:15`
- Modify: `target-vad/live_test.py:10`
- Modify: `target-vad/pipeline/target_vad_pipeline.py:8`

- [ ] **Step 1: Move the directory**

Run: `git mv target-vad/audio target-vad/core/audio`
Expected: no stdout. `target-vad/audio/` no longer exists; `target-vad/core/audio/` does.

If you see an error about pathspec, verify the `git mv` worked by listing both paths; if not, fall back to `mv target-vad/audio target-vad/core/audio` followed by staging the changes manually.

- [ ] **Step 2: Update import in `enroll.py`**

In `target-vad/enroll.py` line 15, change:

```python
from audio.mic_stream import MicrophoneStream
```

to:

```python
from core.audio.mic_stream import MicrophoneStream
```

- [ ] **Step 3: Update import in `live_test.py`**

In `target-vad/live_test.py` line 10, change:

```python
from audio.mic_stream import MicrophoneStream
```

to:

```python
from core.audio.mic_stream import MicrophoneStream
```

- [ ] **Step 4: Update import in `pipeline/target_vad_pipeline.py`**

In `target-vad/pipeline/target_vad_pipeline.py` line 8, change:

```python
from audio.mic_stream import MicrophoneStream
```

to:

```python
from core.audio.mic_stream import MicrophoneStream
```

- [ ] **Step 5: Clean stale __pycache__ if present**

Run: `rm -rf target-vad/audio` (in case `git mv` left an empty husk; safe no-op if already gone).
Run: `find target-vad -name '__pycache__' -type d -prune -exec rm -rf {} +` to clear any stale cache pointing at old paths.

- [ ] **Step 6: Run tests**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`.

- [ ] **Step 7: Commit**

```bash
git add target-vad/core/audio target-vad/enroll.py target-vad/live_test.py target-vad/pipeline/target_vad_pipeline.py
git commit -m "$(cat <<'EOF'
refactor: move audio/ into core/

Consumers updated: enroll.py, live_test.py, pipeline/target_vad_pipeline.py.
No behavior change; 23/23 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Move `vad/` into `core/`

**Files:**
- Move: `target-vad/vad/` → `target-vad/core/vad/`
- Modify: `target-vad/enroll.py:18`
- Modify: `target-vad/live_test.py:11`
- Modify: `target-vad/main.py:11`
- Modify: `target-vad/pipeline/target_vad_pipeline.py:12`
- Modify: `target-vad/tests/test_vad.py:6`
- Modify: `target-vad/tests/test_pipeline.py:12`

- [ ] **Step 1: Move the directory**

Run: `git mv target-vad/vad target-vad/core/vad`

- [ ] **Step 2: Update import in `enroll.py`**

In `target-vad/enroll.py` line 18, change:

```python
from vad.silero_vad import SileroVAD
```

to:

```python
from core.vad.silero_vad import SileroVAD
```

- [ ] **Step 3: Update import in `live_test.py`**

In `target-vad/live_test.py` line 11, change:

```python
from vad.silero_vad import SileroVAD, SpeechSegment
```

to:

```python
from core.vad.silero_vad import SileroVAD, SpeechSegment
```

- [ ] **Step 4: Update import in `main.py`**

In `target-vad/main.py` line 11, change:

```python
from vad.silero_vad import SpeechSegment
```

to:

```python
from core.vad.silero_vad import SpeechSegment
```

- [ ] **Step 5: Update import in `pipeline/target_vad_pipeline.py`**

In `target-vad/pipeline/target_vad_pipeline.py` line 12, change:

```python
from vad.silero_vad import SileroVAD, SpeechSegment
```

to:

```python
from core.vad.silero_vad import SileroVAD, SpeechSegment
```

- [ ] **Step 6: Update import in `tests/test_vad.py`**

In `target-vad/tests/test_vad.py` line 6, change:

```python
from vad.silero_vad import SileroVAD, SpeechSegment
```

to:

```python
from core.vad.silero_vad import SileroVAD, SpeechSegment
```

- [ ] **Step 7: Update import in `tests/test_pipeline.py`**

In `target-vad/tests/test_pipeline.py` line 12, change:

```python
from vad.silero_vad import SileroVAD, SpeechSegment
```

to:

```python
from core.vad.silero_vad import SileroVAD, SpeechSegment
```

- [ ] **Step 8: Clean stale __pycache__**

Run: `rm -rf target-vad/vad` (no-op if already gone).
Run: `find target-vad -name '__pycache__' -type d -prune -exec rm -rf {} +`.

- [ ] **Step 9: Run tests**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`.

- [ ] **Step 10: Commit**

```bash
git add target-vad/core/vad target-vad/enroll.py target-vad/live_test.py target-vad/main.py target-vad/pipeline/target_vad_pipeline.py target-vad/tests/test_vad.py target-vad/tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
refactor: move vad/ into core/

Consumers updated: enroll.py, live_test.py, main.py,
pipeline/target_vad_pipeline.py, tests/test_vad.py, tests/test_pipeline.py.
No behavior change; 23/23 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Move `speaker/` into `core/`

**Files:**
- Move: `target-vad/speaker/` → `target-vad/core/speaker/`
- Modify: `target-vad/core/speaker/verifier.py:8` (internal cross-module import within speaker/)
- Modify: `target-vad/enroll.py:16-17`
- Modify: `target-vad/main.py:12`
- Modify: `target-vad/pipeline/target_vad_pipeline.py:9-11`
- Modify: `target-vad/live_test.py:51` (inline late import)
- Modify: `target-vad/tests/test_pipeline.py:9-11`
- Modify: `target-vad/tests/test_verifier.py:9-10`

- [ ] **Step 1: Move the directory**

Run: `git mv target-vad/speaker target-vad/core/speaker`

- [ ] **Step 2: Update internal import in `core/speaker/verifier.py`**

In `target-vad/core/speaker/verifier.py` line 8, change:

```python
from speaker.enrollment_store import EnrollmentStore
```

to:

```python
from core.speaker.enrollment_store import EnrollmentStore
```

- [ ] **Step 3: Update imports in `enroll.py`**

In `target-vad/enroll.py` lines 16-17, change:

```python
from speaker.embedder import EmbeddingExtractor
from speaker.enrollment_store import EnrollmentStore
```

to:

```python
from core.speaker.embedder import EmbeddingExtractor
from core.speaker.enrollment_store import EnrollmentStore
```

Then in the same file, find the late import inside `cmd_test` (around line 123):

```python
from speaker.verifier import SpeakerVerifier
```

Change to:

```python
from core.speaker.verifier import SpeakerVerifier
```

- [ ] **Step 4: Update import in `main.py`**

In `target-vad/main.py` line 12, change:

```python
from speaker.verifier import VerificationResult
```

to:

```python
from core.speaker.verifier import VerificationResult
```

- [ ] **Step 5: Update imports in `pipeline/target_vad_pipeline.py`**

In `target-vad/pipeline/target_vad_pipeline.py` lines 9-11, change:

```python
from speaker.embedder import EmbeddingExtractor
from speaker.enrollment_store import EnrollmentStore
from speaker.verifier import SpeakerVerifier, VerificationResult
```

to:

```python
from core.speaker.embedder import EmbeddingExtractor
from core.speaker.enrollment_store import EnrollmentStore
from core.speaker.verifier import SpeakerVerifier, VerificationResult
```

- [ ] **Step 6: Update inline late import in `live_test.py`**

In `target-vad/live_test.py` line 51 (inside `main()`), change:

```python
        from speaker.embedder import EmbeddingExtractor
```

to:

```python
        from core.speaker.embedder import EmbeddingExtractor
```

- [ ] **Step 7: Update imports in `tests/test_pipeline.py`**

In `target-vad/tests/test_pipeline.py` lines 9-11, change:

```python
from speaker.embedder import EmbeddingExtractor
from speaker.enrollment_store import EnrollmentStore
from speaker.verifier import SpeakerVerifier, VerificationResult
```

to:

```python
from core.speaker.embedder import EmbeddingExtractor
from core.speaker.enrollment_store import EnrollmentStore
from core.speaker.verifier import SpeakerVerifier, VerificationResult
```

- [ ] **Step 8: Update imports in `tests/test_verifier.py`**

In `target-vad/tests/test_verifier.py` lines 9-10, change:

```python
from speaker.enrollment_store import EnrollmentStore
from speaker.verifier import SpeakerVerifier, VerificationResult, cosine_similarity
```

to:

```python
from core.speaker.enrollment_store import EnrollmentStore
from core.speaker.verifier import SpeakerVerifier, VerificationResult, cosine_similarity
```

- [ ] **Step 9: Clean stale __pycache__**

Run: `rm -rf target-vad/speaker` (no-op if already gone).
Run: `find target-vad -name '__pycache__' -type d -prune -exec rm -rf {} +`.

- [ ] **Step 10: Run tests**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`.

- [ ] **Step 11: Commit**

```bash
git add target-vad/core/speaker target-vad/enroll.py target-vad/main.py target-vad/pipeline/target_vad_pipeline.py target-vad/live_test.py target-vad/tests/test_pipeline.py target-vad/tests/test_verifier.py
git commit -m "$(cat <<'EOF'
refactor: move speaker/ into core/

Updated speaker/verifier.py internal import and all consumers:
enroll.py, main.py, pipeline/target_vad_pipeline.py, live_test.py,
tests/test_pipeline.py, tests/test_verifier.py. No behavior change;
23/23 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Re-namespace `config.yaml` under `core:`

**Files:**
- Modify: `target-vad/config.yaml`
- Modify: `target-vad/enroll.py` (multiple `config["..."]` accesses)
- Modify: `target-vad/main.py:81`
- Modify: `target-vad/pipeline/target_vad_pipeline.py` (multiple)
- Modify: `target-vad/live_test.py:32-33`

**Background:** The existing tests pass dict literals to constructors, not the YAML file, so they'll continue to pass without modification. The change is to runtime entry points only.

- [ ] **Step 1: Re-namespace `config.yaml`**

Replace the entire contents of `target-vad/config.yaml` with:

```yaml
core:
  vad:
    sample_rate: 16000
    chunk_duration_ms: 30
    speech_threshold: 0.5
    min_speech_duration_ms: 300
    padding_ms: 200

  speaker:
    threshold: 0.75
    min_segment_duration_ms: 800
    enrollment_utterances: 5
    enrollment_min_self_similarity: 0.6
    enrollment_max_retries: 3

  audio:
    device_index: null
    channels: 1
    sample_rate: 16000
    chunk_size: 480

  paths:
    voiceprints_dir: "./voiceprints"
    silero_model_path: null
```

(Mode-specific blocks `kiosk:` and `diarization:` are NOT added here — they'll be added when those modes are built.)

- [ ] **Step 2: Update `enroll.py` config accesses**

In `target-vad/enroll.py`, replace each occurrence:

| Line (approx) | Before | After |
|---|---|---|
| 42 | `config["speaker"]["enrollment_utterances"]` | `config["core"]["speaker"]["enrollment_utterances"]` |
| 43 | `config["speaker"].get("enrollment_min_self_similarity", 0.6)` | `config["core"]["speaker"].get("enrollment_min_self_similarity", 0.6)` |
| 44 | `config["speaker"].get("enrollment_max_retries", 3)` | `config["core"]["speaker"].get("enrollment_max_retries", 3)` |
| 52 | `SileroVAD(config["vad"])` | `SileroVAD(config["core"]["vad"])` |
| 54 | `EnrollmentStore(config["paths"]["voiceprints_dir"])` | `EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])` |
| 61 | `MicrophoneStream(config["audio"])` | `MicrophoneStream(config["core"]["audio"])` |
| 138 | `EnrollmentStore(config["paths"]["voiceprints_dir"])` | `EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])` |
| 152 | `EnrollmentStore(config["paths"]["voiceprints_dir"])` | `EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])` |
| 163 | `SileroVAD(config["vad"])` | `SileroVAD(config["core"]["vad"])` |
| 165 | `EnrollmentStore(config["paths"]["voiceprints_dir"])` | `EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])` |
| 168 | `SpeakerVerifier(store, config["speaker"]["threshold"])` | `SpeakerVerifier(store, config["core"]["speaker"]["threshold"])` |
| 176 | `MicrophoneStream(config["audio"])` | `MicrophoneStream(config["core"]["audio"])` |

(Line numbers are approximate after the prior tasks' edits — search for the literal `config["..."]` patterns rather than relying on numbers.)

- [ ] **Step 3: Update `main.py` config access**

In `target-vad/main.py` line 81, change:

```python
        config["speaker"]["threshold"] = args.threshold
```

to:

```python
        config["core"]["speaker"]["threshold"] = args.threshold
```

- [ ] **Step 4: Update `pipeline/target_vad_pipeline.py` config accesses**

In `target-vad/pipeline/target_vad_pipeline.py` lines 22-29, change:

```python
        self.vad = SileroVAD(config["vad"])
        self.embedder = EmbeddingExtractor()
        self.store = EnrollmentStore(config["paths"]["voiceprints_dir"])
        self.verifier = SpeakerVerifier(
            self.store, config["speaker"]["threshold"]
        )
        self.mic = MicrophoneStream(config["audio"])
        self.min_segment_ms = config["speaker"].get("min_segment_duration_ms", 800)
```

to:

```python
        self.vad = SileroVAD(config["core"]["vad"])
        self.embedder = EmbeddingExtractor()
        self.store = EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])
        self.verifier = SpeakerVerifier(
            self.store, config["core"]["speaker"]["threshold"]
        )
        self.mic = MicrophoneStream(config["core"]["audio"])
        self.min_segment_ms = config["core"]["speaker"].get("min_segment_duration_ms", 800)
```

- [ ] **Step 5: Update `live_test.py` config accesses**

In `target-vad/live_test.py` lines 32-33, change:

```python
    vad = SileroVAD(config["vad"])
    mic = MicrophoneStream(config["audio"])
```

to:

```python
    vad = SileroVAD(config["core"]["vad"])
    mic = MicrophoneStream(config["core"]["audio"])
```

- [ ] **Step 6: Run tests**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`. (Tests don't read config.yaml; they pass dicts directly to constructors. They should be unaffected by the YAML re-namespacing as long as the Python files compile.)

- [ ] **Step 7: Syntax-check entry points (catch any missed config access)**

Run: `cd target-vad && py -3.14 -c "import ast; [ast.parse(open(f).read()) for f in ('enroll.py','main.py','live_test.py','pipeline/target_vad_pipeline.py')]; print('all syntax OK')"`
Expected: `all syntax OK`.

- [ ] **Step 8: Smoke-test `enroll.py list`** (no mic interaction; just verifies config load + import path)

Run: `cd target-vad && py -3.14 enroll.py list`
Expected: either `Enrolled users:` followed by entries (e.g. `siddharth`), or `No enrolled users.` Either is success — what we're checking is that config loads and imports resolve without error. **Failure mode to catch:** `KeyError: 'speaker'` means a config access was missed.

- [ ] **Step 9: Commit**

```bash
git add target-vad/config.yaml target-vad/enroll.py target-vad/main.py target-vad/pipeline/target_vad_pipeline.py target-vad/live_test.py
git commit -m "$(cat <<'EOF'
refactor: namespace config under core:

config.yaml: nest existing top-level keys (vad, speaker, audio, paths)
under a top-level 'core:' block. All runtime entry points updated to
read config["core"]["..."]. Tests use dict fixtures and are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification

**Files:** none modified; verification only.

- [ ] **Step 1: Final test sweep**

Run: `cd target-vad && py -3.14 -m pytest -v`
Expected: All 23 tests passing, no skips that weren't skipped before.

- [ ] **Step 2: Verify final tree structure**

Run: `ls target-vad/core target-vad/modes`
Expected:
- `target-vad/core/` contains `__init__.py`, `audio/`, `compat.py`, `speaker/`, `vad/`.
- `target-vad/modes/` contains `__init__.py`, `kiosk/`, `diarization/`.

Run: `ls target-vad/`
Expected to NOT contain (these moved): `audio`, `speaker`, `vad`, `compat.py`.
Expected to still contain: `core/`, `modes/`, `pipeline/`, `tests/`, `enroll.py`, `main.py`, `live_test.py`, `config.yaml`, `requirements.txt`, `voiceprints/`.

- [ ] **Step 3: Verify git log shows the per-task commits**

Run: `git log --oneline -10`
Expected: Last 6 commits should be the chore + 5 refactor commits from tasks 1–6, in order.

- [ ] **Step 4: No commit needed for this task**

Verification only. If everything checks, the refactor is complete and the codebase is ready for the kiosk and diarization mode plans.

---

## Self-review notes

**Spec coverage:** every section of [`2026-05-14-shared-speaker-stack.md`](../specs/2026-05-14-shared-speaker-stack.md) "Project layout", "Configuration", "Reused components", "Migration tasks" maps to a task above. Sections that are out of scope for this refactor (DecisionSmoother in `core/speaker/`, structured event logging) are intentionally deferred to the mode plans that first consume them, per YAGNI.

**Type/name consistency:** module imports use `core.audio.mic_stream`, `core.speaker.{embedder,enrollment_store,verifier}`, `core.vad.silero_vad`, `core.compat`. Same path used in every task referencing them.

**Placeholders:** none. Every step lists exact file, line range, before/after content, and command.

**Risk:** the repository wraps its source under a `target-vad/` subfolder. All commands assume that's the working directory or use the `target-vad/...` prefix. Path mistakes at this level would break the move silently — re-read each command before running.
