# Wake-Word Kiosk Talkback — Design (S2)

**Date:** 2026-05-14
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-14-shared-speaker-stack.md`](./2026-05-14-shared-speaker-stack.md)

## Purpose

Implement a streaming, real-time speaker-handling layer for a talkback kiosk. The system idles until any user speaks a wake phrase. The wake-word speaker becomes the "session primary." For the duration of the session, only audio segments that match the session-primary voiceprint are forwarded downstream; everything else (other voices, ambient noise) is suppressed. Sessions end on silence/timeout, after which the system returns to idle and the session voiceprint is discarded.

This spec covers the speaker-handling pipeline only. Downstream STT/LLM/TTS is explicitly out of scope and is consumed via a callback interface.

## Why this design

Two design choices are non-obvious and worth justifying up front:

1. **Session voiceprint is captured fresh from the wake-word audio**, not loaded from enrolled voiceprints. Same-condition matching (snapshot taken seconds before the segments it's matched against, in the same room state with the same mic DSP behavior) is structurally tighter than cross-time matching against pre-enrolled voiceprints. This sidesteps the voiceprint-drift problem the existing TVAD pipeline hits with the C10 in noisy rooms.
2. **Variant A (open kiosk):** anyone can use the kiosk; no enrollment required. Authentication via enrolled voiceprint matching is left for a later variant.

## State machine

```
                    ┌─────────────────────────┐
                    │         IDLE            │
                    │  wake-word detector on  │
                    │  primary = None         │
                    └────────────┬────────────┘
                                 │ wake-word fires
                                 ▼
                    ┌─────────────────────────┐
                    │      CAPTURING          │
                    │  capture wake-word      │
                    │  audio + 1s tail        │
                    └────────────┬────────────┘
                                 │ ECAPA(captured_audio)
                                 ▼
                    ┌─────────────────────────┐
                    │     ACTIVE_SESSION      │
                    │  primary = snapshot     │
                    │  smoother running       │
                    │  VAD streams segments   │
                    └────────────┬────────────┘
                                 │ silence_timeout OR hard_timeout
                                 ▼
                    ┌─────────────────────────┐
                    │      ENDING             │
                    │  flush smoother         │
                    │  emit session_ended     │
                    │  discard primary        │
                    └────────────┬────────────┘
                                 │
                                 └────────────► back to IDLE
```

## Pipeline

```
[Continuous mic input — sounddevice stream]
             │
             ▼
   ┌──────── STATE: IDLE ────────┐
   │  Frames → openwakeword       │
   │  if score >= wake_threshold: │
   │     transition to CAPTURING  │
   └─────────────┬────────────────┘
                 │
                 ▼
   ┌──────── STATE: CAPTURING ────┐
   │  buffer = wake_audio + 1s    │
   │  embedding = ECAPA(buffer)   │
   │  primary = embedding         │
   │  init DecisionSmoother       │
   │  init silence_timer          │
   │  emit session_started        │
   │  → ACTIVE_SESSION            │
   └─────────────┬────────────────┘
                 │
                 ▼
   ┌──────── STATE: ACTIVE_SESSION ────────┐
   │  Silero VAD streams speech segments    │
   │  for each segment:                     │
   │    emb = ECAPA(segment.audio)          │
   │    score = cosine(emb, primary)        │
   │    matched = smoother.update(score)    │
   │    if matched:                         │
   │      callback.on_primary_speech(...)   │
   │      reset silence_timer               │
   │  if silence_timer > silence_timeout:   │
   │    → ENDING                            │
   │  if session_clock > hard_timeout:      │
   │    → ENDING                            │
   └─────────────┬──────────────────────────┘
                 │
                 ▼
   ┌──────── STATE: ENDING ────────┐
   │  emit session_ended            │
   │  primary = None                │
   │  smoother = None               │
   │  → IDLE                        │
   └────────────────────────────────┘
```

## Components

| File | Responsibility |
|---|---|
| `target-vad/kiosk.py` | CLI entry point: arg parsing, instantiates `KioskPipeline`, wires callback |
| `target-vad/modes/kiosk/pipeline.py` | `KioskPipeline` class: owns the state machine, mic loop, callbacks |
| `target-vad/modes/kiosk/wake_word.py` | `WakeWordDetector` class: wraps openwakeword, returns wake events |
| `target-vad/modes/kiosk/session.py` | `Session` dataclass: primary embedding, smoother, timers, start_time |
| `target-vad/modes/kiosk/__init__.py` | (empty) |
| `core/speaker/decision_smoother.py` | `DecisionSmoother` (sliding-window M-of-N) — see shared spec |
| `core/speaker/embedder.py` | (reused) embed wake-word snapshot + each session segment |
| `core/speaker/verifier.py::cosine_similarity` | (reused) per-segment scoring |
| `core/vad/silero_vad.py` | (reused) segment the active session audio stream |
| `core/audio/mic_stream.py` | (reused) continuous mic input |

### `KioskPipeline` interface

```python
class KioskPipeline:
    def __init__(
        self,
        config: dict,
        on_primary_speech: Callable[[SpeechSegment, np.ndarray], None],
        on_session_started: Callable[[], None] = lambda: None,
        on_session_ended: Callable[[str], None] = lambda reason: None,
    ): ...

    def run(self) -> None: ...    # blocks; runs until KeyboardInterrupt
    def stop(self) -> None: ...   # signals run() to exit cleanly
```

`on_primary_speech` receives the speech segment and its embedding. The handler is opaque to the pipeline; out-of-scope concerns (STT, LLM, TTS) live in the handler.

`on_session_ended(reason)`: `reason ∈ {"silence_timeout", "hard_timeout", "stopped"}`.

## CLI

```
py -3.14 kiosk.py [--wake-phrase hey_jarvis] [--config config.yaml] [--log] [--dry-run]
```

- `--wake-phrase`: openwakeword model name. Defaults to `hey_jarvis`. Bundled options: `alexa`, `hey_jarvis`, `hey_mycroft`. Custom phrase training is deferred.
- `--config`: defaults to `./config.yaml`.
- `--log`: enables JSON-lines event log per shared spec.
- `--dry-run`: runs the pipeline, but `on_primary_speech` is replaced with a printer that shows `[PRIMARY] {duration}ms score={score}` instead of forwarding to a real downstream handler. Use for testing the speaker-handling layer in isolation.

CLI prints state transitions in `rich`:

```
[IDLE] Listening for "hey jarvis"...
[WAKE] Detected (score=0.89)
[SESSION STARTED] Primary speaker locked
[PRIMARY] 1840ms score=0.71
[PRIMARY] 2210ms score=0.78
[NON-PRIMARY] 1100ms score=0.42  (suppressed)
...
[SESSION ENDED] reason=silence_timeout (lasted 47s)
[IDLE] Listening for "hey jarvis"...
```

## Configuration

Per the shared spec, the `kiosk:` block in `config.yaml`:

```yaml
kiosk:
  wake_phrase: "hey_jarvis"
  wake_threshold: 0.5                      # openwakeword confidence
  wake_capture_tail_seconds: 1.0           # additional audio captured AFTER the wake phrase ends, appended to the wake-phrase audio itself before embedding (snapshot = wake_audio + tail)
  session_primary_threshold: 0.60          # cosine threshold per segment, fed into smoother
  session_silence_timeout_s: 10
  session_hard_timeout_s: 300
  decision_smoother:
    window_size: 3
    min_matches: 2
    threshold: 0.60                        # same as session_primary_threshold; explicit so they can diverge
```

`session_primary_threshold` and `decision_smoother.threshold` are intentionally exposed separately even though we default them to the same value — gives us tuning flexibility without code changes.

## Error handling

| Failure | Behavior |
|---|---|
| Mic unavailable / disconnected | Exit with clear error; suggest checking `sounddevice` device list |
| openwakeword model file missing | Auto-download on first run if internet available; otherwise exit with download instructions |
| Wake-word fires but the captured audio fails to embed (e.g., all silence after wake) | Log warning, return to IDLE; do NOT start session |
| Session callback handler raises | Log error with traceback; **session continues**, do not crash the pipeline (the kiosk should be resilient to downstream bugs) |
| KeyboardInterrupt | Clean shutdown: emit session_ended if active, close mic, exit 0 |

## Testing approach

`tests/kiosk/` contains:

- **Unit tests:**
  - `test_decision_smoother.py`: M-of-N logic; window slides correctly; reset on instantiation. (Lives in `tests/core/` per shared spec; mentioned here because it's load-bearing for kiosk correctness.)
  - `test_pipeline_state_machine.py`: `KioskPipeline` driven with **mocked** wake-word detector and mocked VAD/embedder (yielding deterministic embeddings). Asserts state transitions: IDLE→CAPTURING→ACTIVE_SESSION→ENDING→IDLE. Tests both timeouts.
  - `test_session_callbacks.py`: callbacks fire in correct order and with correct arguments; downstream handler exception doesn't crash session.
- **Integration test (manual, not in CI):**
  - `test_end_to_end.py.skip`: runs against real mic + real wake-word model. Marked skip; run manually with `--run-mic`.
- **Existing 23 tests must continue to pass.**

## Quality expectations and known limitations

- **Wake-word reliability with C10 DSP:** unknown until empirically tested. Risk: aggressive noise suppression may chop the wake phrase. If false-negative rate is high, mitigation is (a) lower `wake_threshold`, (b) try a different phrase, or (c) accept some failure rate.
- **First-speaker race:** if two people say the wake phrase nearly simultaneously, the captured snapshot is a mixed embedding and won't reliably match either single voice. Both subsequent voices will likely be suppressed. **Acceptable failure mode for v1**; users will retry. (Not a common scenario in a kiosk context.)
- **Session-primary score variance:** even with same-condition matching, expect occasional dips below threshold during a single utterance. The 2-of-3 smoother handles this; a single dip won't end the session.
- **Background talker leakage:** if a non-primary speaker talks loudly enough to be the dominant voice in a VAD segment that *also* contains some primary-speaker audio, the embedding of that mixed segment may match the primary above threshold and leak through. Same overlap problem as S1; out of scope to fully solve.
- **Silence detection vs short pauses:** the 10 s silence timeout is a heuristic. If the primary speaker pauses >10 s mid-thought, the session ends. Tunable; consider longer for slower-paced use cases.

## Out of scope

- Downstream STT, LLM, and TTS. Handled by the consumer of `on_primary_speech`.
- Multi-language wake-word support (use whatever bundled model fits).
- Custom wake-phrase training.
- Speaker authentication via enrolled voiceprints (Variant B; deferred).
- Multi-primary sessions (e.g., two speakers in conversation with the kiosk).
- Echo cancellation between TTS output and mic input (relevant for full talkback but lives downstream).
- Session resumption (each silence timeout closes the session permanently).

## Open questions

None at the time of writing. All decisions resolved 2026-05-14.
