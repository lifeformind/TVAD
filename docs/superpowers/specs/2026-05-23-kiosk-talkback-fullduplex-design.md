# Full-Duplex Kiosk Talkback — Design

**Date:** 2026-05-23
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-14-kiosk-talkback-design.md`](./2026-05-14-kiosk-talkback-design.md), [`2026-05-14-shared-speaker-stack.md`](./2026-05-14-shared-speaker-stack.md)
**Supersedes (partially):** the "Out of scope: downstream STT, LLM, TTS" decision in the 2026-05-14 kiosk spec.

## Purpose

Extend the validated S2 kiosk into a real full-duplex voice assistant. Today `KioskPipeline` locks onto a primary speaker and emits `on_primary_speech(segment, embedding)` to a handler that this project doesn't ship. This spec defines the downstream half — streaming STT → local LLM → streaming TTS → speaker playback — with **acoustic echo cancellation** and **speaker-verified barge-in** so the kiosk can be interrupted mid-response by the session-primary speaker.

This design also folds in two outstanding kiosk roadmap items (Batch 1f): **F4 watchdog** for chunk-independent timeout firing, and **F6 structured JSONL logging** for audit trails. The Batch 2c non-self false-positive validation is **deferred** — but F6's JSONL output is itself the validation harness, so no separate harness code is needed.

The kiosk is now general-purpose: any user who says the wake phrase becomes the session primary. The talkback model is multi-turn within one session (the LLM keeps message history until `silence_timeout`), single-turn between sessions (memory discarded when the session ends).

## Target hardware

NVIDIA DGX Spark (GB10 Grace + Blackwell, 128 GB unified LPDDR5x, aarch64 Linux). All STT/LLM/TTS inference runs on the Blackwell GPU. The existing TVAD CPU-first design for Strix Halo is unchanged; this spec adds a GPU-accelerated path that activates only when `kiosk.talkback_enabled: true`.

## Why this design

Three decisions are non-obvious and worth justifying up front:

1. **Two-layer architecture, not a rewrite.** `KioskPipeline` already passes 252 tests and reliably handles `IDLE → AWAITING_SPEECH → ACTIVE_SESSION`. A unified async pipeline that absorbs that role would be cleaner in a vacuum but throws away validated behavior. Instead, the existing pipeline hands off to a new `TalkbackController` at the moment the session becomes active. The hand-off boundary maps to a real architectural split: wake-detection and full-duplex talkback are different problems with different rhythms.
2. **Speaker-verified barge-in, not raw-VAD barge-in.** Background noise that VAD treats as speech but the ECAPA verifier rejects must not cut TTS. This preserves the kiosk-in-public property: a coworker walking past saying something doesn't derail the active session. Trade-off: a *second human* cannot interrupt the kiosk; only the locked primary speaker can. Configurable via `barge_in.require_speaker_match`.
3. **Software AEC via `webrtc-audio-processing-py`.** The target setup is a plain mic + separate speakers (no hardware AEC mic). Without echo cancellation, the kiosk would hear its own TTS and either (a) trigger spurious barge-ins or (b) feed its own audio back through STT. The WebRTC APM module is the same AEC Chrome uses — battle-tested, ~10 ms frame granularity, well-understood failure modes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      KioskPipeline (unchanged role)          │
│                                                              │
│   IDLE ──wake──> AWAITING_SPEECH ──first VAD segment──> ◯  │
└─────────────────────────────────────────────────────────────┘
                                                          │
                       hand-off: TalkbackHandoff payload  │
                                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     TalkbackController (new)                 │
│                                                              │
│   LISTENING ⇄ SPEAKING ⇄ BARGED_IN ──silence/hard──> ◯    │
└─────────────────────────────────────────────────────────────┘
                                                          │
                                  returns mic to kiosk   │
                                                          ▼
                                                      back to IDLE
```

### Hand-off contract

When `KioskPipeline` is configured with `talkback_enabled: true`, the existing first-segment moment in `_start_session_from_segment` does NOT call `on_primary_speech`. Instead it constructs a `TalkbackHandoff` and calls `controller.run(handoff)` (blocking). When the controller returns a `TalkbackResult`, the kiosk ends the session via the existing `_end_session(reason)` path and goes back to IDLE.

```python
@dataclass
class TalkbackHandoff:
    mic: MicrophoneStream            # the open mic; ownership transferred
    primary_embedding: np.ndarray    # 192-dim L2-normalized ECAPA snapshot
    first_segment: SpeechSegment     # the wake-segment, seeds STT for turn 1
    config: dict                     # the kiosk.talkback subsection

@dataclass
class TalkbackResult:
    reason: str                      # "silence_timeout" | "hard_timeout" | "stopped" | "device_lost"
    turns: int                       # number of complete user/assistant exchanges
    total_duration_s: float
```

Mic ownership is explicit so there is no shared-state confusion about who's reading chunks. `KioskPipeline` regains the mic from `TalkbackResult` and reuses it for the next IDLE → wake cycle.

When `talkback_enabled: false` (the default), `KioskPipeline` behaves exactly as it does today, including emitting `on_primary_speech`. This preserves the existing dry-run and test paths.

## TalkbackController internals

The controller owns one duplex audio runtime for the duration of a conversation. Internally it's a small set of components connected by `asyncio.Queue`s; sounddevice's audio callback thread is the only thing outside the loop, and it shoves chunks into a thread-safe queue.

```
                      ┌─── playback ref ───┐
                      │                     │
   mic stream ─┬─► [AEC] ─► clean mic ──┬─► [VAD + Embed + Verify]
               │                         │             │
               │                         │             ▼
               │                         │     primary-only segments
               │                         │             │
               │                         │             ▼
               │                         │     [STT (streaming, partials + finals)]
               │                         │             │
               │                         │             ▼
               │                         │       [Conversation Manager]  ◄── multi-turn history
               │                         │             │
               │                         │             ▼
               │                         │       [LLM (streaming tokens)]
               │                         │             │
               │                         │             ▼
               │                         │       [Sentence chunker]
               │                         │             │
               │                         │             ▼
               │                         │       [TTS (streaming chunks)]
               │                         │             │
               │                         └─── barge-in signal ──┐
               │                                                ▼
               └───────────────────────────────────────►   [Player] ─► speakers
                                                                │
                                                                └─► playback ref (back to AEC)
```

### Component responsibilities

| Component | Module | Responsibility |
|---|---|---|
| **AEC** | `modes/talkback/aec.py` | Wraps `webrtc-audio-processing-py`. 10 ms / 160-sample frames @ 16 kHz. Takes (mic_frame, playback_ref_frame) → clean_mic_frame. |
| **VAD + Embed + Verify** | reused: `core/vad/silero_vad.py`, `core/speaker/embedder.py`, `core/speaker/verifier.py` | Same primitives the existing kiosk uses. Same `DecisionSmoother` config but instantiated fresh with the hand-off's `primary_embedding` as the reference. |
| **Streaming STT** | `modes/talkback/stt.py` | Wraps `faster-whisper` (CUDA, float16). Feeds clean-mic chunks as they arrive. Emits partial transcripts every `stt.partials_every_ms` (default 300 ms) and a final transcript on Silero VAD's "speech ended" + `end_of_utterance_tail_ms` (default 400 ms). |
| **Conversation Manager** | `modes/talkback/conversation.py` | Owns the LLM message list for this session. Appends user finals, calls LLM, appends assistant responses. Session ends → memory discarded. |
| **LLM client** | `modes/talkback/llm.py` | OpenAI-compatible HTTP client against the configured `llm.base_url`. `stream=true`. Supports cancellation via `aiohttp.ClientSession.close()` for barge-in. |
| **Sentence chunker** | `modes/talkback/chunker.py` | Buffers LLM tokens; emits a chunk on sentence terminator (`.`, `?`, `!`) or `max_chunk_chars` cap (~120). Handles common abbreviation false-positives ("Dr.", "U.S."). Flushes trailing fragment at LLM stream end. |
| **TTS** | `modes/talkback/tts.py` | Streaming sentence-in, audio-out. Default backend `kokoro`, fallback `piper`. Returns audio as float32 @ 24 kHz (Kokoro) or 22 kHz (Piper); resampled to 16 kHz before player. |
| **Player** | `modes/talkback/player.py` | Pushes TTS audio frames to the sounddevice output stream. Maintains a ring buffer so the *same* frames feed back to AEC as playback reference, sample-aligned. Supports immediate flush (drops queued audio, emits one silence frame to settle) on barge-in. |
| **Watchdog** (F4) | `modes/talkback/watchdog.py` | `asyncio.create_task` that wakes every `watchdog.tick_ms` (default 500 ms) and checks silence + hard timeouts independently of chunk arrival. |
| **Event logger** (F6) | `core/logging/jsonl_logger.py` (new) | JSONL writer shared by `KioskPipeline` and `TalkbackController`. Path interpolates `{date}` and `{session_id}`. |

### Barge-in flow

The most important detail; it's what makes this "full-duplex" rather than "streaming half-duplex."

1. While Player is playing TTS, mic chunks continue flowing through AEC → VAD → Embed → Verify.
2. When Verify confirms primary-speaker speech **start** (Silero VAD says "speech started" + the smoother passes on the first usable chunk; minimum `barge_in.min_speech_ms` of sustained speech to debounce):
   - Player flushes immediately (queue cleared, one silence frame to the device).
   - The in-flight LLM HTTP stream is cancelled.
   - The TTS task is cancelled.
   - Any sentences already in the TTS queue are dropped.
   - STT continues capturing the interrupting utterance as the next user turn (turn boundary, **not** session boundary).
3. State transitions: `SPEAKING → BARGED_IN → LISTENING` (LISTENING is reached once the new utterance finalizes through STT).

**Speaker-verified cut, not noise-triggered.** With `barge_in.require_speaker_match: true` (default), background noise or other voices that fail the verifier do NOT cut TTS. This is configurable.

### End-of-utterance detection caveat

Without barge-in concerns, half-duplex assistants wait for Silero VAD's "speech ended" + a generous tail (~500 ms) before sending to LLM. With barge-in we want to start LLM ASAP so the response is ready. The default `end_of_utterance_tail_ms: 400` is the compromise. The "mid-utterance interruptible STT with re-LLM on new content" variant is a v2 problem and explicitly out of scope here.

## CLI

The existing `kiosk.py` CLI stays the entry point. New flag:

```
py -3.14 kiosk.py [--config config.yaml] [--wake-phrase hey_jarvis]
                  [--dry-run] [--talkback]
```

- `--talkback`: force `talkback_enabled: true` at the command line. Without it, the config's value wins (default: `false`). `--dry-run` continues to mean "print the existing kiosk events and DO NOT activate talkback" and is incompatible with `--talkback`. Passing both is a CLI error (exit 2 with a hint), not a silent override.

CLI prints for talkback mode (in addition to the existing kiosk lines):

```
[IDLE] Listening for "hey jarvis"...
[WAKE] phrase=hey_jarvis score=0.892
[SESSION STARTED] Primary speaker locked
[HANDOFF] → TalkbackController (turn 1)
[USER] "what's the weather like today"
[ASSISTANT] "I don't have live data, but I can pull from your calendar..."  ← streamed
[BARGE-IN] cut at 1240ms (primary score=0.71)
[USER] "actually, just summarize my next meeting"
[ASSISTANT] "Your 2pm with Sarah is about the Q3 launch plan..."
[SESSION ENDED] reason=silence_timeout (lasted 47s, 4 turns)
[IDLE] Listening for "hey jarvis"...
```

## Backend choices

All four below are pluggable behind interfaces (one-file swap), but these are the committed shipping defaults.

| Component | Default | Why this default |
|---|---|---|
| **STT** | `faster-whisper` `large-v3`, `float16`, `device=cuda` | Already a project dependency (Phase 2A). Blackwell makes `large-v3` ~10× real-time. Fallback: `distil-large-v3.en` if first-token latency is too high. |
| **LLM server** | `llama.cpp` server (`llama-server`) with the Blackwell CUDA build, OpenAI-compatible `/v1/chat/completions` with `stream=true` | Lightweight, builds cleanly on aarch64 + CUDA, GGUF ecosystem, streaming works out of the box. vLLM is faster at throughput but overkill for single-user. TRT-LLM is a rabbit hole. |
| **LLM model** | **Qwen 2.5 7B Instruct**, Q5_K_M or BF16 | Strong instruction-following at small size, ~150-250 ms first-token on Blackwell, ~5-8 GB VRAM. 128 GB headroom means trivial upgrade to 14B/32B later if quality matters more than latency. |
| **TTS** | **Kokoro-82M**, GPU, streaming sentence-by-sentence | Small, surprisingly high quality, CC-BY licensed, streams cleanly. Piper is the documented fallback when per-sentence latency dominates over voice quality. |
| **AEC** | `webrtc-audio-processing-py` (WebRTC APM, AEC3) | Same algorithm Chrome uses. Battle-tested, ~10 ms frames @ 16 kHz, well-understood failure modes. |
| **Audio I/O** | `sounddevice` **duplex** stream (PortAudio under the hood, ALSA/PipeWire on Linux) | Already a project dep. A single duplex stream gives input + output callbacks on the same clock, which makes AEC reference alignment vastly easier than two separate streams. |

### DGX Spark / aarch64 install notes

These are not design decisions, just facts the implementation plan must capture:

- `webrtc-audio-processing-py` doesn't ship aarch64 wheels. Install `libwebrtc-audio-processing-dev` via apt, then `pip install` from source.
- `ctranslate2` (faster-whisper's runtime) needs a CUDA-12.x build for Blackwell. Pin to a known-good combination.
- `llama.cpp` ships pre-built CUDA binaries for aarch64; recommended over building from source for the LLM server.
- All three (llama.cpp, ctranslate2, PyTorch for Kokoro) must agree on a single CUDA toolchain version. Pin in `requirements.txt`.

### Latency budget

Best-case, after warm-up, on Blackwell:

| Stage | Target |
|---|---|
| End of user speech → STT final | ~250 ms |
| STT final → first LLM token | ~150 ms |
| First LLM token → first TTS audio frame | ~250 ms |
| First TTS audio frame → speaker output | ~50 ms |
| **End-of-speech to first audio out** | **~700 ms** |
| Barge-in detect → TTS silent | **<100 ms** |

These are targets, not guarantees. If they don't land, downgrade in this order: Qwen 7B → 3B; faster-whisper large-v3 → distil-large-v3.en; Kokoro → Piper.

## Configuration

The existing `kiosk:` block (wake_phrase, wake_threshold, decision_smoother, timeouts) is unchanged. New `kiosk.talkback:` subsection is read only when `talkback_enabled: true`.

```yaml
kiosk:
  # ── existing fields preserved ──

  talkback_enabled: false             # default off; enable explicitly or via --talkback

  talkback:
    sample_rate_hz: 16000
    frame_ms: 10                      # pinned for webrtc-audio-processing
    output_device: null               # null = sounddevice default
    input_device: null

    aec:
      enabled: true
      suppression_level: "high"       # "low" | "moderate" | "high"

    stt:
      model: "large-v3"               # or "distil-large-v3.en"
      compute_type: "float16"
      device: "cuda"
      partials_every_ms: 300
      end_of_utterance_tail_ms: 400

    llm:
      base_url: "http://127.0.0.1:8080/v1"
      model: "qwen2.5-7b-instruct-q5_k_m"
      temperature: 0.6
      max_tokens: 512
      system_prompt: |
        You are a concise voice assistant. Replies should be 1-3 sentences,
        natural-sounding, and avoid lists, code blocks, or markdown.

    tts:
      backend: "kokoro"               # "kokoro" | "piper"
      voice: "af_bella"
      device: "cuda"

    chunker:
      sentence_terminators: [".", "?", "!"]
      max_chunk_chars: 120

    barge_in:
      enabled: true
      require_speaker_match: true
      min_speech_ms: 120

    watchdog:
      tick_ms: 500

    logging:
      jsonl_path: "logs/kiosk-{date}-{session_id}.jsonl"
      include_partial_transcripts: false
```

## F4 — Watchdog (idle-mic timeout fix)

Today `KioskPipeline` checks `silence_timeout` and `hard_timeout` only inside `_handle_active_chunk`, so a stalled mic = no timeouts ever fire.

**For the existing `KioskPipeline`** (chunk-iteration based): add a `threading.Thread` started in `run()` that ticks every `watchdog.tick_ms`. It reads `self._state`, `self._session.started_at`, and `self._session.last_speech_at` and calls `self._end_session(reason)` when a threshold is exceeded. Stop signal is a `threading.Event` shared with the main loop; thread is joined in the `finally` block. The thread holds no locks except for a brief read of the session timestamps; race with the main thread updating `last_speech_at` is benign (worst case: one extra tick of delay).

**For the new `TalkbackController`** (asyncio): `asyncio.create_task(self._watchdog())` that does `while running: await asyncio.sleep(tick); check_timeouts()`. Cancelled cleanly on shutdown.

Both call into the same shared end-session logic — no duplicated timeout policy.

## F6 — JSONL audit logging

A single `EventLogger` class in `core/logging/jsonl_logger.py` (new package). Constructor takes the path template + session-id generator. Method `log(event: str, payload: dict)` writes one JSON line per call with auto-injected `ts` (ISO8601 UTC), `session_id`, `event`, `payload`. Atomic line append (`open(path, "a")` is sufficient; the OS handles line-level atomicity for writes under PIPE_BUF).

The existing `KioskPipeline.on_event` callback is wired to `logger.log` in production. The new `TalkbackController` calls it directly.

Events emitted per session:

```
wake_detected               { phrase, score }
session_started             { snapshot_norm }
handoff_to_talkback         { primary_embedding_norm }            ← new
turn_started                { turn_number }                       ← talkback only
partial_transcript          { text, is_final: false }             ← talkback only, opt-in
user_turn_complete          { text, turn_number }                 ← talkback only
llm_request_sent            { messages_count, model }             ← talkback only
llm_response_started        { time_to_first_token_ms }            ← talkback only
llm_response_complete       { tokens, latency_ms }                ← talkback only
tts_started                 { sentence_count }                    ← talkback only
tts_completed               { audio_duration_ms }                 ← talkback only
barge_in                    { during_state, primary_score, cut_at_ms }  ← talkback only
watchdog_fired              { reason }                            ← new
segment_scored              { score, duration_ms, decision }
session_ended               { reason, turns, total_duration_ms }
```

Partial transcripts are off by default because they're noisy. Turn it on when debugging.

**F6 is also the 2c validation harness.** Once a second human is available, sessions where the second voice is an interloper produce `segment_scored` events with cosine scores tagged self/non-self in the JSONL. The non-self false-positive distribution becomes a `jq` one-liner. No separate harness code.

## Error handling

| Failure | Behavior |
|---|---|
| `llama.cpp` server unreachable at session start | Log `llm_unavailable`; play a pre-recorded "I can't reach my brain right now" clip via TTS if it's alive; return to IDLE without entering talkback |
| `llama.cpp` server crashes mid-stream | Cancel TTS, log; flush player; stay in LISTENING (don't kill session) |
| STT crashes mid-utterance | Log `stt_error`; drop this turn; stay in LISTENING |
| TTS crashes mid-sentence | Log `tts_error`; flush player; stay in LISTENING (user hears their response cut off, which is recoverable) |
| `webrtc-audio-processing` init fails | Hard exit with install-hint message. AEC isn't optional in this config. |
| Sounddevice loses the audio device mid-session | Watchdog catches it via silence timeout; session ends with reason `device_lost` |
| LLM stream stalls past `llm.max_tokens` worth of time | Cancel the request, log `llm_stall`, stay in LISTENING |
| Barge-in fires but the player flush leaves residual audio | Log `barge_in_residual_ms`; not a session-ending error |

All errors are logged as JSONL events. The pipeline survives every error in this table except APM init failure.

## Testing approach

Three layers, scaled by what they can verify without hardware.

### Layer 1 — Pure unit tests (no audio, no models, no network)

Everything below uses fakes/mocks and runs synchronously in < 5 s total. Marked default (no marker).

| Test file | What it pins down |
|---|---|
| `tests/kiosk/test_handoff.py` | `KioskPipeline` with `talkback_enabled=true` and a fake `TalkbackController` — assert hand-off fires exactly once on first ACTIVE_SESSION segment with the right `TalkbackHandoff` payload. |
| `tests/kiosk/talkback/test_controller_state.py` | TalkbackController driven with fake STT/LLM/TTS/Player. Assert state transitions: LISTENING → SPEAKING → BARGED_IN → LISTENING; LISTENING → ENDING on silence/hard timeout. |
| `tests/kiosk/talkback/test_conversation_manager.py` | Message-list assembly across multi-turn within one session. Verify system prompt is first; user/assistant alternation; reset between sessions. |
| `tests/kiosk/talkback/test_sentence_chunker.py` | Token stream in → sentence chunks out. Pin behavior on `.`/`?`/`!`, max-chars cutoff, abbreviation edge cases ("Dr.", "U.S."), trailing-fragment flush. |
| `tests/kiosk/talkback/test_barge_in.py` | Fake TTS-in-progress + inject verified primary speech. Assert: player flushed, LLM cancelled, TTS cancelled, next STT path open within one tick. Also: non-primary speech does NOT cut when `require_speaker_match: true`. |
| `tests/kiosk/talkback/test_watchdog.py` | Fake clock, no chunk arrival. Assert silence + hard timeouts fire within one `tick_ms`. Tested for both `KioskPipeline` and `TalkbackController`. |
| `tests/core/logging/test_jsonl_logger.py` | Path templating, ISO8601 timestamps, session-id propagation, line-atomic append, file rotation per session. |

### Layer 2 — Component integration (real backends, no mic/speaker)

Marked `@pytest.mark.integration`. Run with `pytest -m integration`. Each gated on the backend being installed.

- **STT integration:** feed the committed 1-second speech fixture (already in repo from Phase 4's 1d work) through streaming STT, assert text is plausible.
- **LLM integration:** ping the local `llama.cpp` server with a fixed prompt + `stream=true`, assert tokens arrive and first-token latency is under a generous bound (1 s).
- **TTS integration:** generate one sentence with Kokoro, assert audio comes back at expected sample rate and approximate length.
- **AEC integration:** synthetic — generate a known sine on the playback reference, mix into mic input at known SNR, run through APM, assert > 15 dB suppression at the sine's frequency. No real audio hardware needed.

### Layer 3 — End-to-end with mic + speaker (manual, not in CI)

`tests/kiosk/talkback/test_e2e_live.py.skip`. One golden conversation: wake → handoff → LLM response → TTS playback → barge-in → second turn → silence timeout → IDLE. Verifies F6 JSONL output sequence + F4 watchdog under manual mic disconnect.

### Existing 252 tests must remain green

The hand-off change is gated on `talkback_enabled` (default `false` in `config.yaml`), so behavior in the existing test suite is unchanged.

## Quality expectations and known limitations

- **AEC effectiveness depends on the room.** WebRTC APM is good but not perfect. Highly reverberant rooms degrade the AEC; this can produce barge-in false positives where the kiosk hears its own echo as primary speech. Mitigations: raise `barge_in.min_speech_ms`, lower `aec.suppression_level`, or use a closer-talking mic.
- **First-token latency dominates perceived responsiveness.** If the latency budget slips beyond ~1.5 s, the kiosk feels broken. Downgrade path (smaller model, distilled STT, Piper TTS) is the primary mitigation.
- **Barge-in is not interrupt-by-anyone.** Only the locked primary can cut TTS. This is by design (kiosk-in-public property) but it means a co-located helper cannot break in. Flip `barge_in.require_speaker_match` if your use case wants the opposite.
- **Single conversation session, no cross-session memory.** Each silence timeout discards the message list. Persistent memory is out of scope; revisit when there's a real ask.
- **End-of-utterance detection is imperfect.** Slow speakers with thoughtful pauses may have their turn finalized prematurely (after `end_of_utterance_tail_ms`). The 400 ms default is the compromise; tune up if needed.
- **Hot-swap of LLM model at runtime is not supported.** Config changes require restart.

## Out of scope

- Tool / function calling. The voice assistant does chat only. Separate spec when it needs to do anything that touches the rest of the system.
- Persistence across sessions (each silence timeout clears memory).
- Speaker authentication via enrolled voiceprints (Variant B from 2026-05-14; still deferred).
- Custom wake-phrase training.
- Multi-user / overlapping sessions.
- Network or multi-room audio.
- Localization beyond English (STT model is multilingual-capable but TTS voice + system prompt assume English).
- Persona presets beyond the configurable `system_prompt`.
- The actual Batch 2c non-self FP validation **run** (the JSONL harness ships; the run waits for a second human).
- Replacing the existing CPU-only TVAD offline pipeline behavior. This spec adds a GPU path that activates only when talkback is enabled.

## Open questions

None at the time of writing. All decisions resolved 2026-05-23.

## Implementation order (informational, not binding)

The writing-plans pass will refine this into a TDD task list. Rough sequencing for context:

1. Hand-off scaffolding: `TalkbackHandoff`/`TalkbackResult` types, `talkback_enabled` config flag, `KioskPipeline` change with a no-op `TalkbackController` stub. Existing tests stay green; one new test for the hand-off itself.
2. `EventLogger` (F6) — useful immediately, no dependencies. Wire `KioskPipeline.on_event` to it.
3. `KioskPipeline` watchdog thread (F4 — the existing-pipeline half).
4. `Player` + sounddevice duplex stream + playback-reference ring buffer. No AEC yet; just plays silence and records the reference. Smoke-testable.
5. AEC wrapper with the synthetic-sine Layer 2 test.
6. Streaming STT wrapper (real `faster-whisper`).
7. LLM client (HTTP streaming) + Conversation Manager + sentence chunker.
8. TTS wrapper (Kokoro first, Piper as the swap).
9. Full `TalkbackController` state machine + `TalkbackController` watchdog (F4 — the new-pipeline half).
10. Barge-in wiring (cancellation across LLM/TTS/Player).
11. Layer 1 unit tests across the new modules.
12. Layer 2 integration tests behind `-m integration`.
13. Manual Layer 3 end-to-end + first real conversation.
