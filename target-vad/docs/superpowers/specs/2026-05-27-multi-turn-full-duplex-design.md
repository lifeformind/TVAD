# Multi-Turn Full-Duplex Conversation Loop

**Date:** 2026-05-27
**Status:** Approved

## Problem

The TalkbackController currently processes only the first speech segment from the wake-word handoff, generates one LLM response with TTS playback, then waits for the watchdog to fire `silence_timeout` — ending the session. The mic stream continues buffering in the background but nobody consumes those chunks. There is no second turn.

## Goal

Enable multi-turn voice conversation within a single wake-word session. The mic stays hot during TTS playback (true full-duplex). Speaker-verified barge-in allows the primary speaker to interrupt the assistant mid-response.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mic/VAD sharing | Reuse pipeline's instances via handoff | No duplicate model memory; pipeline loop is blocked during handoff so no contention |
| Duplex mode | True full-duplex (mic hot during playback) | Enables real-time barge-in; AEC cleans echo |
| Barge-in verification | Speaker embedding check via pipeline's ECAPA embedder | Prevents echo/bystander from cutting TTS |
| Mic reading architecture | Async task with run_in_executor | Keeps everything in asyncio-land; clean shutdown via task cancellation; negligible executor overhead per ~30ms chunk |

## Handoff Contract Changes

`TalkbackHandoff` gains two fields:

```python
@dataclass
class TalkbackHandoff:
    mic: Any                       # existing — MicrophoneStream (already open)
    primary_embedding: np.ndarray  # existing — locked speaker snapshot
    first_segment: Any             # existing — SpeechSegment that triggered session
    config: dict                   # existing — talkback config subtree
    vad: Any                       # NEW — pipeline's SileroVAD instance
    embedder: Any                  # NEW — pipeline's EmbeddingExtractor
```

Pipeline passes its own `vad` and `embedder`. Pipeline's `run()` loop blocks at `controller.run(handoff)`, so there is no concurrent access.

## Controller Architecture

Three concurrent async tasks inside `_run_async`:

| Task | Responsibility |
|------|---------------|
| `_listen_loop` | Reads mic chunks via executor → AEC (during SPEAKING only) → VAD → pushes SpeechSegments to `_segment_queue` |
| `_response_task` | Streams LLM → sentence chunker → TTS → sd.play. One at a time. Cancellable for barge-in |
| Watchdog | Unchanged — ticks and fires silence_timeout / hard_timeout |

The main `_run_async` loop coordinates: pulls segments from the queue, decides action based on state, manages task lifecycle.

## State Machine

```
                    ┌──────────────┐
    wake+segment    │              │  silence_timeout / hard_timeout
   ──────────────►  │  LISTENING   │  ──────────────────────────────► END
                    │              │
                    └──────┬───────┘
                           │ segment transcribed
                           ▼
                    ┌──────────────┐
                    │              │  response complete
                    │  SPEAKING    │  ──────────────► LISTENING
                    │              │
                    └──────┬───────┘
                           │ primary speaker barge-in
                           ▼
                    ┌──────────────┐
                    │  BARGED_IN   │  → cancel response → transcribe → SPEAKING
                    └──────────────┘
```

Key change: **SPEAKING → LISTENING → SPEAKING cycles** instead of SPEAKING → sleep-wait → END.

## Listen Loop Data Flow

```
mic.stream() ─── [executor] ───► chunk
                                   │
                    ┌──────────────┤
                    │ if SPEAKING  │ if LISTENING
                    ▼              ▼
              AEC(chunk, ref)    chunk (passthrough)
                    │              │
                    └──────┬───────┘
                           ▼
                     VAD.process_chunk()
                           │
                    segments (if any)
                           ▼
                    _segment_queue.put()
```

AEC uses `Player.get_reference_frame()` for the playback reference. Mic chunks (480 samples) are split into AEC-frame-sized pieces (160 samples for 10ms at 16kHz).

## Barge-In Flow

During SPEAKING state, when a segment arrives from the queue:

1. Extract speaker embedding via `embedder.extract(segment.audio)` in executor
2. Compare against `primary_embedding` via `cosine_similarity` 
3. If score >= threshold (from `config.barge_in`): cancel `_response_task`, call `sd.stop()`, flush player queue, transition to BARGED_IN, transcribe segment, start new response
4. If score < threshold: ignore (background noise, echo residual, or bystander)

## Playback Cancellation

Currently `_generate_response` calls `sd.play(audio, blocking=True)` in an executor. For barge-in cancellability:

- On barge-in: call `sd.stop()` to halt playback immediately
- The cancelled task's executor-running `sd.play` returns early once `sd.stop()` is called
- Player queue is flushed via `player.flush()`
- LLM stream cancelled via `llm.cancel()`

## Silence Timer Behavior

`_last_speech_at` resets when the listen loop detects a speech segment (not when transcription completes). This prevents the watchdog from firing during STT latency. The silence timeout only triggers when the user genuinely stops talking.

## Files Changed

| File | Change |
|------|--------|
| `modes/talkback/handoff.py` | Add `vad`, `embedder` fields to TalkbackHandoff |
| `modes/talkback/controller.py` | Rewrite `_run_async`: add `_listen_loop`, segment queue, multi-turn main loop, barge-in handling |
| `modes/kiosk/pipeline.py` | Pass `self.vad` and `self.embedder` in TalkbackHandoff construction |
| `tests/kiosk/talkback/test_controller.py` | Update for multi-turn scenarios |
| New: `tests/kiosk/talkback/test_multi_turn.py` | Multi-turn conversation tests |
| New: `tests/kiosk/talkback/test_barge_in.py` | Barge-in with speaker verification tests |

## Config

No new config keys. Existing keys used:

- `barge_in.enabled` — gate on barge-in checking
- `barge_in.require_speaker_match` — gate on embedding verification
- `barge_in.min_speech_ms` — minimum segment duration before barge-in triggers
- `aec.enabled` — gate on AEC processing during SPEAKING
- `silence_timeout_s` — inactivity timeout (resets each turn)
- `hard_timeout_s` — absolute session wall clock limit
