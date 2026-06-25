# Conversation Director — Design (talkback rebuild)

**Status:** DRAFT for build — decisions encoded, not open for re-litigation
**Date:** 2026-06-21 (§4, §7, §9, §12, §13, §14 revised 2026-06-24 — FOCUS moved from
acoustic pVAD to camera presence+identity; see
`2026-06-24-director-floor-control-design.md`. §13 step 5 camera floor control
SHIPPED 2026-06-25 — Director-07 merged to master, see
`docs/notes/2026-06-24-director-07-live.md`.)
**Mode:** Ground-up rebuild of the talkback conversation core (single-threaded async "Conversation Director")
**Target:** LOCAL/OFFLINE on one NVIDIA DGX Spark (GB10 Grace-Blackwell, ~128GB unified, ONE GPU, aarch64). Python 3.12.

---

## 1. Problem

The kiosk must converse like a human in a crowd: lock onto one enrolled speaker, ignore everyone else, let that user interrupt mid-reply and resume, keep talking through "mhm" but stop for "wait, why?", and never time out while it is itself talking. The current `KioskPipeline` + `TalkbackController` stack cannot deliver this, for four structural reasons that motivate the rebuild:

1. **Double-managed session (the live bug, HARD REQ 5).** Two components own the session lifecycle. `KioskPipeline._start_session_from_segment()` calls `self._talkback_controller.run(handoff)` at **pipeline.py:200**, which *blocks the pipeline thread for the entire conversation*. The pipeline therefore never re-enters `_handle_active_chunk()` and never refreshes `Session.last_speech_at` (only updated at **pipeline.py:242**). Meanwhile a daemon **watchdog thread (pipeline.py:104-113)** keeps reading `Session.silence_duration()` against a frozen `last_speech_at`, so it fires `_end_session("silence_timeout")` **~30s after session START**, regardless of activity. `_end_session()` prints `[SESSION ENDED]`/`[IDLE]` but does **not** stop the controller — talkback keeps answering with resume state intact, no wake required, while the pipeline FSM is already back in IDLE and listening for a new wake word. Two silence timers (`kiosk.session_silence_timeout_s` and `kiosk.talkback.silence_timeout_s`, both 30s — **config.yaml:30 and 52**) race. **Note the bug has two independent causes that BOTH must be removed:** (a) the blocking call's side effect — the pipeline never refreshes `last_speech_at` — and (b) the watchdog thread reading that frozen value. Deleting only the watchdog leaves a component that can never measure real silence; deleting only the blocking call leaves a second timeout authority. The Director eliminates both by being the sole owner.

2. **Cut-before-STT in the wrong order (historical) and a phantom state today.** The old `BARGED_IN` design cut the reply on raw VAD onset, before transcribing or classifying — so any cough or bystander killed the turn. The validated Spike-1 reorder (*duck → proximity-gate → min-duration reject → speaker-match → transcribe → classify → cut|restore*) now lives in the controller, but it is **not** a separate resting state: it runs entirely inside the `SPEAKING` branch of `_handle_segment` (**controller.py:650-742**), with the onset-duck in `_listen_loop` (**controller.py:839-849**). `TalkbackState.BARGED_IN` is set only *after* the cut+drain+cancel, inside `_handle_barge_in` (**controller.py:871**), and is immediately left for `SPEAKING` again (**controller.py:736**). So there is no named "ducked, capturing, deciding" state — the logic is bolted into `SPEAKING`, which makes it untestable in isolation and impossible to reason about. The Director promotes this phase to a first-class `EVALUATING` state (Section 4, Section 13).

3. **ECAPA on the hot path.** Per-turn ECAPA verification (108ms p95 on GB10 CPU) is too slow for a <100ms reflex and unreliable on <2-3s segments — it rejects the real user (see MEMORY: `ecapa-short-segment-unreliable.md`). Today it runs synchronously inside the barge-in branch (**controller.py:681-693**, `run_in_executor` but awaited on the decision path). Speaker identity belongs off the synchronous decision path.

4. **Timeout semantics.** Silence must accrue only while *waiting for the user*, never while the kiosk thinks or speaks. The split-brain timers above make this impossible to reason about.

The fix is one component that owns wake → session → teardown and all timers, with an explicit, testable FSM and identity/STT pushed into parallel workers.

---

## 2. Goals and Non-goals

**Goals (the 5 hard requirements):**

1. **FOCUS.** Once an enrolled speaker triggers the kiosk, lock onto that speaker; ignore all other speakers and background noise.
2. **INTERRUPT + RESUME.** The focused user can interrupt mid-reply with a clarifying question, get it answered, then have the kiosk continue its prior point ("as I was saying…").
3. **BACKCHANNEL vs QUESTION.** Keep talking through acknowledgments ("okay/yeah/mhm"); stop and answer genuine questions/commands ("why?/wait/stop").
4. **HUMAN-LIKE SILENCE.** Suspend the silence timeout while thinking/speaking; count true user-silence only while WAITING; restart the grace window each time the kiosk yields the floor; emit an "are you still there?" nudge before ending.
5. **SINGLE SESSION OWNERSHIP.** The Director is the sole owner of session lifecycle and all timers (one watchdog), subsuming wake → handoff → session → teardown. No second timeout authority.

**Non-goals / out of scope:**

- Speaker *enrollment capture UX* beyond reading a finalized embedding for the session and (new, Section 7) holding out one pre-finalize utterance embedding for verify-before-serve. The Director does not own the multi-utterance capture flow.
- Multi-party turn allocation (we lock to *one* speaker by design).
- Cloud/online fallback of any kind (fully local/offline).
- Training a custom TS-VAD (that is V2; V1 ships a borrowed pVAD).
- Non-English (English-only, matching `language="en"` in `StreamingStt._transcribe_sync`, **stt.py:45**).
- Wake-word redesign (`WakeWordDetector` is reused as-is).

---

## 3. Architecture overview

**Core principle: parallelism in the workers, decision-making serialized.** A single asyncio event loop runs the **Director**, which owns an explicit FSM and a **Context** blackboard and is the *only* mutator of state. Every model (STT, TTS, LLM, pVAD, ECAPA, Smart Turn) runs as a **stateless worker** in parallel, communicating *only* through an async **event bus**: workers emit events in, the Director emits commands out. This kills the race classes by construction — no two code paths mutate state concurrently, because only the Director mutates, and it does so one event at a time.

```
                         ┌─────────────────────────────────────────────┐
   IDLE/AWAIT (thin)     │            CONVERSATION DIRECTOR              │
  ┌───────────────┐      │   single asyncio loop · SOLE state mutator   │
  │   WakeGate    │ run()│  ┌──────────┐         ┌──────────────────┐   │
  │ wake + 1st seg│─────▶│  │   FSM    │◀───────▶│  Context (board) │   │
  │  + ECAPA snap │      │  │ IDLE/LIS │ mutate  │ embedding, msgs, │   │
  └───────────────┘      │  │ THK/SPK/ │         │ gen_id, stack,   │   │
                         │  │  EVAL    │         │ _started_at,     │   │
                         │  └────┬─────┘         │ _last_speech_at, │   │
                         │       │ commands      │ _nudged_cycle    │   │
                         │   ┌───┴────────── ASYNC EVENT BUS ───────┴┐  │
                         └───┼───────────────────────────────────────┼──┘
                             │            │           │          │
                    ┌────────▼──┐  ┌──────▼─────┐ ┌───▼────┐ ┌───▼──────┐
   MIC ─chunks──────▶ Ingestion │  │ STT worker │ │  LLM   │ │ Playback │──▶ SPKR
   (executor thread) │  (CPU):  │  │  (GPU):    │ │ main+  │ │ +AEC     │
                     │ Silero + │  │ NeMo /     │ │arbiter │ │ worker:  │
                     │ pVAD +   │  │ openai-    │ │ (GPU,  │ │ TTS(GPU) │
                     │ SmartTurn│  │ whisper    │ │  warm) │ │ +Player  │
                     │ + AEC    │  │ (NOT       │ └────────┘ │ +sd.Out  │
                     │ + ECAPA  │  │ faster-w)  │            │ (executor│
                     │ (→exec)  │  └────────────┘            │  thread) │
                     └──────────┘                            └──────────┘
```

**Concurrency model:**

- **Async core.** One event loop. The FSM transition function is synchronous and runs to completion per event — no `await` inside the mutator. As today, `Director.run(handoff)` spins a fresh loop with `asyncio.new_event_loop()` (**controller.py:247-255**) and runs `_run_async` to completion; from the WakeGate's blocking call it is fully synchronous and returns only at true session end (Section 4, Req-5 proof).
- **Warm LLM servers.** The main LLM (gemma) and the small arbiter LLM are separate local `llama.cpp /v1/chat/completions` servers, reached via two `LlmClient` instances (`modes/talkback/llm.py`). Both warm-loaded; `ping()` gates session start. See Section 9 for the close-then-ping lifecycle change required to run two clients.
- **CPU-pinned reflex specialists.** Silero VAD, the pVAD crowd-focus model, AEC, and Smart Turn run on **CPU** (insulated from GPU contention per the spike). Synchronous-but-short calls (Smart Turn ~27-55ms, ECAPA ~108ms) go through `run_in_executor` so the loop never blocks.
- **Audio I/O threads bridge to the loop.** `MicrophoneStream.stream()` (blocking generator) is consumed via `run_in_executor`; playback writes to `sd.OutputStream` inside a dedicated executor future. Both bridge thread→loop by posting events onto the bus.

---

## 4. State machine & floor control

Five explicit states. `EVALUATING` is a **new** state, not a rename of `BARGED_IN` — see the migration note below.

| State | Meaning | Floor owner | Silence accrues? |
|---|---|---|---|
| **IDLE** | No session (lives in the thin WakeGate) | — | n/a |
| **LISTENING** | Waiting for / capturing the user's turn | User | **YES** |
| **THINKING** | LLM generating, no audible TTS yet | Kiosk | NO |
| **SPEAKING** | TTS audio playing out | Kiosk | NO |
| **EVALUATING** | Near-field onset during SPEAKING: ducked, capturing, deciding whether to yield | Contested | NO |

```
        wake + 1st VAD seg + ECAPA snapshot (WakeGate → Director.run)
                              │
                              ▼
        ┌───────────────▶ LISTENING ◀───────────── restore (backchannel/bystander/too-short)
        │   user turn end │      ▲                          │
        │  (endpoint)     ▼      │ cut & answer            duck on near-field onset
        │             THINKING   │ (question/command)        │
   yield floor            │      │                           ▼
   (reset silence,        │ 1st  └──────────────────────  EVALUATING
    clear _nudged)        │ TTS                            (proximity → min-dur →
        │                 ▼ frame                           transcribe → classify)
        └──────────── SPEAKING ──────near-field onset──────▶
                  reply complete → LISTENING (yield floor, reset silence, clear _nudged)
```

**THINKING entry/exit (new state — defined precisely, since the code has none today).** The current `_handle_segment` transitions `LISTENING → SPEAKING` directly (**controller.py:642**). The Director inserts THINKING between them:

- **Enter THINKING** on the event `llm_request_sent` — emitted the instant the Director dispatches the generation task (`asyncio.create_task(_generate_and_speak(...))`, today **controller.py:644-647**). This is the moment the floor leaves the user.
- **Exit THINKING → SPEAKING** on the event `first_tts_frame_written` — emitted by the playback worker when the *first* post-gain frame reaches `sd.write()` for this `gen_id`. Until then no audio is audible.
- **Silence semantics:** `_silence_duration()` already returns `0.0` for every state that is not LISTENING (**controller.py:150-151**), so THINKING and EVALUATING are *already covered* — no new branch is needed; just update the docstring (which names `BARGED_IN`) to name THINKING/EVALUATING. Floor=Kiosk, silence=NO follows automatically.

**Floor control rules:**

- On **SPEAKING**, `SileroVAD.is_speaking` going True at a near-field onset (gated by RMS ≥ `proximity_rms`, **controller.py:839-849**) triggers an *immediate duck* (gain drop to `barge_in.duck_level`) and transition **SPEAKING → EVALUATING** — before any STT. This is the <100ms reflex (Section 6).
- **EVALUATING** resolves to exactly one of: **CUT** (→ THINKING then SPEAKING, answer the interjection) or **RESTORE** (→ SPEAKING, un-duck and keep talking). Default-to-cut on *semantic* ambiguity, but **never** cut on the structural reject gates below (too-short, far, speaker-mismatch, empty/low-confidence) — those always RESTORE. Auto-resume (Section 8) backs any wrong cut.
- Every transition **into LISTENING** is "yielding the floor": it resets `_last_speech_at` (**controller.py:140-143**) *and* clears the per-cycle `_nudged` flag (Section 5).

**Camera presence input (added 2026-06-24, §7-revised).** A `VisionWorker` emits
`OwnerPresenceEvent(status ∈ {PRESENT, ABSENT, UNAVAILABLE})` onto the bus on debounced
changes. The reducer records it in Context (no transition) and adds **one new
end-condition inside `_on_tick`**: a sustained valid `ABSENT` (≥ `owner_absent_grace_s`)
with an active-talk guard → `EndSession("owner_absent")` (frees the kiosk fast when the
owner leaves; a stranger reads as `ABSENT`, so this also covers owner-changed). This is
purely additive: the five states are unchanged, the §5 silence timeout/nudge are
unchanged, and the watchdog stays the sole timeout authority (§4a) — it is a *condition*
inside the existing tick, not a competing timer. `UNAVAILABLE`/absent-camera degrades to
today's audio-only behavior. Detail: `2026-06-24-director-floor-control-design.md`.

**MIGRATION NOTE (corrects "rename BARGED_IN").** Promoting EVALUATING is a **split of the SPEAKING branch**, not a rename:

- The onset-duck block (**controller.py:839-849**, currently inside `_listen_loop` while `state == SPEAKING`) emits a new `near_field_onset` event; the reducer consumes it to transition `SPEAKING → EVALUATING` and emit a `duck` command.
- The entire decision body (**controller.py:650-742**: proximity pre-gate 659-669, min-duration reject 673-679, speaker-match 681-693, transcribe 700, classify 705-710, cut path 715-742) moves out of the `SPEAKING` branch of `_handle_segment` and becomes the body of the **EVALUATING** branch.
- The old `BARGED_IN` enum value and `_handle_barge_in`'s `_transition(BARGED_IN)` (**controller.py:871**) are deleted; the cut path transitions `EVALUATING → THINKING` (via `llm_request_sent`) → `SPEAKING`, never through a `BARGED_IN` resting state.

**Single session ownership.** The Director subsumes wake → handoff → session → teardown:

- The thin **WakeGate** retains only IDLE + AWAIT_FIRST_SEGMENT (wake detection, `awaiting_speech_timeout_s`, first-segment ECAPA snapshot). It builds one `DirectorHandoff(mic, primary_embedding, holdout_embedding, first_segment, config, vad, embedder)` and makes one blocking call `director.run(handoff)` — the same call shape as today's `controller.run()` — then resets to IDLE on return. It owns **no** `Session` object, **no** silence timer, **no** hard-timeout timer.
- The pipeline watchdog thread (**pipeline.py:104-113**) is **deleted outright** — no replacement. The Director runs the *single* `AsyncWatchdog` (Section 5). There is exactly one timeout authority.

---

## 4a. Req-5 single-ownership proof (post-conditions, not just deletions)

The single-owner property must be *provable*, not asserted. After the rebuild the following post-conditions hold and are grep-checkable in CI:

1. **WakeGate holds no session state and no timeout path.** Grep asserts the absence of `_watchdog`, `_start_watchdog`, `_stop_watchdog`, `_end_session`, `Session(`, `last_speech_at`, `silence_timeout`, `hard_timeout`, and any `_silence_duration` in the WakeGate module. The WakeGate's only fields are wake/first-segment capture state.
2. **`director.run(handoff)` is fully synchronous from the WakeGate's view.** It spins its own loop (`asyncio.new_event_loop()`, **controller.py:247**) and `loop.run_until_complete(self._run_async(handoff))`, returning a `DirectorResult` only at true session end (timeout, hard cap, lockout, or KeyboardInterrupt). There is no concurrent pipeline activity during the conversation — the WakeGate thread is parked inside this call.
3. **On return the WakeGate does exactly one thing: reset to IDLE.** No `_end_session`, no second "reason" authority, no re-entrant teardown. The session reason originates solely from `DirectorResult.reason` (the Director's `_handle_timeout`, **controller.py:855-863**).
4. **Both root causes of the live bug are gone.** (a) The blocking call no longer starves a `last_speech_at` refresher, because the Director *is* the thing refreshing `_last_speech_at` (**controller.py:852**) on every segment; (b) the pipeline watchdog reading a frozen value is deleted. The deleted config keys `kiosk.session_silence_timeout_s` / `kiosk.session_hard_timeout_s` (**config.yaml:30-31**) are removed so no second timeout value can be read.

---

## 5. Timeout & nudge model

One timer, one authority: the Director's `AsyncWatchdog`. Today this class is **single-shot** — its `_loop` checks hard-cap then silence and `return`s after firing `on_timeout` (**watchdog.py:43-49**). The nudge requires a **new, non-terminal** path. This is genuinely new work; Section 10 lists the watchdog as "reused core, extended," NOT "as-is."

**New `AsyncWatchdog` contract:**

```python
AsyncWatchdog(
    tick_s,
    on_timeout,              # terminal: stops the loop (hard_cap or silence)
    on_nudge,                # NON-terminal: must NOT return/stop the loop
    get_silence_duration,    # 0.0 outside LISTENING (controller.py:150-151)
    get_session_duration,
    silence_timeout_s,
    hard_timeout_s,
    nudge_lead_s,            # NEW: fire nudge this many seconds before silence_timeout
    is_nudged,              # NEW: callback → bool, "already nudged this LISTENING cycle?"
    mark_nudged,            # NEW: callback, set the per-cycle flag
)
```

`_loop` becomes:

```python
while True:
    await sleep(tick_s)
    if get_session_duration() >= hard_timeout_s:
        on_timeout("hard_timeout"); return          # terminal
    sil = get_silence_duration()
    if sil >= silence_timeout_s:
        on_timeout("silence_timeout"); return        # terminal
    if sil >= (silence_timeout_s - nudge_lead_s) and not is_nudged():
        mark_nudged(); on_nudge()                    # NON-terminal: loop continues
```

**Director-side semantics:**

- **Silence accrues only in LISTENING.** `get_silence_duration()` returns `0.0` in every state except LISTENING — the exact semantics already at **controller.py:150-151** (unchanged; only the docstring's `BARGED_IN` reference is updated to THINKING/EVALUATING).
- **Reset + nudge-flag clear on yielding the floor.** Entering LISTENING sets `_last_speech_at = time.monotonic()` (**controller.py:140-143**) *and* `self._nudged_cycle = False`. This ties the once-per-cycle nudge flag to the same transition that resets the silence clock, so the nudge can fire **at most once per LISTENING entry** and re-arms on the next entry. `is_nudged`/`mark_nudged` read/write `self._nudged_cycle`.
- **Still-there nudge.** `on_nudge` makes the Director speak "Are you still there?" via **direct TTS** (no LLM round-trip), inside the playback worker, without leaving LISTENING (silence keeps accruing toward the real timeout). If the user replies, the normal LISTENING reset clears `_nudged_cycle`; if not, the silence timeout fires as usual.
- **Hard cap.** `hard_timeout_s` bounds total session length regardless of activity and is checked *before* the silence/nudge checks.
- **Authoritative config values.** `kiosk.talkback.silence_timeout_s = 30` (**config.yaml:52**), `kiosk.talkback.hard_timeout_s = 300` (**config.yaml:53**), and a **new** `kiosk.talkback.nudge_lead_s` (default **5**, so the nudge fires at silence=25s of a 30s window). The invariant `0 < nudge_lead_s < silence_timeout_s` is asserted at session start. The `_run_async` fallback default `silence_timeout = config.get("silence_timeout_s", 10.0)` (**controller.py:275**) is retained but never reached for the shipped config (the key is present). The now-dead `kiosk.session_silence_timeout_s` / `kiosk.session_hard_timeout_s` (**config.yaml:30-31**, consumed only by the deleted pipeline watchdog) are **removed**.

---

## 6. Turn-taking brain (reflex + arbiter)

A hybrid brain, mirroring FireRedChat (arXiv 2509.06502): a **reflex** layer makes the hot-path duck/keep/cut call with no LLM; a small **arbiter** LLM resolves only ambiguous semantic cases, off the hot path.

**Reflex layer (CPU, <100ms):** the only thing on the synchronous decision path.

- **Onset → immediate duck.** `SileroVAD.is_speaking` per mic chunk; on near-field onset during SPEAKING (RMS-gated, **controller.py:839-849**), duck and enter EVALUATING. No model call yet.
- **Turn completion in LISTENING.** After Silero yields a `SpeechSegment`, run `SmartTurnDetector.endpoint_prob()` (`modes/talkback/endpointing.py`, ~27-55ms p95 CPU) via `run_in_executor`. Below threshold (e.g. 0.5) → keep accumulating; at/above → the turn is complete, advance to THINKING. `NullTurnDetector` (always 1.0) is the test/CI stub via the `TurnDetector` Protocol.
- **Interjection classification.** In EVALUATING, after the captured **endpointed segment** is transcribed (see flow below), call `classify_interjection(text)` (`modes/talkback/intent.py:57`) — a **pure** ~microsecond lexical function, called *synchronously* in the loop. BACKCHANNEL → RESTORE; INTERRUPT → CUT. Default is INTERRUPT (default-to-cut).

**EVALUATING flow** (the validated Spike-1 reorder, now an explicit state, with the real gates that exist in code):

```
SPEAKING + near-field onset (is_speaking ∧ rms ≥ proximity_rms)
   └─▶ duck (gain → duck_level)                          [reflex, immediate; ctrl 839-849]
        └─▶ accumulate the near-field VAD segment        [endpointed, NOT a streaming partial]
             └─▶ PROXIMITY pre-gate: rms < proximity_rms?  ──yes──▶ RESTORE   [ctrl 659-669]
             └─▶ MIN-DURATION reject:
                   duration_ms < barge_in.verify_window_ms? ──yes──▶ RESTORE   [ctrl 673-679]
                   (config value 700ms — see "verify_window_ms" note below)
             └─▶ SPEAKER-match (off hot path, run_in_executor ECAPA OR pVAD is_target):
                   score < barge_in.speaker_threshold? ──yes──▶ RESTORE        [ctrl 681-693]
             └─▶ transcribe_segment(audio) → (text, mean_word_prob)            [ctrl 700; EXTENDED sig]
             └─▶ EMPTY or mean_word_prob < conf_floor? ──yes──▶ RESTORE
                   (classify_interjection maps empty→BACKCHANNEL anyway, intent.py:59)
             └─▶ classify_interjection(text)                                    [ctrl 705]
                   ├─ BACKCHANNEL ─▶ RESTORE (un-duck, keep talking)            [ctrl 708-710]
                   └─ INTERRUPT   ─▶ CUT (drain+cancel → answer; push resume)   [ctrl 715-742]
```

**Endpointed, not "partial."** The validated reorder transcribes the **full endpointed VAD segment** (`self._stt.transcribe_segment(segment.audio)`, **controller.py:700**); there is **no** streaming-partial path in the reused code, and the spike never built one. The spec ships V1 on full-segment EVALUATING transcription. The latency budget for "duck → classify → cut" is therefore: onset-duck (immediate) + segment endpoint wait (VAD-bounded) + ECAPA/pVAD (executor, off the synchronous tick) + STT (GPU worker) + `classify_interjection` (µs). The **<100ms reflex budget applies only to the onset-duck** (the perceptible reaction); the cut/restore decision completes after the captured segment endpoints and does not block the audio loop.

**`verify_window_ms` — the short-interjection reject gate (must be documented).** Before any speaker-match or STT, EVALUATING rejects segments shorter than `barge_in.verify_window_ms` as `too_short_to_verify` and RESTOREs (**controller.py:673-679**). The **code default is 1200ms but the shipped config is 700ms** (**config.yaml:131**); 1200 rejected most real barge-ins. **Single source of truth: `config.yaml` (700ms).** Implementers must know short interjections are dropped before classification — usable short barge-ins depend on this value, guarded against bystanders by the proximity gate (see MEMORY: `aec-noop-playback-bypass.md`).

**Empty/low-confidence guard (requires an extended STT signature).** `classify_interjection` Stage 0 already maps empty/punctuation-only text to BACKCHANNEL (`intent.py:59`) — never cut on empty STT. To *also* RESTORE on low-confidence garbage, the Director needs per-segment confidence, which **`transcribe_segment` does not return today** (it returns a bare `str`, **stt.py:37-51**, and does not pass `word_timestamps=True`). The signature is therefore **extended** on every backend (Section 9) to `transcribe_segment(audio) -> TranscriptResult(text: str, mean_word_prob: float)`; the Director RESTOREs when `mean_word_prob < conf_floor` (new config `barge_in.conf_floor`, default 0.5). This is new work, not a property of the reused file.

**Arbiter LLM (off hot path).** When the reflex result is genuinely ambiguous (e.g. lexical default-to-cut on an unrecognized phrase that *might* be a backchannel), the Director may consult the small arbiter `LlmClient` *after* it has already made the safe reflex call. The arbiter never sits in the duck-reaction path; its job is to catch wrong cuts early and trigger the auto-resume net (Section 8) faster, or confirm a borderline keep. This split — specialist reflex + LLM arbiter off the turn-taking path — is 2026 best practice (LiveKit/Pipecat/OpenAI).

Built and reused unchanged: `intent.py` (`classify_interjection`, tested, pure) and `endpointing.py` (`TurnDetector`/`NullTurnDetector`/`SmartTurnDetector`).

---

## 7. Speaker focus in a crowd

> **REVISED 2026-06-24 — FOCUS is now PHYSICAL, not acoustic.** The original §7
> mechanism (the FireRedChat pVAD gating audio on a per-frame `is_target`) is
> **retired**: its ECAPA `spkemb` conditioning proved inert on our embeddings — it
> degenerates to a plain energy VAD and bystanders leak (memory
> `pvad-conditioning-inert`; shipped disabled in `config.yaml`). A vision spike then
> proved, live on the GB10, that cheap CPU-only **camera presence (YuNet) + enrolled
> identity (SFace)** discriminate the owner from a stranger at kiosk distance (0.73
> cosine margin; `docs/notes/2026-06-23-vision-presence.md`). **FOCUS (Req 1) is now
> delivered by the camera as the floor-control authority — owner present → keep
> serving; owner gone → free the kiosk; stranger → not the owner — while audio is
> content only.** The detailed design is
> `docs/superpowers/specs/2026-06-24-director-floor-control-design.md` (Sub-project 2).
> The subsections below are kept for the parts that survive (enrollment hardening,
> verify-before-serve) and annotated for the parts that change (frame-level mechanism,
> safety-net/lockout role).

The kiosk stands in a crowd; FOCUS (Req 1) is the hardest requirement. **Physical
presence is the authority** (camera); audio identity is a relaxed, deferred backstop,
with the expensive/unreliable ECAPA pushed entirely off the hot path.

**Enrollment hardening — reconciled with live config.** Short enrollment makes ECAPA unreliable (EER ~8.9% @1s vs ~2.0% @10s; TI unreliable below 2s). The Director requires a hardened enrollment before serving. The existing config is `core.speaker.enrollment_utterances = 5`, `enrollment_min_self_similarity = 0.6`, `min_segment_duration_ms = 800` (**config.yaml:11-13**); `EmbeddingExtractor.MIN_DURATION_SAMPLES = 12800` = 800ms (**embedder.py:15**, zero-pads below that). We **adopt the existing keys and raise two thresholds**, explicitly overriding:

- **Utterance count:** keep `enrollment_utterances = 5` (already ≥ the 3-utterance floor we wanted; no change).
- **Per-utterance floor:** keep `min_segment_duration_ms = 800` as the hard reject (the extractor zero-pads below it, which is what we want to forbid for enrollment) — **the earlier draft's "reject <1s" is dropped** in favor of the existing 800ms floor, to avoid inventing a second incompatible threshold.
- **Self-similarity:** **raise `enrollment_min_self_similarity` from 0.6 to 0.80** (config change, documented in `config.yaml` comment) — 0.6 admits drifty enrollments that later false-reject the user; 0.80 matches the ~2% EER operating point on ≥5s cumulative audio. Reject (prompt re-enroll, bounded by `enrollment_max_retries = 3`, **config.yaml:14**) below 0.80 and log the self-similarity.

**Verify-before-serve — fixed to survive `finalize_enrollment`.** The earlier draft scored the finalized embedding against a "held-out enrollment utterance," but **`finalize_enrollment` deletes the per-utterance file** (`os.remove(utt_path)`, **enrollment_store.py:99**) — so no held-out utterance survives. **Fix (two-part):**

1. **Capture the holdout BEFORE finalize.** The enrollment flow holds out *one* utterance embedding (taken from the utterances file *before* `finalize_enrollment` is called) and passes it through to the session as `DirectorHandoff.holdout_embedding`. No change to `finalize_enrollment` semantics is required for V1.
2. **Score at session start.** The Director scores `cosine(primary_embedding, holdout_embedding)` and refuses to start the session (return to IDLE, re-enroll prompt) if it is below 0.80. This is the verify-before-serve check, using data captured *before* the destructive finalize step.

(Alternative, if the enrollment flow cannot be changed: modify `finalize_enrollment` to retain the utterances file instead of `os.remove` at **enrollment_store.py:99**. V1 uses the holdout-before-finalize path to avoid touching shared enrollment infra.)

**Frame-level target-speaker mechanism — V1 (ship now): CAMERA floor control
(REVISED).** ~~FireRedChat pVAD~~ is retired (inert conditioning, above). FOCUS is
instead a **separate `VisionWorker`** running CPU-only YuNet detection + SFace identity
at ~3 fps, emitting a single `OwnerPresenceEvent(status ∈ {PRESENT, ABSENT,
UNAVAILABLE}, now)` onto the event bus on debounced status changes. "Present" means the
**enrolled owner's face** is in the central zone (SFace cosine ≥ `identity_threshold`,
~0.40, vs a face embedding captured at wake). The reducer records presence in Context
and adds **one end-condition inside `_on_tick`**: a sustained valid `ABSENT` (≥
`owner_absent_grace_s`), with an active-talk guard, → `EndSession("owner_absent")`. A
stranger reads as `ABSENT` (owner-changed = swap). This is an **add-on**: the §5
silence timeout/nudge are unchanged, and the watchdog remains the sole timeout
authority (§4a preserved — a condition, not a competing timer). `UNAVAILABLE` (camera
glitch) ⇒ vision ignored, fall back to the §5 audio timeout (fail-safe). With vision
disabled / no camera / no face reference, the worker is `None` and the runtime is
byte-for-byte today's Director (no-regression guarantee). Full design + config in
`2026-06-24-director-floor-control-design.md`.

**Audio identity is deferred behind a seam (REVISED).** Content acceptance for V1 is
proximity-only: camera owner-present + near-field RMS. The case the camera *cannot*
catch — a bystander speaking right beside the present owner (face present, voice not the
owner's) — is handled by a flag-gated audio seam (default **off**) built from the
existing `SafetyNet` + `Lockout` (see the revised safety-net paragraph below),
revisited only after live data shows it is a real problem in the space.

**Frame-level acoustic mechanism — V2: now OPTIONAL (REVISED).** With the camera
delivering FOCUS, the bespoke trained TS-VAD drops from "needed for real FOCUS" to an
**optional** enhancement (it would only add same-distance acoustic discrimination the
camera can't, i.e. the bystander-beside-owner case the §9 seam targets more cheaply).
Retained here for reference: a noise-robust causal-Conformer TS-VAD (arXiv 2501.03184
design, ~124k params, 310ms context, FiLM, ECAPA-conditioned) on 2000h+ clean speech
with MUSAN/RIR augmentation at 0-30dB SNR and 2-4-speaker mixtures, optionally with
in-session self-augmentation (arXiv 2601.12769). **NeMo Streaming Sortformer remains
rejected** (anonymous diarization, no enrollment conditioning; torchaudio has no working
aarch64 CUDA build on GB10).

**Rolling-window safety net → the deferred audio seam (REVISED).** Session-hijack
detection is now the camera's job (a stranger reads as `ABSENT` → the session frees on
the §4 owner-absent path). The accumulated-window ECAPA verifier (`SafetyNet`) +
`Lockout` are therefore **repurposed, not primary**: they become the **flag-gated
audio seam** (`vision.audio_safety_net.enabled`, default **off**) for the one case the
camera cannot see — a bystander speaking right beside the present owner. When enabled,
`SafetyNet.accumulate(is_target audio)` embeds every ~`verify_window_ms` (2000ms) off
the hot path (`run_in_executor`, 108ms p95 fine here) and `Lockout` (WARN → EJECT →
IDLE, never a permanent lockout) is its action arm. Default-off until live data shows
the leak is real. The near-field **RMS proximity gate** (`barge_in.proximity`,
auto-calibrated from primary enrollment RMS when `rms_threshold: null`) stays as the
content near-field check and as the crash-fallback for `is_target`.

---

## 8. Interruption-resume

Req 2: interrupt mid-reply, answer the clarifier, then continue the prior point. This ports the real mechanism faithfully.

- **Bounded interrupted-stack.** When EVALUATING resolves to CUT during SPEAKING, the Director finalizes the truncated assistant turn (store `partial + " [interrupted]"` via `ConversationManager.add_assistant_turn`, exactly **controller.py:766-767**) and pushes a **resume frame** (`{"query", "partial"}`, mirroring `_store_interruption` at **controller.py:753-777**) onto a small, **bounded** interrupted-stack on the Context blackboard (cap depth 2-3; drop oldest on overflow). The one-shot `_pending_steer` instruction is set (**controller.py:769-774**). Then transition through THINKING to SPEAKING and answer the clarifier as a normal turn.
- **LLM-steered continuation.** After the clarifier is answered and the floor returns to LISTENING, the next generation injects the resume steer via `_maybe_inject_resume_steer` (**controller.py:779-796**, called at **controller.py:641** before the LLM request): the continuation is **LLM-generated** ("As I was saying, …"), not a replayed audio buffer. `_pending_steer` is one-shot and cleared after injection.
- **Auto-resume safety net.** Because the reflex defaults to CUT on *semantic* ambiguity, a *wrong* cut (we cut on what was actually a backchannel/bystander that slipped past the gates) is recovered automatically: if the post-cut LISTENING turn yields no genuine user content (empty / `mean_word_prob < conf_floor` / backchannel), the Director pops the resume frame and continues the prior point without a wasted exchange. The arbiter LLM (Section 6) can flag a likely-wrong cut early to trigger this faster.
- **Resume on the barge-in path.** Resume is driven from the same EVALUATING→CUT path that handles barge-in, so interruption and resume share one code path and one stack — no separate "resume mode."

---

## 9. Model stack & GB10 placement

**Streaming STT (the real gap) — and an important grounding correction.** `modes/talkback/stt.py` is **not** generic "segment-level Whisper": `StreamingStt` wraps **faster-whisper / CTranslate2** with `device="cuda"`, `model="large-v3"` by default (**stt.py:17-19, 29-35**) — the *same* backend that has **no aarch64 CUDA wheel** on GB10 (CPU-only ~270ms in the spike). So "reuse `StreamingStt` as-is on the LISTENING path" is **infeasible**: on this box it would either fail to load CUDA or fall back to ~270ms CPU per full turn. **Both** STT paths must therefore be **re-backed** off faster-whisper. The split:

- **LISTENING path (no hard latency constraint, but faster-whisper still unusable on CUDA here):** re-back `StreamingStt` onto one of openai-whisper (torch CUDA), NeMo, or a CTranslate2 CUDA *source build*. The `StreamingStt` **class name/interface is kept** (callers unchanged) but its internals are swapped; it is **not** reused as-is.
- **EVALUATING path (barge-in classification):** same re-backed engine behind the **extended** `transcribe_segment(audio) -> TranscriptResult(text, mean_word_prob)` signature (Section 6).

**Library disambiguation (the spike measured openai-whisper, NOT faster-whisper):** the spike's `whisper-tiny(CUDA) 38.8ms p95` was **openai-whisper (torch)** — a *different* library and codepath from the `faster_whisper` in `stt.py`. Wherever a whisper number appears below, it refers to **openai-whisper (torch)**. `StreamingStt`'s current faster-whisper backend has **no** validated CUDA path on GB10.

- **PRIMARY: NVIDIA NeMo Nemotron streaming** (`nemotron-speech-streaming-en-0.6b` / `parakeet-unified-en-0.6b`). DGX Spark is a listed target (282× RTFx; ~24ms TTFT at 80ms chunks); exposes per-word `word_confidence` to populate `mean_word_prob`. Install via NeMo-from-source on the PyTorch 25.10 (2.9) container — **not** 2.10/25.12 (breaks NeMo/Lhotse); pin `lhotse>=1.32.2`; NIM is x86-only, do not use it. **Unproven on this box** (Section 14).
- **FALLBACK (the only CUDA STT proven on GB10): openai-whisper (torch), chunked.** CUDA confirmed in the spike (`tiny` 38.8ms p95); emits per-segment text with `word_timestamps=True` `probability`, averaged into `mean_word_prob`. Use `base.en`/`tiny` for the EVALUATING reflex, `large-v3-turbo` for LISTENING final quality if NeMo stalls. **Caveat:** the spike was a standalone script, not an integrated worker — "fallback already working" means the *backend* is proven, but the worker integration (extended signature, streaming chunk loop, gen_id tagging) is still net-new work.
- **Reserve:** parakeet.cpp REST subprocess (no in-process Python API; adds a 1-5ms hop) only if NeMo's in-process path proves too fragile.

**TTS:** `TtsEngine` (`modes/talkback/tts.py`), Kokoro-82M on GPU, 24k→16k resample, reused as-is, one warm instance. Also used directly (no LLM) for the "are you still there?" nudge (Section 5).

**LLMs — two warm clients with a fixed lifecycle.** main = gemma; arbiter = a small model; both as **warm** local `llama.cpp` servers behind two `LlmClient` instances. Today a single client calls `await self._llm.close()` *before* `ping()` at session start (**controller.py:264**). For two clients this must be **duplicated and coordinated**: close-then-ping is run **independently per client at session start**, and **only the main client is closed/cancelled on CUT** — the arbiter client must **not** be closed mid-turn (it has no in-flight generation to cancel and is needed for the next ambiguous call). On CUT: bump `gen_id`, `main_llm.cancel()` + `Task.cancel()` on the main generation, then `_drain_playback()`; the arbiter client is left warm.

**Co-residency & contention budget.** Many models share 128GB unified memory; one GPU is time-shared. The spike measured the reflex specialists clearing <100ms *under real gemma GPU load*: openai-whisper-tiny(CUDA) 38.8ms p95, Smart Turn(CPU) 55.4ms p95; ECAPA(CPU) 108ms p95 stays off the synchronous path. Placement rule: **reflex specialists on CPU, insulated from GPU contention; ECAPA off hot path.**

**Combined hot-path latency spike (NEW cutover gate).** The individual numbers above were measured *separately*. The Director adds the pVAD onto the same CPU hot path. Before cutover, a **combined** spike must fire Silero + pVAD + Smart Turn + `classify_interjection` **on one chunk** under live gemma GPU load and confirm the aggregate onset-to-duck reaction stays <100ms on GB10 CPU. This is a hard cutover criterion (Section 13), not the per-model numbers already in hand.

**GPU contention at a cut (no scheduler today — flagged as a real path).** At a cut, EVALUATING-STT + TTS + main LLM (+ arbiter) may collide on one GPU. The reused components have **no** GPU priority queue. V1 mitigation is structural, not scheduled: reflex stays on CPU; on CUT the main LLM is cancelled (freeing GPU) and TTS is drained *before* the new generation, so in practice EVALUATING-STT rarely overlaps a live TTS. A GPU priority mechanism is deferred (Section 14) — this remains an unvalidated worst-case path.

| Component | Device | Hot path? | Notes |
|---|---|---|---|
| Silero VAD | CPU | yes (onset) | `is_speaking` drives duck-at-onset |
| ~~pVAD (FireRedChat)~~ → VisionWorker | CPU | floor-control (off audio hot path) | RETIRED 2026-06-24; replaced by YuNet+SFace camera presence at ~3 fps in a separate thread (§7-revised) — not on the audio reflex loop at all |
| Smart Turn v3 | CPU | yes (endpoint) | 27-55ms p95, `run_in_executor` |
| AEC (WebRTC APM) | CPU | yes (per-frame) | runs in Ingestion worker; reference invariant in Section 10 |
| `classify_interjection` | CPU (in-loop) | yes | pure, ~µs, synchronous |
| ECAPA embedder | CPU | **no** | 108ms p95, `run_in_executor`, safety net only |
| STT (EVALUATING) | **GPU** | decision off-loop | NeMo / openai-whisper (NOT faster-whisper), run in worker |
| STT (LISTENING) | GPU | no | re-backed `StreamingStt`, full-turn |
| Kokoro TTS | GPU | no | one warm instance |
| Main LLM (gemma) | GPU | no | warm `llama.cpp` server |
| Arbiter LLM | GPU | no | warm, off hot path, never closed mid-turn |

---

## 10. Component reuse

The Director **reuses** the two genuine assets no framework offers — the race-fixed PortAudio teardown and the interrupt-then-resume stack — and borrows specialists rather than hand-rolling them. Reuse map (verdicts from the reuse survey):

| Component | File | Verdict | Adapter |
|---|---|---|---|
| `Player` | `player.py:14` | as-is | teardown invariants copied verbatim into playback worker |
| `AecProcessor` | `aec.py:54` | as-is | same `process_frame(mic, ref)`; reference invariant below |
| `SentenceChunker` | `chunker.py:17` | as-is | one per generation |
| `ConversationManager` | `conversation.py:7` | as-is | `add_assistant_turn` discipline preserved |
| `classify_interjection` | `intent.py:57` | as-is | sync in EVALUATING |
| `SmartTurnDetector`/`TurnDetector`/`Null` | `endpointing.py:49,81,69` | as-is | caller wraps in executor |
| `StreamingStt` | `stt.py:12` | **re-backed (NOT as-is)** | keep class/interface; swap faster-whisper→openai-whisper/NeMo; **extend** `transcribe_segment` to return `TranscriptResult(text, mean_word_prob)` |
| `TtsEngine` | `tts.py:13` | as-is | also direct-call for nudge |
| `LlmClient` (×2) | `llm.py:13` | as-is, lifecycle coordinated | per-client close-then-ping; only main closed on cut |
| `MicrophoneStream` | `core/audio/mic_stream.py:11` | as-is | Director owns mic lifetime |
| `SileroVAD` | `core/vad/silero_vad.py:20` | as-is | `is_speaking` drives duck |
| `EmbeddingExtractor` | `core/speaker/embedder.py:12` | as-is (CPU) | all calls in executor; warm at init; 800ms min (MIN_DURATION_SAMPLES) |
| `EnrollmentStore` | `core/speaker/enrollment_store.py:17` | as-is for read; holdout captured **before** finalize | verify-before-serve uses pre-finalize holdout (line 99 deletes utterances) |
| `DecisionSmoother` | `core/speaker/decision_smoother.py:7` | as-is | one per session; safety-net only |
| `AsyncWatchdog` | `watchdog.py:7` | **extended (NOT as-is)** | add `nudge_lead_s`/`on_nudge`/`is_nudged`/`mark_nudged`; nudge path is non-terminal (Section 5) |
| `WakeWordDetector` | `modes/kiosk/wake_word.py:9` | as-is | WakeGate IDLE loop |
| `Session` | `modes/kiosk/session.py:11` | **subsume** | folded into Context `_started_at`/`_last_speech_at` |
| `TalkbackHandoff`/`Result` | `handoff.py:16,27` | rename → `DirectorHandoff`/`DirectorResult` | +`holdout_embedding` field |
| `KioskPipeline` | `pipeline.py:22` | **replace** | watchdog (104-113) + active-chunk paths deleted; thin WakeGate remains |
| `TalkbackController` | `controller.py:45` | **replace** | becomes the Director FSM (SPEAKING branch split into EVALUATING) |

**Race-fixed teardown INVARIANTS — preserve verbatim** (cross-thread `sd.write`/`stream.close` segfault PortAudio):

1. **`_write_lock`** held around *every* `sd.write()` (**controller.py:221-228**) **and** around `stream.stop()/close()` (**controller.py:237-244**) — write and close mutually exclusive across threads.
2. **`_play_gen`** generation counter (**controller.py:106**, aligned with the Context `gen_id`): `_play_audio()` checks `gen != self._play_gen` before each frame and again inside `_write_lock` (**controller.py:216,222**); a barge-in bumps it via `_drain_playback()`. This is the *only* safe way to stop an in-flight executor write (executor tasks aren't cancellable once running).
3. **`record_reference()` and AEC reference alignment across the worker boundary.** The AEC reference must be the *same post-gain frame* that goes to `sd.write()`, recorded inside the *same* `_write_lock` acquisition, so the ring buffer stays sample-aligned. **Cross-worker restatement:** the per-frame AEC consume (`Player.get_reference_frame` + `AecProcessor.process_frame`, today in `_listen_loop` at **controller.py:819-832**) runs in the **Ingestion** worker, while `record_reference` runs in the **Playback** worker under `_write_lock`. The invariant is preserved by keeping **record + write co-located in the Playback worker under one `_write_lock`** (never split record from write across workers); the Ingestion worker only *reads* the reference ring via `get_reference_frame`. The ring buffer is the single shared, lock-internal hand-off; no other cross-worker coupling of these two calls is permitted.
4. **`_drain_playback()`** (**controller.py:189-203**): bump `_play_gen`, then `await asyncio.shield(_play_future)` so the executor thread fully exits before close or before a new `_play_audio()`. **Must be awaited before `_close_out_stream()`** — two threads on one `OutputStream` segfault.
5. **Two-path teardown:** async drain first (normal case), then synchronous, lock-guarded, idempotent `_close_out_stream()` (**controller.py:231-244**) as a KeyboardInterrupt/exception backstop (**controller.py:254**).
6. `_play_future` is set on every `run_in_executor` launch and **not** cleared on barge-in — it must survive until the next generation confirms the prior thread exited.

---

## 11. Concurrency, error handling & robustness

- **Single-mutator FSM.** Only the Director mutates state, one event at a time, in a synchronous transition function. No worker touches FSM state. This is the structural cure for the double-manage race.
- **Per-turn generation ids.** Every LLM/TTS/playback generation carries a `gen_id` (the Context's monotone counter, aligned with `_play_gen`). Stale events (from a generation the user already cut) are dropped by `gen_id` mismatch — late STT/TTS/token events from a cancelled turn cannot corrupt the live turn. On CUT: bump `gen_id`, `main_llm.cancel()`, cancel the wrapping `asyncio.Task` (CancelledError breaks the `async for`), then `_drain_playback()`. The arbiter client is **not** cancelled (Section 9).
- **Worker isolation + supervisors.** Each worker runs as a supervised task; an exception emits a `worker_failed` event and is restarted, rather than killing the loop. The pVAD worker has a defined degraded mode (fall back to RMS proximity gate, `is_target := rms ≥ proximity_rms`); STT/TTS failures surface as turn errors, not crashes.
- **State watchdogs.** Beyond the silence/hard timers, the Director arms bounded guards: **stuck-EVALUATING** (if no STT/decision result within N ms, force RESTORE — never hang ducked), **stuck-THINKING** (LLM stream stalled → abort turn, apologize, return to LISTENING), and a **duck-safety** guard (if ducked but not in EVALUATING, force gain restore via `_restore_volume`, **controller.py:746-751**).
- **Graceful teardown stops everything.** On session end (timeout, hard cap, lockout, or interrupt), the Director runs full teardown — cancel `listen_task` and `response_task`, `await _drain_playback()`, `await watchdog.stop()`, `_close_out_stream()`, `main_llm.close()` + `arbiter_llm.close()` — *before* returning `DirectorResult`. There is no second component still answering: the no-orphan-after-end property is guaranteed because the Director is the sole owner (Section 4a).

---

## 12. Testing strategy

Deterministic dialogue policy was a core constraint; the FSM is built to be tested without models.

- **Director FSM as a pure reducer.** Factor the transition function as `reduce(state, event) -> (state, [commands])` with **no I/O**. Feed synthetic event streams (`SpeechSegment`, `SpeakerFrame`, `near_field_onset`, `llm_request_sent`, `first_tts_frame_written`, `TranscriptResult`, `endpoint_prob`, watchdog ticks, `on_nudge`) and assert resulting state + emitted commands. No GPU, no audio, fully deterministic, runs in CI. Inject `NullTurnDetector` via the `TurnDetector` Protocol.
- **Reflex/arbiter contract tests.** `classify_interjection` truth table (backchannel set, force-interrupt set, empty→backchannel, default→interrupt); Smart Turn threshold behavior; the EVALUATING gate ladder (too-short reject at `verify_window_ms`, far-RMS reject, speaker-mismatch reject, empty/`mean_word_prob<conf_floor`→RESTORE).
- **THINKING-state transitions.** Assert `LISTENING → THINKING` on `llm_request_sent` and `THINKING → SPEAKING` on `first_tts_frame_written`; assert `_silence_duration()==0` in THINKING/SPEAKING/EVALUATING and `>0` only in LISTENING.
- **Nudge mechanics (Req-4 regression).** Drive the watchdog with a synthetic clock: assert the nudge fires exactly once when `silence ≥ silence_timeout_s − nudge_lead_s`, does **not** stop the loop, does **not** re-fire on the next tick, and **re-arms** after a `LISTENING` re-entry clears `_nudged_cycle`; then assert the terminal `silence_timeout` still fires.
- **Scenario E2E** (recorded audio fixtures): backchannel ignored (keep talking), genuine question cuts and is answered, interrupt-then-resume continues the prior point ("As I was saying…"), **timeout suspended while speaking then nudge then end** (the explicit Req-4 regression), bystander/non-target speaker refused (camera owner-absent → `EndSession("owner_absent")`; §7-revised), short interjection below `verify_window_ms` dropped, session-hijack ejection.
- **Concurrency/teardown regression.** Cut-during-playback (assert no PortAudio segfault, `_drain_playback` awaited before close), **no-orphan-after-end** (assert nothing answers after `DirectorResult` returns — the exact Req-5 bug; plus the grep post-conditions of Section 4a), KeyboardInterrupt mid-playback hits the synchronous backstop, stale-`gen_id` events dropped, AEC reference stays sample-aligned across the Ingestion/Playback worker boundary.
- **Enrollment / verify-before-serve.** Assert session refuses to start when `cosine(primary, holdout) < 0.80`; assert the holdout embedding is captured *before* `finalize_enrollment` deletes utterances (**enrollment_store.py:99**).

---

## 13. Migration / build sequence

Build the Director as a **new module alongside** the existing controller, reuse the built pieces, then cut over.

1. **Scaffold + split SPEAKING into EVALUATING.** New `Director` (rename of `TalkbackController` core) with the 5-state FSM. **Delete** the `BARGED_IN` enum and its `_transition` (**controller.py:871**). **Split** the SPEAKING branch: the onset-duck (**controller.py:839-849**) emits `near_field_onset` → reducer enters EVALUATING; the decision body (**controller.py:650-742**: proximity 659-669, min-dur reject 673-679, speaker-match 681-693, transcribe 700, classify 705-710, cut 715-742) becomes the EVALUATING branch. Insert THINKING between LISTENING and SPEAKING (`llm_request_sent` / `first_tts_frame_written`). Pure-reducer factoring first, FSM unit tests green before wiring any model.
2. **Single watchdog + nudge.** Extend `AsyncWatchdog` (new `nudge_lead_s`/`on_nudge`/`is_nudged`/`mark_nudged`; non-terminal nudge path, Section 5). Add `_nudged_cycle`, cleared on every LISTENING entry. Keep `_silence_duration → 0` outside LISTENING (update its docstring). Add `kiosk.talkback.nudge_lead_s` (5).
3. **Subsume the pipeline session lifecycle.** Reduce `KioskPipeline` to the thin **WakeGate** (IDLE + AWAIT_FIRST_SEGMENT + handoff). **Delete** `_watchdog_loop`/`_start_watchdog`/`_stop_watchdog` (**pipeline.py:91-114**), `_handle_active_chunk`/`_process_session_segment`, and the `Session` field. Remove `kiosk.session_silence_timeout_s`/`session_hard_timeout_s` (**config.yaml:30-31**). Land the Section-4a grep post-condition test.
4. **Re-back STT off faster-whisper.** Swap `StreamingStt` internals to openai-whisper (torch CUDA, proven) or NeMo; **extend** `transcribe_segment → TranscriptResult(text, mean_word_prob)`; wire the empty/low-confidence RESTORE guard. (This is prerequisite to *both* LISTENING and EVALUATING working on GB10 — faster-whisper has no CUDA wheel here.)
5. **Crowd focus — CAMERA floor control (REVISED 2026-06-24; was pVAD).**
   **✅ DONE — Director-07 SHIPPED & merged to master 2026-06-25** (commits up to
   `a1c453a`; verdict `docs/notes/2026-06-24-director-07-live.md`; all 3 live checks
   pass — owner-absent fires fast, no-regression with vision off, fail-safe when the
   camera is unavailable). The pVAD path shipped and was disabled (inert conditioning).
   FOCUS is now a separate `VisionWorker` (YuNet + SFace, CPU, ~3 fps) that self-enrolls
   the owner at session start and emits `OwnerPresenceEvent`; the reducer adds the
   owner-absent end-condition inside `_on_tick` (§4-revised). Built per
   `2026-06-24-director-floor-control-design.md` (events+reducer pure first, then
   classifier, worker, self-enrollment, assembly `_build_vision` + runtime start/stop).
   The flag-gated `SafetyNet`/`Lockout` audio seam is a **reserved config flag with no
   consumer yet** (deferred, default off — §9-revised).
   **⏳ STILL OPEN from the original step 5** (NOT part of Director-07, not yet built):
   capture holdout-before-finalize and add verify-before-serve at session start (raise
   `enrollment_min_self_similarity` 0.6→0.80).
6. **Interrupt-resume polish.** Bounded interrupted-stack, LLM-steered continuation (`_store_interruption`/`_maybe_inject_resume_steer` ported verbatim), auto-resume net, two-client LLM lifecycle, arbiter wiring.

**Cutover criteria:** FSM reducer tests + nudge regression + all E2E scenarios green; cut-during-playback and no-orphan-after-end (incl. Section-4a greps) pass; **combined** reflex hot-path measured <100ms under live gemma load on GB10; pVAD latency spike confirms CPU budget *and* bare-model load works; verify-before-serve + nudge-then-end verified; STT re-backed and loading on CUDA. The old `TalkbackController`/pipeline paths are removed only after the Director clears every scenario.

---

## 14. Risks & open questions

- **Streaming-STT integration (highest).** `StreamingStt` ships faster-whisper, which has **no aarch64 CUDA wheel** — so *both* STT paths need re-backing, not "reuse as-is." NeMo on aarch64 is version-fragile (pin PyTorch 2.9/container 25.10, `lhotse>=1.32.2`; 2.10 breaks it; NIM is x86-only); the in-process streaming loop must be ported from NeMo example scripts; a `word_confidence` length-mismatch bug exists in some parakeet variants. The only CUDA STT *proven* on this box is openai-whisper (torch), and even that was a spike script, not an integrated worker with the extended `TranscriptResult` signature. **Mitigation:** ship LISTENING + EVALUATING on openai-whisper-chunked if NeMo stalls; treat the worker integration (signature, chunk loop, gen_id) as real work, not done.
- **~~pVAD un-benchmarked on GB10 CPU~~ — RESOLVED/RETIRED (2026-06-24).** The pVAD is
  dead (inert conditioning) and FOCUS moved to the camera, so this risk and the
  "combined hot-path latency" gate it implied are **off the critical path**: the camera
  presence path is CPU-only at ~3 fps (detection ~2% of a core) and was spike-validated
  to co-exist with the full LLM/TTS/STT stack (contention PASS,
  `docs/notes/2026-06-23-vision-presence.md`). No pVAD CPU budget remains to clear.
- **NEW: camera dependency (placement, lighting, two-faces).** FOCUS now depends on a
  camera. The spike was at a desk; real kiosk mount height/backlighting/distance can
  shift the identity margin (generous 0.73 headroom; `identity_threshold` is config,
  re-measure on-site). Two people at the kiosk (owner + companion leaning in) could let
  the larger central face win — mitigated by the zone/size gate + debounce, revisited
  if observed. Camera failure fails safe (degrades to today's audio-only timeout).
- **GPU contention at a cut.** STT(EVAL) + TTS + main + arbiter LLM may hit one GPU at the worst instant; the reused components have **no** GPU priority queue. V1 mitigation is structural (cancel main LLM + drain TTS on cut so EVAL-STT rarely overlaps live TTS); a real scheduler is deferred. This remains an unvalidated worst-case path.
- **Trained TS-VAD data (V2) — now OPTIONAL (2026-06-24).** With the camera delivering
  FOCUS, the bespoke noise-robust TS-VAD (2000h+ augmented data) is no longer needed for
  FOCUS — it drops to an optional enhancement for same-distance acoustic discrimination
  (the bystander-beside-owner case the §9 audio seam targets more cheaply). V1 ships on
  the camera, not a borrowed/trained acoustic model.
- **Two-client LLM lifecycle.** The current single-client close-then-ping (**controller.py:264**) must be duplicated per client and the arbiter must never be closed mid-turn. Mis-coordination would either stall the arbiter or cancel it during an ambiguity call.
- **Single-maintainer burden.** A hand-rolled FSM + borrowed specialists is a lot of surface for one engineer. The pure-reducer design and Protocol-based mock injection are the deliberate countermeasures (most logic testable without hardware).
- **Conditions that would change the approach:** if a maintained framework shipped the race-fixed PortAudio teardown *and* an interrupt-resume stack, the build-vs-buy verdict would flip toward buy; if torchaudio aarch64 CUDA gets fixed, Streaming Sortformer could be reconsidered for diarization-assisted focus (still needs enrollment conditioning); if ECAPA short-segment reliability were solved, the pVAD layer could simplify.