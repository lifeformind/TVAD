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
    Knob(TB + "barge_in.onset_floor_speaking", BARGE, "Onset floor while speaking (RMS)", "float",
         "Extra RMS floor for duck-at-onset while TTS plays; 0 disables.",
         "Residual TTS echo ducked replies constantly (2026-09-02); owner barge-ins run 0.5+.",
         min=0.0, max=1.0, step=0.01),
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
    Knob(TB + "stt.model", VOICE, "STT model (openai-whisper backend)", "select",
         "Whisper model for turn transcription (openai-whisper backend).",
         "GB10 p95: tiny 67 / base.en 84 / small.en 189 / medium.en 415 ms.",
         choices=("tiny", "base.en", "small.en", "medium.en")),
    Knob(TB + "stt.end_of_utterance_tail_ms", VOICE, "Utterance tail (ms)", "int",
         "Audio kept after the endpoint before transcribing.",
         min=0, max=2000, step=50),
    Knob(TB + "llm.temperature", VOICE, "LLM temperature", "float",
         "Sampling temperature.", min=0.0, max=2.0, step=0.05),
    Knob(TB + "llm.max_tokens", VOICE, "LLM max tokens", "int",
         "Reply cap.", min=32, max=2048, step=32),
    Knob(TB + "llm.no_markdown_grammar", VOICE, "No-markdown grammar", "bool",
         "Ban markdown characters at the decoder (GBNF).",
         "Replaces regex stripping as the primary defense.",
         strict_bool=True),
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
