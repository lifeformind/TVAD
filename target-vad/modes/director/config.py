"""DirectorConfig — all decision thresholds, copied from spec section 5/6 and
config.yaml (kiosk.talkback.*). Frozen: the reducer never mutates config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectorConfig:
    # Timeouts / nudge (spec section 5)
    silence_timeout_s: float = 30.0      # config.yaml:52
    hard_timeout_s: float = 300.0        # config.yaml:53
    nudge_lead_s: float = 5.0            # NEW: nudge fires at silence == 30-5 = 25s
    # Turn-taking (spec section 6)
    endpoint_threshold: float = 0.5      # Smart Turn endpoint_prob >= this => turn complete
    min_speech_ms: float = 120.0         # barge_in.min_speech_ms (config.yaml)
    verify_window_ms: float = 700.0      # barge_in.verify_window_ms (config.yaml:131)
    speaker_threshold: float = 0.20      # barge_in.speaker_threshold (config.yaml:130)
    conf_floor: float = 0.5             # NEW: mean_word_prob below this => RESTORE
    duck_level: float = 0.35            # barge_in.duck_level: partial duck keeps the reply tail audible when an interjection is rejected (was 0.15 = near-silent -> lost tails)
    # Camera floor control (Director-07, spec §8). Presence is an ADD-ON: these
    # never touch the silence timeout above; they only add an owner-absent end.
    owner_absent_grace_s: float = 3.0   # sustained ABSENT this long => free the kiosk
    active_talk_guard_s: float = 3.0    # never owner-absent-end within this of owner speech
