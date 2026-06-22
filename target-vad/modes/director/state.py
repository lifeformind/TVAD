"""The Director's five explicit FSM states (spec section 4)."""

import enum


class State(enum.Enum):
    IDLE = "IDLE"            # no session (lives in the thin WakeGate, Plan 03)
    LISTENING = "LISTENING"  # waiting for / capturing the user's turn
    THINKING = "THINKING"    # LLM generating, no audible TTS yet
    SPEAKING = "SPEAKING"    # TTS audio playing out
    EVALUATING = "EVALUATING"  # near-field onset during SPEAKING: ducked, deciding
