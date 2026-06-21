"""DirectorResult — what DirectorRuntime returns at true session end.

Re-exports the SINGLE DirectorResult defined in modes/talkback/handoff.py (the
binding handoff contract, Plan 02/03). Keeping one class means the runtime's
return value and the WakeGate's consumed result are the SAME type — no
isinstance mismatch across the assembly seam (Plan 03). It is frozen there."""

from modes.talkback.handoff import DirectorResult

__all__ = ["DirectorResult"]
