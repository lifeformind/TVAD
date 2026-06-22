# STT Backend Selection — GB10 (DGX Spark)

**Date:** 2026-06-22
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128GB unified)
**Spike:** `bench/stt_backend_probe.py` (Director Plan 04, Task 1)
**Clip:** `self.wav` first 3.0s (real speech, 16kHz mono)
**Environment:** torch 2.12.0+cu130, CUDA available, openai-whisper installed; NeMo NOT installed.

## Candidates probed

| Backend | Loaded? | p50 (ms) | p95 (ms) | Per-word confidence | Notes |
|---|---|---|---|---|---|
| openai-whisper `base.en` (CUDA) | YES | 75.7 | 84.3 | YES | torch-native; word_timestamps→.probability. Transcribed "Okay, I'm trying to record myself speaking." (correct) |
| openai-whisper `tiny` (CUDA)    | YES | 62.8 | 67.1 | YES | fastest; transcribed "Okay, I'm trying to record myself. Speak." (slightly less accurate than base.en) |
| NeMo `parakeet-tdnn-0.6b-v2` (CUDA) | NO | — | — | — | NOT installed by probe; `No module named 'nemo'`. Install is human-gated per spec §9 |
| NeMo `parakeet-tdt-0.6b-v2` (CUDA)  | NO | — | — | — | same — not installed |

## Decision

**PRIMARY = openai-whisper (model: `base.en`, device: cuda).**

Rationale: openai-whisper is the only CUDA STT present and proven on this box; it
loads with per-word `.probability` (required for the empty/low-confidence RESTORE
guard) and transcribes the real clip correctly. `base.en` at 84.3ms p95 is well
within budget for post-endpoint segment STT (this runs AFTER endpointing, not on
the <100ms reflex hot path), and is more accurate than `tiny` (which dropped
"speaking" → "Speak."). NeMo is not installed and would be a separate, deliberate,
human-gated install (spec §9: PyTorch 2.9 / container 25.10, lhotse>=1.32.2, NIM
x86-only/forbidden) — no reason to take that on when the proven path clears the bar.

Note: the 84ms p95 here exceeds the 38.8ms whisper-tiny figure in
`2026-06-21-gb10-contention.md` because this probe enables `word_timestamps=True`
(an extra alignment pass) to obtain per-word confidence; that cost buys the
confidence number the RESTORE guard needs and is acceptable off the reflex path.

## Consequences for Plan 04

- PRIMARY = openai-whisper → **Tasks 2–6 ship as written.**
- Task 7 (NeMo swap) is a **no-op** — not triggered.
- faster-whisper is removed regardless (no aarch64 CUDA wheel).
