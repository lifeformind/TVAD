# SOTA Upgrade Research — 2026-09-02

Five parallel web-research passes (STT, TTS, speaker/target-gating, duplex/turn-taking,
on-device LLM) against the September 2026 state of the art, checked for GB10/aarch64
viability. Compiled into a Type 1/2/3 improvement plan; full per-area reports with all
sources are appended.

**Decisions taken (same day):**
- Quick wins first: GBNF grammar + llama-server prompt caching → AS-Norm/QMF calibration
  → Gemma 4 26B-A4B swap → Parakeet via NeMo. TTS swap held pending on-box TTFA test.
- **Type 3.1 approved but deferred** (DOA-conditioned extraction front end) until quick
  wins land and the user live-tests the upgraded system.
- Because 3.1 is committed: cancel the multitalker-Parakeet overlap evaluation, further
  cone-vote threshold tuning, and the embedding-free pVAD track — 3.1 replaces all three.
- D11 merged to master 2026-09-02 without its clean gate run; those scenarios fold into
  the post-upgrade comprehensive live test.

Current stack at time of research: openwakeword wake → Silero VAD → openai-whisper
small.en (CUDA) → llama.cpp gemma-3-4b-it → Kokoro TTS; speechbrain ECAPA-voxceleb
speaker gate; Smart Turn v3.2 endpointing; ReSpeaker XVF-3000 (ch0 processed + hardware
DOA); YuNet+SFace camera presence. Hardware: NVIDIA GB10 / DGX Spark class (aarch64 +
Blackwell, 128GB unified, ~273GB/s — decode is bandwidth-bound).

---

## Type 1 — simple upgrades of existing components

1. **LLM: gemma-3-4b → Gemma 4 26B-A4B MoE** (Apr 2026, Apache 2.0, 3.8B active).
   ~52 t/s measured on DGX Spark via llama.cpp GGUF; best-in-class hallucination
   resistance (attacks the garbled-STT-fabrication problem — stronger LLMs are measurably
   more resilient to ASR errors per VoiceBench). One-line GGUF swap. Conservative
   fallback: Gemma 4 E4B (~100 t/s class). Alternative: Qwen3.6-35B-A3B-MTP (~90+ t/s
   with multi-token prediction; needs thinking-mode off for voice).
2. **GBNF grammar kills markdown at the decoder.** llama.cpp `grammar` param per request;
   excludes `*`, `#`, backticks etc. Makes markdown impossible instead of stripped
   post-hoc. Near-zero overhead, streams fine.
3. **Prompt-cache discipline on llama-server.** `id_slot` + `cache_prompt: true`,
   byte-stable system-prompt prefix with dynamic content at the END. Reported TTFT
   400ms → <50ms on cached prefixes. GB10 prefill ~1.6–2K t/s, so TTFT is then solved.
4. **STT: whisper small.en → Parakeet-TDT-0.6B-v2 via NeMo.** ~6.05% vs ~8.6% avg WER
   (~30% relative cut). Proven on GB10 (community deploy in NGC PyTorch container —
   avoid Riva/nemo2riva, broken on aarch64). NeMo entropy-based word confidence > whisper
   mean_word_prob as a gate (verify TDT confidence health; parakeet-rnnt is the safe
   decoder). Zero-dependency stopgap: whisper `large-v3-turbo` (~1 WER point for a config
   line). CTranslate2 still has no aarch64 CUDA wheel (confirmed Mar 2026) — the
   faster-whisper ban stands.
5. **Speaker embedder: ECAPA-voxceleb → ReDimNet2 or ERes2NetV2.** ReDimNet2 (MIT,
   torch.hub, Jul 2026): best accuracy/compute, best far-field evidence (2.7% EER on
   VOiCES). ERes2NetV2 (Apache 2.0): published 2s EER 1.48% (vs ECAPA several× worse at
   short durations). Either ≈ a day: swap embed fn, re-measure thresholds.
6. **TTS: Kokoro → Chatterbox Turbo (eyes open).** Kokoro is frozen (Jan 2025, no
   successor). Chatterbox Turbo (MIT, 350M, English): clear quality jump, `[laugh]`-class
   tags, proven on DGX Spark (dep trap: pinned deps pull CPU torch — install `--no-deps`).
   Regressions: TTFA ~300–500ms vs Kokoro near-instant; loses misaki inline-IPA
   pronunciation control. Benchmark before committing. Runner-up: Qwen3-TTS 0.6B/1.7B
   (Apache 2.0, claimed 97ms first packet, unverified on aarch64).
7. **Diarization: pyannote 3.1 → community-1/4.0** (Sep 2025) — offline mode only.

## Type 2 — new techniques within the current Director structure

1. **Score calibration first: AS-Norm + duration/SNR QMF + enrollment work.** The 0.15
   threshold squeeze (owner 0.23–0.47, stranger 0.07) is a calibration problem: AS-Norm
   against an imposter cohort passed through OUR far-field channel; threshold as a
   function of duration+SNR instead of one constant; far-field enrollment augmentation
   (RIR simulation of the clean seed); running enrollment centroid absorbing
   confidently-verified far-field turns (+4.8 F1 in the hardest condition in the Huawei
   pVAD paper). ~a day each, no retraining. Do BEFORE the embedder swap so the A/B is fair.
2. **Backchannel-aware barge-in — the planned reorder, validated.** HRI 2025 paper
   (arXiv 2501.01568, code released): overlap detect → STT → LLM classifies interjection
   → per-type policy; 88.78% classification, ~94% of live interruptions handled well,
   text-only features. Layering: keep duck-don't-cut; ~0ms lexical gate (≤2 words ∧
   backchannel stoplist → RESTORE); LLM classifies the remainder. After a false cut,
   **resume from the exact cut point, never restart** (IHBench).
3. **Speculative LLM during the endpoint window.** Kick STT at Silero speech offset, run
   the LLM speculatively while Smart Turn + 400ms tail decide, gate only TTS start on
   turn-complete; discard/regenerate if more speech arrives. LiveKit default; their
   measured p95 ~1.3s → ~600ms on a comparable stack. No new models.
4. **Semantic endpointing second opinion: UltraVAD** (open weights, 0.7B, ~65–110ms GPU)
   takes dialogue history — knows a pause after five digits of a phone number isn't
   end-of-turn. Invoke only when Smart Turn is uncertain (0.3–0.7). Smart Turn v3.2 is
   still head of its line (no v4).
5. ~~Multitalker Parakeet for overlap~~ — **cancelled by Type 3.1 decision** (was:
   `nvidia/multitalker-parakeet-streaming-0.6b-v1` + Sortformer transcribes overlapped
   speakers separately; keep as fallback if 3.1 stalls).
6. **SoulX-Duplug sidecar** (Soul-AILab, Apache 2.0, 0.6B, paper Mar 2026): streaming
   state predictor for cascades, five states per 160ms chunk incl. native
   `user_backchannel` and semantic complete/incomplete; ~250ms decisions. Consumed like
   Smart Turn. Benchmark first: EN accuracy (zh home turf), aarch64 build.
7. **Cheap duplex polish:** pre-rendered "mm-hmm" at long user mid-turn pauses (CHI 2025
   validated); filler clip when LLM TTFT >~700ms; `[low-confidence transcript]` tags in
   the LLM prompt + clarification-asking system-prompt clause with 2–3 few-shot
   garbled→clarify examples.
8. **Streaming STT: nemotron-speech-streaming-en-0.6b** — cache-aware streaming
   transducer, 80ms–1.12s latency presets, RNNT confidence on partials, same NeMo stack
   as Parakeet. Enabler for item 3's partial-transcript speculation. Needs a GB10 spike
   (only public attempt died at Riva conversion, not NeMo inference).

## Type 3 — full redesign

1. **Target-speaker-first front end: extract, don't gate.** ★ APPROVED, DEFERRED.
   DOA-conditioned neural extraction behind the mic array (DSENet, arXiv 2507.20926,
   code jingkangqi/DSENet, 1.4M params, validated on near-ReSpeaker 3-mic 30mm geometry)
   so downstream stages only hear the locked customer. Direction-conditioning sidesteps
   the embedding weakness that killed pVAD. Work: raw 4-ch XVF-3000 capture (firmware
   check), retrain on simulated 4-mic ReSpeaker geometry (pyroomacoustics), causal
   retrain for live. What it retires: multitalker-ASR eval, cone-vote decision logic
   (tracker survives to AIM the extractor), embedding-free pVAD track. What survives:
   voice verification + calibration (only defense vs same-bearing impostor), proximity
   floor as fail-safe, all behavior-layer work. If frame-level voice conditioning is ever
   retried: FiLM or embedding-free (USEF-TP cross-attention / HyWA hypernetwork,
   88.9%→9.9% false barge-in), never fixed-embedding concatenation.
2. **Probabilistic focus engine instead of stacked boolean gates.** Fuse DOA bearing,
   face ID, voice score (calibrated), proximity into one continuously-updated
   owner-attention posterior with one threshold and principled abstention — replacing the
   five hand-tuned per-gate thresholds.
3. **Fully streaming spine + duplex behavior layer.** Nemotron streaming ASR (or Kyutai
   STT w/ built-in semantic VAD) → speculative Gemma 4 MoE → Kyutai TTS (consumes LLM
   tokens word-by-word, ~220ms, no sentence buffering; interruption = stop feeding
   tokens). SoulX-Duplug states drive turn policy. Kyutai's **Unmute** repo is an open
   working implementation of almost exactly this cascade — reference architecture. Keep
   the race-fixed Director as orchestrator. Native S2S stays ruled out on GB10.

**Key unverified flags:** Nemotron / Kyutai / SoulX-Duplug / Qwen3-TTS have no confirmed
GB10 runs (Parakeet, Kokoro, Chatterbox do); HyWA and Huawei-pVAD are papers without
released code; DSENet real-time capability unstated (likely needs causal retrain);
vendor-run benchmark numbers flagged inline below.

---

# Appendix A — On-device LLM (full report)

The landscape shifted decisively in Feb–Apr 2026: **small-active-parameter MoE models**
(Gemma 4 26B-A4B, Qwen3.5/3.6-35B-A3B) are the clear winners on bandwidth-bound hardware
like the GB10, and llama.cpp gained mainline multi-token prediction (MTP) speculative
decoding in May 2026.

## Model candidates

**Gemma 4 26B MoE ("26B-A4B") — top recommendation.** Released April 2, 2026, Apache 2.0
(license upgrade from Gemma 3's custom terms). Sizes: E2B, E4B, 26B MoE (3.8B active),
31B dense. #6 open model on Arena text leaderboard (31B is #3); 26B MoE ≈ 88.3 AIME with
only 3.8B active. Same lineage/chat template family as gemma-3-4b — behavior and
prompting carry over. Wins AA-Omniscience hallucination evals vs Qwen3.5 and is the
better pick for instruction-following/structured output (Qwen leads agentic). GB10 speed
(measured, multiple sources): ~52 t/s decode sustained single-stream via llama.cpp,
still ~52 t/s at 32K depth; 45 t/s on NVIDIA forums; theoretical ceiling ~143 t/s.
Drop-in GGUF swap, day-one llama.cpp/Ollama support.

**Gemma 4 E4B — conservative small swap.** Direct gemma-3-4b successor: ~4B active,
128K ctx, native audio input, multimodal, Apache 2.0. Should hold ~100 t/s-class decode
on GB10 (inferred, not benchmarked on Spark). Better hallucination behavior than
Qwen3.5-4B; concise (Qwen's hybrid thinking emits 2–5× more tokens — bad for TTFT).

**Qwen3.5-35B-A3B / Qwen3.6-35B-A3B.** Qwen3.5 family Feb–Mar 2026 (Apache 2.0, 262K
ctx, vision); 35B-A3B = 3B active, ~19GB at Q4. Qwen3.6-35B-A3B (~May 2026) ships an MTP
head: with llama.cpp mainline MTP (PR #22673), 1.4–2.2× decode speedup, 70–85%
acceptance; on an L4 (~300GB/s, similar bandwidth) a Q4_K_XL + MTP setup hits 91–99 t/s
— expect similar on GB10. Smartest per-token in class; hybrid-thinking verbosity needs
no-think config for voice; Ollama doesn't support it (llama.cpp does).

**Deprioritized:** Llama 4 Scout (109B/17B active — ~2× slower decode, mixed chat
reception); Phi-4-mini / Phi-4-reasoning-vision-15B (no advantage); Ministral 3;
gpt-oss-20B (~61 t/s tg / ~2000 t/s pp measured on Spark, but reasoning-channel output
format awkward for terse voice).

## GB10/DGX Spark field reports

Canonical thread: ggml-org discussion #16578 — gpt-oss-120B: 60 t/s tg / 1956 t/s pp;
gpt-oss-20B: 61 t/s; Qwen3-Coder-30B-A3B Q8: 44 t/s tg / 1654 t/s pp; Feb 2026 builds
cut cold model load 104s → 22s (kernel 6.17.1 fix). Qwen3-30B-A3B Q4 ~89 t/s; 32B dense
~10 t/s — confirms the bandwidth math. Prefill is compute-bound and strong (1.6–2K t/s):
a 500-token prompt ≈ 0.3s. NVFP4 (native Blackwell 4-bit): TRT-LLM/vLLM/NIM only, NOT
mainline llama.cpp (PRs in flight); +46% over FP8 at concurrency 10 under TRT-LLM, but
single-stream voice on llama.cpp: stick with Q4_K_M/Q8 GGUF. Community fork
croll83/llama.cpp-dgx claims NVFP4 + "DFlash MTP" on GB10 — unverified.

## Latency techniques (ranked)

1. **Prompt/KV caching across turns** — biggest TTFT win, free. llama-server reuses KV by
   prefix similarity per slot; pin conversation with `id_slot`, `cache_prompt: true`,
   `--cache-idle-slots` / host-memory prompt cache. Reported 400ms → <50ms first-token on
   cached prefixes. Any byte changed early in the prompt invalidates — dynamic content at
   the END.
2. **MTP / speculative decoding** — built-in with Qwen3.6-MTP (1.4–2.2×). Classic draft
   speculation (`--model-draft`) ~1.5–2× on dense models (Gemma 4 31B drafted by E2B);
   adds little to A3/4B MoEs. Draft and target must share a tokenizer.
3. **GBNF grammar for markdown suppression** — grammar excluding `*`, `#`, backticks,
   `-`-at-line-start makes markdown impossible at the decoder; near-zero overhead; works
   streaming. The correct fix for the markdown-leak problem.
4. Keep max_tokens small; stream sentence-by-sentence into TTS — at 52 t/s a 1–3 sentence
   reply completes in <1.5s.

## Robustness to noisy ASR input

- VoiceBench (TACL): stronger backend LLMs are measurably more resilient to ASR errors —
  the single most effective lever is model quality (supports the 26B-MoE upgrade).
- Prompting alone doesn't fix ASR *error correction* (ASR-EC benchmark), but for
  *graceful clarification*: explicit refusal/clarification emphasis + few-shot examples
  yields substantial gains (AbstentionBench, Abstain-R1; "Knowing but Not Showing" —
  models recognize ambiguity but rarely ask). Recipe: system-prompt clause "input comes
  from speech recognition and may contain misheard words; if a request seems nonsensical,
  say what you heard and ask a short clarifying question" + 2–3 few-shot garbled→clarify.
- Passing ASR confidence/N-best into the prompt (RobustGER-style); cheap version:
  `[low-confidence transcript]` tag when STT confidence is low.
- No credible "voice-optimized instruct" text-LLM exists to swap in.

**Unverified:** croll83 fork; "52 t/s with NVFP4" blog title (GGUF numbers are the
reliable ones); E4B Spark throughput (inferred); "Gemma 4 31B beats 400B rivals"
headlines.

Sources: github.com/ggml-org/llama.cpp/discussions/16578 ·
blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ ·
medium.com/@james-tang/benchmarking-gemma-4-26b-a4b-on-the-dgx-spark-dc8245292095 ·
forums.developer.nvidia.com/t/guide-uncensored-gemma-4-26b-at-45-tok-s-on-dgx-spark-actually-feels-great-to-use/366442 ·
artificialanalysis.ai/articles/qwen3-5-small-models ·
x.com/neural_avb/status/2040305916512440399 · unsloth.ai/docs/models/mtp ·
github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4 ·
github.com/ggml-org/llama.cpp/discussions/13606 (KV cache reuse) ·
github.com/ggml-org/llama.cpp/discussions/20574 (host-memory prompt caching) ·
forums.developer.nvidia.com/t/llama-cpp-nvfp4-native-support-on-blackwell/368430 ·
arxiv.org/html/2410.17196v3 (VoiceBench) · arxiv.org/html/2412.03075v1 (ASR-EC) ·
arxiv.org/pdf/2605.25284 (Knowing but Not Showing) · arxiv.org/pdf/2506.09038
(AbstentionBench) · arxiv.org/pdf/2604.17073 (Abstain-R1)

---

# Appendix B — Speech-to-text (full report)

Baseline: openai-whisper small.en (244M), ~189ms p95 on 3s utterances, Open ASR
Leaderboard avg WER ~8.6% (approximate).

## Batch / drop-in candidates

**NVIDIA Parakeet-TDT 0.6B v2 (English) — top drop-in.** 600M FastConformer + TDT,
punctuation/caps, word timestamps. 6.05% avg WER (was #1 English at release), RTFx ~3380
(A100 batch). v3 (multilingual, 25 langs) is 6.34% — for English-only, v2 wins.
**Proven on this exact hardware:** community deployed parakeet-tdt-0.6b-v3 on DGX Spark
(ARM64, CUDA 13) via NeMo inside `nvcr.io/nvidia/pytorch:25.11-py3`: RTFx ~15×, 90–96%
GPU util, ~4.7GB (github.com/mARTin-B78/dgx-spark-parakeet-asr). NeMo has a full
confidence framework (frame/token/word, max_prob or entropy) — a better hallucination
gate than mean_word_prob. Caveat: TDT decoders had confidence bugs (NeMo issue #8737);
verify, or use parakeet-rnnt. Effort: moderate (nemo_toolkit[asr] in NGC container; same
batch-per-segment call pattern).

**Whisper large-v3-turbo — the lazy drop-in.** 809M, 4 decoder layers, ~7.75% WER. Runs
on the existing openai-whisper stack: `model="large-v3-turbo"`. Plausibly 300–500ms p95
on GB10 (unverified). Same hallucination-on-noise weakness.

**Canary-Qwen 2.5B:** 5.63% avg WER, current #1, but ~6.5× slower than Parakeet
(LLM-decoder hybrid). Batch-oriented; skip for a 3s-utterance kiosk.

**Qwen3-ASR (Jan 2026):** 0.6B/1.7B, Apache 2.0, 52 langs; 1.7B claims SOTA-competitive.
No native streaming in open release; unverified on GB10; multilingual wasted here.
Watch-list.

**Moonshine v2 / Streaming (Feb 2026):** ergodic streaming encoder; medium ~245M/6.65%
WER beats whisper small.en *while streaming*; edge-CPU-targeted; HF transformers
support. Confidence undocumented. Interesting low-cost partials source.

**Kyutai STT (stt-1b-en_fr, stt-2.6b-en):** truly streaming (Delayed Streams Modeling):
1B = 0.5s fixed delay, word timestamps, punctuation, and built-in **semantic VAD**
(predicts user completion — useful for turn-taking); 2.6B en-only = 2.5s delay, higher
accuracy. PyTorch or Rust server (semantic VAD in the Rust server only). Unverified on
GB10; no confidence output. Effort: moderate-to-rework.

## Streaming (true partials)

**NVIDIA Nemotron Speech Streaming — top streaming candidate.**
`nemotron-speech-streaming-en-0.6b` (~Jan 2026): cache-aware FastConformer-RNNT built
for voice agents — each frame encoded exactly once, latency presets 80ms / 160ms / 560ms
/ 1.12s; sub-100ms streaming latency, ~7.2–7.8% WER incl. AMI (far-field meetings).
Multilingual sibling nemotron-3.5-asr-streaming-0.6b exists (40 locales; its English
FLEURS WER is worse — use the en-only model). GB10 status: the one public attempt failed
at the **Riva/nemo2riva conversion step** (aarch64 onnxruntime hell), NOT NeMo PyTorch
inference; the Parakeet precedent says plain NeMo streaming inference should work. Needs
a spike. RNNT → NeMo confidence on partials.

**Whisper streaming wrappers:** SimulStreaming (AlignAtt policy, IWSLT 2025 winner,
~5× faster than whisper_streaming) and WhisperLiveKit (server, OpenAI-compatible API,
would need the PyTorch backend on GB10). Viable but a kludge: repeated re-decoding,
~1–2s partials, more compute than a native streaming transducer. Only if staying on
Whisper weights.

**sherpa-onnx streaming Zipformer:** mature, CPU-real-time (fine on Grace cores), but
worse English WER than Parakeet-class and GPU on aarch64 = build onnxruntime from source
(prebuilt ORT 1.24.4 CUDA binaries for Spark exist on the NVIDIA forum). Fallback only.

## Overlap / target-speaker ASR

- **nvidia/multitalker-parakeet-streaming-0.6b-v1**: streaming multi-talker ASR
  transcribing overlapped speech per speaker, no enrollment; learnable speaker kernels +
  streaming Sortformer diarization; latency 0.08–1.12s; cpWER 15.8–37.4% on AMI/CH109/
  Mixer6 (hard-mode audio). The only shipped open-weights fix for overlap garble.
  [Post-research note: superseded by the Type 3.1 extraction decision; keep as fallback.]
- **Enrollment-based TS-ASR** (CONF-TSASR, SQ-Whisper, SC-SOT): research-grade; no
  production-ready open checkpoint with a plug-in x-vector interface verified as of Sept
  2026; models are picky about their matching embedding extractor (cf. the pVAD failure).
- Nothing in single-speaker SOTA handles overlap natively.

## Platform checks

- **CTranslate2 aarch64 CUDA: still NO** (Mar 2026 bug report on DGX Spark-class
  hardware, speaches #620). Source-compile for SM 12.1 is the only route.
- **NeMo on GB10: YES, proven** (NGC PyTorch aarch64 container, plain NeMo inference;
  avoid Riva/nemo2riva).
- GB10 gotcha: ONNX/cuDNN batch-load crash class fixed by driver 580.173 — check driver
  before benchmarking.

**Unverified:** exact small.en leaderboard WER; nemotron-3.5 release date;
Nemotron/Kyutai/Qwen3-ASR end-to-end on GB10; TDT confidence health; the 15× RTF Spark
figure may include API overhead.

Sources: huggingface.co/nvidia/parakeet-tdt-0.6b-v2 ·
huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · arxiv.org/abs/2509.14128 ·
github.com/mARTin-B78/dgx-spark-parakeet-asr ·
forums.developer.nvidia.com/t/multilingual-speech-to-text-stt-asr-with-nvidia-parakeet-tdt-0-6b-v3-for-the-dgx-spark/365554 ·
huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b ·
huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b ·
forums.developer.nvidia.com/t/asr-on-spark-with-nemotron-speech-streaming-en-0-6b/358614 ·
huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 ·
github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Streaming_Multitalker_ASR.ipynb ·
github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_Confidence_Estimation.ipynb ·
github.com/NVIDIA-NeMo/NeMo/issues/8737 · kyutai.org/stt ·
huggingface.co/kyutai/stt-1b-en_fr · github.com/kyutai-labs/delayed-streams-modeling ·
arxiv.org/abs/2602.12241 (Moonshine v2) ·
huggingface.co/moonshine-ai/moonshine-streaming-small · github.com/QwenLM/Qwen3-ASR ·
arxiv.org/abs/2601.21337 · huggingface.co/openai/whisper-large-v3-turbo ·
github.com/ufal/SimulStreaming · github.com/QuentinFuxa/WhisperLiveKit ·
github.com/speaches-ai/speaches/issues/620 · opennmt.net/CTranslate2/installation.html ·
forums.developer.nvidia.com/t/onnx-runtime-gpu-inference-on-dgx-spark-gx10-build-guide-and-prebuilt-binaries/366157 ·
docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html ·
huggingface.co/spaces/hf-audio/open_asr_leaderboard ·
northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks ·
emergentmind.com/topics/target-speaker-automatic-speech-recognition-asr ·
arxiv.org/abs/2412.05589 (SQ-Whisper) · arxiv.org/abs/2308.05218 (CONF-TSASR) ·
arxiv.org/abs/2506.12672 (SC-SOT) ·
forums.developer.nvidia.com/t/gb10-dgx-spark-cudnn-fe-failure-11-and-cublas-status-internal-error-under-batch-load-fixed-by-driver-580-173/380948

---

# Appendix C — Speaker verification / target-speaker gating (full report)

## Embedding-model candidates (ECAPA replacement)

**ReDimNet (Interspeech 2024) / ReDimNet2 (Interspeech 2026) — top pick.** IDRnD B0–B6
(1.1M–15M params). ReDimNet-B6 ft_lm: 0.40% Vox1-O / 1.05% Vox1-H; ReDimNet2-B6: 0.29%
Vox1-O (vs speechbrain ECAPA ~0.9–1.2%). VoxBlink2-pretrained variants: **2.7% EER on
VOiCES** (far-field/reverb — direct proxy for the ReSpeaker condition). No published
1–2s table (gap), but Huawei's Interspeech 2025 robust-pVAD chose ReDimNet-L0 as its
*frame-level* speaker encoder. Integration: torch.hub load (PalabraAI/redimnet2, MIT; or
IDRnD/ReDimNet — license unconfirmed), 16kHz mono, plain PyTorch → aarch64+CUDA fine.
~a day.

**ERes2NetV2 (Alibaba 3D-Speaker) — best documented short-duration numbers.** Vox1-O:
0.61% full, 0.98% @3s, **1.48% @2s** — the only candidate with published 2s trials;
designed for short-duration SV. Apache 2.0, ModelScope checkpoints (use the
VoxCeleb-trained one for English, not the zh "common" models). ~a day via 3D-Speaker.

**CAM++**: 0.65–0.73% Vox1-O, very cheap, ONNX exports; ERes2NetV2 beats it. Fallback.
**SimAM-ResNet34/100 (wespeaker, VoxBlink2 110k speakers)**: expected better far-field/
accent generalization; no short-utterance table. **WavLM-large + ECAPA head**: ~0.38%
Vox1-O, most noise-robust, ~101M params/~10× latency — viable turn-end-only on Blackwell.
**TitaNet-Large: skip** (2021, reproduction disputed). **pyannote community-1/4.0**
(Sep 2025): upgrade the offline diarization mode — big noisy-audio gains, VBx clustering.

Note: wespeaker large-margin (LM) checkpoints are tuned for >3s audio — for 1–2s windows
prefer non-LM variants or A/B both.

## Target-speaker VAD that actually discriminates

The FireRedChat pVAD failure (target 0.61 vs bystander 0.55) matches a documented
pattern: **fixed-utterance-embedding conditioning is brittle under enrollment/test
domain mismatch** (near-field enroll vs far-field test). The 2025 literature moved to
embedding-free / cross-attention conditioning and FiLM:

1. **Robust pVAD (Huawei/WHU, Interspeech 2025)** — closest blueprint: embedding-free,
   ReDimNet-L0 frame-level encoder + FiLM fusion + 2-Conformer encoder, dual decoders;
   **dynamic enrollment update** during inference; hard-sample (similar-imposter)
   training cut far-field imposter FPR 29.9% → 10.1%. Their embedding-based baseline
   degraded exactly like ours. Code NOT released — retrain-from-recipe (weeks).
2. **USEF-TP (CSL 2025, arXiv 2501.03612)** — joint TSE + personal VAD,
   embedding-free via cross-attention on raw enrollment audio. Sister repo ZBang/USEF-TSE
   has checkpoints (CC BY-NC 4.0 — non-commercial); pVAD-variant checkpoints unconfirmed.
3. **HyWA (Huawei, arXiv 2510.12947)** — hypernetwork turns a speaker embedding into
   weight deltas for a frozen stock VAD at enrollment time; zero speech-time overhead.
   On real full-duplex barge-in data: **false interruptions 88.9% → 9.9%**, AUROC
   +10.3pts. Purpose-built for our barge-in problem; code release unconfirmed.
4. **DN-APC (arXiv 2501.03184)** — causal sub-150k-param Conformer TS-VAD; key finding:
   **FiLM conditioning beats concatenation/addition/multiplication**. FireRedChat likely
   used naive concatenation — a FiLM retrofit is the cheapest possible frame-gating retry.

2025 TS-VAD models are 150K–5M params, causal, 10ms-frame — trivially always-on for GB10.

## Direction-conditioned extraction (uses the XVF-3000 DOA)

- **DSENet (arXiv 2507.20926, Jul 2025)**: end-to-end DOA-guided extraction — conditions
  on DOA (cyclic positional encoding) + learnable beamwidth mask; **1.4M params, 18.3 dB
  SI-SDRi**, validated on a 3-mic 30mm circular array (near ReSpeaker's 4-mic 32mm ring).
  **Code: github.com/jingkangqi/DSENet.** Condition on the owner's locked DOA cone
  instead of a voice embedding → sidesteps the ECAPA weakness. Causality/real-time not
  stated (likely offline; needs causal retrain). Retraining on simulated 4-mic geometry
  via pyroomacoustics is standard. Needs raw multi-channel capture (XVF-3000 6-ch
  firmware exposes ch1–4 raw — already flashed).
- **Neural Directional Filtering (Fraunhofer, arXiv 2409.13502)**: "software
  supercardioid" — higher-order directivity from a small array; no code found.
- Related: MIMO-DBnet (2212.03401), Locate-and-Beamform (2305.10821), beamformer-guided
  TSE (2303.08702); survey: emergentmind.com/topics/spatial-target-speaker-extraction.
- DOA gating fails when bystander and owner share a bearing — additive to, not a
  replacement for, proximity+face+voice fusion.

## Short-segment verification tricks (likely missing)

1. **AS-Norm** — normalize live scores against an imposter cohort passed through OUR
   far-field channel; converts miscalibrated raw cosine into a z-like score with a much
   wider margin. Standard in every VoxSRC-winning system. ~half a day, no retraining.
2. **QMF (quality measure functions)** — calibrate with log duration, SNR, embedding
   magnitude, cohort mean: threshold becomes a learned function of duration+SNR instead
   of a fixed 0.15.
3. **Enrollment augmentation / multi-condition enrollment** — pyroomacoustics RIRs +
   noise on the clean seed; and/or enroll near-field + far-field embeddings, score
   against max/mean.
4. **Dynamic enrollment update** — running centroid absorbing confidently-verified
   far-field turns, weighted by segment length (+4.8 F1 in the hardest far-field mix in
   the Huawei paper).

**Unverified:** ReDimNet 1–2s EERs; HyWA/Huawei-pVAD code; DSENet real-time; USEF-TSE
checkpoints are CC BY-NC; IDRnD original-repo license; TitaNet reproduction.

Sources: github.com/IDRnD/ReDimNet · github.com/IDRnD/redimnet/blob/master/EVALUATION.md ·
github.com/PalabraAI/redimnet2 · arxiv.org/pdf/2407.18223 ·
isca-archive.org/interspeech_2024/chen24l_interspeech.pdf (ERes2NetV2) ·
github.com/modelscope/3D-Speaker · arxiv.org/pdf/2303.00332 (CAM++) ·
github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md ·
huggingface.co/nvidia/speakerverification_en_titanet_large ·
isca-archive.org/interspeech_2025/lin25_interspeech.pdf (Robust pVAD) ·
arxiv.org/abs/2501.03612 (USEF-TP) · github.com/ZBang/USEF-TSE ·
arxiv.org/html/2510.12947 (HyWA) · arxiv.org/abs/2501.03184 (DN-APC) ·
arxiv.org/html/2507.20926v1 (DSENet) · github.com/jingkangqi/DSENet ·
arxiv.org/abs/2409.13502 (Neural Directional Filtering) · arxiv.org/pdf/2212.03401 ·
arxiv.org/pdf/2305.10821 · arxiv.org/pdf/2303.08702 ·
sciencedirect.com/science/article/abs/pii/S0167639315000643 (duration/noise calibration) ·
sri.com QMF publication · arxiv.org/pdf/2308.08766 (DKU-MSXF VoxSRC23) ·
arxiv.org/pdf/2308.12526 (UNISOUND) · arxiv.org/pdf/2509.19721 (short-segment SV MRE) ·
arxiv.org/pdf/2110.05777 (WavLM SSL SV) ·
huggingface.co/pyannote/speaker-diarization-community-1

---

# Appendix D — Turn-taking / duplex behavior (full report)

**Name check: "SoulX-Duplug" CONFIRMED** — github.com/Soul-AILab/SoulX-Duplug,
arXiv 2603.14877 (Mar 2026, Interspeech 2026 submission), weights
Soul-AILab/SoulX-Duplug-0.6B, from Soul App's AI lab.

## Endpointing / turn-taking models

- **Smart Turn: v3.2 is current, no v4.** Whisper-Tiny base, ~8M params, 8MB int8,
  10–100ms CPU, 23 languages, BSD-2. Already at the head of this line.
- **LiveKit Turn Detector v1.0 (Jun 2026)**: dual-branch audio+LLM semantic fusion with
  prosody/timing branch; at 300ms latency 9.9% false-cutoff vs 12.9% Deepgram Flux vs
  27.7% ultraVAD (vendor benchmark). Full v1 cloud-only; **v1-mini is open Apache-2.0**
  but SDK-entangled.
- **UltraVAD (open-sourced ~Aug 2025)**: 0.7B Llama-backbone + Ultravox audio projector;
  takes dialogue history text + last user audio turn → end-of-turn probability; ~65–110ms
  on an A6000; 26 languages. Run speculatively at short silences. The most practical
  *semantic* endpointing upgrade for a hand-rolled cascade.
- **VAP (Voice Activity Projection)**: predicts both parties' future voice activity 2s
  ahead; uniquely predicts backchannel opportunities and turn continuations; CPU
  real-time; academic; expects clean stereo (one channel per speaker — feed user-mic +
  TTS-reference). Experiment, not component.
- Research watch: hierarchical EOT + primary-speaker segmentation (arXiv 2603.13379:
  1.14M params, 36ms, claims 87.7% recall vs 58.9% Smart Turn v3 — no code released);
  Next-Turn (2606.18094); Deepgram Flux (cloud-only).

## Duplex behavior layers over cascades

- **SoulX-Duplug — direct match.** Plug-and-play streaming state predictor in front of
  any half-duplex cascade; unifies VAD + streaming ASR + turn detection. Five states per
  160ms chunk: user_idle / user_nonidle / **user_backchannel** / user_complete /
  user_incomplete — natively the backchannel-vs-interruption distinction, plus semantic
  endpointing. Qwen3-0.6B backbone + frozen GLM-4-Voice tokenizer; SenseVoice-Small for
  the English streaming-ASR leg; ~240–250ms decision latency; beats Freeze-Omni on
  Full-Duplex-Bench. Apache 2.0; ships a streaming inference server. Integration:
  sidecar the Director subscribes to (like Smart Turn). Caveats: tested on L20 (GB10
  should fit; aarch64 builds unverified); brings its own ASR for the streaming leg;
  EN-vs-zh accuracy needs benchmarking.
- **Freeze-Omni** (ICML 2025): the state-head-over-frozen-backbone pattern SoulX-Duplug
  productized; not for this hardware itself.
- **LiveKit Adaptive Interruption Handling**: CNN on the first ~200–300ms of overlap
  (216ms median trigger, ≤30ms inference; 86% precision / 100% recall at 500ms overlap;
  rejects 51% of VAD false positives). Cloud-only — the design lesson (acoustic
  snap-judgment before STT), not reusable code.
- **LiveKit false-interruption resume**: if barge-in fires but STT yields no words within
  a timeout, the agent **resumes from where it left off** — the production-standard
  refinement over restart.
- **Telnyx interrupt_prediction_threshold** (~0.4): commercial table stakes for
  backchannels not cutting the assistant.
- Benchmarks to mine for test cases: Full-Duplex-Bench, τ-Voice (2603.13686), IHBench
  (2606.19595 — for filler interruptions the correct behavior is continue exactly where
  cut off), Awesome-Full-Duplex-SDM list.
- **Assistant-side backchannels**: CHI 2025 (older adults) — pre-recorded "mm-hmm"/"yeah"
  at pause points measurably improves engagement; trivial cascade add. Contextual fillers
  during LLM latency: research only (Thinking-While-Speaking 2511.07397; 2507.22352).

## Backchannel vs interruption classification

- Taxonomy consensus: Competitive / Cooperative / Topic-change / Backchannel; energy-based
  barge-in structurally cannot separate them.
- **Interruption Handling for Conversational Robots (arXiv 2501.01568, HRI 2025, code
  released)** — the closest published blueprint to the planned fix: overlap detect → STT
  → LLM classifies interjection into 4 types → per-type policy (agreement → acknowledge +
  continue; clarification → answer + resume; disruptive → yield or summarize-then-yield).
  **88.78% classification accuracy; 93.69% of 111 live interruptions handled well;
  text-only features** — validates the STT-first reorder. gemma-class local LLM suffices.
- Tiny/fast options: lexical gate (~0ms — backchannel vocabulary is tiny: mm-hmm, uh-huh,
  yeah, okay, right; word-count + stoplist on the first partial); DistilBERT-class
  fine-tune <20ms GPU (no off-the-shelf English checkpoint found; cf. arXiv 2407.14940);
  acoustic-first pre-STT (LiveKit's model is closed; SoulX-Duplug's backchannel state is
  the open equivalent).
- Recovery rule (IHBench): after a backchannel/false cut, resume mid-sentence exactly.

## Cascade latency tricks (2025–26 production defaults)

- **Preemptive/speculative generation — LiveKit default**: LLM starts on partial/early
  transcript before end-of-turn confirmation; discarded/regenerated if context changes;
  TTS gated on turn confirmation. Their stack: ~1.2–1.4s p95 → ~500–650ms p95.
- Pipecat: preemptive generation still an open proposal as of Dec 2025 (issue #3321).
- Cascade-native equivalent for batch STT: kick whisper at Silero speech offset (not
  after Smart Turn + 400ms tail), run the LLM speculatively during the endpoint window,
  gate only TTS start on turn-complete — converts the serial tail to parallel, no new
  models.
- Fillers while thinking: play a short "hmm" clip when LLM TTFT exceeds ~700ms.

**Unverified:** hierarchical-EOT claims; LiveKit vendor benchmarks; pipecat preemptive
status after Dec 2025; SoulX-Duplug English performance and GB10 buildability; pipecat
min-words strategy name.

Sources: github.com/pipecat-ai/smart-turn · huggingface.co/pipecat-ai/smart-turn-v3 ·
daily.co/blog/improved-accuracy-in-smart-turn-v3-1 · github.com/Soul-AILab/SoulX-Duplug ·
arxiv.org/abs/2603.14877 · huggingface.co/Soul-AILab/SoulX-Duplug-0.6B ·
livekit.com/blog/solving-end-of-turn-detection · huggingface.co/livekit/turn-detector ·
ultravox.ai/blog/ultravad-is-now-open-source-introducing-the-first-context-aware-audio-native-endpointing-model ·
huggingface.co/fixie-ai/ultraVAD · github.com/inokoj/VAP-Realtime ·
arxiv.org/abs/2401.04868 · arxiv.org/abs/2603.13379 · arxiv.org/pdf/2606.18094 ·
github.com/VITA-MLLM/Freeze-Omni · livekit.com/blog/adaptive-interruption-handling ·
docs.livekit.io/agents/logic/turns · telnyx.com/release-notes/interruption-prediction-voice-ai-assistants ·
hamming.ai/resources/voice-agent-interruption-handling-runbook ·
deepgram.com/learn/backchannels-vs-interruptions-voice-agents ·
arxiv.org/html/2501.01568v2 (HRI 2025) · arxiv.org/abs/2407.14940 ·
arxiv.org/pdf/2606.19595 (IHBench) · arxiv.org/pdf/2603.13686 (τ-Voice) ·
dl.acm.org/doi/10.1145/3706598.3714228 (CHI 2025) ·
docs.livekit.io/agents/multimodality/audio ·
livekit.com/blog/understand-and-improve-agent-latency ·
github.com/pipecat-ai/pipecat/issues/3321 · arxiv.org/pdf/2511.07397 ·
arxiv.org/pdf/2507.22352 · github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM

---

# Appendix E — Text-to-speech (full report)

**Kokoro is frozen.** English model still Kokoro-82M v1.0 (Jan 2025; only a v1.1-zh
since). Still the latency/efficiency king (sub-0.3s synthesis) and confirmed
GPU-accelerated on DGX Spark, but the field passed it on naturalness. Underused feature:
**misaki accepts inline IPA in brackets + custom lexicon entries** — a pronunciation-fix
channel that exists today and is LOST on any swap to the LLM-token models.

## Candidates

**Kyutai TTS 1.6B (Delayed Streams Modeling, Jul 2025).** The only model that streams on
the *text* side: consumes LLM tokens word-by-word and starts speaking before the
sentence exists. ~220ms end-to-end; 32 concurrent streams at 350ms on one L40S. Quality
clearly above Kokoro; limited explicit style control; occasional loss of whole-sentence
prosody (inherent to word-by-word). Weights CC-BY-4.0, code MIT/Apache. EN+FR. **No
voice cloning** (embedding model withheld) — pick from the tts-voices repo; fine for a
fixed kiosk persona. Integration: moderate — replaces the sentence chunker entirely. The
companion **Unmute** repo (github.com/kyutai-labs/unmute) is an open working
implementation of this exact cascade shape (streaming STT + LLM + streaming TTS with
semantic VAD, interruption support) — worth mining regardless. Unverified on GB10.

**Chatterbox Turbo / Nano (Resemble, Turbo Dec 2025).** Turbo 350M English single-step
decoder; Nano 110M (3× realtime on 8 CPU cores); Multilingual V3 500M; all MIT. Quality
a class above Kokoro (vendor blind tests ~63–65% preference over ElevenLabs —
directional). Zero-shot cloning, exaggeration knob, trained `[laugh]`/`[chuckle]`/
`[cough]` tags. First-chunk ~472ms on RTX 4090, RTF ~0.5 ("~150ms streaming" figure is
likely their hosted API — unverified). **Community-proven on DGX Spark** with one trap:
pinned torch deps silently pull CPU wheels — install `--no-deps` and reinstall deps
manually. Near drop-in (same sentence-chunk→stream shape). PerTh watermarking built in.

**Qwen3-TTS (Jan 2026).** 0.6B/1.7B, Apache 2.0, 10 langs. Claimed 97ms first packet;
SEED test-en WER 1.24; 3s voice cloning + natural-language voice/emotion instructions.
Streaming text *input* undocumented; newest and least battle-tested; unverified on GB10.
Runner-up — an afternoon to benchmark.

**Orpheus 3B (Mar 2025).** Llama-3B → SNAC codec, Apache 2.0; ~200ms streaming claimed
*under vLLM on a strong GPU*; rich emotive tags. On GB10 it's a second 3B AR model
fighting the chat LLM for bandwidth — most operationally expensive option. Skip for now.

**Sesame CSM-1B (Mar 2025).** Unique conversation-context prosody conditioning (pass
prior turns text+audio), but stock repo slow, needs fine-tuning for a consistent voice
(the famous demo quality was never released). Research reference, not a kiosk engine.

**Maya1 (Nov 2025).** 3B, Apache 2.0, 20+ inline emotion tags, voice-from-description.
Same second-3B-AR-model concern; latency unverified. Skip for now.

**F5-TTS / Zonos / Dia / VibeVoice — wrong shape** for low-latency turn-by-turn (not
streaming-native / stalled / long-form multi-speaker targets).

## Latency techniques

(a) Stream text into TTS instead of sentence-buffering (Kyutai productionizes;
S5-TTS/SpeakStream-class research shows sub-50ms TTFA possible). (b) Chunked codec
decode (Mimi/SNAC ~80ms frames). (c) Single-step decoders (Chatterbox Turbo's 10→1).
Cheap wins without switching models: cut the FIRST chunk at the first comma/clause
instead of 120-char sentence; pre-decode chunk N+1 while N plays.

## Duplex/expressive notes

Interruption-friendly incremental synthesis: Kyutai best (stop feeding tokens); any
chunk-streamer keeps duck/cut client-side. Trained laughter/backchannel tokens: Orpheus,
Maya1, Chatterbox. Assistant "mm-hmm": pre-render clips, don't synthesize live.
Conversation-history prosody: only Sesame CSM explicitly. Phoneme/SSML: only Kokoro
(misaki IPA + lexicon); LLM-token models fix pronunciation by respelling text — a real
regression when leaving Kokoro (keep it as fallback or respell in the text-cleanup
stage).

**Unverified:** Chatterbox 150ms; Qwen3-TTS 97ms (vendor) + streaming-text-input;
Maya1 real-time; Kyutai/Qwen3 on GB10 (Kokoro and Chatterbox are the only two with
community-confirmed DGX Spark runs); all "beats ElevenLabs" numbers are vendor-run.

Sources: kyutai.org/tts · github.com/kyutai-labs/delayed-streams-modeling ·
github.com/kyutai-labs/unmute · erogol.substack.com/p/model-check-kyutaitts-streaming-text ·
huggingface.co/kyutai/tts-voices · github.com/kyutai-labs/moshi/issues/404 ·
github.com/resemble-ai/chatterbox · huggingface.co/ResembleAI/chatterbox-turbo ·
replicate.com/resemble-ai/chatterbox-turbo · resemble.ai/learn/models/chatterbox-turbo ·
github.com/QwenLM/Qwen3-TTS · huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base ·
github.com/canopyai/Orpheus-TTS · github.com/SesameAILabs/csm ·
github.com/davidbrowne17/csm-streaming · github.com/SesameAILabs/csm/issues/80 ·
marktechpost.com (Maya1, 2025-11-11) · github.com/SWivid/F5-TTS/issues/700 ·
github.com/microsoft/VibeVoice · huggingface.co/hexgrad/Kokoro-82M ·
github.com/remsky/Kokoro-FastAPI ·
forums.developer.nvidia.com/t/running-kokoro-tts-on-nvidia-dgx-spark-arm64-gb10/368846 ·
github.com/bidual/awesome-dgx-spark · inferless.com (12-model TTS comparison) ·
arxiv.org/html/2606.21882 (S5-TTS) · tryspeakeasy.io/blog/open-source-text-to-speech-2026
