# aarch64/GB10 Availability of Borrowed Turn-Taking Specialists

**Date**: 2026-06-21  
**Host**: Ubuntu 24.04.4 LTS, aarch64 (NVIDIA GB10 / DGX Spark)  
**Python**: 3.12.3 (`/usr/bin/python3`)  
**pip**: 24.0 (system pip, no project venv found — requirements.txt present but venv not yet built)  
**CUDA**: Available — `torch 2.12.0+cu130`, 1× NVIDIA GB10 device  
**Network egress**: OPEN — `curl -sI https://huggingface.co` → HTTP/2 200  

---

## Probe Results

### 0. General environment

| Command | Result |
|---------|--------|
| `python3 --version` | Python 3.12.3 |
| `pip3 --version` | pip 24.0 (system) |
| `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `2.12.0+cu130 True NVIDIA GB10` |
| `curl -sI https://huggingface.co \| head -3` | HTTP/2 200 — network egress is OPEN |

**Nothing was installed during this probe run.** All packages below were checked with `pip index versions` or `pip download --no-deps --dest /tmp/wheelprobe`.

---

### 1. onnxruntime

**Command run**:
```
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

**Result**:
```
1.26.0 ['AzureExecutionProvider', 'CPUExecutionProvider']
```

**Additional probe** (`python3 -c "import onnxruntime as ort; print(ort.get_all_providers())"`):
```
['NvTensorRTRTXExecutionProvider', 'TensorrtExecutionProvider', 'CUDAExecutionProvider',
 'MIGraphXExecutionProvider', 'OpenVINOExecutionProvider', 'DnnlExecutionProvider', ...
 'CPUExecutionProvider']
```

**Notes**:
- onnxruntime **1.26.0 is already installed** system-wide.
- `get_available_providers()` returns `[AzureExecutionProvider, CPUExecutionProvider]` — CUDA EP is compiled in (`get_all_providers()` includes `CUDAExecutionProvider` and `TensorrtExecutionProvider`) but is not reported as *available* at runtime with this wheel, which is the standard `onnxruntime` CPU wheel. The CUDA EP being in `all_providers` but not `available_providers` is expected for the CPU-only wheel build.
- `pip3 index versions onnxruntime` shows versions up to 1.27.0; `onnxruntime-gpu` returns "No matching distribution found" — there is no `onnxruntime-gpu` aarch64 wheel on PyPI. The CUDA path for ORT on aarch64 requires building from source or using NVIDIA's ONNX Runtime container.
- For the Smart Turn and LiveKit models (both CPU ONNX), the installed CPU wheel is fully sufficient.

**aarch64 wheel availability**: CPU wheel — AVAILABLE and installed. GPU wheel — must build from source (not blocking for these models).

**VERDICT**: AVAILABLE (CPU, installed as 1.26.0)

---

### 2. Smart Turn v3 (Pipecat semantic endpointer)

**Probe 1 — pipecat-ai installability on aarch64**:
```
pip3 download --no-deps --dest /tmp/wheelprobe pipecat-ai 2>&1 | tail
```
Result:
```
Downloading pipecat_ai-1.4.0-py3-none-any.whl (11.2 MB)
Saved /tmp/wheelprobe/pipecat_ai-1.4.0-py3-none-any.whl
Successfully downloaded pipecat-ai
```

**Key finding — model is BUNDLED in the wheel**:
`unzip -l pipecat_ai-1.4.0-py3-none-any.whl | grep onnx`:
```
8679182  pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx   (8.3 MB)
2327524  pipecat/audio/vad/data/silero_vad.onnx
```
The `smart-turn-v3.2-cpu.onnx` model ships **inside** the `pipecat-ai` wheel — no separate HuggingFace download required.

**Probe 2 — HuggingFace repo existence**:
```
curl -sI https://huggingface.co/pipecat-ai/smart-turn → HTTP/2 200
curl -sI https://huggingface.co/pipecat-ai/smart-turn-v3 → HTTP/2 200
```
HF repo `pipecat-ai/smart-turn-v3` hosts: `smart-turn-v3.0.onnx`, `smart-turn-v3.1-cpu.onnx`, `smart-turn-v3.1-gpu.onnx`, `smart-turn-v3.2-cpu.onnx`, `smart-turn-v3.2-gpu.onnx` plus benchmarks.

**Model details**:
- Model: `smart-turn-v3.2-cpu.onnx`, **8.3 MB**, int8 quantized
- Architecture: Whisper log-mel feature extraction → ONNX classifier (`LocalSmartTurnAnalyzerV3` in `pipecat/audio/turn/smart_turn/local_smart_turn_v3.py`)
- Input: 16 kHz audio frames (uses `soxr` for resampling)
- Runtime: `onnxruntime` CPU (already installed)
- Dependencies needed beyond onnxruntime: `soxr`, `numpy`, `loguru` (all lightweight)

**pipecat-ai** is a pure-Python `py3-none-any` wheel — installs on any architecture. The `LocalSmartTurnAnalyzerV3` class can be extracted and used standalone without the full pipecat framework.

**aarch64 wheel availability**: py3-none-any — installs everywhere. Model bundled in wheel.

**VERDICT**: AVAILABLE — model bundled in pipecat-ai wheel, runs on CPU onnxruntime (already installed), no separate download needed.

---

### 3. LiveKit turn-detector

**Probe 1 — versions**:
```
pip3 index versions livekit-agents → 1.6.2 (latest)
pip3 index versions livekit-plugins-turn-detector → 1.6.2 (latest)
```

**Probe 2 — wheel download**:
```
pip3 download --no-deps --dest /tmp/wheelprobe livekit-plugins-turn-detector
→ Saved livekit_plugins_turn_detector-1.6.2-py3-none-any.whl (10 kB)
```

**Key finding — plugin is DEPRECATED**:
From `livekit_plugins_turn_detector-1.6.2.dist-info/METADATA`:
> ⚠️ **Deprecated.** This plugin is deprecated and will be removed in a future release. Use `livekit.agents.inference.TurnDetector` instead — it ships with `livekit-agents`, requires no additional install, and replaces both the English and Multilingual text-based models with a unified **audio** end-of-turn detector.

**Architecture (text-based, not audio)**:
- `models.py`: `HG_MODEL = "livekit/turn-detector"`, `ONNX_FILENAME = "model_q8.onnx"`
- HF repo `livekit/turn-detector` — confirmed live (HTTP/2 200); contains `model_quantized.onnx` + tokenizer files
- This is a **text/transcript-based** EOU model (GPT-2 style transformer, quantized ONNX), not an audio VAD
- Model revision for English: `v1.2.2-en`; multilingual: `v0.4.1-intl`
- Model file size: not reported by HF API without auth, but `model_q8.onnx` is typically ~40–80 MB (GPT-2 small quantized)
- Requires: `onnxruntime>=1.18`, `transformers>=4.47.1`, `numpy>=1.26`, `livekit-agents>=1.6.2`
- The replacement `livekit.agents.inference.TurnDetector` (in `livekit-agents` 1.6+) is the new unified audio EOU detector

**Self-hostable vs cloud-coupled**: The ONNX model is **self-hostable** — downloaded via `huggingface_hub.hf_hub_download` at startup. No LiveKit cloud dependency for inference. However, `livekit-agents` pulls in the full LiveKit WebRTC stack which is heavyweight.

**aarch64 wheel availability**: py3-none-any — installs everywhere. Model fetched from HuggingFace at runtime (network required first run).

**VERDICT**: AVAILABLE but DEPRECATED (text-based; the newer audio EOU in livekit-agents 1.6+ is the real successor — requires livekit-agents install)

---

### 4. Krisp-style backchannel classifier

**Finding A — "krisp" PyPI package is unrelated**:
```
pip3 download --no-deps --dest /tmp/wheelprobe krisp → Downloaded krisp-0.1.6-py3-none-any.whl
```
The `krisp` PyPI package is a **bioinformatics tool** (CRISPR diagnostic assay design — `krisp_fasta`, `krisp_vcf`). Completely unrelated to Krisp audio noise reduction.

**Finding B — Real Krisp VIVA SDK is proprietary**:
Pipecat ships `pipecat/audio/turn/krisp_viva_turn.py` which `import krisp_audio` — this is Krisp's proprietary native SDK (`krisp_audio` Python bindings), not available on PyPI. The pipecat module logs: *"In order to use KrispVivaTurn, you need to install krisp_audio."*

**Finding C — Open-source backchannel classifier (Thai/Japanese only)**:
```
pip3 index versions backchannel-classifier → 0.4.0 (1 version)
pip3 download --no-deps --dest /tmp/wheelprobe backchannel-classifier → 65 kB wheel, saved
```
`backchannel-classifier 0.4.0` is a scikit-learn pickle model (~4 KB EN model, ~219 KB JA model), but it is **Thai and Japanese only** — uses regex feature extraction on Thai Unicode characters. Not applicable for English.

**Finding D — Alternatives for English backchannel detection**:
No open-source English backchannel classifier was found on PyPI. Available approaches for English:
1. **Lexical/regex baseline**: Rules on short utterances ("uh-huh", "yeah", "right", "ok", "mm-hmm", "sure" etc.) — already a natural fit given the project's existing transcription pipeline.
2. **SpeechBrain** (already in requirements.txt, `speechbrain>=1.0.0`): Can be used with an Intent Classification model or fine-tuned on backchannel labels. No pre-trained English backchannel model shipped by default, but the framework is available.
3. **HuggingFace transformers**: A small BERT/DistilBERT classifier could be fine-tuned on datasets like the Switchboard backchannel corpus, but this requires training data and effort.
4. **Audio-feature heuristic**: Short duration (<300ms) + low word count + filler acoustics — implementable with existing VAD + Whisper.

**VERDICT**: PROPRIETARY (Krisp VIVA SDK — not publicly installable). Open English alternative: lexical baseline is the only immediately available option; SpeechBrain could support a trained classifier with effort.

---

## Summary Table

| Candidate | Command | aarch64 wheel | Model source & size | CPU runnable | CUDA runnable | VERDICT |
|-----------|---------|---------------|---------------------|--------------|---------------|---------|
| **onnxruntime 1.26.0** | `import onnxruntime as ort; ort.get_available_providers()` → `[Azure, CPU]` | Installed (CPU wheel) | N/A — runtime only | YES | NO (GPU wheel not on PyPI; needs build-from-source) | **AVAILABLE** |
| **Smart Turn v3.2** (pipecat-ai) | `pip download pipecat-ai` → `pipecat_ai-1.4.0-py3-none-any.whl` (11 MB, contains 8.3 MB ONNX) | py3-none-any ✓ | Bundled in wheel (`smart-turn-v3.2-cpu.onnx`, 8.3 MB int8); also on HF `pipecat-ai/smart-turn-v3` | YES (CPU ONNX) | YES (separate `-gpu.onnx` on HF, but needs CUDA ORT) | **AVAILABLE — NEEDS-DOWNLOAD** (wheel not yet installed) |
| **LiveKit turn-detector** (text EOU) | `pip index versions livekit-plugins-turn-detector` → 1.6.2; wheel is 10 kB py3-none-any | py3-none-any ✓ | HF `livekit/turn-detector` → `model_q8.onnx` (~40–80 MB GPT-2 quantized); fetched at runtime | YES (CPU ONNX) | NO tested | **AVAILABLE but DEPRECATED** (text-based; heavyweight dep chain) |
| **Krisp VIVA backchannel/turn** | `pip index versions krisp` → bioinformatics tool (wrong package); pipecat requires `krisp_audio` native SDK | Not on PyPI | Proprietary SDK (`krisp_audio` Python bindings) | UNKNOWN | UNKNOWN | **PROPRIETARY** |
| **backchannel-classifier (Thai/JP)** | `pip download backchannel-classifier` → 65 kB; sklearn pickle | py3-none-any ✓ | Bundled sklearn model (~4 KB) | YES | N/A | **UNAVAILABLE** for English use-case |
| **English lexical backchannel baseline** | Regex/dict matching on Whisper transcript | N/A | No model needed | YES | N/A | **AVAILABLE** (implement in code) |

---

## RECOMMENDATION

### Task 5 — Endpointer choice

**Use Smart Turn v3.2 (`pipecat-ai` / `LocalSmartTurnAnalyzerV3`).**

Rationale:
- The `smart-turn-v3.2-cpu.onnx` model (8.3 MB, int8) is **bundled inside the `pipecat-ai` wheel** — no separate model download needed; a single `pip3 install pipecat-ai` (or extracting the ONNX file directly from the downloaded wheel) gets the model onto disk.
- It runs on the already-installed `onnxruntime 1.26.0` CPU wheel with no additional dependencies beyond `soxr` and `numpy`.
- It is audio-based (Whisper mel features), not transcript-based — it fires at utterance end based on audio cues, which is exactly what this pipeline needs for low-latency end-of-turn detection.
- The `LocalSmartTurnAnalyzerV3` class can be lifted out of pipecat and used standalone, keeping the dependency footprint minimal.
- The deprecated `livekit-plugins-turn-detector` is text-based (needs transcription first), adds the full LiveKit WebRTC stack, and is on a deprecation path — avoid.

**Installation action for Task 5**: `pip3 install pipecat-ai` (or extract `smart-turn-v3.2-cpu.onnx` from the downloaded wheel at `/tmp/wheelprobe/pipecat_ai-1.4.0-py3-none-any.whl`). Nothing else needs installing.

### Task 6 — Backchannel approach

**Use the lexical baseline; do not attempt Krisp VIVA.**

Rationale:
- Krisp VIVA is a proprietary native SDK (`krisp_audio`) — not publicly installable, no PyPI package, requires a commercial agreement with Krisp.
- The `backchannel-classifier` PyPI package is Thai/Japanese-only — useless for English.
- A lexical/regex classifier on Whisper transcripts (short duration + token set: "yeah", "uh-huh", "right", "ok", "mm-hmm", "sure", "got it", "I see", etc.) is immediately implementable, zero new dependencies, and matches how the existing pipeline already processes speech.
- If SpeechBrain fine-tuning is later desired, the framework is already in `requirements.txt` and HuggingFace network egress is open for pulling Switchboard-style corpora.

**Conclusion**: The lexical baseline is the appropriate Task 6 starting point. It gives a measurable F1 against which a trained model can later be compared, without blocking progress on any package availability issue.

---

## Task 6 — lexical backchannel baseline

**Run command**: `PYTHONPATH=/path/to/target-vad python3 bench/backchannel_eval.py bench/backchannel_labels.json`

### Dataset

- **Total cases**: 54
- **Class balance**: 29 BACKCHANNEL / 25 INTERRUPT (54/46 split — reasonably balanced)
- **Real-log utterances**: 10 drawn from `logs/kiosk-*.jsonl` `user_turn_complete` events (`"Thank you."`, `"Yeah."`, `"Okay."`, `"Stop."`, `"Can you stop?"`, `"Is there more to the story?"`, `"What is bioluminescence?"`, `"Can you go back to the story about Flika?"`, `"so uh"`, `"Yeah."`)
- **Hand-written cases**: 44, covering pure backchannels, multi-word acks (go on, no problem, no worries), plain questions/commands, ack-pivot combinations, and deliberate hard/ambiguous cases the lexical approach plausibly gets wrong

### Measured accuracy

```
lexical accuracy: 48/54 = 88.9%
confusion (gold,pred) BB/BI/IB/II: {'BB': 24, 'BI': 5, 'IB': 1, 'II': 24}
```

- **Precision (BACKCHANNEL)**: 24/25 = 96.0%  (almost no false backchannels — the "default-to-cut" design is conservative)
- **Recall (BACKCHANNEL)**: 24/29 = 82.8%  (misses ~17% of true backchannels)
- **False interrupt rate** (BI): 5/29 = 17.2%  (user says something harmless; system cuts anyway)
- **False backchannel rate** (IB): 1/25 = 4.0%  (system stays talking when user wanted to interrupt)

### Misclassifications (6 total)

| Gold | Pred | Text | What it reveals |
|------|------|------|-----------------|
| BACKCHANNEL | INTERRUPT | `'interesting'` | Common listener-reaction word not in the backchannel token set; defaults to INTERRUPT. Easy fix: add "interesting" to BACKCHANNEL set. |
| BACKCHANNEL | INTERRUPT | `'no'` | `'no'` is in FORCE_INTERRUPT to catch corrections; however in isolation ("no no, I get it") it is often a filler. The ambiguity is real and hard to resolve without prosody. |
| BACKCHANNEL | INTERRUPT | `'yeah no'` | Conversational filler ("yeah no totally"); `'no'` triggers FORCE_INTERRUPT. Same `'no'` ambiguity as above. |
| BACKCHANNEL | INTERRUPT | `'and'` | Single-word continuation prompt (like "go on"); not in backchannel set, so defaults to INTERRUPT. Relatively rare in practice. |
| BACKCHANNEL | INTERRUPT | `'so uh'` | From real log — hesitation filler before the user resumed their question; `'so'` is not in any set, defaults to INTERRUPT. |
| INTERRUPT | BACKCHANNEL | `'right?'` | The word `'right'` is a backchannel token, but with rising intonation it is a confirmation-seek. Text alone cannot distinguish. Only false-backchannel (IB) miss — most dangerous type. |

**What the errors reveal**: The classifier's errors fall into two clear clusters:
1. **Unknown single-token words** (`interesting`, `and`, `so uh`) that default to INTERRUPT. These are benign errors in a "default-to-cut" system — the user gets interrupted slightly too often, not silenced.
2. **`'no'` ambiguity**: The word sits at the boundary of correction vs filler. Prosody (short, unstressed "no" vs sharp stressed "NO") would resolve this; text cannot.
3. **Intonation-dependent cases** (`right?`): The one IB miss. This is the most dangerous error type because it means the system keeps talking when the user wanted to interject. Intonation-aware models would catch this.

### Caveats

- **Small sample (54 cases)**: The error rate estimates have wide confidence intervals (±6–8 pp at 95%). The 88.9% accuracy is indicative, not definitive.
- **Lab-authored utterances**: 44/54 cases are hand-written to stress-test the classifier, not drawn from production traffic. Real-world accuracy may be higher on common backchannels and lower on unusual phrasings.
- **No prosody signal**: All hard cases (`right?`, `no`, `yeah no`, `interesting`) would likely be resolved correctly by a prosody-aware or audio-based model.

### Recommendation

**Keep lexical for v0.** The 88.9% accuracy is adequate for a barge-in gate where false interrupts are annoying but recoverable (auto-resume self-heals) and false backchannels are rare (4% IB rate). The `'no'` + intonation cluster is the clearest gap; any borrowed model would need to beat ~89% accuracy on a balanced English set and specifically improve IB recall to justify added complexity. The three-line fixes (`interesting`, `and` added to BACKCHANNEL; `no` moved to a "context-sensitive" tier) would push lexical accuracy to ~94% without a model change, and should be considered before reaching for a borrowed model.
