# Target VAD — Claude Code Implementation Plan
## AMD Strix Halo | Local-First | Air-Gapped Compatible

---

## CONTEXT FOR CLAUDE CODE

You are building a **Target VAD** system — a speaker-gated voice activity detection pipeline that:
1. Listens to a microphone audio stream continuously
2. Detects speech segments using Silero VAD (ONNX, CPU)
3. Extracts speaker embeddings using SpeechBrain ECAPA-TDNN
4. Compares against enrolled voiceprints stored locally
5. Only forwards audio to a downstream callback (TTS trigger) if a registered speaker is detected

**Hardware target:** AMD Strix Halo (iGPU + NPU available, but target CPU-first for maximum compatibility)
**Runtime:** Python 3.10+
**Constraint:** Fully local, no cloud calls, air-gapped safe
**Latency target:** <50ms overhead on top of VAD chunk size

---

## PROJECT STRUCTURE TO CREATE

```
target-vad/
├── README.md
├── requirements.txt
├── config.yaml                  # thresholds, device, chunk sizes
├── main.py                      # CLI entry point
├── enroll.py                    # enrollment CLI tool
│
├── vad/
│   ├── __init__.py
│   └── silero_vad.py            # Silero VAD wrapper (ONNX)
│
├── speaker/
│   ├── __init__.py
│   ├── embedder.py              # ECAPA-TDNN embedding extractor
│   ├── verifier.py              # cosine similarity + threshold logic
│   └── enrollment_store.py     # load/save voiceprints (.npy files)
│
├── pipeline/
│   ├── __init__.py
│   └── target_vad_pipeline.py  # orchestrates VAD → verify → callback
│
├── audio/
│   ├── __init__.py
│   └── mic_stream.py           # PyAudio microphone stream
│
├── voiceprints/                 # auto-created, stores enrolled .npy files
│   └── .gitkeep
│
└── tests/
    ├── test_vad.py
    ├── test_verifier.py
    └── test_pipeline.py
```

---

## STEP 1 — Environment Setup

### 1.1 Create `requirements.txt`

```
torch>=2.1.0
torchaudio>=2.1.0
speechbrain>=1.0.0
onnxruntime>=1.17.0
numpy>=1.24.0
scipy>=1.11.0
pyaudio>=0.2.13
pyyaml>=6.0
rich>=13.0.0        # pretty CLI output
```

**NOTE FOR CLAUDE CODE:** Do NOT pin to CUDA-specific torch. Use the standard PyPI torch which defaults to CPU. On Strix Halo, ROCm torch can be substituted manually but CPU is the safe default for this prototype.

### 1.2 Create `config.yaml`

```yaml
vad:
  sample_rate: 16000
  chunk_duration_ms: 30        # 30ms chunks (Silero optimal)
  speech_threshold: 0.5        # Silero confidence threshold
  min_speech_duration_ms: 300  # ignore fragments shorter than this
  padding_ms: 200              # pad before/after speech for context

speaker:
  threshold: 0.75              # cosine similarity threshold (tune this)
  min_segment_duration_ms: 800 # don't verify segments shorter than this
  enrollment_utterances: 5     # number of utterances to average on enroll

audio:
  device_index: null           # null = system default mic
  channels: 1
  sample_rate: 16000
  chunk_size: 480              # 30ms at 16kHz

paths:
  voiceprints_dir: "./voiceprints"
  silero_model_path: null      # null = auto-download and cache locally
```

---

## STEP 2 — VAD Module

### File: `vad/silero_vad.py`

Implement a `SileroVAD` class that:

- Downloads the Silero VAD ONNX model on first run and caches it locally (use `torch.hub.load` with `trust_repo=True` OR download the ONNX directly from the Silero GitHub releases)
- Exposes a `is_speech(chunk: np.ndarray, sample_rate: int) -> float` method returning a confidence score (0.0–1.0)
- Maintains a stateful iterator: `process_stream(audio_generator) -> Iterator[SpeechSegment]` that yields complete speech segments (with start/end timestamps) when speech is detected and then ends
- Implements the padding logic from config (prepend/append silence padding to segments)
- Has a `reset()` method to clear internal state between sessions

```python
# Key interface to implement:
class SileroVAD:
    def __init__(self, config: dict): ...
    def is_speech(self, chunk: np.ndarray) -> float: ...
    def process_stream(self, audio_gen) -> Iterator[SpeechSegment]: ...
    def reset(self): ...

@dataclass
class SpeechSegment:
    audio: np.ndarray       # float32, 16kHz, mono
    start_ms: float
    end_ms: float
    duration_ms: float
```

**IMPORTANT:** Silero VAD requires chunks of exactly 512 samples at 16kHz (32ms) or 256 samples (16ms). Buffer incoming audio to meet this requirement before passing to the model.

---

## STEP 3 — Speaker Embedding Module

### File: `speaker/embedder.py`

Implement an `EmbeddingExtractor` class that:

- Loads SpeechBrain ECAPA-TDNN from `speechbrain/spkrec-ecapa-voxceleb` on first call, caching the model locally at `~/.cache/target-vad/speechbrain/`
- Exposes `extract(audio: np.ndarray, sample_rate: int) -> np.ndarray` returning a 192-dim L2-normalised embedding vector
- Handles short segments gracefully: if segment < 800ms, pad with reflected audio to reach minimum length, then extract
- Runs on CPU (do not assume CUDA)

```python
class EmbeddingExtractor:
    def __init__(self, cache_dir: str): ...
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray: ...
    def _pad_if_short(self, audio: np.ndarray, min_samples: int) -> np.ndarray: ...
```

---

## STEP 4 — Enrollment Store

### File: `speaker/enrollment_store.py`

Implement an `EnrollmentStore` class that:

- Manages a directory of `.npy` voiceprint files, one per registered user
- File naming: `{username}.npy` — stores the averaged embedding across all enrollment utterances
- Exposes:
  - `enroll(username: str, embedding: np.ndarray)` — adds utterance embedding, re-averages stored voiceprint
  - `get_all() -> Dict[str, np.ndarray]` — returns all enrolled voiceprints
  - `get(username: str) -> Optional[np.ndarray]`
  - `delete(username: str)`
  - `list_users() -> List[str]`
- Stores intermediate utterance embeddings during enrollment in a `{username}_utterances.npy` (shape: [N, 192]) and computes the mean as the final voiceprint
- On `finalize_enrollment(username)`: compute mean of all utterances, L2-normalize, save as `{username}.npy`, delete utterances file

---

## STEP 5 — Speaker Verifier

### File: `speaker/verifier.py`

Implement a `SpeakerVerifier` class that:

- Takes `EnrollmentStore` and config threshold as inputs
- Exposes `verify(embedding: np.ndarray) -> VerificationResult`
- Computes cosine similarity between input embedding and all enrolled voiceprints
- Returns the best match if above threshold

```python
@dataclass
class VerificationResult:
    is_registered: bool
    matched_user: Optional[str]   # None if no match
    confidence: float             # best cosine similarity score
    all_scores: Dict[str, float]  # scores against all enrolled users

class SpeakerVerifier:
    def __init__(self, store: EnrollmentStore, threshold: float): ...
    def verify(self, embedding: np.ndarray) -> VerificationResult: ...
    def update_threshold(self, threshold: float): ...
```

Cosine similarity implementation (do not use sklearn, keep dependencies minimal):
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

## STEP 6 — Microphone Stream

### File: `audio/mic_stream.py`

Implement a `MicrophoneStream` class that:

- Uses PyAudio to open the microphone at 16kHz, mono, 16-bit PCM
- Exposes a generator `stream() -> Iterator[np.ndarray]` that yields float32 numpy chunks of `chunk_size` samples
- Converts int16 PCM to float32 in range [-1.0, 1.0]
- Handles device selection via `device_index` (None = default)
- Has `start()`, `stop()`, and context manager support (`__enter__`/`__exit__`)
- Implements a ring buffer to handle PyAudio callback timing jitter

```python
class MicrophoneStream:
    def __init__(self, config: dict): ...
    def stream(self) -> Iterator[np.ndarray]: ...
    def start(self): ...
    def stop(self): ...
    def __enter__(self): ...
    def __exit__(self, *args): ...
```

---

## STEP 7 — Main Pipeline

### File: `pipeline/target_vad_pipeline.py`

Implement the `TargetVADPipeline` class that wires all modules together:

```python
class TargetVADPipeline:
    def __init__(self, config: dict):
        self.vad = SileroVAD(config['vad'])
        self.embedder = EmbeddingExtractor(cache_dir=...)
        self.store = EnrollmentStore(config['paths']['voiceprints_dir'])
        self.verifier = SpeakerVerifier(self.store, config['speaker']['threshold'])
        self.mic = MicrophoneStream(config['audio'])

    def run(self, on_registered_speech: Callable[[SpeechSegment, VerificationResult], None]):
        """
        Main loop:
        1. Stream mic audio
        2. Feed chunks to Silero VAD
        3. When VAD yields a SpeechSegment:
           a. Check duration >= min_segment_duration_ms
           b. Extract embedding
           c. Verify against enrolled voiceprints
           d. If registered: call on_registered_speech callback
           e. Log result with latency breakdown
        """
        ...

    def stop(self): ...
```

The `on_registered_speech` callback is the integration point for TalkBack Local or any TTS system. It receives the full `SpeechSegment` audio and the `VerificationResult`.

---

## STEP 8 — Enrollment CLI

### File: `enroll.py`

Build a CLI tool using `rich` for pretty output:

```
Usage: python enroll.py --user <name> [--utterances 5]

Commands:
  enroll   Record N utterances from mic and enroll a new user
  list     List all enrolled users
  delete   Delete a user's voiceprint
  test     Run a live verification test (not enrolled, just shows scores)
```

Flow for `enroll`:
1. For each of N utterances:
   - Print "Speak now..." with countdown
   - Record until silence detected (use VAD)
   - Extract embedding
   - Store intermediate embedding
2. After N utterances: finalize (average, normalize, save)
3. Print confirmation with embedding stats

---

## STEP 9 — Main Entry Point

### File: `main.py`

```
Usage: python main.py [--config config.yaml] [--threshold 0.75] [--verbose]

Options:
  --config      Path to config file (default: config.yaml)
  --threshold   Override cosine similarity threshold
  --verbose     Print per-segment scores for all enrolled users
  --demo        Demo mode: print "REGISTERED" or "UNKNOWN" instead of triggering TTS
```

In demo mode, the `on_registered_speech` callback simply prints:
```
[REGISTERED] user=siddharth | confidence=0.83 | duration=1.2s | latency=31ms
[UNKNOWN]    best_match=siddharth | score=0.61 | duration=0.9s
```

---

## STEP 10 — Latency Instrumentation

Add timing instrumentation throughout the pipeline. At every `on_registered_speech` call, log:

```python
{
  "vad_latency_ms": ...,        # time VAD took to yield segment
  "embed_latency_ms": ...,      # embedding extraction time
  "verify_latency_ms": ...,     # cosine similarity time
  "total_overhead_ms": ...,     # sum of above (excludes speech duration)
  "segment_duration_ms": ...,
  "matched_user": ...,
  "confidence": ...
}
```

Print this as a live table using `rich.table` when `--verbose` is set.

---

## STEP 11 — Tests

### `tests/test_vad.py`
- Test with a synthetic sine wave (should not trigger VAD)
- Test with a white noise burst (may or may not trigger — document threshold behaviour)
- Test with a real speech WAV file (must trigger, check segment boundaries)

### `tests/test_verifier.py`
- Enroll a synthetic embedding (random unit vector)
- Verify same embedding → expect match
- Verify orthogonal embedding → expect no match
- Verify near-similar embedding → test threshold boundary

### `tests/test_pipeline.py`
- Mock the MicrophoneStream to feed pre-recorded WAV chunks
- End-to-end test: enroll from WAV, then verify from WAV → expect match
- Use pytest fixtures for temp voiceprints directory

---

## IMPLEMENTATION ORDER FOR CLAUDE CODE

Execute in this exact order to enable incremental testing:

1. `requirements.txt` + `config.yaml`
2. `audio/mic_stream.py` → test: can you hear audio?
3. `vad/silero_vad.py` → test: does VAD trigger on voice?
4. `speaker/embedder.py` → test: does it return a 192-dim vector?
5. `speaker/enrollment_store.py` → test: save/load .npy files
6. `speaker/verifier.py` → test: same embedding = high score
7. `enroll.py` → run enrollment for one user
8. `pipeline/target_vad_pipeline.py` → wire it all together
9. `main.py` with `--demo` flag → full end-to-end test
10. `tests/` → write and run pytest suite

---

## AMD STRIX HALO NOTES FOR CLAUDE CODE

- **Do not use CUDA-specific code.** All torch operations should work on CPU.
- SpeechBrain will use `device='cpu'` — explicitly set this in `from_hparams()` call: `run_opts={"device": "cpu"}`
- Silero VAD via `torch.hub.load` requires internet on first run. For air-gapped use, download the ONNX file from `https://github.com/snakers4/silero-vad/releases` and load from local path. Implement both paths.
- PyAudio on Linux may require `portaudio19-dev`. Add an OS check and print install instruction if import fails.
- If ROCm PyTorch is installed on the Strix Halo, the code will automatically use the iGPU for torch operations — no code changes needed, ROCm is transparent to the API.

---

## SUCCESS CRITERIA

The prototype is complete when:

- [ ] `python enroll.py --user siddharth` records 5 utterances and saves a voiceprint
- [ ] `python main.py --demo --verbose` runs continuously and correctly labels registered vs unknown speech
- [ ] Overhead latency (VAD + embed + verify) is logged below 50ms on CPU
- [ ] An unknown speaker talking does NOT trigger the callback
- [ ] The registered user talking consistently DOES trigger the callback
- [ ] All pytest tests pass
