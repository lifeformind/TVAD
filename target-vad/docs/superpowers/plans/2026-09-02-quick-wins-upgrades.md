# Quick-Wins Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four agreed quick-win upgrades in one pass: (A) native llama-server with GBNF markdown suppression + prompt caching, (B) Gemma 4 26B-A4B model swap + clarification-asking prompt, (C) AS-Norm speaker-score calibration (shadow mode) + running enrollment centroid, (D) Parakeet-TDT STT backend via NeMo behind a config selector.

**Architecture:** Each workstream slots into an existing seam: the launcher script swaps the serving binary; `LlmClient` grows three payload fields; `SafetyNet` gains an optional normalizer and centroid update (reducer untouched — it stays pure); STT gets a second backend class satisfying the existing 4-point `StreamingStt` contract, selected in `kiosk.py`. Workstreams are independent — they can be executed in any order, but the task numbering below is the recommended one (server before payload fields; calibration before any threshold retuning).

**Tech Stack:** llama.cpp (native `llama-server`, CUDA sm_121), GGUF (Gemma 4 26B-A4B Q4_K_M), GBNF grammars, numpy (AS-Norm), NVIDIA NeMo ASR (`nemo_toolkit[asr]`, parakeet-tdt-0.6b-v2), pytest.

**Spec:** `target-vad/docs/notes/2026-09-02-sota-upgrade-research.md` (Type 1 items 1–4 + Type 2 items 1, 7-partial). The embedder swap (ReDimNet2/ERes2NetV2) is deliberately NOT in this plan — it waits for live shadow-mode calibration data so the A/B is fair.

## Global Constraints

- All commands run from `target-vad/` unless stated. Tests: `python3 -m pytest tests -q` (there is no `python` on PATH — always `python3`). Full suite is 858 pass / 2 skip green at plan time; it must stay green after every task.
- aarch64/GB10: no faster-whisper/CTranslate2 CUDA (no wheel — confirmed 2026-03). NeMo runs via plain PyTorch (proven on this hardware); NEVER via Riva/nemo2riva (broken on aarch64).
- New boolean config keys use the strict-bool idiom: compare `is True`, warn to stderr on present-but-non-bool (the "flase" lesson — see `assembly.py:129-135` for the pattern).
- Reducer purity: `modes/director/reducer.py` never reads config dicts. Any threshold the reducer compares goes through the frozen `DirectorConfig` dataclass (`modes/director/config.py`) populated in `assembly.py:_director_config_from`.
- New-feature defaults: OFF/no-op in code defaults, enabled in shipped `config.yaml` (byte-for-byte no-regression when the key is absent).
- New config keys follow the four-step convention: (1) key in `config.yaml` with a rationale comment; (2) read at its construction site; (3) `tune/knobs.py` registration if live-tunable (backend selectors and device pins are deliberately excluded); (4) meta-test assertions — `tests/tune/test_knobs.py` auto-checks knobs against the real config.yaml, and `tests/director/test_assembly.py::test_shipped_config_yaml_matches_live_readers` (line ~221) must be extended for keys read by assembly.
- Underscore kwargs (`_embedder=None` etc.) are test seams only; production code omits them.
- Commit after every task. Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- `kiosk-stack.sh` has no automated tests; its verification is `bash -n` plus the live checks written into each task.

---

## Workstream A — Native llama-server, GBNF grammar, prompt caching

### Task 1: Build native llama-server for GB10

**Files:**
- Modify: `kiosk-stack.sh` (config block lines 8-24, dispatch table lines ~245-257)

**Interfaces:**
- Produces: `$HOME/.local/opt/llama.cpp/build/bin/llama-server` binary; `cmd_build_server` subcommand; `LLAMA_SERVER_BIN` variable consumed by Task 2.

- [ ] **Step 1: Add the binary path variable and build subcommand**

In the config block (after `CHAT_FORMAT`), add:

```bash
# Native llama.cpp server (replaces python -m llama_cpp.server; needed for
# Gemma 4 MoE day-one support, per-request GBNF grammar, and slot prompt
# caching). Built by `kiosk-stack.sh build-server`.
LLAMA_CPP_DIR="$HOME/.local/opt/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
```

Add the function next to `cmd_build` (llama-cpp-python rebuild, lines ~73-84 — leave that untouched as the rollback path):

```bash
cmd_build_server() {
  # Native llama.cpp with CUDA for the GB10 (Blackwell, sm_121).
  if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  fi
  git -C "$LLAMA_CPP_DIR" pull --ff-only
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-server -j "$(nproc)"
  "$LLAMA_SERVER_BIN" --version
}
```

Register `build-server) cmd_build_server ;;` in the dispatch table.

- [ ] **Step 2: Syntax-check and build**

Run: `bash -n kiosk-stack.sh` (expect silence), then `./kiosk-stack.sh build-server`.
Expected: build completes; `--version` prints a build hash. If cmake can't find CUDA, prepend `CUDACXX=/usr/local/cuda/bin/nvcc`.

- [ ] **Step 3: Manual smoke against the CURRENT model**

```bash
"$HOME/.local/opt/llama.cpp/build/bin/llama-server" \
  --model "$(ls $HOME/.cache/models/models--unsloth--gemma-3-4b-it-GGUF/snapshots/*/gemma-3-4b-it-Q5_K_M.gguf | head -1)" \
  --host 127.0.0.1 --port 8081 -ngl 999 --ctx-size 4096 --jinja &
sleep 20
curl -sf http://127.0.0.1:8081/v1/models
curl -s http://127.0.0.1:8081/v1/chat/completions -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":20}'
kill %1
```

Expected: a models list and a sensible chat reply (the `--jinja` flag makes the server use the GGUF's embedded chat template — required for Gemma).

- [ ] **Step 4: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): build-server subcommand — native llama-server (CUDA sm_121)"
```

### Task 2: Launch the native server from the stack script

**Files:**
- Modify: `kiosk-stack.sh:140-160` (`start_llm_bg`), `kiosk-stack.sh:180-207` (`ensure_llm`)

**Interfaces:**
- Consumes: `LLAMA_SERVER_BIN` from Task 1.
- Produces: llama-server on 127.0.0.1:8080 with `--slots` prompt caching; same `/v1` API surface the kiosk already talks to. `N_CTX` raised for the bigger model later.

- [ ] **Step 1: Replace the launch command**

In `start_llm_bg()`, replace the `python3 -m llama_cpp.server` invocation with:

```bash
  if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "llama-server not built — run: $0 build-server" >&2; exit 5
  fi
  nohup "$LLAMA_SERVER_BIN" \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --ctx-size "$N_CTX" \
    -ngl 999 \
    --parallel 1 \
    --cache-reuse 256 \
    --jinja \
    >"$LLM_LOG" 2>&1 &
  echo $! > "$PID_FILE"; WE_STARTED_LLM=1
```

Notes to preserve in comments: `--interrupt_requests` does not exist here and is not needed — native llama-server does not abort in-flight streams when `/v1/models` is probed (that was the llama-cpp-python bug fixed in 4741b27). `--cache-reuse 256` enables KV prefix reuse within the single slot. `N_GPU_LAYERS`/`CHAT_FORMAT` variables become dead — delete them and their comment lines.

- [ ] **Step 2: Syntax check + live cycle**

Run: `bash -n kiosk-stack.sh`, then `./kiosk-stack.sh stop; ./kiosk-stack.sh start` and confirm: kiosk boots (its startup `ping()` passes), one full voice exchange works, `./kiosk-stack.sh stop` reaps the server (`.llm.pid` gone, port free).

- [ ] **Step 3: Run the suite (regression net)**

Run: `python3 -m pytest tests -q` — Expected: 858 pass / 2 skip (nothing in the suite touches the launcher, this catches accidental damage elsewhere).

- [ ] **Step 4: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): serve via native llama-server (cache-reuse, --jinja; interrupt_requests obsolete)"
```

### Task 3: GBNF grammar + cache fields in LlmClient

**Files:**
- Create: `modes/talkback/gbnf.py`
- Modify: `modes/talkback/llm.py:16-28` (`__init__`), `llm.py:39-45` (payload), `kiosk.py:229-235` (construction), `config.yaml:105-115` (llm block), `tune/knobs.py` (~line 220, LLM knobs)
- Test: `tests/kiosk/talkback/test_llm.py`, `tests/kiosk/talkback/test_gbnf.py`

**Interfaces:**
- Produces: `SPEECH_GRAMMAR: str` (module constant in `gbnf.py`); `LlmClient.__init__(..., grammar: str | None = None, cache_prompt: bool = False)`. Payload gains `"grammar"` (only when set), `"cache_prompt"`, `"id_slot": 0` (only when cache_prompt). Config key `kiosk.talkback.llm.no_markdown_grammar` (strict bool), `kiosk.talkback.llm.cache_prompt` (strict bool).

- [ ] **Step 1: Write the grammar module**

```python
"""GBNF grammar for llama-server: make markdown structurally impossible.

The decoder simply cannot emit *, #, backtick, _ or ~ — the characters
strip_markdown_for_speech (speech_text.py) spends most of its regexes on.
The stripper stays as belt-and-suspenders for what a char class can't
express (bullet dashes at line starts, [link](url) syntax).
"""

# Any character except markdown marker characters; explicitly allows
# newlines, punctuation, and unicode (GBNF negated classes are by
# codepoint, so non-ASCII passes).
SPEECH_GRAMMAR = 'root ::= [^*#`_~]*'
```

- [ ] **Step 2: Write the failing tests**

`tests/kiosk/talkback/test_gbnf.py`:

```python
from modes.talkback.gbnf import SPEECH_GRAMMAR


def test_grammar_excludes_markdown_markers():
    for ch in "*#`_~":
        assert ch in SPEECH_GRAMMAR  # present in the negated class

def test_grammar_is_a_single_root_rule():
    assert SPEECH_GRAMMAR.startswith("root ::=")
```

In `tests/kiosk/talkback/test_llm.py`, add (follow the existing `FakeStreamResponse`/mock-session convention at lines 11-34; capture the payload via `mock_session.post.call_args`):

```python
@pytest.mark.asyncio
async def test_payload_includes_grammar_and_cache_fields():
    client = LlmClient("http://x/v1", "m", grammar="root ::= [^*]*", cache_prompt=True)
    fake = FakeStreamResponse([])
    session = MagicMock()
    session.post = MagicMock(return_value=fake)
    client._session = session
    async for _ in client.stream([{"role": "user", "content": "hi"}]):
        pass
    payload = session.post.call_args.kwargs["json"]
    assert payload["grammar"] == "root ::= [^*]*"
    assert payload["cache_prompt"] is True
    assert payload["id_slot"] == 0

@pytest.mark.asyncio
async def test_payload_omits_grammar_and_cache_when_disabled():
    client = LlmClient("http://x/v1", "m")   # defaults: grammar=None, cache_prompt=False
    fake = FakeStreamResponse([])
    session = MagicMock()
    session.post = MagicMock(return_value=fake)
    client._session = session
    async for _ in client.stream([{"role": "user", "content": "hi"}]):
        pass
    payload = session.post.call_args.kwargs["json"]
    assert "grammar" not in payload
    assert "cache_prompt" not in payload and "id_slot" not in payload
```

(Adapt the exact post-capture idiom to how the existing tests in that file grab the payload — mirror them, don't invent.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/kiosk/talkback/test_gbnf.py tests/kiosk/talkback/test_llm.py -q`
Expected: FAIL (`ModuleNotFoundError` / `TypeError: unexpected keyword argument 'grammar'`)

- [ ] **Step 4: Implement**

`llm.py` — extend `__init__` signature: `def __init__(self, base_url, model, temperature=0.6, max_tokens=512, grammar=None, cache_prompt=False)`, store `self._grammar`, `self._cache_prompt`. After the payload dict (`llm.py:39-45`) add:

```python
        if self._grammar:
            payload["grammar"] = self._grammar
        if self._cache_prompt:
            payload["cache_prompt"] = True
            payload["id_slot"] = 0  # single conversation = single slot; keeps the KV prefix warm across turns
```

`kiosk.py:229-235` — plumb config (strict-bool idiom):

```python
    grammar = None
    if llm_cfg.get("no_markdown_grammar", False) is True:
        from modes.talkback.gbnf import SPEECH_GRAMMAR
        grammar = SPEECH_GRAMMAR
    llm = LlmClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        model=llm_cfg.get("model", "gemma-3-4b-it"),
        temperature=llm_cfg.get("temperature", 0.6),
        max_tokens=llm_cfg.get("max_tokens", 512),
        grammar=grammar,
        cache_prompt=llm_cfg.get("cache_prompt", False) is True,
    )
```

`config.yaml` llm block — add with rationale comments:

```yaml
      # GBNF grammar bans *, #, backtick, _, ~ at the DECODER, so markdown
      # can't be emitted at all; strip_markdown_for_speech stays as the net
      # for bullets/links. Only a real boolean true enables.
      no_markdown_grammar: true
      # llama-server slot prompt caching: keeps the conversation's KV prefix
      # warm across turns (id_slot 0 + cache_prompt). Reported TTFT
      # 400ms -> <50ms on cached prefixes. Requires the native llama-server
      # launched by kiosk-stack.sh (Task 2).
      cache_prompt: true
```

`tune/knobs.py` — next to the existing LLM knobs (~line 220):

```python
    Knob(TB + "llm.no_markdown_grammar", VOICE, "No-markdown grammar", "bool",
         "Ban markdown characters at the decoder (GBNF).",
         "Replaces regex stripping as the primary defense.",
         strict_bool=True),
```

(If `strict_bool` knobs are asserted as "exactly the documented keys" in `tests/tune/test_knobs.py::test_strict_bools_are_exactly_the_documented_keys` (line ~57), add this key to that documented set.)

- [ ] **Step 5: Run tests to verify they pass, then the full suite**

Run: `python3 -m pytest tests/kiosk/talkback tests/tune -q` then `python3 -m pytest tests -q`
Expected: all pass. If `tests/director/test_assembly.py::test_shipped_config_yaml_matches_live_readers` asserts the llm block's key set, extend it with the two new keys.

- [ ] **Step 6: Live check**

Restart the stack, ask for "a list of three fruits". Expected: spoken reply contains no asterisks/hashes read aloud, and `logs/llm.log` shows no grammar errors. Second turn in a session should log a prompt-cache hit (llama-server logs `n_past` reuse).

- [ ] **Step 7: Commit**

```bash
git add modes/talkback/gbnf.py modes/talkback/llm.py kiosk.py config.yaml tune/knobs.py tests/
git commit -m "feat(llm): GBNF no-markdown grammar + slot prompt caching in LlmClient"
```

---

## Workstream B — Gemma 4 26B-A4B swap

### Task 4: Model download, launcher vars, prompt hardening

**Files:**
- Modify: `kiosk-stack.sh:8-24` (MODEL_REPO/MODEL_GLOB/N_CTX), `config.yaml:105-115` (llm.model, system_prompt)

**Interfaces:**
- Consumes: native llama-server launch from Task 2.
- Produces: Gemma 4 26B-A4B serving on :8080; clarification-asking system prompt.

- [ ] **Step 1: Verify the exact GGUF repo name, then download**

The research (Appendix A of the spec) names the model but GGUF repo naming varies. Run:

```bash
python3 -c "from huggingface_hub import list_models; [print(m.id) for m in list_models(search='gemma-4-26b', limit=20)]"
```

Pick the Unsloth or ggml-org GGUF repo for **gemma-4-26b-a4b-it** (instruction-tuned MoE). Then set in `kiosk-stack.sh`:

```bash
MODEL_REPO="unsloth/gemma-4-26b-a4b-it-GGUF"   # verified via list_models above
MODEL_GLOB="*Q4_K_M*.gguf"
N_CTX=8192
```

and run `./kiosk-stack.sh download` (existing subcommand, `hf download`). Expect ~16GB. If no 26B-A4B GGUF exists yet under any publisher, STOP this task and fall back to `unsloth/gemma-4-E4B-it-GGUF` (the 4B successor) — record which was used in the commit message.

- [ ] **Step 2: Update config.yaml model name and system prompt**

`llm.model: "gemma-4-26b-a4b-it"` (informational — llama-server serves whatever is loaded; the client echoes this in the payload).

Append to the existing `system_prompt` block scalar (keep the current text — it is live-validated for story length):

```yaml
        Your input comes from speech recognition and may occasionally
        contain misheard words. If a request seems nonsensical or contains
        an improbable phrase, don't guess: briefly say what you heard and
        ask one short clarifying question instead of answering.
```

(The few-shot garbled→clarify examples from the spec are NOT added here — with a 26B-class model the instruction alone is the right first step; add examples only if live testing shows it ignoring the clause.)

- [ ] **Step 3: Live smoke + throughput measurement**

`./kiosk-stack.sh stop && ./kiosk-stack.sh start`. First load of a 16GB model takes minutes — watch `logs/llm.log`. Then:

```bash
time curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 30 in words."}],"max_tokens":256}' | tail -c 300
```

Expected: coherent output; decode ≥ 35 t/s (research measured ~45-52 on this hardware; llama-server logs print exact t/s). One full voice exchange through the kiosk. If tokens/sec < 20 or load OOMs, fall back to E4B and note it.

- [ ] **Step 4: Suite + shipped-config assertions**

Run: `python3 -m pytest tests -q`. If `test_shipped_config_yaml_matches_live_readers` or any test pins `llm.model`'s old value, update the assertion to the new name.

- [ ] **Step 5: Commit**

```bash
git add kiosk-stack.sh config.yaml tests/
git commit -m "feat(llm): Gemma 4 26B-A4B (Q4_K_M) + ASR-aware clarification prompt"
```

---

## Workstream C — Speaker-score calibration (AS-Norm shadow) + running centroid

### Task 5: AsNorm scorer

**Files:**
- Create: `core/speaker/calibration.py`
- Test: `tests/core/test_calibration.py`

**Interfaces:**
- Produces: `class AsNorm: __init__(self, cohort: np.ndarray, top_k: int = 50)`; `score(self, enroll: np.ndarray, test: np.ndarray) -> float` (adaptive symmetric normalization; inputs are L2-normalized embeddings, any dim); `raw(enroll, test) -> float` (plain cosine, convenience). Consumed by Tasks 6-7.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from core.speaker.calibration import AsNorm


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)

def _cohort(n=100, d=8, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, d)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def test_same_embedding_scores_higher_than_orthogonal():
    norm = AsNorm(_cohort())
    a = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    b = _unit([0, 1, 0, 0, 0, 0, 0, 0])
    assert norm.score(a, a) > norm.score(a, b)

def test_normalized_scale_is_zscore_like():
    # A raw score equal to the cohort mean should normalize to ~0;
    # a genuine match lands far above.
    norm = AsNorm(_cohort())
    a = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    assert norm.score(a, a) > 3.0          # many std devs above imposters

def test_top_k_larger_than_cohort_is_clamped():
    small = _cohort(n=5)
    norm = AsNorm(small, top_k=50)
    a = _unit([1, 1, 0, 0, 0, 0, 0, 0])
    assert np.isfinite(norm.score(a, a))

def test_raw_matches_cosine():
    norm = AsNorm(_cohort())
    a = _unit([1, 2, 3, 0, 0, 0, 0, 0])
    b = _unit([3, 2, 1, 0, 0, 0, 0, 0])
    assert abs(norm.raw(a, b) - float(np.dot(a, b))) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/core/test_calibration.py -q` — Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
"""AS-Norm: adaptive symmetric score normalization for speaker scores.

Raw cosine on the far-field array channel is miscalibrated (owner band
0.23-0.47 vs stranger 0.07 live, forcing threshold 0.15 — see spec
Appendix C). Normalizing each score against a cohort of imposter
embeddings pushed through the same channel converts it to a z-like score
with a much wider genuine/imposter margin. Standard in VoxSRC-winning
systems (DKU-MSXF 2023 et al.).
"""
import numpy as np


class AsNorm:
    def __init__(self, cohort: np.ndarray, top_k: int = 50):
        # cohort: (N, D) L2-normalized imposter embeddings (build_cohort.py)
        self._cohort = np.asarray(cohort, dtype=np.float32)
        self._top_k = max(1, min(int(top_k), len(self._cohort)))

    def _stats(self, emb: np.ndarray) -> tuple[float, float]:
        scores = self._cohort @ emb
        top = np.sort(scores)[-self._top_k:]
        return float(top.mean()), float(top.std() + 1e-6)

    def raw(self, enroll: np.ndarray, test: np.ndarray) -> float:
        return float(np.dot(enroll, test))

    def score(self, enroll: np.ndarray, test: np.ndarray) -> float:
        s = self.raw(enroll, test)
        mu_e, sd_e = self._stats(enroll)
        mu_t, sd_t = self._stats(test)
        return 0.5 * ((s - mu_e) / sd_e + (s - mu_t) / sd_t)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m pytest tests/core/test_calibration.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/speaker/calibration.py tests/core/test_calibration.py
git commit -m "feat(speaker): AsNorm adaptive score normalization (top-K cohort)"
```

### Task 6: Cohort builder bench script

**Files:**
- Create: `bench/build_cohort.py`
- Test: `tests/bench/test_build_cohort.py`

**Interfaces:**
- Consumes: `EmbeddingExtractor.extract(audio: np.ndarray) -> np.ndarray` (`core/speaker/embedder.py:40`).
- Produces: `voiceprints/cohort.npy` — (N, 192) float32 L2-normalized rows. Core function `build_cohort(wav_paths: list, embedder, augment: bool = False, seed: int = 0) -> np.ndarray` importable for tests.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from bench.build_cohort import build_cohort, _reverb


class _Emb:
    def extract(self, audio, sample_rate=16000):
        v = np.zeros(4, dtype=np.float32)
        v[0] = 1.0 if float(np.mean(np.abs(audio))) > 0.01 else 0.0
        v[1] = 1.0
        return v / np.linalg.norm(v)


def test_build_cohort_rows_are_unit_norm(tmp_path):
    import soundfile as sf
    for i in range(3):
        sf.write(tmp_path / f"u{i}.wav", np.random.default_rng(i).normal(0, 0.1, 16000).astype(np.float32), 16000)
    cohort = build_cohort(sorted(tmp_path.glob("*.wav")), _Emb())
    assert cohort.shape[0] == 3
    assert np.allclose(np.linalg.norm(cohort, axis=1), 1.0, atol=1e-5)

def test_augment_doubles_rows(tmp_path):
    import soundfile as sf
    sf.write(tmp_path / "u.wav", np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32), 16000)
    cohort = build_cohort(sorted(tmp_path.glob("*.wav")), _Emb(), augment=True)
    assert cohort.shape[0] == 2   # clean + reverberated

def test_reverb_changes_signal_but_keeps_length():
    x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    y = _reverb(x, seed=1)
    assert y.shape == x.shape and not np.allclose(x, y)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/bench/test_build_cohort.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
"""Build an AS-Norm imposter cohort from a directory of wav files.

Feed it NON-owner speech: podcast clips, LibriSpeech test-clean samples,
recordings of other people — ideally captured THROUGH the array (same
channel as live scoring). --augment adds a synthetic-reverb copy of each
clip to approximate the far-field channel for near-field source material.

Usage:
    python3 bench/build_cohort.py --wav-dir /path/to/cohort_wavs \
        --out voiceprints/cohort.npy [--augment]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _reverb(audio: np.ndarray, seed: int = 0, rt60_ms: int = 300, sr: int = 16000) -> np.ndarray:
    # Cheap synthetic room tail: exponentially-decaying noise impulse
    # response (no pyroomacoustics dependency).
    rng = np.random.default_rng(seed)
    n = int(sr * rt60_ms / 1000)
    ir = rng.normal(0, 1, n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    wet = np.convolve(audio, ir)[: len(audio)].astype(np.float32)
    peak = float(np.max(np.abs(wet))) or 1.0
    return wet / peak * float(np.max(np.abs(audio)))


def build_cohort(wav_paths, embedder, augment: bool = False, seed: int = 0) -> np.ndarray:
    rows = []
    for i, path in enumerate(wav_paths):
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        variants = [audio] + ([_reverb(audio, seed=seed + i)] if augment else [])
        for v in variants:
            emb = embedder.extract(v)
            rows.append(emb / (np.linalg.norm(emb) or 1.0))
    return np.stack(rows).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", default="voiceprints/cohort.npy")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    from core.speaker.embedder import EmbeddingExtractor
    paths = sorted(Path(args.wav_dir).glob("*.wav"))
    if len(paths) < 20:
        print(f"WARNING: only {len(paths)} wavs — AS-Norm wants >=100 cohort rows for stable stats")
    cohort = build_cohort(paths, EmbeddingExtractor(), augment=args.augment)
    np.save(args.out, cohort)
    print(f"wrote {cohort.shape} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, then build the real cohort**

Run: `python3 -m pytest tests/bench/test_build_cohort.py -q` — Expected: PASS.
Then (manual, once): assemble ≥100 non-owner clips — quickest source is LibriSpeech test-clean (`wget https://www.openslr.org/resources/12/test-clean.tar.gz`, take ~150 flac→wav 3-10s utterances from distinct speakers) plus `other.wav` — and run with `--augment`. Commit does NOT include the cohort or wavs (`voiceprints/` content stays untracked; add `voiceprints/cohort.npy` to `.gitignore` if `git status` shows it).

- [ ] **Step 5: Commit**

```bash
git add bench/build_cohort.py tests/bench/test_build_cohort.py .gitignore
git commit -m "feat(bench): build_cohort — AS-Norm imposter cohort from wav dir (+synthetic reverb)"
```

### Task 7: Wire AsNorm into SafetyNet (shadow) and barge-in (mode-gated)

**Files:**
- Modify: `modes/director/safety_net.py` (`SafetyVerdict`, `SafetyNet.__init__`, `maybe_verify`), `modes/director/workers/safety_net.py:46` (`_drain` emit), `modes/director/events.py` (`SpeakerWindowVerdict`), `modes/director/assembly.py:124-149` (`_build_safety_net`), `assembly.py:163-165` + `assembly.py:379` (barge-in score_fn + threshold), `modes/director/reducer.py:212-230` (`safety_diag_line`), `config.yaml` turn_gate block
- Test: `tests/director/test_safety_net.py`, `tests/director/test_assembly.py`, `tests/director/test_safety_diag.py`

**Interfaces:**
- Consumes: `AsNorm` (Task 5); cohort file from Task 6.
- Produces: `SafetyVerdict(score, smoother_ok, window_rms, norm_score: float | None = None)`; `SafetyNet.__init__(..., normalizer=None, norm_decides: bool = False)`; config `turn_gate.score_norm: {mode: off|shadow|on, cohort_path, top_k, speaker_threshold_norm}` and `barge_in.speaker_threshold_norm`. Mode semantics: **off** = today's behavior; **shadow** = normalized score computed and logged, raw score still decides; **on** = normalized score decides (SafetyNet smoother threshold becomes `speaker_threshold_norm`, and the barge-in `score_fn` injected into `IngestionWorker` becomes `normalizer.score` with `DirectorConfig.speaker_threshold` mapped from `barge_in.speaker_threshold_norm`). The reducer is untouched in all three modes.

- [ ] **Step 1: Write the failing tests**

In `tests/director/test_safety_net.py` add (reuse that file's existing fake-embedder conventions):

```python
class _FixedNorm:
    def score(self, enroll, test):
        return 7.0

def test_shadow_mode_logs_norm_but_raw_decides():
    emb = _Emb()          # existing fake: extract -> unit ones
    primary = emb.extract(np.zeros(4))
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.5,
                    normalizer=_FixedNorm(), norm_decides=False, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.norm_score == 7.0
    assert v.score == pytest.approx(1.0)      # raw cosine, and it decided
    assert v.smoother_ok is True

def test_on_mode_normalized_score_feeds_smoother():
    emb = _Emb()
    primary = emb.extract(np.zeros(4))
    class _LowNorm:
        def score(self, enroll, test): return -3.0
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.0,
                    normalizer=_LowNorm(), norm_decides=True, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.smoother_ok is False             # -3.0 < threshold 0.0 despite raw cosine 1.0

def test_no_normalizer_is_todays_behavior():
    emb = _Emb()
    net = SafetyNet(emb, emb.extract(np.zeros(4)), verify_window_ms=100, threshold=0.5)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.norm_score is None and v.smoother_ok is True
```

In `tests/director/test_assembly.py` add mapping tests following the file's `_build_safety_net` test style: mode `"shadow"` with an existing cohort file (write a tiny `(4, 192)` unit-norm `.npy` to `tmp_path`) yields a `SafetyNet` whose `_normalizer` is set and `_norm_decides` is False; mode `"on"` yields `_norm_decides` True and smoother threshold == `speaker_threshold_norm`; mode `"off"` or a MISSING cohort file yields `_normalizer is None` (fail-open, warn to stderr — assert via capsys).

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_safety_net.py tests/director/test_assembly.py -q` — Expected: new tests FAIL (unexpected kwargs).

- [ ] **Step 3: Implement**

`safety_net.py`:

```python
@dataclass(frozen=True)
class SafetyVerdict:
    score: float
    smoother_ok: bool
    window_rms: float
    norm_score: Optional[float] = None
```

`SafetyNet.__init__` gains `normalizer=None, norm_decides: bool = False`; in `maybe_verify()` (after `score = _cosine(emb, self._primary)`):

```python
        norm_score = None
        if self._normalizer is not None:
            norm_score = float(self._normalizer.score(self._primary, emb))
        deciding = norm_score if (self._norm_decides and norm_score is not None) else score
        smoother_ok = self._smoother.update(deciding)
```

and return `SafetyVerdict(score, smoother_ok, window_rms, norm_score)`. Keep `.score` = raw always (DIAG continuity; the smoother threshold's meaning is chosen by assembly).

`workers/safety_net.py:_drain` — pass `norm_score` through into `E.SpeakerWindowVerdict`; add the field (default `None`) to the event dataclass in `events.py`. `reducer.py:safety_diag_line` — append `norm=<x.xx>` when the field is not None (reducer only FORMATS it; decisions still flow through `smoother_ok`/`score` exactly as today).

`assembly.py:_build_safety_net` — read the new block:

```python
    sn_cfg = tg.get("score_norm", {})
    mode = sn_cfg.get("mode", "off")
    normalizer = None
    if mode in ("shadow", "on"):
        path = sn_cfg.get("cohort_path", "./voiceprints/cohort.npy")
        try:
            cohort = np.load(path)
            normalizer = AsNorm(cohort, top_k=sn_cfg.get("top_k", 50))
        except (OSError, ValueError) as exc:
            print(f"[assembly] score_norm: cohort load failed ({exc}) — falling back to raw scores", file=sys.stderr)
            mode = "off"
    threshold = sn_cfg.get("speaker_threshold_norm", 0.0) if mode == "on" else tg.get("speaker_threshold", 0.30)
```

and pass `threshold=threshold, normalizer=normalizer, norm_decides=(mode == "on")` into `SafetyNet`. For barge-in (assembly.py:379 + `_director_config_from`): when `mode == "on"`, inject `score_fn=normalizer.score` into `IngestionWorker` and map `DirectorConfig.speaker_threshold` from `tb_cfg["barge_in"].get("speaker_threshold_norm", 0.0)`; otherwise keep today's `cosine_similarity` + `barge_in.speaker_threshold`.

`config.yaml` — in `turn_gate:` add:

```yaml
      # AS-Norm score calibration (spec Appendix C): normalize every ECAPA
      # score against an imposter cohort recorded through OUR channel
      # (bench/build_cohort.py). mode: off = raw scores (today) | shadow =
      # normalized score computed + logged, raw still decides | on =
      # normalized score decides, using the *_norm thresholds. Ship shadow
      # first; flip to on only after a bench/speaker_scores.py sweep of
      # live shadow logs picks the norm thresholds.
      score_norm:
        mode: "shadow"
        cohort_path: "./voiceprints/cohort.npy"
        top_k: 50
        speaker_threshold_norm: 0.0    # z-like scale; tune from shadow logs
```

and in `barge_in:` add `speaker_threshold_norm: 0.0  # used only when turn_gate.score_norm.mode is "on"`.

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest tests -q` — Expected: green. Update `test_shipped_config_yaml_matches_live_readers` for the new keys, and any `SafetyVerdict`/`SpeakerWindowVerdict` constructor calls in existing tests (the new fields default, so most won't need edits).

- [ ] **Step 5: Live shadow check**

Run the kiosk with the real cohort present; one session. Expected: DIAG lines now carry `norm=` values; behavior otherwise identical (shadow). Note the owner's norm range vs. any bystander noise for the later threshold sweep.

- [ ] **Step 6: Commit**

```bash
git add modes/director/ config.yaml tests/
git commit -m "feat(speaker): AS-Norm calibration — shadow mode wired through SafetyNet + barge-in"
```

### Task 8: Running enrollment centroid

**Files:**
- Modify: `modes/director/safety_net.py` (`SafetyNet.__init__`, `maybe_verify`), `modes/director/assembly.py:_build_safety_net`, `config.yaml` turn_gate block
- Test: `tests/director/test_safety_net.py`

**Interfaces:**
- Produces: `SafetyNet.__init__(..., update_alpha: float = 0.0, update_margin: float = 0.10)`. After a window whose DECIDING score clears `threshold + update_margin`, the primary embedding becomes `l2norm((1 - alpha) * primary + alpha * window_emb)`. `alpha == 0.0` disables (code default — no regression). Config `turn_gate.enrollment_update_alpha` / `enrollment_update_margin`.

- [ ] **Step 1: Write the failing tests**

```python
def test_centroid_moves_toward_confident_matches():
    class _DriftEmb:
        def extract(self, audio, sample_rate=16000):
            v = np.array([1.0, 0.3, 0.0, 0.0], dtype=np.float32)
            return v / np.linalg.norm(v)
    emb = _DriftEmb()
    primary = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.5,
                    update_alpha=0.5, update_margin=0.1, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    net.maybe_verify()
    assert net._primary[1] > 0.0                     # moved toward the window
    assert np.linalg.norm(net._primary) == pytest.approx(1.0, abs=1e-5)

def test_no_update_below_margin():
    class _WeakEmb:
        def extract(self, audio, sample_rate=16000):
            v = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)  # cosine 0.6 vs primary
            return v / np.linalg.norm(v)
    primary = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    net = SafetyNet(_WeakEmb(), primary, verify_window_ms=100, threshold=0.55,
                    update_alpha=0.5, update_margin=0.1, sr=16000)   # 0.6 < 0.55+0.1
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    net.maybe_verify()
    assert np.allclose(net._primary, primary)

def test_alpha_zero_never_updates():
    # same _DriftEmb as above, update_alpha left at default 0.0
    ...
    assert np.allclose(net._primary, primary)
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/director/test_safety_net.py -q` → FAIL.

- [ ] **Step 3: Implement**

In `maybe_verify()`, after computing `deciding` (Task 7) and before building the verdict:

```python
        if self._update_alpha > 0.0 and deciding >= self._decide_threshold + self._update_margin:
            mixed = (1.0 - self._update_alpha) * self._primary + self._update_alpha * emb
            n = float(np.linalg.norm(mixed))
            if n > 0.0:
                self._primary = (mixed / n).astype(np.float32)
```

(`self._decide_threshold` is the threshold the smoother uses — store it in `__init__`.) Note in a comment: the margin is the poison guard — only windows comfortably above the eject threshold may teach the centroid (spec Appendix C: +4.8 F1 far-field in the Huawei dynamic-enrollment result). The barge-in path keeps its session-start `primary_embedding` copy; only SafetyNet's copy drifts — acceptable for now, note it in the commit message.

`assembly.py`: pass `update_alpha=tg.get("enrollment_update_alpha", 0.0), update_margin=tg.get("enrollment_update_margin", 0.10)`.

`config.yaml` turn_gate:

```yaml
      # Running enrollment centroid: windows scoring >= threshold+margin
      # fold into the primary voiceprint (EMA), absorbing far-field drift.
      # 0.0 disables. Margin is the poison guard.
      enrollment_update_alpha: 0.10
      enrollment_update_margin: 0.10
```

- [ ] **Step 4: Run the full suite** — `python3 -m pytest tests -q` → green (+ shipped-config assertions updated).

- [ ] **Step 5: Commit**

```bash
git add modes/director/safety_net.py modes/director/assembly.py config.yaml tests/
git commit -m "feat(speaker): running enrollment centroid in SafetyNet (margin-guarded EMA)"
```

---

## Workstream D — Parakeet STT via NeMo

### Task 9: NeMo install + Parakeet spike (GO/NO-GO gate)

**Files:**
- Modify: `bench/stt_backend_probe.py:118-152` (`probe_nemo`)

**Interfaces:**
- Produces: a GO/NO-GO verdict for Task 10, printed latency + confidence-availability facts.

- [ ] **Step 1: Install NeMo (host pip, NOT the NGC container first)**

```bash
python3 -m pip install -U "nemo_toolkit[asr]"
python3 -c "import nemo.collections.asr as a; print(a.__name__, 'ok')"
```

If the import fails on aarch64 dependency builds, retry inside the NGC container per the spec (Appendix B: `nvcr.io/nvidia/pytorch:25.11-py3` is the community-proven path) — but container-only viability means Task 10 becomes a service, which is OUT of quick-win scope: in that case record NO-GO and stop this workstream.

- [ ] **Step 2: Extend probe_nemo to actually run Parakeet**

Replace the "not installed" stub with:

```python
def probe_nemo(model_name: str, clip: np.ndarray) -> None:
    import time
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model = model.to("cuda").eval()
    # Warm-up, then timed runs
    model.transcribe(audio=[clip], verbose=False)
    t0 = time.perf_counter()
    for _ in range(5):
        hyps = model.transcribe(audio=[clip], return_hypotheses=True, verbose=False)
    dt = (time.perf_counter() - t0) / 5
    hyp = hyps[0]
    print(f"model={model_name} mean_latency={dt*1000:.0f}ms text={hyp.text!r}")
    print(f"word_confidence available: {getattr(hyp, 'word_confidence', None) is not None}")
    print(f"hypothesis attrs: {[a for a in dir(hyp) if 'conf' in a.lower()]}")
```

- [ ] **Step 3: Run the spike**

```bash
python3 bench/stt_backend_probe.py --nemo-model nvidia/parakeet-tdt-0.6b-v2
```

(adapt to the script's existing CLI; it already loads `self.wav`). GO criteria: transcribes `self.wav` correctly, mean latency ≤ 400ms on a ~3s clip, runs on CUDA (check `nvidia-smi` during). Record whether `word_confidence` is populated; if not, note which confidence attrs exist (the spec flags TDT confidence bugs — `nvidia/parakeet-rnnt-1.1b` is the fallback decoder to also probe if TDT confidence is empty).

- [ ] **Step 4: Commit the probe + a GO/NO-GO note**

```bash
git add bench/stt_backend_probe.py
git commit -m "bench: probe_nemo runs Parakeet live — latency + confidence availability"
```

Write the verdict (GO/NO-GO, measured numbers, which confidence field works) into the commit body — Task 10 reads it.

### Task 10: NemoStt backend + selector

**Files:**
- Create: `modes/talkback/stt_nemo.py`
- Modify: `kiosk.py:224-228` (backend selector), `config.yaml:94-103` (stt block)
- Test: `tests/kiosk/talkback/test_stt_nemo.py`, `tests/director/test_kiosk_entrypoint.py`

**Interfaces:**
- Consumes: spike facts from Task 9 (exact confidence attribute).
- Produces: `class NemoStt` satisfying the 4-point StreamingStt contract: `__init__(self, model: str = "nvidia/parakeet-tdt-0.6b-v2", device: str = "cuda")`; `_ensure_model()` (idempotent — `kiosk.py:245` calls it eagerly); `async transcribe_segment(self, audio: np.ndarray) -> TranscriptResult` (float32 mono 16kHz in; `mean_word_prob ∈ [0,1]`, `0.0` on empty). Config: `stt.backend` becomes LIVE (`"openai-whisper" | "nemo"`), plus `stt.nemo_model`.

- [ ] **Step 1: Write the failing tests**

`tests/kiosk/talkback/test_stt_nemo.py` — mirror `test_stt.py`'s `__new__` + fake-model bypass convention (`test_stt.py:43-47`):

```python
import numpy as np
import pytest

from modes.talkback.stt_nemo import NemoStt
from modes.director.transcript import TranscriptResult


class _Hyp:
    def __init__(self, text, word_confidence):
        self.text = text
        self.word_confidence = word_confidence

class _FakeNemoModel:
    def __init__(self, hyp):
        self._hyp = hyp
    def transcribe(self, audio, return_hypotheses=True, verbose=False):
        return [self._hyp]


def _make(hyp):
    stt = NemoStt.__new__(NemoStt)
    stt._model = _FakeNemoModel(hyp)
    stt._device = "cpu"
    return stt


@pytest.mark.asyncio
async def test_transcribe_returns_text_and_mean_confidence():
    stt = _make(_Hyp(" hello there ", [0.9, 0.7]))
    result = await stt.transcribe_segment(np.zeros(16000, dtype=np.float32))
    assert isinstance(result, TranscriptResult)
    assert result.text == "hello there"
    assert result.mean_word_prob == pytest.approx(0.8)

@pytest.mark.asyncio
async def test_missing_confidence_falls_back_to_one_for_nonempty_text():
    stt = _make(_Hyp("hi", None))
    result = await stt.transcribe_segment(np.zeros(16000, dtype=np.float32))
    assert result.mean_word_prob == 1.0

@pytest.mark.asyncio
async def test_empty_text_scores_zero():
    stt = _make(_Hyp("", None))
    result = await stt.transcribe_segment(np.zeros(16000, dtype=np.float32))
    assert result.text == "" and result.mean_word_prob == 0.0
```

Selector test in `tests/director/test_kiosk_entrypoint.py` (follow that file's existing patterns for exercising `kiosk.py` construction): config with `stt.backend: "nemo"` constructs a `NemoStt`; `"openai-whisper"` (and absent) constructs `StreamingStt`; an unknown backend raises `SystemExit`.

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/kiosk/talkback/test_stt_nemo.py -q` → FAIL.

- [ ] **Step 3: Implement `stt_nemo.py`**

```python
"""Parakeet-TDT STT backend via NeMo (spec Appendix B: ~6.05% vs ~8.6% WER).

Satisfies the StreamingStt contract (modes/talkback/stt.py): eager
_ensure_model() warm-up, whole-segment async transcribe_segment ->
TranscriptResult. NeMo import stays inside methods so the module imports
on machines without nemo (conftest does not stub it).
"""
import asyncio

import numpy as np

from modes.director.transcript import TranscriptResult


class NemoStt:
    def __init__(self, model: str = "nvidia/parakeet-tdt-0.6b-v2", device: str = "cuda"):
        self._model_name = model
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
        self._model = model.to(self._device).eval()
        # Confidence: enable per the Task-9 spike findings. If the spike
        # showed word_confidence needs a decoding-config change, apply it
        # here via self._model.change_decoding_strategy(...) exactly as
        # the spike's working invocation did.

    async def transcribe_segment(self, audio: np.ndarray) -> TranscriptResult:
        self._ensure_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> TranscriptResult:
        hyps = self._model.transcribe(
            audio=[audio], return_hypotheses=True, verbose=False)
        hyp = hyps[0]
        text = (hyp.text or "").strip()
        conf = getattr(hyp, "word_confidence", None)
        if not text:
            prob = 0.0
        elif conf:
            prob = float(np.clip(np.mean(conf), 0.0, 1.0))
        else:
            prob = 1.0   # no confidence stream -> don't false-trip the conf_floor gate
        return TranscriptResult(text=text, mean_word_prob=prob)
```

`kiosk.py:224-228` selector:

```python
    stt_cfg = tb_cfg.get("stt", {})
    backend = stt_cfg.get("backend", "openai-whisper")
    if backend == "nemo":
        from modes.talkback.stt_nemo import NemoStt
        stt = NemoStt(
            model=stt_cfg.get("nemo_model", "nvidia/parakeet-tdt-0.6b-v2"),
            device=stt_cfg.get("device", "cuda"),
        )
    elif backend == "openai-whisper":
        stt = StreamingStt(
            model=stt_cfg.get("model", "base.en"),
            device=stt_cfg.get("device", "cuda"),
        )
    else:
        console.print(f"[red]Unknown stt.backend: {backend!r}[/red]")
        sys.exit(3)
```

`config.yaml` stt block: flip `backend: "nemo"` ONLY if Task 9 was GO (else leave `"openai-whisper"`); either way update the comment to say the key is now live, and add:

```yaml
      nemo_model: "nvidia/parakeet-tdt-0.6b-v2"   # used when backend: "nemo"; spike-verified 2026-09
```

Do NOT register backend as a knob (`tune/knobs.py:5-7` deliberately excludes backend selectors). The existing `stt.model` knob stays whisper-only — its label/doc should say "(openai-whisper backend)".

- [ ] **Step 4: Run tests, then the full suite** — targeted tests PASS, `python3 -m pytest tests -q` green (extend shipped-config assertions for the stt block if pinned).

- [ ] **Step 5: Live check (if GO)** — full stack, one exchange; confirm transcripts in the JSONL log carry sane `mean_word_prob` values and STT latency feels ≤ whisper small.en.

- [ ] **Step 6: Commit**

```bash
git add modes/talkback/stt_nemo.py kiosk.py config.yaml tests/
git commit -m "feat(stt): NemoStt Parakeet backend behind stt.backend selector"
```

---

## After all tasks

1. Full suite green, then one end-to-end live session exercising: wake → exchange (Gemma 4 replies, no markdown spoken, second-turn TTFT visibly snappier) → story → barge-in → clean stop.
2. The comprehensive live test (user-run) then covers the deferred D11 gate scenarios: wake in quiet, podcast after first exchange, out-of-cone rejects with no ducking — plus the new shadow `norm=` DIAG values for the later threshold sweep (`bench/speaker_scores.py`).
3. Follow-ups explicitly NOT in this plan: flipping `score_norm.mode` to `"on"` (needs the shadow-data sweep), embedder swap (needs the same data), TTS swap (held), Type 3.1 extraction front end (deferred — see memory).

## Self-review notes

- Spec coverage: Type 1.1 (Task 4), 1.2 (Task 3), 1.3 (Tasks 2-3), 1.4 (Tasks 9-10), Type 2.1 (Tasks 5-8), 2.7-partial (Task 4 prompt clause). Type 1.5/1.6/1.7 intentionally out (stated in header).
- Known execution-time verifications (not placeholders — external facts): exact Gemma 4 GGUF repo id (Task 4 Step 1 resolves it), NeMo confidence attribute (Task 9 resolves it, Task 10 consumes), llama-server flag names against the built version (`--cache-reuse`, `--jinja` — check `llama-server --help` if the build is newer than the research).
- Type consistency: `TranscriptResult` frozen dataclass reused (never redefined); `SafetyVerdict.norm_score` defaults `None` so existing constructor calls stand; `AsNorm.score(enroll, test)` argument order matches the `score_fn(embedding, primary)` injection at `assembly.py:379` — note the ingestion worker calls `score_fn(embedding, self._primary)`, i.e. (test, enroll); AS-Norm is symmetric in its two stats terms, so the order difference is harmless — a comment in assembly should say so.
