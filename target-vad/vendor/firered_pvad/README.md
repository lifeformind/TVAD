# Vendored: FireRedChat personal-VAD (pVAD)

**Provenance for the crowd-focus pVAD used by the Director (Plan 05).**

- **Upstream:** [`FireRedTeam/FireRedChat-pvad`](https://huggingface.co/FireRedTeam/FireRedChat-pvad) (HuggingFace)
- **Paper:** FireRedChat — arXiv:2509.06502
- **License:** Apache-2.0 (see upstream `NOTICE`)
- **Checkpoint:** `pvad.onnx`
  - **sha256:** `2114fd3c3fa87b560eaf4cad6a6e1a0a73aefba08da05521a27bfe2382ef4bdd`
  - **size:** 3,940,567 bytes (~3.9 MB)
- **Speaker-embedding conditioning:** `speechbrain/spkrec-ecapa-voxceleb` (192-dim,
  L2-normalized) — the SAME ECAPA our `core/speaker/embedder.py` already loads, so
  our enrollment embeddings drop straight in as the `spkemb` input.

## Why ONNX, not a vendored `nn.Module`

The original Plan-05 Task-1 assumed a PyTorch `.pt` checkpoint to be loaded into a
hand-ported `nn.Module`. The actual artifact upstream is a **self-contained ONNX
streaming model** — so we run it directly via `onnxruntime` (already installed for
Smart Turn). This removes the plan's biggest risks: no architecture reconciliation,
and no hand-built mel front-end (the log-mel is computed *inside* the graph).

The binary is NOT committed here — it is fetched and cached via
`huggingface_hub.hf_hub_download(repo_id="FireRedTeam/FireRedChat-pvad",
filename="pvad.onnx")` on first use, exactly like whisper / Kokoro / Silero in this
project. `modes/director/pvad/loader.py` is the loader.

## ONNX I/O (streaming, CPU)

One call consumes ONE 10 ms (160-sample @ 16 kHz) raw-audio frame and threads the
carried state buffers:

| Inputs | shape | meaning |
|---|---|---|
| `input_audio` | `(batch, 160)` | one 10 ms raw-audio frame (mel is in-graph) |
| `spkemb` | `(batch, 192)` | enrolled ECAPA embedding (conditioning) |
| `mel_buffer` | `(batch, 80, 15)` | carried mel ring state |
| `gru_buffer` | `(2, batch, 256)` | carried GRU hidden state |

| Outputs | shape | meaning |
|---|---|---|
| `linear_out` | `(batch, 1)` | pre-sigmoid logit |
| `sigmoid_out` | `(batch, 1)` | target-speaker probability for this frame |
| `mel_buffer_out` | `(batch, 80, 15)` | updated mel ring (feed back next call) |
| `gru_buffer_out` | `(2, batch, 256)` | updated GRU hidden (feed back next call) |

Measured on GB10 (CPU, 1 intra-op thread): **~0.26 ms p95 per 10 ms frame** →
~5 ms for a 200 ms chunk. See `docs/notes/2026-06-22-pvad.md` for the cutover verdict.
