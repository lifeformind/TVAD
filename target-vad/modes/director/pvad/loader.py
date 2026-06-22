"""Load the FireRedChat pVAD ONNX model on CPU.

The upstream artifact is a self-contained ONNX *streaming* model (see
vendor/firered_pvad/README.md), so there is no nn.Module / state_dict to
reconcile — we run it via onnxruntime. CPU-only (intra_op single-threaded) so it
stays insulated from gemma's GPU contention (spec §9 placement rule). The
checkpoint is fetched + cached via hf_hub_download on first use, matching how
whisper / Kokoro / Silero are loaded in this project.
"""

import onnxruntime as ort

PVAD_REPO = "FireRedTeam/FireRedChat-pvad"
PVAD_FILE = "pvad.onnx"


def resolve_pvad_path(model_path: str = None, local_files_only: bool = False) -> str:
    """Return a filesystem path to pvad.onnx, downloading+caching it if needed."""
    if model_path:
        return model_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=PVAD_REPO, filename=PVAD_FILE,
                           local_files_only=local_files_only)


def load_pvad(model_path: str = None) -> ort.InferenceSession:
    """Build a CPU onnxruntime session for the streaming pVAD.

    Single intra-op thread keeps the per-frame cost predictable and off the GPU.
    """
    path = resolve_pvad_path(model_path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])
