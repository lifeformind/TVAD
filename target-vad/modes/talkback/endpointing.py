"""Pluggable turn-end detector interface for the talkback pipeline.

Two implementations are provided:

  NullTurnDetector   — always reports complete (probability 1.0). No external
                       deps; safe for unit tests and CI.

  SmartTurnDetector  — thin wrapper around the Smart Turn v3.2 ONNX model
                       bundled inside pipecat-ai (≈8.3 MB int8 weights).
                       Uses onnxruntime CPUExecutionProvider.

Both conform to the TurnDetector typing.Protocol:

    endpoint_prob(audio, sample_rate) -> float in [0, 1]

where a value close to 1.0 means "speaker has almost certainly finished" and
a value close to 0.0 means "speaker is still talking / mid-utterance".

Preprocessing (SmartTurnDetector)
----------------------------------
Mirrors pipecat.audio.turn.smart_turn.local_smart_turn_v3:

  1. Resample input to 16 kHz (via soxr) if needed.
  2. Truncate to the last 8 s or zero-pad the beginning to reach 8 s
     (128 000 samples).
  3. Compute Whisper-style log-mel features: 80-band mel, Slaney filterbank,
     STFT with N_FFT=400, hop=160 — via pipecat's vendored
     compute_whisper_log_mel_features (do_normalize=True).
  4. Add batch dim → shape (1, 80, 800) as tensor "input_features".
  5. Model output[0][0] is a sigmoid probability (Complete = high).

Source references:
  pipecat/audio/turn/smart_turn/local_smart_turn_v3.py  — inference loop
  pipecat/audio/turn/smart_turn/_whisper_features.py    — feature extraction
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

_MODEL_SAMPLE_RATE = 16_000
_MODEL_AUDIO_SECONDS = 8
_MODEL_SAMPLES = _MODEL_SAMPLE_RATE * _MODEL_AUDIO_SECONDS  # 128 000


@runtime_checkable
class TurnDetector(Protocol):
    """Structural protocol for turn-end detectors.

    Any object that implements endpoint_prob with the correct signature is a
    valid TurnDetector — no inheritance required.
    """

    def endpoint_prob(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return the probability (in [0, 1]) that the speaker has finished.

        Args:
            audio: 1-D float32 waveform, already at *sample_rate* Hz.
            sample_rate: Sample rate of *audio* in Hz.

        Returns:
            Float in [0, 1].  High → turn complete; Low → still talking.
        """
        ...


class NullTurnDetector:
    """Fallback detector that always reports turn complete (probability 1.0).

    Safe for unit tests, CI environments, and any deployment where the Smart
    Turn model is unavailable.  Satisfies the TurnDetector protocol.
    """

    def endpoint_prob(self, audio: np.ndarray, sample_rate: int) -> float:  # noqa: ARG002
        """Always return 1.0 — treat every endpointed segment as complete."""
        return 1.0


class SmartTurnDetector:
    """End-of-turn detector backed by the Smart Turn v3.2 ONNX model.

    Loads the ONNX session at construction time.  Subsequent calls to
    endpoint_prob are synchronous (no async required from the caller).

    Args:
        model_path: Absolute path to the ONNX file.  When None (default) the
                    model bundled inside the installed pipecat-ai wheel is used.
        cpu_count:  Number of intra-op threads for onnxruntime (default 1).
    """

    def __init__(self, model_path: str | None = None, *, cpu_count: int = 1) -> None:
        from pipecat.audio.turn.smart_turn._whisper_features import (
            compute_whisper_log_mel_features,
        )

        self._compute_features = compute_whisper_log_mel_features

        import onnxruntime as ort

        # Resolve model path — default to the pipecat-bundled asset.
        if model_path is None:
            model_path = _bundled_model_path()

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cpu_count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def endpoint_prob(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return P(speaker finished) from Smart Turn v3.2.

        Preprocessing pipeline (matches pipecat's LocalSmartTurnAnalyzerV3):
          1. Resample to 16 kHz if sample_rate != 16000 (soxr HQ).
          2. Pad/truncate to exactly 8 s (128 000 samples).
          3. Whisper log-mel features → (80, 800) float32 with do_normalize=True.
          4. Expand to (1, 80, 800) and run ONNX inference.
          5. Return outputs[0][0] as Python float.

        Args:
            audio: 1-D float32 array at *sample_rate* Hz.
            sample_rate: Sample rate of *audio* in Hz.

        Returns:
            Float in [0, 1].
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"endpoint_prob expects 1-D audio; got shape {audio.shape}")

        # Step 1 — resample to 16 kHz if needed.
        if sample_rate != _MODEL_SAMPLE_RATE:
            import soxr

            audio = soxr.resample(audio, sample_rate, _MODEL_SAMPLE_RATE, quality="HQ")

        # Step 2 — truncate last 8 s or zero-pad the beginning to 8 s.
        if len(audio) > _MODEL_SAMPLES:
            audio = audio[-_MODEL_SAMPLES:]
        elif len(audio) < _MODEL_SAMPLES:
            pad = _MODEL_SAMPLES - len(audio)
            audio = np.pad(audio, (pad, 0), mode="constant", constant_values=0)

        # Step 3 — Whisper log-mel features, shape (80, 800).
        log_mel = self._compute_features(audio, do_normalize=True)

        # Step 4 — add batch dimension → (1, 80, 800).
        input_features = np.expand_dims(log_mel, axis=0)

        # Step 5 — ONNX inference.  Output shape is (batch, 1); use .item() to
        # extract a Python float from the 1-element numpy array at outputs[0][0].
        outputs = self._session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bundled_model_path() -> str:
    """Return the absolute path to the pipecat-bundled ONNX model.

    Mirrors the fallback chain in LocalSmartTurnAnalyzerV3.__init__ so we
    don't hard-code the site-packages layout.
    """
    model_name = "smart-turn-v3.2-cpu.onnx"
    package_path = "pipecat.audio.turn.smart_turn.data"

    try:
        import importlib_resources as impresources  # type: ignore[import-untyped]

        return str(impresources.files(package_path).joinpath(model_name))
    except ImportError:
        pass

    from importlib import resources as impresources  # type: ignore[no-redef]

    try:
        with impresources.path(package_path, model_name) as f:  # type: ignore[attr-defined]
            return str(f)
    except Exception:
        return str(impresources.files(package_path).joinpath(model_name))
