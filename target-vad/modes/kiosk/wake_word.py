"""WakeWordDetector — thin wrapper over openwakeword with a clean Optional[float] API."""

from typing import Optional

import numpy as np
from openwakeword.model import Model


class WakeWordDetector:
    """Buffers audio chunks to openwakeword's 1280-sample frame size and
    returns the wake-phrase confidence when it crosses the threshold.

    Handles model-name suffix variation (e.g. 'hey_jarvis_v0.1') by matching
    on substring. Other models in the predictions dict are ignored.

    On Python 3.14 / Windows, tflite-runtime is unavailable; we explicitly
    request the ONNX inference backend.
    """

    OWW_FRAME_SAMPLES = 1280  # 80ms at 16 kHz, what openwakeword expects

    def __init__(self, model_name: str, threshold: float):
        self.model_name = model_name
        self.threshold = threshold
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self._buffer = np.array([], dtype=np.float32)

    def process(self, chunk: np.ndarray) -> Optional[float]:
        """Append chunk to internal buffer, run predict on full 1280-sample frames.

        Returns the highest matching confidence above threshold, or None.
        """
        self._buffer = np.concatenate([self._buffer, chunk])
        best_score: Optional[float] = None
        while len(self._buffer) >= self.OWW_FRAME_SAMPLES:
            frame_f32 = self._buffer[: self.OWW_FRAME_SAMPLES]
            self._buffer = self._buffer[self.OWW_FRAME_SAMPLES :]
            # openwakeword expects int16 PCM
            frame_i16 = (frame_f32 * 32767.0).clip(-32768, 32767).astype(np.int16)
            preds = self._model.predict(frame_i16)
            for key, score in preds.items():
                if self.model_name in key.lower():
                    if score >= self.threshold:
                        if best_score is None or score > best_score:
                            best_score = float(score)
        return best_score

    def reset(self) -> None:
        """Clear the internal audio buffer and reset openwakeword model state."""
        self._buffer = np.array([], dtype=np.float32)
        if hasattr(self._model, "reset"):
            self._model.reset()
