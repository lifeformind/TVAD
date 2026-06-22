"""Per-session pVAD driver over the FireRedChat ONNX streaming model.

update_speaker(emb) binds the enrolled ECAPA embedding (192-dim, L2-normed) and
resets the streaming state. push(chunk, ts) slices the raw audio into 10ms
(160-sample) frames, runs one ONNX step per frame threading the mel_buffer +
gru_buffer state across calls, aggregates the per-frame target-speaker
probabilities to ~agg_ms, and emits SpeakerFrame events. The leftover sub-frame
tail is buffered so successive pushes are continuous (no dropped audio).
"""

import numpy as np

from .types import SpeakerFrame

_FRAME_SAMPLES = 160          # 10ms @ 16kHz — the ONNX input_audio length
_MEL_SHAPE = (1, 80, 15)
_GRU_SHAPE = (2, 1, 256)
_OUTPUTS = ["sigmoid_out", "mel_buffer_out", "gru_buffer_out"]


class VADStream:
    def __init__(self, session, *, sr=16000, agg_ms=50, hop_ms=10, threshold=0.5):
        self._sess = session
        self._sr = sr
        self._threshold = threshold
        self._frames_per_agg = max(1, round(agg_ms / hop_ms))
        self._spk = None
        self._mel = None
        self._gru = None
        self._buf = np.zeros(0, dtype=np.float32)

    def update_speaker(self, embedding: np.ndarray) -> None:
        """Bind the enrolled ECAPA embedding (192-dim) and reset streaming state."""
        spk = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(spk))
        if norm > 0:
            spk = spk / norm
        self._spk = spk
        self._reset_state()

    def reset(self) -> None:
        self._spk = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._mel = np.zeros(_MEL_SHAPE, dtype=np.float32)
        self._gru = np.zeros(_GRU_SHAPE, dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray, ts: float) -> list:
        if self._spk is None:
            raise RuntimeError("update_speaker() must be called before push()")
        chunk = np.asarray(chunk, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        self._buf = np.concatenate([self._buf, chunk])

        probs = []
        while self._buf.size >= _FRAME_SAMPLES:
            frame = self._buf[:_FRAME_SAMPLES].reshape(1, -1)
            self._buf = self._buf[_FRAME_SAMPLES:]
            sig, self._mel, self._gru = self._sess.run(_OUTPUTS, {
                "input_audio": frame,
                "spkemb": self._spk,
                "mel_buffer": self._mel,
                "gru_buffer": self._gru,
            })
            probs.append(float(np.ravel(sig)[-1]))

        out = []
        n = self._frames_per_agg
        for i in range(0, len(probs), n):
            group = probs[i:i + n]
            if not group:
                continue
            conf = sum(group) / len(group)
            out.append(SpeakerFrame(
                ts=ts, is_target=conf >= self._threshold,
                confidence=conf, rms=rms,
            ))
        return out
