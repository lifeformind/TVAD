"""Silero VAD wrapper — speech activity detection with segment extraction."""

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    audio: np.ndarray       # float32, 16kHz, mono
    start_ms: float
    end_ms: float
    duration_ms: float


class SileroVAD:
    """Wraps Silero VAD for streaming speech detection."""

    # Silero VAD requires exactly 512 samples at 16kHz (32ms)
    SILERO_CHUNK_SAMPLES = 512

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.speech_threshold = config.get("speech_threshold", 0.5)
        self.min_speech_duration_ms = config.get("min_speech_duration_ms", 300)
        self.padding_ms = config.get("padding_ms", 200)

        self._model = None
        self._load_model(config.get("silero_model_path"))
        self.reset()

    def _load_model(self, model_path: Optional[str] = None):
        """Load Silero VAD model — from local path or torch.hub."""
        if model_path:
            # Air-gapped: load from local ONNX/JIT file
            self._model = torch.jit.load(model_path, map_location="cpu")
            self._model.eval()
        else:
            # Download via torch.hub (requires internet on first run)
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )

    def reset(self):
        """Clear internal state between sessions."""
        if self._model is not None:
            self._model.reset_states()
        self._input_buffer = np.array([], dtype=np.float32)
        self._speech_buffer = []
        self._is_speaking = False
        self._speech_start_sample = 0
        self._current_sample = 0
        self._padding_samples = int(self.padding_ms * self.sample_rate / 1000)
        self._min_speech_samples = int(
            self.min_speech_duration_ms * self.sample_rate / 1000
        )
        self._pre_speech_buffer = np.array([], dtype=np.float32)

    def is_speech(self, chunk: np.ndarray) -> float:
        """Return speech probability (0.0-1.0) for a chunk of audio.

        The chunk must be exactly SILERO_CHUNK_SAMPLES (512) samples.
        """
        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()
        return prob

    def process_chunk(self, chunk: np.ndarray) -> list:
        """Stateful: feed one audio chunk, return any speech segments completed by this chunk.

        Buffers incoming audio to 512-sample Silero frames, tracks speech/silence
        transitions, and returns a list of completed SpeechSegments (possibly empty).
        Caller is responsible for the loop. See process_stream() for a generator-style adapter.
        """
        completed: list[SpeechSegment] = []
        self._input_buffer = np.concatenate([self._input_buffer, chunk])

        while len(self._input_buffer) >= self.SILERO_CHUNK_SAMPLES:
            frame = self._input_buffer[: self.SILERO_CHUNK_SAMPLES]
            self._input_buffer = self._input_buffer[self.SILERO_CHUNK_SAMPLES:]

            prob = self.is_speech(frame)

            if prob >= self.speech_threshold:
                if not self._is_speaking:
                    # Speech onset
                    self._is_speaking = True
                    self._speech_start_sample = self._current_sample
                    if len(self._pre_speech_buffer) > 0:
                        pad = self._pre_speech_buffer[-self._padding_samples:]
                        self._speech_buffer = [pad]
                    else:
                        self._speech_buffer = []
                self._speech_buffer.append(frame)
            else:
                if self._is_speaking:
                    # Speech offset — add padding then yield
                    self._speech_buffer.append(frame)
                    self._is_speaking = False

                    speech_audio = np.concatenate(self._speech_buffer)
                    duration_samples = len(speech_audio)
                    duration_ms = duration_samples / self.sample_rate * 1000

                    if duration_ms >= self.min_speech_duration_ms:
                        start_ms = self._speech_start_sample / self.sample_rate * 1000
                        end_ms = start_ms + duration_ms
                        completed.append(SpeechSegment(
                            audio=speech_audio,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            duration_ms=duration_ms,
                        ))
                    self._speech_buffer = []

            # Keep a rolling pre-speech buffer for padding
            self._pre_speech_buffer = np.concatenate(
                [self._pre_speech_buffer, frame]
            )
            max_pre = self._padding_samples * 2
            if len(self._pre_speech_buffer) > max_pre:
                self._pre_speech_buffer = self._pre_speech_buffer[-max_pre:]

            self._current_sample += self.SILERO_CHUNK_SAMPLES

        return completed

    def process_stream(self, audio_gen: Iterator[np.ndarray]) -> Iterator[SpeechSegment]:
        """Process an audio stream and yield complete speech segments.

        Thin generator wrapper over process_chunk(). Maintained for backward
        compatibility with the legacy pipeline.
        """
        for chunk in audio_gen:
            for segment in self.process_chunk(chunk):
                yield segment
