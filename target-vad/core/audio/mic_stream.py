"""Microphone audio stream using sounddevice with ring buffer."""

import collections
import threading
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd


class MicrophoneStream:
    """Streams audio from the microphone as float32 numpy chunks."""

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.chunk_size = config.get("chunk_size", 480)
        self.device_index = config.get("device_index", None)

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._buffer: collections.deque = collections.deque(maxlen=100)
        self._buffer_event = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — pushes float32 chunks into ring buffer."""
        self._buffer.append(indata[:, 0].copy())
        self._buffer_event.set()

    def start(self):
        """Open the microphone stream."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            dtype="float32",
            device=self.device_index,
            callback=self._audio_callback,
        )
        self._running = True
        self._stream.start()

    def stop(self):
        """Close the microphone stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._buffer.clear()

    def stream(self) -> Iterator[np.ndarray]:
        """Yield float32 numpy chunks from the microphone."""
        while self._running:
            self._buffer_event.wait(timeout=0.1)
            self._buffer_event.clear()
            while self._buffer:
                yield self._buffer.popleft()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
