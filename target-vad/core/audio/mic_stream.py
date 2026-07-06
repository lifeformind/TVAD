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
        # Which captured column becomes the mono stream. On the ReSpeaker the
        # device must be opened at its full 6 channels with use_channel 0:
        # column 0 (PipeWire FL) is the XVF-3000's PROCESSED output
        # (beamformed + hardware AEC + NS). Opening 1 channel instead makes
        # PipeWire DOWNMIX all six — raw capsules AND the ch5 playback
        # reference — which more than doubles the kiosk's own-TTS bleed
        # (measured 2026-07-06: mono-downmix tone energy 0.0153 vs pure-ch0
        # 0.0068).
        self.use_channel = config.get("use_channel", 0)

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._buffer: collections.deque = collections.deque(maxlen=100)
        self._buffer_event = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — pushes float32 chunks into ring buffer."""
        self._buffer.append(indata[:, self.use_channel].copy())
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

    def read_available(self) -> list:
        """Non-blocking: pop and return ALL currently-buffered chunks (possibly an
        empty list). deque.popleft is atomic vs the callback's append (CPython), so
        this is a race-free drain that needs no Event handshake — the async caller
        polls it with asyncio.sleep instead of blocking an executor thread on the
        generator (which deadlocked under the Director's concurrency)."""
        out = []
        while self._buffer:
            out.append(self._buffer.popleft())
        return out

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
