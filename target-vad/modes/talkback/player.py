"""Player — async audio output with ring buffer for AEC playback reference.

Pushes TTS audio frames to the sounddevice output and maintains a ring buffer
so the same frames feed back to AEC as playback reference, sample-aligned.
Supports immediate flush for barge-in (drops queued audio).
"""

import asyncio
import threading

import numpy as np


class Player:
    """Async audio player with AEC reference ring buffer."""

    def __init__(self, sample_rate: int = 16000, ring_buffer_seconds: float = 2.0):
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._ring_buffer_size = int(sample_rate * ring_buffer_seconds)
        self._ring_buffer = np.zeros(self._ring_buffer_size, dtype=np.float32)
        self._ring_write_pos = 0
        self._ring_samples_written = 0
        self._ring_lock = threading.Lock()
        self._flushed = False
        self._pending_count = 0

    @property
    def pending_frames(self) -> int:
        return self._pending_count

    @property
    def is_playing(self) -> bool:
        return self._pending_count > 0

    @property
    def is_flushed(self) -> bool:
        return self._flushed

    async def enqueue(self, audio: np.ndarray) -> None:
        await self._queue.put(audio)
        self._pending_count += 1

    def flush(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._pending_count -= 1
            except asyncio.QueueEmpty:
                break
        self._pending_count = 0
        self._flushed = True

    async def get_next_frame(self) -> np.ndarray | None:
        frame = await self._queue.get()
        if frame is not None:
            self._pending_count -= 1
            self._record_to_ring_buffer(frame)
        return frame

    def record_reference(self, audio: np.ndarray) -> None:
        """Record an actually-played (post-gain) frame as the AEC reference.

        The playback loop calls this with each frame as it goes to the output
        device, so get_reference_frame() returns what the mic is actually
        hearing — which is what makes AEC cancel real echo.
        """
        self._record_to_ring_buffer(audio)

    def _record_to_ring_buffer(self, audio: np.ndarray) -> None:
        with self._ring_lock:
            n = len(audio)
            end = self._ring_write_pos + n
            if end <= self._ring_buffer_size:
                self._ring_buffer[self._ring_write_pos:end] = audio
            else:
                first = self._ring_buffer_size - self._ring_write_pos
                self._ring_buffer[self._ring_write_pos:] = audio[:first]
                remainder = n - first
                self._ring_buffer[:remainder] = audio[first:]
            self._ring_write_pos = end % self._ring_buffer_size
            self._ring_samples_written += n

    def get_reference_frame(self, num_samples: int) -> np.ndarray | None:
        with self._ring_lock:
            if self._ring_samples_written < num_samples:
                return None
            read_pos = (self._ring_write_pos - num_samples) % self._ring_buffer_size
            if read_pos + num_samples <= self._ring_buffer_size:
                return self._ring_buffer[read_pos:read_pos + num_samples].copy()
            first = self._ring_buffer_size - read_pos
            return np.concatenate([
                self._ring_buffer[read_pos:],
                self._ring_buffer[:num_samples - first],
            ])

    def reset_flush(self) -> None:
        self._flushed = False
