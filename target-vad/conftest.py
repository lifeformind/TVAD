"""Root conftest — patches for environments missing audio/ML hardware libs.

Unit tests inject fakes for all hardware-dependent components (mic, VAD,
embedder, wake detector). These mocks only exist so the top-level imports
don't crash on headless machines.
"""

import sys
from unittest.mock import MagicMock

_OPTIONAL_MODULES = [
    "sounddevice",
    "torch",
    "torchaudio",
    "speechbrain",
    "speechbrain.inference",
    "speechbrain.inference.speaker",
    "openwakeword",
    "openwakeword.model",
    "faster_whisper",
    "kokoro",
    "webrtc_audio_processing",
    "librosa",
]

for _mod in _OPTIONAL_MODULES:
    try:
        __import__(_mod)
    except (ImportError, OSError):
        sys.modules.setdefault(_mod, MagicMock())
