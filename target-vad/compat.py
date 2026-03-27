"""Compatibility patches for dependency version mismatches."""

import torchaudio

# torchaudio 2.9+ removed list_audio_backends(); SpeechBrain 1.0.3 requires it
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["torchcodec"]
