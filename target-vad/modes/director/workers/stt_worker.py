"""SttWorker — executes the Director's transcription commands.

TranscribeUserTurn / TranscribeInterjection carry no audio (the reducer is pure):
the audio to transcribe is whatever the Ingestion worker last staged here for
that purpose. The worker runs StreamingStt.transcribe_segment and emits the
matching *Transcribed event, coercing today's bare-str return into a
TranscriptResult via wrap_transcript (spec sections 6 & 9). Plan 04 swaps the
engine internals for real per-word confidence with no change here."""

import numpy as np

from modes.director.bus import EventBus
from modes.director.transcript import wrap_transcript
from modes.director import events as E
from modes.director import commands as C


class SttWorker:
    def __init__(self, stt, bus: EventBus):
        self._stt = stt
        self._bus = bus
        self._pending_user_audio = None
        self._pending_interjection_audio = None

    def set_pending_user_audio(self, audio: np.ndarray) -> None:
        self._pending_user_audio = audio

    def set_pending_interjection_audio(self, audio: np.ndarray) -> None:
        self._pending_interjection_audio = audio

    async def execute(self, command) -> None:
        if isinstance(command, C.TranscribeUserTurn):
            audio = self._pending_user_audio
            self._pending_user_audio = None
            tr = await self._transcribe(audio)
            await self._bus.emit(E.UserTurnTranscribed(text=tr.text,
                                                       mean_word_prob=tr.mean_word_prob))
        elif isinstance(command, C.TranscribeInterjection):
            audio = self._pending_interjection_audio
            self._pending_interjection_audio = None
            tr = await self._transcribe(audio)
            await self._bus.emit(E.InterjectionTranscribed(text=tr.text,
                                                           mean_word_prob=tr.mean_word_prob))

    async def _transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return wrap_transcript("")        # empty -> reducer keeps listening / RESTOREs
        raw = await self._stt.transcribe_segment(audio)
        return wrap_transcript(raw)
