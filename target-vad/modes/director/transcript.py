"""TranscriptResult — the extended STT return (spec sections 6 & 9).

Today StreamingStt.transcribe_segment returns a bare str (stt.py:37-51). Plan 04
re-backs it to return per-segment confidence. wrap_transcript() bridges the gap:
a bare str becomes TranscriptResult(text, mean_word_prob=1.0) so the empty/low-
confidence RESTORE guard in the reducer composes today and tightens for free
once Plan 04 supplies real word probabilities."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    mean_word_prob: float


def wrap_transcript(raw: Optional[Union[str, TranscriptResult]]) -> TranscriptResult:
    if isinstance(raw, TranscriptResult):
        return raw
    if raw is None:
        return TranscriptResult(text="", mean_word_prob=1.0)
    return TranscriptResult(text=str(raw), mean_word_prob=1.0)
