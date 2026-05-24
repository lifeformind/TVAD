"""SentenceChunker — buffers streaming LLM tokens, emits on sentence boundaries.

Handles common abbreviation false positives so "Dr. Smith" doesn't split mid-title.
"""

import re

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
    "vs", "etc", "inc", "ltd", "corp",
    "u", "s", "a",
}

SENTENCE_TERMINATORS = {".", "?", "!"}


class SentenceChunker:
    """Feed tokens one at a time; get back complete sentence chunks."""

    def __init__(
        self,
        sentence_terminators: list[str] | None = None,
        max_chunk_chars: int = 120,
    ):
        self._terminators = set(sentence_terminators or SENTENCE_TERMINATORS)
        self._max_chunk_chars = max_chunk_chars
        self._buffer = ""

    def feed(self, token: str) -> str | None:
        self._buffer += token

        if len(self._buffer) >= self._max_chunk_chars:
            return self._emit()

        stripped = self._buffer.rstrip()
        if not stripped:
            return None

        last_char = stripped[-1]
        if last_char not in self._terminators:
            return None

        if last_char == "." and self._is_abbreviation(stripped):
            return None

        return self._emit()

    def flush(self) -> str | None:
        if self._buffer.strip():
            return self._emit()
        return None

    def reset(self) -> None:
        self._buffer = ""

    def _emit(self) -> str:
        chunk = self._buffer.strip()
        self._buffer = ""
        return chunk

    def _is_abbreviation(self, text: str) -> bool:
        match = re.search(r"(\w+)\.$", text)
        if match:
            word = match.group(1).lower()
            return word in ABBREVIATIONS
        return False
