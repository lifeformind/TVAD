"""RollingContext — running transcript window for faster-whisper's initial_prompt.

Maintains the last N characters of accumulated transcript text. Each segment
appends its result; the head is truncated as needed so the next initial_prompt
fits the budget. Joined with single spaces, no punctuation manipulation.
"""


class RollingContext:
    """A character-bounded FIFO of transcript text.

    Always keeps at most `max_chars` characters, truncated from the LEFT so the
    most recent text is preserved (whisper benefits more from recent context
    than from old). Empty appends are no-ops. `max_chars=0` is supported and
    keeps the buffer empty regardless of appends.
    """

    def __init__(self, max_chars: int):
        if max_chars < 0:
            raise ValueError(f"max_chars must be >= 0, got {max_chars}")
        self._max_chars = max_chars
        self._text = ""

    def append(self, chunk: str) -> None:
        """Append `chunk` to the running context, separated by a space."""
        if not chunk or self._max_chars == 0:
            return
        if self._text:
            new = self._text + " " + chunk
        else:
            new = chunk
        if len(new) > self._max_chars:
            new = new[-self._max_chars:]
        self._text = new

    def text(self) -> str:
        """Return the current context string."""
        return self._text

    def reset(self) -> None:
        """Clear the context."""
        self._text = ""
