"""Tests for the rolling-context helper used as faster-whisper's initial_prompt."""

import pytest

from modes.transcription.rolling_context import RollingContext


class TestRollingContext:
    def test_empty_at_start(self):
        ctx = RollingContext(max_chars=200)
        assert ctx.text() == ""

    def test_single_append(self):
        ctx = RollingContext(max_chars=200)
        ctx.append("hello world")
        assert ctx.text() == "hello world"

    def test_multiple_appends_joined_by_space(self):
        ctx = RollingContext(max_chars=200)
        ctx.append("hello")
        ctx.append("world")
        assert ctx.text() == "hello world"

    def test_truncates_to_max_chars_from_the_left(self):
        """When the running context exceeds max_chars, keep the LAST max_chars (most recent)."""
        ctx = RollingContext(max_chars=10)
        ctx.append("0123456789")  # exactly 10 chars, fits
        assert ctx.text() == "0123456789"
        ctx.append("ABC")  # now 14 chars total: "0123456789 ABC"
        # Keep the last 10 chars: "456789 ABC"
        assert ctx.text() == "456789 ABC"
        assert len(ctx.text()) <= 10

    def test_append_empty_string_is_noop(self):
        ctx = RollingContext(max_chars=200)
        ctx.append("hello")
        ctx.append("")
        assert ctx.text() == "hello"

    def test_max_chars_zero_keeps_text_empty(self):
        ctx = RollingContext(max_chars=0)
        ctx.append("anything")
        assert ctx.text() == ""

    def test_reset_clears_text(self):
        ctx = RollingContext(max_chars=200)
        ctx.append("hello")
        ctx.reset()
        assert ctx.text() == ""
        ctx.append("world")
        assert ctx.text() == "world"
