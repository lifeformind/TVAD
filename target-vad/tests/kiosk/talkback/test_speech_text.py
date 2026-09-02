"""Tests for strip_markdown_for_speech — keep markdown out of the TTS audio.

The LLM emits markdown emphasis (*slope*, **bold**, `code`) despite the system
prompt; a voice assistant must never vocalize those markers. This strips them
deterministically before synthesis.
"""

from modes.talkback.speech_text import strip_markdown_for_speech


def test_strips_italic_asterisks():
    assert strip_markdown_for_speech(
        "find the *area* under the line.") == "find the area under the line."


def test_strips_bold():
    assert strip_markdown_for_speech(
        "this is **very** important") == "this is very important"


def test_strips_inline_code():
    assert strip_markdown_for_speech("run `pytest` now") == "run pytest now"


def test_strips_link_keeps_text():
    assert strip_markdown_for_speech(
        "see [the docs](http://x.com) please") == "see the docs please"


def test_strips_leading_header():
    assert strip_markdown_for_speech("## Summary") == "Summary"


def test_strips_leading_bullet():
    assert strip_markdown_for_speech("- first item") == "first item"


def test_underscore_emphasis():
    assert strip_markdown_for_speech(
        "this is _emphasized_ text") == "this is emphasized text"


def test_plain_text_unchanged():
    assert strip_markdown_for_speech(
        "Hello there. How are you?") == "Hello there. How are you?"


def test_removes_stray_asterisk_from_split_emphasis():
    # emphasis split across streamed chunks leaves a lone marker -> never spoken
    assert "*" not in strip_markdown_for_speech("the *slope of a line")


def test_lone_marker_becomes_empty():
    assert strip_markdown_for_speech("*") == ""


def test_preserves_apostrophes_and_hyphens_and_endash():
    assert strip_markdown_for_speech(
        "it's well-known – really") == "it's well-known – really"


def test_strips_fullwidth_homoglyph_markers():
    # A decoder routing around an ASCII-only markdown ban can emit fullwidth
    # lookalikes (U+FF0A/FF03/FF40/FF3F/FF5E) instead — these must never
    # reach TTS either.
    result = strip_markdown_for_speech("＃heading ＊bold＊")
    for ch in "＊＃｀＿～":
        assert ch not in result


def test_collapses_whitespace_left_by_removals():
    assert strip_markdown_for_speech("a  *b*   c") == "a b c"
