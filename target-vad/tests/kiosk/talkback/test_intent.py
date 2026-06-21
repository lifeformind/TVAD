"""Tests for the pure backchannel-vs-question interjection classifier."""

import pytest

from modes.talkback.intent import Interjection, classify_interjection


@pytest.mark.parametrize("text", [
    "okay", "ok", "yeah", "yep", "mhm", "right", "sure", "cool",
    "got it", "yeah got it", "uh-huh", "mm-hmm", "i see", "makes sense",
])
def test_pure_backchannels_do_not_interrupt(text):
    assert classify_interjection(text) == Interjection.BACKCHANNEL


@pytest.mark.parametrize("text", ["", "   ", ".", "?!"])
def test_empty_or_punctuation_only_is_backchannel(text):
    # whisper-tiny near-silence -> never cut on garbage
    assert classify_interjection(text) == Interjection.BACKCHANNEL


@pytest.mark.parametrize("text", ["you", "thank you", "thanks"])
def test_common_stt_hallucinations_are_backchannel(text):
    assert classify_interjection(text) == Interjection.BACKCHANNEL


@pytest.mark.parametrize("text", [
    "why", "why?", "wait", "stop", "what do you mean", "hold on",
    "can you repeat that", "but is that the fast one", "actually no",
])
def test_questions_and_commands_interrupt(text):
    assert classify_interjection(text) == Interjection.INTERRUPT


def test_force_token_wins_over_backchannel_tokens():
    # a force-interrupt token anywhere forces a cut even amid backchannels
    assert classify_interjection("okay but why") == Interjection.INTERRUPT


def test_multiple_content_words_interrupt():
    # not in either lexicon, >=2 content words -> treat as a real turn
    assert classify_interjection("tell me about the blue door") == Interjection.INTERRUPT


@pytest.mark.parametrize("text", ["no problem", "no worries", "go on"])
def test_no_problem_and_go_on_are_backchannel(text):
    assert classify_interjection(text) == Interjection.BACKCHANNEL


def test_bare_no_still_interrupts():
    assert classify_interjection("no") == Interjection.INTERRUPT


def test_none_input_is_backchannel():
    assert classify_interjection(None) == Interjection.BACKCHANNEL
