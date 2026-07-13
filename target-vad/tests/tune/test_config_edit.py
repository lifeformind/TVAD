"""Targeted line edits to config.yaml: values change, everything else is
byte-identical. All-or-nothing — any failure leaves the text untouched
(the function is pure; the caller only writes the returned string)."""

from pathlib import Path

import pytest
import yaml

from tune.config_edit import ConfigEditError, get_path, set_values

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture()
def text():
    return REAL_CONFIG.read_text()


def _reload(edited, path):
    return get_path(yaml.safe_load(edited), path)


# ---- happy paths, one per scalar kind ----

def test_float_edit_touches_exactly_one_line(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0})
    assert _reload(edited, "kiosk.talkback.turn_gate.doa.cone_deg") == 25.0
    diff = [(a, b) for a, b in zip(text.split("\n"), edited.split("\n")) if a != b]
    assert len(diff) == 1
    old_line, new_line = diff[0]
    assert "cone_deg" in old_line
    # the inline comment survives verbatim
    assert old_line.split("#", 1)[1] == new_line.split("#", 1)[1]


def test_int_edit(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.doa.min_in_cone_samples": 4})
    assert _reload(edited, "kiosk.talkback.turn_gate.doa.min_in_cone_samples") == 4


def test_strict_bool_renders_literal_true_false(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.reject_bystanders": False})
    line = [l for l in edited.split("\n") if "reject_bystanders:" in l][0]
    assert "reject_bystanders: false" in line


def test_none_renders_null_and_null_becomes_number(text):
    # rms_threshold is null in the file today; set it, then set it back
    key = "kiosk.talkback.barge_in.proximity.rms_threshold"
    edited = set_values(text, {key: 0.04})
    assert _reload(edited, key) == 0.04
    back = set_values(edited, {key: None})
    assert _reload(back, key) is None
    assert "rms_threshold: null" in back


def test_string_edit_is_quoted(text):
    edited = set_values(text, {"kiosk.wake_phrase": "hey_jarvis"})
    assert _reload(edited, "kiosk.wake_phrase") == "hey_jarvis"
    line = [l for l in edited.split("\n") if l.strip().startswith("wake_phrase")][0]
    assert '"hey_jarvis"' in line


def test_block_scalar_edit_keeps_header_and_indents(text):
    key = "kiosk.talkback.llm.system_prompt"
    edited = set_values(text, {key: "Line one.\nLine two."})
    assert _reload(edited, key).rstrip("\n") == "Line one.\nLine two."
    lines = edited.split("\n")
    hdr = [i for i, l in enumerate(lines) if l.strip().startswith("system_prompt")][0]
    assert lines[hdr].rstrip().endswith("|")
    assert lines[hdr + 1] == "        Line one."          # key at 6 -> body at 8
    # the section after the block is intact
    assert _reload(edited, "kiosk.talkback.tts.voice") == "af_bella"


def test_multi_change_in_one_call(text):
    edited = set_values(text, {
        "kiosk.talkback.turn_gate.speaker_threshold": 0.18,
        "kiosk.talkback.barge_in.duck_level": 0.5,
    })
    assert _reload(edited, "kiosk.talkback.turn_gate.speaker_threshold") == 0.18
    assert _reload(edited, "kiosk.talkback.barge_in.duck_level") == 0.5


def test_same_key_name_different_sections_are_independent(text):
    # speaker_threshold exists under BOTH turn_gate and barge_in
    edited = set_values(text, {"kiosk.talkback.barge_in.speaker_threshold": 0.25})
    assert _reload(edited, "kiosk.talkback.barge_in.speaker_threshold") == 0.25
    assert _reload(edited, "kiosk.talkback.turn_gate.speaker_threshold") == 0.15


def test_comments_and_untouched_lines_are_byte_identical(text):
    edited = set_values(text, {"kiosk.talkback.silence_timeout_s": 45})
    kept = [l for l in text.split("\n") if "silence_timeout_s" not in l]
    kept_after = [l for l in edited.split("\n") if "silence_timeout_s" not in l]
    assert kept == kept_after


# ---- refusals (all-or-nothing) ----

def test_unknown_path_refused(text):
    with pytest.raises(ConfigEditError, match="no_such"):
        set_values(text, {"kiosk.no_such_key": 1})


def test_section_path_refused(text):
    with pytest.raises(ConfigEditError):
        set_values(text, {"kiosk.talkback.turn_gate.doa": 1})


def test_one_bad_path_aborts_the_whole_save(text):
    with pytest.raises(ConfigEditError):
        set_values(text, {
            "kiosk.talkback.turn_gate.doa.cone_deg": 25.0,
            "kiosk.bogus": 1,
        })
    # pure function: caller's text is untouched by construction


def test_duplicate_line_ambiguity_refused():
    dup = "a:\n  x: 1\n  x: 2\n"
    with pytest.raises(ConfigEditError, match="multiple"):
        set_values(dup, {"a.x": 3})


def test_unsupported_value_type_refused(text):
    with pytest.raises(ConfigEditError, match="unsupported"):
        set_values(text, {"kiosk.wake_threshold": [1, 2]})


def test_edit_never_disturbs_other_leaves(text):
    edited = set_values(text, {"kiosk.talkback.turn_gate.doa.cone_deg": 25.0})
    before, after = yaml.safe_load(text), yaml.safe_load(edited)
    before["kiosk"]["talkback"]["turn_gate"]["doa"]["cone_deg"] = 25.0
    assert before == after
