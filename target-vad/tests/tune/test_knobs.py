"""Registry meta-tests: the registry must stay true to the real config.yaml
(catches drift when config evolves) and internally sane."""

from pathlib import Path

import yaml

from tune.config_edit import get_path
from tune.knobs import BY_PATH, KNOBS, TABS, Knob, as_json

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
CONFIG = yaml.safe_load(REAL_CONFIG.read_text())

KIND_TYPES = {
    "float": (int, float), "int": (int,), "bool": (bool,),
    "select": (str,), "text": (str,), "textarea": (str,),
}


def test_every_knob_path_exists_in_real_config():
    for k in KNOBS:
        get_path(CONFIG, k.path)  # raises on drift


def test_every_knob_value_matches_its_kind():
    for k in KNOBS:
        v = get_path(CONFIG, k.path)
        if v is None:
            assert k.nullable, f"{k.path} is null but not nullable"
            continue
        assert isinstance(v, KIND_TYPES[k.kind]), f"{k.path}: {v!r} not {k.kind}"
        if k.kind != "bool":
            assert not isinstance(v, bool), f"{k.path}: bool in a {k.kind} knob"


def test_selects_include_the_current_value():
    for k in KNOBS:
        if k.kind == "select":
            assert get_path(CONFIG, k.path) in k.choices, k.path


def test_numeric_knobs_have_ranges_containing_current_value():
    for k in KNOBS:
        if k.kind in ("float", "int"):
            assert k.min is not None and k.max is not None and k.min < k.max, k.path
            v = get_path(CONFIG, k.path)
            if v is not None:
                assert k.min <= v <= k.max, f"{k.path}: {v} outside [{k.min},{k.max}]"


def test_tabs_are_the_declared_set_and_nonempty():
    assert set(k.tab for k in KNOBS) == set(TABS)
    for t in TABS:
        assert any(k.tab == t for k in KNOBS)


def test_strict_bools_are_exactly_the_documented_keys():
    strict = {k.path for k in KNOBS if k.strict_bool}
    assert strict == {
        "kiosk.talkback.turn_gate.require_speaker_match",
        "kiosk.talkback.turn_gate.reject_bystanders",
        "kiosk.talkback.turn_gate.doa.enabled",
    }


def test_paths_unique_and_by_path_complete():
    assert len({k.path for k in KNOBS}) == len(KNOBS)
    assert BY_PATH == {k.path: k for k in KNOBS}


def test_as_json_is_serializable_and_ordered_like_knobs():
    rows = as_json()
    assert [r["path"] for r in rows] == [k.path for k in KNOBS]
    assert all(isinstance(r["choices"], list) for r in rows)


def test_excluded_keys_are_not_registered():
    for banned in ("kiosk.talkback.output_device", "core.audio.device_index",
                   "kiosk.talkback.stt.backend", "kiosk.talkback.llm.base_url",
                   "kiosk.talkback.aec.enabled", "kiosk.talkback.crowd_focus.enabled",
                   "core.audio.sample_rate", "core.vad.sample_rate",
                   "kiosk.talkback.sample_rate_hz"):
        assert banned not in BY_PATH, banned
