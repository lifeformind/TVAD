from modes.director.assembly import _director_config_from
from modes.director.config import DirectorConfig


def test_default_is_false():
    assert DirectorConfig().reject_bystanders is False


def test_mapping_true_enables():
    cfg = _director_config_from({"turn_gate": {"reject_bystanders": True}})
    assert cfg.reject_bystanders is True


def test_mapping_absent_key_is_false():
    cfg = _director_config_from({"turn_gate": {}})
    assert cfg.reject_bystanders is False


def test_mapping_malformed_value_is_false():
    # the 'flase' lesson: a non-bool (typo/string/int) must NOT enable
    for bad in ("flase", "true", 1, "yes"):
        cfg = _director_config_from({"turn_gate": {"reject_bystanders": bad}})
        assert cfg.reject_bystanders is False, bad


def test_doa_keys_map_from_turn_gate_doa():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({"turn_gate": {"doa": {
        "cone_deg": 25, "bearing_ema_alpha": 0.5}}})
    assert cfg.doa_cone_deg == 25.0
    assert cfg.doa_bearing_ema_alpha == 0.5


def test_doa_keys_default_when_absent():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({})
    assert cfg.doa_cone_deg == 20.0
    assert cfg.doa_bearing_ema_alpha == 0.3
