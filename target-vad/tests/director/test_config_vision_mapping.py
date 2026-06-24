from modes.director.assembly import _director_config_from


def test_vision_floor_control_mapped():
    cfg = _director_config_from({"vision": {"owner_absent_grace_s": 4.0,
                                            "active_talk_guard_s": 2.0}})
    assert cfg.owner_absent_grace_s == 4.0
    assert cfg.active_talk_guard_s == 2.0


def test_vision_floor_control_defaults_when_absent():
    cfg = _director_config_from({})
    assert cfg.owner_absent_grace_s == 3.0
    assert cfg.active_talk_guard_s == 3.0
