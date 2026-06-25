from modes.director.assembly import _build_vision


def test_disabled_returns_none():
    assert _build_vision({"vision": {"enabled": False}}, bus=object()) is None


def test_missing_vision_block_returns_none():
    assert _build_vision({}, bus=object()) is None


def test_enabled_but_no_cv2_returns_none(monkeypatch):
    import modes.director.assembly as A
    monkeypatch.setattr(A, "_cv2_available", lambda: False)
    assert _build_vision({"vision": {"enabled": True}}, bus=object()) is None


def test_malformed_enabled_string_is_treated_as_off():
    # The live `enabled: flase` typo: YAML parses it as a truthy STRING, which
    # must NOT fail-open the feature. Only a real boolean True enables vision.
    assert _build_vision({"vision": {"enabled": "flase"}}, bus=object()) is None


def test_truthy_non_bool_enabled_is_treated_as_off():
    # Strict: any non-bool (even an arguably-affirmative string) is off, never on.
    for bad in ("true", "yes", 1, [1]):
        assert _build_vision({"vision": {"enabled": bad}}, bus=object()) is None


def test_malformed_enabled_warns(capsys):
    # A malformed value must be visible, not silently off.
    _build_vision({"vision": {"enabled": "flase"}}, bus=object())
    assert "flase" in capsys.readouterr().err
