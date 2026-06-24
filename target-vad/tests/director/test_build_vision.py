from modes.director.assembly import _build_vision


def test_disabled_returns_none():
    assert _build_vision({"vision": {"enabled": False}}, bus=object()) is None


def test_missing_vision_block_returns_none():
    assert _build_vision({}, bus=object()) is None


def test_enabled_but_no_cv2_returns_none(monkeypatch):
    import modes.director.assembly as A
    monkeypatch.setattr(A, "_cv2_available", lambda: False)
    assert _build_vision({"vision": {"enabled": True}}, bus=object()) is None
