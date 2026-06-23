import pathlib
import importlib.util
import numpy as np

# Import the bench module by path (it lives outside any package).
_spec = importlib.util.spec_from_file_location(
    "vpp", pathlib.Path(__file__).resolve().parents[2] / "bench" / "vision_presence_probe.py")
vpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpp)


def test_ensure_model_skips_existing(tmp_path):
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"already here")
    # Should NOT attempt any network fetch when a non-empty file already exists.
    out = vpp.ensure_model("http://invalid.invalid/should-not-be-fetched.onnx", dest)
    assert out == dest
    assert dest.read_bytes() == b"already here"


def test_iou_overlap_and_disjoint():
    assert vpp.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert vpp.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    # Half-overlap on x: intersection 5x10=50, union 100+100-50=150 -> 1/3.
    assert abs(vpp.iou((0, 0, 10, 10), (5, 0, 10, 10)) - (50 / 150)) < 1e-6


def test_box_in_zone_center_and_size():
    # frame 320x240; central zone x in [0.2,0.8]. A big centered box passes.
    assert vpp.box_in_zone((130, 80, 60, 80), 320, 240) is True
    # Off to the left edge (center x ~ 0.06) -> outside zone.
    assert vpp.box_in_zone((0, 80, 40, 80), 320, 240) is False
    # Centered but tiny (far away) -> fails min_area_frac.
    assert vpp.box_in_zone((150, 110, 12, 16), 320, 240) is False


def test_presence_debouncer_hysteresis():
    d = vpp.PresenceDebouncer(present_after_s=1.0, absent_after_s=2.0)
    assert d.update(True, 0.0) == "absent"      # not yet continuous 1s
    assert d.update(True, 0.5) == "absent"
    assert d.update(True, 1.0) == "present"      # 1s continuous -> present
    assert d.update(False, 1.5) == "present"     # brief miss, < 2s -> still present
    assert d.update(False, 3.0) == "present"     # 1.5s of absence (<2s from first miss)
    assert d.update(False, 3.6) == "absent"      # >=2s continuous absence -> absent


def test_cosine_basic():
    assert abs(vpp.cosine(np.array([1.0, 0]), np.array([1.0, 0])) - 1.0) < 1e-6
    assert abs(vpp.cosine(np.array([1.0, 0]), np.array([0, 1.0]))) < 1e-6


def test_separation_report_clean_split():
    # self scores high, cross scores low, fully separable.
    rep = vpp.separation_report([0.7, 0.8, 0.75], [0.1, 0.2, 0.15])
    assert rep["separated"] is True
    assert rep["self_accept_rate"] == 1.0
    assert rep["cross_reject_rate"] == 1.0
    assert rep["cross_max"] < rep["threshold"] <= rep["self_min"]


def test_separation_report_overlap_not_separated():
    # overlapping distributions -> no threshold gives 100/100.
    rep = vpp.separation_report([0.4, 0.55, 0.3], [0.35, 0.5, 0.45])
    assert rep["separated"] is False
    assert min(rep["self_accept_rate"], rep["cross_reject_rate"]) < 1.0


def test_gate_is_noop_when_non_interactive(monkeypatch, capsys):
    # Force non-interactive stdin; _gate must print the message and NOT block.
    monkeypatch.setattr(vpp.sys.stdin, "isatty", lambda: False, raising=False)
    vpp._gate("PERSON A: stand at the kiosk")     # must return immediately
    out = capsys.readouterr().out
    assert "PERSON A: stand at the kiosk" in out
