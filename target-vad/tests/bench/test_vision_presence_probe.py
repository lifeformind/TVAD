import pathlib
import importlib.util

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
