import numpy as np
from bench.build_cohort import build_cohort, _reverb


class _Emb:
    def extract(self, audio, sample_rate=16000):
        v = np.zeros(4, dtype=np.float32)
        v[0] = 1.0 if float(np.mean(np.abs(audio))) > 0.01 else 0.0
        v[1] = 1.0
        return v / np.linalg.norm(v)


def test_build_cohort_rows_are_unit_norm(tmp_path):
    import soundfile as sf
    for i in range(3):
        sf.write(tmp_path / f"u{i}.wav", np.random.default_rng(i).normal(0, 0.1, 16000).astype(np.float32), 16000)
    cohort = build_cohort(sorted(tmp_path.glob("*.wav")), _Emb())
    assert cohort.shape[0] == 3
    assert np.allclose(np.linalg.norm(cohort, axis=1), 1.0, atol=1e-5)

def test_augment_doubles_rows(tmp_path):
    import soundfile as sf
    sf.write(tmp_path / "u.wav", np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32), 16000)
    cohort = build_cohort(sorted(tmp_path.glob("*.wav")), _Emb(), augment=True)
    assert cohort.shape[0] == 2   # clean + reverberated

def test_reverb_changes_signal_but_keeps_length():
    x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    y = _reverb(x, seed=1)
    assert y.shape == x.shape and not np.allclose(x, y)
