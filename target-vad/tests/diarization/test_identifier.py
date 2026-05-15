"""Tests for ClusterIdentifier — mocks the embedder, uses real cosine math + sampling."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.diarization.identifier import ClusterIdentifier


SR = 16000


def unit_vec(seed: int, dim: int = 192) -> np.ndarray:
    """Deterministic random unit vector for tests."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


@pytest.fixture
def fake_embedder():
    """Embedder mock — returns whatever extract() is told to return.

    Default behavior is overridden per-test by setting .extract.side_effect.
    """
    m = MagicMock()
    return m


@pytest.fixture
def fake_store():
    """Enrollment-store mock with .get_all() returning a dict of name → voiceprint."""
    m = MagicMock()
    m.get_all.return_value = {}
    return m


class TestClusterIdentifierBasics:
    def test_matches_enrolled_speaker(self, fake_embedder, fake_store):
        """Embedder returns alice's voiceprint exactly → cluster labels as id 'alice'."""
        alice_vp = unit_vec(seed=1)
        bob_vp = unit_vec(seed=2)
        fake_store.get_all.return_value = {"alice": alice_vp, "bob": bob_vp}
        fake_embedder.extract.return_value = alice_vp

        identifier = ClusterIdentifier(
            embedder=fake_embedder,
            enrollment_store=fake_store,
            threshold=0.55,
            max_sample_seconds=30,
        )
        audio = make_silence(10.0)
        clusters = {"SPEAKER_00": [(0.0, 5.0)]}
        labels = identifier.label_clusters(audio, sample_rate=SR, clusters=clusters)
        assert labels == {"SPEAKER_00": "alice"}

    def test_below_threshold_is_unknown(self, fake_embedder, fake_store):
        """Embedder returns a vector orthogonal to all voiceprints → 'unknown'."""
        alice_vp = np.zeros(192, dtype=np.float32); alice_vp[0] = 1.0
        bob_vp = np.zeros(192, dtype=np.float32); bob_vp[1] = 1.0
        fake_store.get_all.return_value = {"alice": alice_vp, "bob": bob_vp}
        orthogonal = np.zeros(192, dtype=np.float32); orthogonal[2] = 1.0
        fake_embedder.extract.return_value = orthogonal

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(make_silence(5.0), SR, {"SPEAKER_00": [(0.0, 2.0)]})
        assert labels == {"SPEAKER_00": "unknown"}

    def test_at_threshold_inclusive(self, fake_embedder, fake_store):
        """A cosine exactly at threshold counts as a match (>=)."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}
        # Construct an embedding whose cosine with alice_vp == 0.55 exactly:
        # take alice_vp scaled by 0.55 plus an orthogonal component scaled by sqrt(1-0.55^2).
        ortho = unit_vec(seed=99)
        ortho = ortho - np.dot(ortho, alice_vp) * alice_vp
        ortho = ortho / np.linalg.norm(ortho)
        target = 0.55 * alice_vp + np.sqrt(1 - 0.55 ** 2) * ortho
        target = target / np.linalg.norm(target)
        fake_embedder.extract.return_value = target

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(make_silence(5.0), SR, {"SPEAKER_00": [(0.0, 2.0)]})
        assert labels == {"SPEAKER_00": "alice"}

    def test_no_enrolled_users_all_unknown(self, fake_embedder, fake_store):
        """Empty enrollment store → every cluster labeled 'unknown'."""
        fake_store.get_all.return_value = {}
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(
            make_silence(10.0), SR,
            {"SPEAKER_00": [(0.0, 2.0)], "SPEAKER_01": [(3.0, 4.0)]},
        )
        assert labels == {"SPEAKER_00": "unknown", "SPEAKER_01": "unknown"}
        fake_embedder.extract.assert_not_called()  # nothing to compare against

    def test_picks_best_match_among_multiple(self, fake_embedder, fake_store):
        """Highest-cosine enrolled id wins."""
        alice_vp = unit_vec(seed=1)
        bob_vp = unit_vec(seed=2)
        fake_store.get_all.return_value = {"alice": alice_vp, "bob": bob_vp}
        # Embedding closer to bob than to alice — verify bob wins
        bob_like = 0.9 * bob_vp + 0.1 * alice_vp
        bob_like = bob_like / np.linalg.norm(bob_like)
        fake_embedder.extract.return_value = bob_like

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(make_silence(5.0), SR, {"S0": [(0.0, 2.0)]})
        assert labels == {"S0": "bob"}


class TestClusterIdentifierAudioExtraction:
    def test_passes_concatenated_audio_to_embedder(self, fake_embedder, fake_store):
        """Cluster audio = concat of all chosen segments from the waveform."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}
        fake_embedder.extract.return_value = alice_vp

        # Construct a waveform with distinct values in two regions so we can verify
        # which samples were passed to the embedder.
        audio = np.zeros(10 * SR, dtype=np.float32)
        audio[1 * SR:2 * SR] = 1.0      # segment A: 1.0–2.0 s
        audio[3 * SR:4 * SR] = 2.0      # segment B: 3.0–4.0 s

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        identifier.label_clusters(audio, SR, {"S0": [(1.0, 2.0), (3.0, 4.0)]})

        # The embedder should have been called exactly once with the concatenated
        # audio of the two segments (1s + 1s = 2s = 32000 samples).
        assert fake_embedder.extract.call_count == 1
        passed_audio = fake_embedder.extract.call_args.args[0]
        assert passed_audio.shape == (2 * SR,)
        # First half is all 1.0, second half is all 2.0
        assert np.all(passed_audio[:SR] == 1.0)
        assert np.all(passed_audio[SR:] == 2.0)

    def test_subsamples_long_cluster(self, fake_embedder, fake_store):
        """Cluster longer than max_sample_seconds should be subsampled."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}
        fake_embedder.extract.return_value = alice_vp

        audio = np.zeros(120 * SR, dtype=np.float32)
        # 12 segments of 5 s each = 60 s total; cap at 10 s
        segments = [(i * 10.0, i * 10.0 + 5.0) for i in range(12)]
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=10,
        )
        identifier.label_clusters(audio, SR, {"S0": segments})

        passed_audio = fake_embedder.extract.call_args.args[0]
        # Concatenated subsample must be <= max_sample_seconds * SR (a bit of slack for boundary)
        assert len(passed_audio) <= 10 * SR + SR  # allow up to one segment's slack


class TestClusterIdentifierErrors:
    def test_embedder_raising_labels_unknown(self, fake_embedder, fake_store):
        """If embedder.extract() raises, label that cluster as 'unknown' and continue."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}

        def raise_then_succeed(audio, sample_rate=SR):
            if raise_then_succeed.called:
                return alice_vp
            raise_then_succeed.called = True
            raise RuntimeError("simulated embedder failure")
        raise_then_succeed.called = False
        fake_embedder.extract.side_effect = raise_then_succeed

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(
            make_silence(20.0), SR,
            {"S0": [(0.0, 2.0)], "S1": [(5.0, 7.0)]},
        )
        # Cluster ordering in dict is insertion-order; S0 fails, S1 succeeds.
        assert labels["S0"] == "unknown"
        assert labels["S1"] == "alice"

    def test_empty_clusters_returns_empty_map(self, fake_embedder, fake_store):
        fake_store.get_all.return_value = {"alice": unit_vec(1)}
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        assert identifier.label_clusters(make_silence(5.0), SR, {}) == {}
