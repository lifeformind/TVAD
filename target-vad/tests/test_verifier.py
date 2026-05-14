"""Tests for speaker verifier and enrollment store."""

import os
import tempfile

import numpy as np
import pytest

from core.speaker.enrollment_store import EnrollmentStore
from core.speaker.verifier import SpeakerVerifier, VerificationResult, cosine_similarity


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(temp_dir):
    return EnrollmentStore(temp_dir)


@pytest.fixture
def verifier(store):
    return SpeakerVerifier(store, threshold=0.75)


def random_unit_vector(dim: int = 192) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = random_unit_vector()
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.zeros(192, dtype=np.float32)
        b = np.zeros(192, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = random_unit_vector()
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector(self):
        v = random_unit_vector()
        z = np.zeros(192, dtype=np.float32)
        assert cosine_similarity(v, z) == 0.0


class TestEnrollmentStore:
    def test_enroll_and_finalize(self, store):
        emb1 = random_unit_vector()
        emb2 = random_unit_vector()
        store.enroll("alice", emb1)
        store.enroll("alice", emb2)
        assert store.utterance_count("alice") == 2

        store.finalize_enrollment("alice")
        vp = store.get("alice")
        assert vp is not None
        assert len(vp) == 192
        # Should be L2-normalized
        assert np.linalg.norm(vp) == pytest.approx(1.0, abs=1e-5)

    def test_list_users(self, store):
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        store.enroll("bob", random_unit_vector())
        store.finalize_enrollment("bob")
        users = store.list_users()
        assert "alice" in users
        assert "bob" in users

    def test_delete(self, store):
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        store.delete("alice")
        assert store.get("alice") is None
        assert "alice" not in store.list_users()

    def test_get_all(self, store):
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        all_vp = store.get_all()
        assert "alice" in all_vp
        assert len(all_vp["alice"]) == 192


class TestSpeakerVerifier:
    def test_same_embedding_matches(self, store, verifier):
        emb = random_unit_vector()
        store.enroll("alice", emb)
        store.finalize_enrollment("alice")

        result = verifier.verify(emb)
        assert result.is_registered
        assert result.matched_user == "alice"
        assert result.confidence > 0.99

    def test_orthogonal_embedding_no_match(self, store, verifier):
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        store.enroll("alice", emb)
        store.finalize_enrollment("alice")

        query = np.zeros(192, dtype=np.float32)
        query[1] = 1.0
        result = verifier.verify(query)
        assert not result.is_registered

    def test_near_threshold(self, store):
        """Test behavior near the similarity threshold."""
        emb = random_unit_vector()
        store.enroll("alice", emb)
        store.finalize_enrollment("alice")

        # Add a small perturbation (small enough to stay above 0.9 similarity)
        noise = np.random.randn(192).astype(np.float32) * 0.05
        perturbed = emb + noise
        perturbed = perturbed / np.linalg.norm(perturbed)

        # With a moderate threshold, perturbed vector should still match
        low_verifier = SpeakerVerifier(store, threshold=0.75)
        result = low_verifier.verify(perturbed)
        assert result.is_registered

        # With a very high threshold, should not match
        high_verifier = SpeakerVerifier(store, threshold=0.999)
        result = high_verifier.verify(perturbed)
        assert not result.is_registered

    def test_no_enrolled_users(self, store, verifier):
        emb = random_unit_vector()
        result = verifier.verify(emb)
        assert not result.is_registered
        assert result.matched_user is None
        assert result.all_scores == {}

    def test_update_threshold(self, verifier):
        verifier.update_threshold(0.9)
        assert verifier.threshold == 0.9
