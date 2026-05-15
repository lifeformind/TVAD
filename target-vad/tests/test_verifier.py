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

    def test_register_and_get_name(self, store):
        store.register("alice_smith", "Alice")
        assert store.get_name("alice_smith") == "Alice"

    def test_get_name_unregistered_raises(self, store):
        with pytest.raises(KeyError):
            store.get_name("nobody")

    def test_finalize_enrollment_auto_registers(self, store):
        emb = random_unit_vector()
        store.enroll("siddharth", emb)
        store.finalize_enrollment("siddharth")
        # Auto-registered with name == id
        assert store.get_name("siddharth") == "siddharth"

    def test_register_overrides_auto_name(self, store):
        emb = random_unit_vector()
        store.enroll("siddharth", emb)
        store.finalize_enrollment("siddharth")  # auto-registers name="siddharth"
        store.register("siddharth", "Siddharth Jain")
        assert store.get_name("siddharth") == "Siddharth Jain"

    def test_orphan_npy_ignored_by_get_all(self, store, temp_dir):
        # Drop a stray .npy file directly into the voiceprints dir, never registered
        import numpy as np
        orphan = random_unit_vector()
        np.save(os.path.join(temp_dir, "stranger.npy"), orphan)
        # Also enroll one legit user
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        all_vp = store.get_all()
        assert "alice" in all_vp
        assert "stranger" not in all_vp

    def test_orphan_npy_ignored_by_list_users(self, store, temp_dir):
        import numpy as np
        np.save(os.path.join(temp_dir, "stranger.npy"), random_unit_vector())
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        assert store.list_users() == ["alice"]

    def test_delete_removes_users_json_entry(self, store):
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        store.delete("alice")
        with pytest.raises(KeyError):
            store.get_name("alice")

    def test_users_json_persists_across_instances(self, store, temp_dir):
        store.enroll("alice", random_unit_vector())
        store.finalize_enrollment("alice")
        store.register("alice", "Alice Smith")
        # Fresh store instance pointing at the same dir
        from core.speaker.enrollment_store import EnrollmentStore as ES
        fresh = ES(temp_dir)
        assert fresh.get_name("alice") == "Alice Smith"
        assert fresh.list_users() == ["alice"]


class TestSpeakerVerifier:
    def test_same_embedding_matches(self, store, verifier):
        emb = random_unit_vector()
        store.enroll("alice", emb)
        store.finalize_enrollment("alice")
        store.register("alice", "Alice Smith")  # explicit display name

        result = verifier.verify(emb)
        assert result.is_registered
        assert result.matched_id == "alice"
        assert result.matched_name == "Alice Smith"
        assert result.confidence > 0.99

    def test_match_falls_back_to_id_when_no_display_name(self, store, verifier):
        emb = random_unit_vector()
        store.enroll("bob", emb)
        store.finalize_enrollment("bob")  # auto-registers name == "bob"

        result = verifier.verify(emb)
        assert result.is_registered
        assert result.matched_id == "bob"
        assert result.matched_name == "bob"

    def test_orthogonal_embedding_no_match(self, store, verifier):
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        store.enroll("alice", emb)
        store.finalize_enrollment("alice")

        query = np.zeros(192, dtype=np.float32)
        query[1] = 1.0
        result = verifier.verify(query)
        assert not result.is_registered
        assert result.matched_id is None
        assert result.matched_name is None

    def test_near_threshold(self, store):
        emb = random_unit_vector()
        store.enroll("alice", emb)
        store.finalize_enrollment("alice")

        noise = np.random.randn(192).astype(np.float32) * 0.05
        perturbed = emb + noise
        perturbed = perturbed / np.linalg.norm(perturbed)

        low_verifier = SpeakerVerifier(store, threshold=0.75)
        result = low_verifier.verify(perturbed)
        assert result.is_registered
        assert result.matched_id == "alice"

        high_verifier = SpeakerVerifier(store, threshold=0.999)
        result = high_verifier.verify(perturbed)
        assert not result.is_registered
        assert result.matched_id is None

    def test_no_enrolled_users(self, store, verifier):
        emb = random_unit_vector()
        result = verifier.verify(emb)
        assert not result.is_registered
        assert result.matched_id is None
        assert result.matched_name is None
        assert result.all_scores == {}

    def test_update_threshold(self, verifier):
        verifier.update_threshold(0.9)
        assert verifier.threshold == 0.9
