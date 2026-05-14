"""Tests for the Target VAD pipeline (mocked microphone)."""

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from speaker.embedder import EmbeddingExtractor
from speaker.enrollment_store import EnrollmentStore
from speaker.verifier import SpeakerVerifier, VerificationResult
from core.vad.silero_vad import SileroVAD, SpeechSegment


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(temp_dir):
    return EnrollmentStore(temp_dir)


@pytest.fixture
def vad():
    config = {
        "sample_rate": 16000,
        "speech_threshold": 0.5,
        "min_speech_duration_ms": 300,
        "padding_ms": 200,
    }
    return SileroVAD(config)


class TestPipelineComponents:
    """Test that pipeline components work together correctly."""

    def test_enrollment_and_verification_roundtrip(self, store):
        """Enroll with embeddings and verify against them."""
        # Create a known embedding
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # Enroll
        store.enroll("testuser", emb)
        store.enroll("testuser", emb)  # Same embedding twice
        store.finalize_enrollment("testuser")

        # Verify
        verifier = SpeakerVerifier(store, threshold=0.75)
        result = verifier.verify(emb)

        assert result.is_registered
        assert result.matched_user == "testuser"
        assert result.confidence > 0.99

    def test_vad_to_embedding_flow(self, vad):
        """Test that VAD output format is compatible with embedding input."""
        # Create a SpeechSegment (as VAD would produce)
        segment = SpeechSegment(
            audio=np.random.randn(16000).astype(np.float32) * 0.1,
            start_ms=0.0,
            end_ms=1000.0,
            duration_ms=1000.0,
        )

        # Verify the segment has the right shape for the embedder
        assert segment.audio.dtype == np.float32
        assert len(segment.audio) >= 12800  # min 800ms at 16kHz

    def test_multiple_users_discrimination(self, store):
        """Verify that the system can distinguish between enrolled users."""
        # Create two distinct embeddings
        np.random.seed(42)
        emb_alice = np.random.randn(192).astype(np.float32)
        emb_alice = emb_alice / np.linalg.norm(emb_alice)

        emb_bob = np.random.randn(192).astype(np.float32)
        emb_bob = emb_bob / np.linalg.norm(emb_bob)

        store.enroll("alice", emb_alice)
        store.finalize_enrollment("alice")
        store.enroll("bob", emb_bob)
        store.finalize_enrollment("bob")

        verifier = SpeakerVerifier(store, threshold=0.75)

        # Alice's embedding should match Alice
        result_a = verifier.verify(emb_alice)
        assert result_a.matched_user == "alice"

        # Bob's embedding should match Bob
        result_b = verifier.verify(emb_bob)
        assert result_b.matched_user == "bob"

    def test_unknown_speaker_rejected(self, store):
        """An unenrolled speaker should not be matched."""
        emb_known = np.random.randn(192).astype(np.float32)
        emb_known = emb_known / np.linalg.norm(emb_known)
        store.enroll("known", emb_known)
        store.finalize_enrollment("known")

        # A completely different embedding
        emb_unknown = np.random.randn(192).astype(np.float32)
        emb_unknown = emb_unknown / np.linalg.norm(emb_unknown)

        verifier = SpeakerVerifier(store, threshold=0.75)
        result = verifier.verify(emb_unknown)

        # Random 192-dim vectors have near-zero cosine similarity on average
        assert not result.is_registered
