"""Tests for in-session enrollment: enroll_from_intros + SessionEnrollmentView."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.diarization.intro_enrollment import (
    IntroVoiceprint,
    SessionEnrollmentView,
    enroll_from_intros,
)
from modes.diarization.manifest import ManifestEntry


SR = 16000


def unit_vec(seed: int, dim: int = 192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def fake_embedder():
    """Embedder mock — extract.return_value is set per-test."""
    return MagicMock()


@pytest.fixture
def fake_store():
    """Persistent store mock with .get_all() / .get_name()."""
    m = MagicMock()
    m.get_all.return_value = {}
    m.get_name.return_value = ""
    return m


class TestEnrollFromIntros:
    def test_returns_one_voiceprint_per_manifest_entry(self, fake_embedder):
        emb_a = unit_vec(1)
        emb_b = unit_vec(2)
        fake_embedder.extract.side_effect = [emb_a, emb_b]

        audio = np.zeros(60 * SR, dtype=np.float32)
        manifest = [
            ManifestEntry(id="alice", name="Alice", start=0.0, end=10.0),
            ManifestEntry(id="bob", name="Bob", start=10.0, end=25.0),
        ]
        result = enroll_from_intros(audio, SR, manifest, fake_embedder)
        assert len(result) == 2
        assert result[0] == IntroVoiceprint(id="alice", name="Alice", embedding=pytest.approx(emb_a))
        assert result[1] == IntroVoiceprint(id="bob", name="Bob", embedding=pytest.approx(emb_b))

    def test_slices_correct_audio_window(self, fake_embedder):
        # Construct audio with two distinct regions
        audio = np.zeros(60 * SR, dtype=np.float32)
        audio[5 * SR:10 * SR] = 1.0   # 5s of 1.0
        audio[20 * SR:30 * SR] = 2.0  # 10s of 2.0

        fake_embedder.extract.return_value = unit_vec(1)

        manifest = [
            ManifestEntry(id="alice", name="Alice", start=5.0, end=10.0),
            ManifestEntry(id="bob", name="Bob", start=20.0, end=30.0),
        ]
        enroll_from_intros(audio, SR, manifest, fake_embedder)

        # First call: 5 seconds of all 1.0
        first_call_audio = fake_embedder.extract.call_args_list[0].args[0]
        assert first_call_audio.shape == (5 * SR,)
        assert np.all(first_call_audio == 1.0)

        # Second call: 10 seconds of all 2.0
        second_call_audio = fake_embedder.extract.call_args_list[1].args[0]
        assert second_call_audio.shape == (10 * SR,)
        assert np.all(second_call_audio == 2.0)

    def test_clamps_end_past_audio_end(self, fake_embedder, caplog):
        fake_embedder.extract.return_value = unit_vec(1)
        audio = np.zeros(30 * SR, dtype=np.float32)  # 30 s of audio
        manifest = [ManifestEntry(id="alice", name="Alice", start=20.0, end=60.0)]  # past end
        result = enroll_from_intros(audio, SR, manifest, fake_embedder)
        # Should have clamped to len(audio) and not raised
        assert len(result) == 1
        passed = fake_embedder.extract.call_args.args[0]
        assert passed.shape == (10 * SR,)  # 20s..30s -> 10s

    def test_empty_manifest_returns_empty(self, fake_embedder):
        result = enroll_from_intros(np.zeros(10 * SR, dtype=np.float32), SR, [], fake_embedder)
        assert result == []
        fake_embedder.extract.assert_not_called()

    def test_warns_on_start_past_audio_end(self, fake_embedder, caplog):
        import logging
        fake_embedder.extract.return_value = unit_vec(1)
        audio = np.zeros(30 * SR, dtype=np.float32)  # 30 s of audio
        manifest = [ManifestEntry(id="ghost", name="Ghost", start=45.0, end=60.0)]
        with caplog.at_level(logging.WARNING, logger="modes.diarization.intro_enrollment"):
            result = enroll_from_intros(audio, SR, manifest, fake_embedder)
        # Entry is not silently dropped — voiceprint is still produced
        assert len(result) == 1
        assert result[0].id == "ghost"
        # Warning surfaces the issue
        assert any("past audio end" in rec.message for rec in caplog.records)

    def test_skips_and_warns_on_short_slice(self, fake_embedder, capsys):
        fake_embedder.extract.return_value = unit_vec(1)
        audio = np.zeros(30 * SR, dtype=np.float32)
        # 500 ms slice — below ECAPA's 800 ms minimum
        manifest = [ManifestEntry(id="brief", name="Brief", start=1.0, end=1.5)]
        result = enroll_from_intros(audio, SR, manifest, fake_embedder)
        # Short-slice entry must be skipped, not embedded
        assert len(result) == 0
        fake_embedder.extract.assert_not_called()
        # Warning must appear in rich console output
        captured = capsys.readouterr()
        assert "brief" in captured.out
        assert "< 0.8s" in captured.out or "0.50s" in captured.out

    def test_short_slice_skipped_but_other_entries_kept(self, fake_embedder, capsys):
        emb_a = unit_vec(1)
        fake_embedder.extract.return_value = emb_a
        audio = np.zeros(30 * SR, dtype=np.float32)
        # First entry is too short; second is fine
        manifest = [
            ManifestEntry(id="brief", name="Brief", start=1.0, end=1.5),
            ManifestEntry(id="alice", name="Alice", start=2.0, end=12.0),
        ]
        result = enroll_from_intros(audio, SR, manifest, fake_embedder)
        # Only the valid entry is enrolled
        assert len(result) == 1
        assert result[0].id == "alice"
        fake_embedder.extract.assert_called_once()


class TestSessionEnrollmentView:
    def test_empty_intros_returns_persistent(self, fake_store):
        fake_store.get_all.return_value = {"alice": unit_vec(1)}
        fake_store.get_name.side_effect = lambda i: {"alice": "Alice"}[i]

        view = SessionEnrollmentView(fake_store, [])
        assert view.get_all() == {"alice": pytest.approx(unit_vec(1))}
        assert view.get_name("alice") == "Alice"

    def test_intros_override_persistent_with_same_id(self, fake_store):
        persistent_alice = unit_vec(1)
        session_alice = unit_vec(2)
        fake_store.get_all.return_value = {"alice": persistent_alice}
        fake_store.get_name.side_effect = lambda i: "Persistent Alice"

        intros = [IntroVoiceprint(id="alice", name="Session Alice", embedding=session_alice)]
        view = SessionEnrollmentView(fake_store, intros)

        all_vp = view.get_all()
        assert all_vp["alice"] is session_alice  # intro shadows
        assert view.get_name("alice") == "Session Alice"

    def test_intros_add_new_ids(self, fake_store):
        fake_store.get_all.return_value = {"alice": unit_vec(1)}
        fake_store.get_name.side_effect = lambda i: {"alice": "Alice"}[i]

        intros = [IntroVoiceprint(id="bob", name="Bob", embedding=unit_vec(3))]
        view = SessionEnrollmentView(fake_store, intros)

        all_vp = view.get_all()
        assert set(all_vp.keys()) == {"alice", "bob"}
        assert view.get_name("bob") == "Bob"
        assert view.get_name("alice") == "Alice"

    def test_get_name_unknown_raises_keyerror(self, fake_store):
        fake_store.get_all.return_value = {}
        fake_store.get_name.side_effect = KeyError("not registered")
        view = SessionEnrollmentView(fake_store, [])
        with pytest.raises(KeyError):
            view.get_name("nobody")
