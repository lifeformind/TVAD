"""Enrollment store — manages voiceprint .npy files on disk."""

import os
from typing import Dict, List, Optional

import numpy as np


class EnrollmentStore:
    """Manages speaker voiceprint enrollment and storage."""

    def __init__(self, voiceprints_dir: str):
        self.voiceprints_dir = voiceprints_dir
        os.makedirs(voiceprints_dir, exist_ok=True)

    def _voiceprint_path(self, username: str) -> str:
        return os.path.join(self.voiceprints_dir, f"{username}.npy")

    def _utterances_path(self, username: str) -> str:
        return os.path.join(self.voiceprints_dir, f"{username}_utterances.npy")

    def enroll(self, username: str, embedding: np.ndarray):
        """Add an utterance embedding during enrollment.

        Appends to the utterances file for later finalization.
        """
        utt_path = self._utterances_path(username)
        if os.path.exists(utt_path):
            existing = np.load(utt_path)
            updated = np.vstack([existing, embedding.reshape(1, -1)])
        else:
            updated = embedding.reshape(1, -1)
        np.save(utt_path, updated)

    def finalize_enrollment(self, username: str):
        """Compute mean of all utterance embeddings, L2-normalize, and save as voiceprint."""
        utt_path = self._utterances_path(username)
        if not os.path.exists(utt_path):
            raise FileNotFoundError(f"No utterances found for '{username}'")

        utterances = np.load(utt_path)
        mean_emb = np.mean(utterances, axis=0)
        # L2 normalize
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        np.save(self._voiceprint_path(username), mean_emb)
        # Clean up utterances file
        os.remove(utt_path)

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return all enrolled voiceprints."""
        result = {}
        for name in self.list_users():
            result[name] = np.load(self._voiceprint_path(name))
        return result

    def get(self, username: str) -> Optional[np.ndarray]:
        """Get a single user's voiceprint."""
        path = self._voiceprint_path(username)
        if os.path.exists(path):
            return np.load(path)
        return None

    def delete(self, username: str):
        """Delete a user's voiceprint and any pending utterances."""
        for path in [self._voiceprint_path(username), self._utterances_path(username)]:
            if os.path.exists(path):
                os.remove(path)

    def list_users(self) -> List[str]:
        """List all enrolled usernames."""
        users = []
        for f in os.listdir(self.voiceprints_dir):
            if f.endswith(".npy") and not f.endswith("_utterances.npy"):
                users.append(f[:-4])
        return sorted(users)

    def utterance_count(self, username: str) -> int:
        """Return the number of utterances recorded so far for enrollment."""
        utt_path = self._utterances_path(username)
        if os.path.exists(utt_path):
            return len(np.load(utt_path))
        return 0
