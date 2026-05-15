"""Enrollment store — manages voiceprint .npy files + a users.json id→name registry.

The .npy file at <id>.npy is the voiceprint. The display name lives in
voiceprints/users.json — a small JSON dict {id: display_name}. The registry
is the authoritative list of who is enrolled: orphan .npy files without a
users.json entry are ignored by get_all() and list_users(). finalize_enrollment
auto-registers id→id so single-arg enroll.py flows keep working unchanged.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np


class EnrollmentStore:
    """Manages speaker voiceprint enrollment and storage."""

    USERS_FILENAME = "users.json"

    def __init__(self, voiceprints_dir: str):
        self.voiceprints_dir = voiceprints_dir
        os.makedirs(voiceprints_dir, exist_ok=True)

    # ---- paths -----------------------------------------------------------

    def _voiceprint_path(self, id: str) -> str:
        return os.path.join(self.voiceprints_dir, f"{id}.npy")

    def _utterances_path(self, id: str) -> str:
        return os.path.join(self.voiceprints_dir, f"{id}_utterances.npy")

    def _users_path(self) -> str:
        return os.path.join(self.voiceprints_dir, self.USERS_FILENAME)

    # ---- users.json registry --------------------------------------------

    def _read_users(self) -> Dict[str, str]:
        path = self._users_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_users(self, users: Dict[str, str]) -> None:
        with open(self._users_path(), "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)

    def register(self, id: str, name: str) -> None:
        """Upsert an id→name entry in users.json without touching the .npy file."""
        users = self._read_users()
        users[id] = name
        self._write_users(users)

    def get_name(self, id: str) -> str:
        """Return the display name for an id. Raises KeyError if not registered."""
        users = self._read_users()
        if id not in users:
            raise KeyError(f"id '{id}' is not registered in {self.USERS_FILENAME}")
        return users[id]

    # ---- enrollment ------------------------------------------------------

    def enroll(self, id: str, embedding: np.ndarray) -> None:
        """Add an utterance embedding during enrollment.

        Appends to the utterances file for later finalization.
        """
        utt_path = self._utterances_path(id)
        if os.path.exists(utt_path):
            existing = np.load(utt_path)
            updated = np.vstack([existing, embedding.reshape(1, -1)])
        else:
            updated = embedding.reshape(1, -1)
        np.save(utt_path, updated)

    def finalize_enrollment(self, id: str) -> None:
        """Compute mean of all utterance embeddings, L2-normalize, save as voiceprint.

        Auto-registers id→id in users.json if no entry exists yet, so callers
        that don't explicitly call register() still produce a discoverable user.
        """
        utt_path = self._utterances_path(id)
        if not os.path.exists(utt_path):
            raise FileNotFoundError(f"No utterances found for '{id}'")

        utterances = np.load(utt_path)
        mean_emb = np.mean(utterances, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        np.save(self._voiceprint_path(id), mean_emb)
        os.remove(utt_path)

        # Auto-register with name == id only if not already registered.
        users = self._read_users()
        if id not in users:
            users[id] = id
            self._write_users(users)

    def utterance_count(self, id: str) -> int:
        """Return the number of utterances recorded so far for enrollment."""
        utt_path = self._utterances_path(id)
        if os.path.exists(utt_path):
            return len(np.load(utt_path))
        return 0

    # ---- lookup ---------------------------------------------------------

    def get(self, id: str) -> Optional[np.ndarray]:
        """Get a single user's voiceprint (registered or orphan). Returns None if no .npy."""
        path = self._voiceprint_path(id)
        if os.path.exists(path):
            return np.load(path)
        return None

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return embeddings for all REGISTERED ids only. Orphan .npy files are ignored."""
        result = {}
        for id in self.list_users():
            vp = self.get(id)
            if vp is not None:
                result[id] = vp
        return result

    def list_users(self) -> List[str]:
        """List all registered ids (sorted). Orphan .npy files are ignored."""
        return sorted(self._read_users().keys())

    def delete(self, id: str) -> None:
        """Delete a user's voiceprint, any pending utterances, and the users.json entry."""
        for path in [self._voiceprint_path(id), self._utterances_path(id)]:
            if os.path.exists(path):
                os.remove(path)
        users = self._read_users()
        if id in users:
            del users[id]
            self._write_users(users)
