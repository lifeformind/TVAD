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

    def _voiceprint_path(self, user_id: str) -> str:
        return os.path.join(self.voiceprints_dir, f"{user_id}.npy")

    def _utterances_path(self, user_id: str) -> str:
        return os.path.join(self.voiceprints_dir, f"{user_id}_utterances.npy")

    def _users_path(self) -> str:
        return os.path.join(self.voiceprints_dir, self.USERS_FILENAME)

    # ---- users.json registry --------------------------------------------

    def _read_users(self) -> Dict[str, str]:
        path = self._users_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{self.USERS_FILENAME} at {path!r} is malformed: {exc.msg} at offset {exc.pos}"
                ) from exc

    def _write_users(self, users: Dict[str, str]) -> None:
        with open(self._users_path(), "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, sort_keys=True)

    def register(self, user_id: str, name: str) -> None:
        """Upsert an id→name entry in users.json without touching the .npy file."""
        users = self._read_users()
        users[user_id] = name
        self._write_users(users)

    def get_name(self, user_id: str) -> str:
        """Return the display name for an id. Raises KeyError if not registered."""
        users = self._read_users()
        if user_id not in users:
            raise KeyError(f"id '{user_id}' is not registered in {self.USERS_FILENAME}")
        return users[user_id]

    # ---- enrollment ------------------------------------------------------

    def enroll(self, user_id: str, embedding: np.ndarray) -> None:
        """Add an utterance embedding during enrollment.

        Appends to the utterances file for later finalization.
        """
        utt_path = self._utterances_path(user_id)
        if os.path.exists(utt_path):
            existing = np.load(utt_path)
            updated = np.vstack([existing, embedding.reshape(1, -1)])
        else:
            updated = embedding.reshape(1, -1)
        np.save(utt_path, updated)

    def finalize_enrollment(self, user_id: str) -> None:
        """Compute mean of all utterance embeddings, L2-normalize, save as voiceprint.

        Auto-registers id→id in users.json if no entry exists yet, so callers
        that don't explicitly call register() still produce a discoverable user.
        """
        utt_path = self._utterances_path(user_id)
        if not os.path.exists(utt_path):
            raise FileNotFoundError(f"No utterances found for '{user_id}'")

        utterances = np.load(utt_path)
        mean_emb = np.mean(utterances, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        np.save(self._voiceprint_path(user_id), mean_emb)
        os.remove(utt_path)

        # Auto-register with name == id only if not already registered.
        users = self._read_users()
        if user_id not in users:
            users[user_id] = user_id
            self._write_users(users)

    def utterance_count(self, user_id: str) -> int:
        """Return the number of utterances recorded so far for enrollment."""
        utt_path = self._utterances_path(user_id)
        if os.path.exists(utt_path):
            return len(np.load(utt_path))
        return 0

    def holdout_utterance_embedding(self, user_id: str) -> np.ndarray:
        """Return ONE per-utterance embedding for verify-before-serve (Plan 05).

        Must be called BEFORE finalize_enrollment, which deletes the utterances
        file (os.remove at the end of finalize). Returns the last recorded
        utterance row so the Director can score cosine(primary, holdout) at
        session start. Raises FileNotFoundError if no utterances exist.
        """
        utt_path = self._utterances_path(user_id)
        if not os.path.exists(utt_path):
            raise FileNotFoundError(f"No utterances found for '{user_id}'")
        utterances = np.load(utt_path)
        return np.asarray(utterances[-1], dtype=np.float32)

    # ---- lookup ---------------------------------------------------------

    def get(self, user_id: str) -> Optional[np.ndarray]:
        """Get a single user's voiceprint (registered or orphan). Returns None if no .npy."""
        path = self._voiceprint_path(user_id)
        if os.path.exists(path):
            return np.load(path)
        return None

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return embeddings for all REGISTERED ids only. Orphan .npy files are ignored."""
        result = {}
        for user_id in self.list_users():
            vp = self.get(user_id)
            if vp is not None:
                result[user_id] = vp
        return result

    def list_users(self) -> List[str]:
        """List all registered ids (sorted). Orphan .npy files are ignored."""
        return sorted(self._read_users().keys())

    def delete(self, user_id: str) -> None:
        """Delete a user's voiceprint, any pending utterances, and the users.json entry."""
        for path in [self._voiceprint_path(user_id), self._utterances_path(user_id)]:
            if os.path.exists(path):
                os.remove(path)
        users = self._read_users()
        if user_id in users:
            del users[user_id]
            self._write_users(users)
