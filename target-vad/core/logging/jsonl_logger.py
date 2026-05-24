"""EventLogger — structured JSONL audit logging (F6).

Shared by KioskPipeline and TalkbackController. Writes one JSON line per
event with auto-injected timestamp, session ID, and event name.
"""

import json
import os
from datetime import datetime, timezone


class EventLogger:
    """Appends structured JSONL events to a per-session log file."""

    def __init__(self, path_template: str, session_id: str):
        self._path_template = path_template
        self._session_id = session_id
        self._current_path: str | None = None
        self._resolve_path()

    def _resolve_path(self) -> None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._current_path = self._path_template.format(
            date=date_str, session_id=self._session_id
        )
        os.makedirs(os.path.dirname(self._current_path), exist_ok=True)

    @property
    def current_path(self) -> str | None:
        return self._current_path

    def start_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._resolve_path()

    def log(self, event: str, payload: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": self._session_id,
            "event": event,
            "payload": payload,
        }
        with open(self._current_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
