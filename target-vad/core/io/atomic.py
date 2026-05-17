"""Shared atomic JSON writer - temp file + os.replace for crash-safe writes.

Used by transcribe.py (Phase 2A), sentiment.py (Phase 2B), metrics.py (Phase 3),
and prosody.py (Phase 4). The pattern was duplicated across those entry points
pre-extraction; this module deduplicates it so all consumers get identical
crash-safety semantics (failed write leaves original file untouched).
"""

import json
import os
import tempfile


def atomic_write_json(path: str, data: dict) -> None:
    """Write `data` as JSON to `path` atomically via tempfile + rename.

    Creates a `.tmp.*` file in the same directory as `path`, writes the JSON
    content with `indent=2` and UTF-8 encoding, then atomically renames the
    temp file over `path`. If anything fails before the rename, the temp file
    is removed and the original `path` is left untouched.
    """
    dirname = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", suffix=".json", dir=dirname)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
