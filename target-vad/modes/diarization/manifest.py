"""Introductions-manifest loader and validator.

The manifest is a JSON array of objects with four required keys:
    id (str, non-empty, unique within manifest)
    name (str, non-empty, may collide across entries)
    start (number, >= 0)
    end (number, > start)

See docs/superpowers/specs/2026-05-15-in-session-enrollment-design.md for the
authoritative schema. Validation is strict: any deviation raises ValueError
with a message that points at the offending entry index.
"""

import json
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    name: str
    start: float
    end: float


def load_manifest(path: str) -> List[ManifestEntry]:
    """Parse and validate the manifest at the given path.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: any validation failure (malformed JSON, missing keys,
            wrong types, invalid ranges, duplicate ids). Message includes the
            entry index where applicable.
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest at {path!r}: malformed JSON at offset {exc.pos}: {exc.msg}") from exc

    if not isinstance(data, list):
        raise ValueError(f"manifest at {path!r}: top-level value must be a JSON array, got {type(data).__name__}")

    entries: List[ManifestEntry] = []
    seen_ids: dict = {}  # id -> first index

    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest at {path!r}: entry {i} must be a JSON object, got {type(raw).__name__}")

        for key in ("id", "name", "start", "end"):
            if key not in raw:
                raise ValueError(f"manifest at {path!r}: entry {i} missing required key {key!r}")

        id_val = raw["id"]
        name_val = raw["name"]
        start_val = raw["start"]
        end_val = raw["end"]

        if not isinstance(id_val, str) or not id_val:
            raise ValueError(f"manifest at {path!r}: entry {i} 'id' must be a non-empty string")
        if not isinstance(name_val, str) or not name_val:
            raise ValueError(f"manifest at {path!r}: entry {i} 'name' must be a non-empty string")
        if not isinstance(start_val, (int, float)) or isinstance(start_val, bool):
            raise ValueError(f"manifest at {path!r}: entry {i} 'start' must be a number")
        if not isinstance(end_val, (int, float)) or isinstance(end_val, bool):
            raise ValueError(f"manifest at {path!r}: entry {i} 'end' must be a number")

        start_f = float(start_val)
        end_f = float(end_val)

        if start_f < 0:
            raise ValueError(f"manifest at {path!r}: entry {i} start must be >= 0, got {start_f}")
        if end_f <= start_f:
            raise ValueError(f"manifest at {path!r}: entry {i} end must be > start (start={start_f}, end={end_f})")

        if id_val in seen_ids:
            first = seen_ids[id_val]
            raise ValueError(f"manifest at {path!r}: duplicate id {id_val!r} at entries [{first}, {i}]")
        seen_ids[id_val] = i

        entries.append(ManifestEntry(id=id_val, name=name_val, start=start_f, end=end_f))

    return entries
