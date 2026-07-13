"""Targeted line edits to config.yaml.

Only the scalar value on a known key's line changes; every comment and all
formatting stay byte-identical. Pure function, all-or-nothing: any locate,
render, or round-trip failure raises ConfigEditError and nothing is returned.
The caller (tune.server) writes the returned text atomically."""

from __future__ import annotations

import json
import re

import yaml


class ConfigEditError(Exception):
    pass


_KEY_RE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z_][\w-]*):(?P<after>.*)$")
_BLOCK_HEADERS = ("|", "|-", "|+", ">", ">-", ">+")


def get_path(data: dict, path: str):
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigEditError(f"unknown config path: {path}")
        node = node[part]
    return node


def set_values(text: str, changes: dict[str, object]) -> str:
    if not changes:
        return text
    original = yaml.safe_load(text)
    for path, value in changes.items():
        if isinstance(get_path(original, path), dict):
            raise ConfigEditError(f"{path}: is a section, not a value")
        _render(value, path)  # type-check before touching anything
    lines = text.split("\n")
    found: dict[str, tuple[int, int, str]] = {}
    for idx, path, indent, after in _iter_key_lines(lines):
        if path in changes:
            if path in found:
                raise ConfigEditError(f"{path}: multiple lines match")
            found[path] = (idx, indent, after)
    missing = sorted(set(changes) - set(found))
    if missing:
        raise ConfigEditError("could not locate line(s) for: " + ", ".join(missing))
    # bottom-up so a block-scalar replacement can't shift later indices
    for path in sorted(found, key=lambda p: found[p][0], reverse=True):
        idx, indent, after = found[path]
        if after.strip() in _BLOCK_HEADERS:
            lines = _replace_block(lines, idx, indent, changes[path], path)
        else:
            lines[idx] = _replace_scalar(lines[idx], changes[path], path)
    edited = "\n".join(lines)
    _verify(original, edited, changes)
    return edited


def _iter_key_lines(lines):
    """Yield (index, dotted-path, indent, value-part) for every mapping key,
    tracking nesting by indentation and skipping block-scalar bodies."""
    stack: list[tuple[int, str]] = []
    block_key_indent = None  # indent of an open block scalar's key, else None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if block_key_indent is not None:
            if indent > block_key_indent:
                continue  # inside the block scalar body
            block_key_indent = None
        m = _KEY_RE.match(line)
        if not m:
            continue  # flow continuation / list item — never a registry target
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m["key"]))
        after = m["after"]
        if after.strip() in _BLOCK_HEADERS:
            block_key_indent = indent
        yield i, ".".join(k for _, k in stack), indent, after


def _render(value, path: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # double-quoted, matches the file's style
    raise ConfigEditError(
        f"{path}: unsupported value type {type(value).__name__}")


def _replace_scalar(line: str, value, path: str) -> str:
    m = re.match(r"^(?P<pre>[ ]*[A-Za-z_][\w-]*:[ \t]*)(?P<rest>.*)$", line)
    rest = m["rest"]
    if not rest or rest.lstrip().startswith("#"):
        raise ConfigEditError(f"{path}: no inline value on its line")
    if rest[0] in "\"'":
        end = rest.find(rest[0], 1)
        if end == -1:
            raise ConfigEditError(f"{path}: unterminated quote")
        tail = rest[end + 1:]
    else:
        cm = re.search(r"[ \t]+#", rest)
        end = cm.start() if cm else len(rest.rstrip())
        tail = rest[end:]
    return m["pre"] + _render(value, path) + tail


def _replace_block(lines, idx, key_indent, value, path):
    if not isinstance(value, str):
        raise ConfigEditError(f"{path}: block scalar needs a string")
    end = idx + 1
    while end < len(lines) and (
            not lines[end].strip()
            or len(lines[end]) - len(lines[end].lstrip(" ")) > key_indent):
        end += 1
    while end > idx + 1 and not lines[end - 1].strip():
        end -= 1  # trailing blank lines belong to the document, not the block
    pad = " " * (key_indent + 2)
    body = [pad + l if l else "" for l in value.rstrip("\n").split("\n")]
    return lines[:idx + 1] + body + lines[end:]


def _verify(original: dict, edited_text: str, changes: dict) -> None:
    try:
        after = yaml.safe_load(edited_text)
    except yaml.YAMLError as e:
        raise ConfigEditError(f"edit produced unparseable YAML: {e}") from e
    for path, want in changes.items():
        got = get_path(after, path)
        if _norm(got) != _norm(want):
            raise ConfigEditError(
                f"{path}: round-trip mismatch ({got!r} != {want!r})")
    for path, was in _leaves(original):
        if path in changes:
            continue
        if get_path(after, path) != was:
            raise ConfigEditError(
                f"{path}: unintended change ({was!r} -> {get_path(after, path)!r})")


def _norm(v):
    return v.rstrip("\n") if isinstance(v, str) else v


def _leaves(data: dict, prefix: str = ""):
    for k, v in data.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _leaves(v, p)
        else:
            yield p, v
