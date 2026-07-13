"""Stdlib HTTP server for the tuning console.

Server-side validation mirrors the knob registry (kind, range, choices,
nullable, strict-bool) — the browser page is NOT the trust boundary for the
file the kiosk boots from. Saves go through config_edit.set_values and land
atomically (temp file + os.replace, same directory)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tune import config_edit, knobs
from tune.config_edit import ConfigEditError
from tune.kiosk_proc import KioskProcError

import yaml

_STATIC = Path(__file__).parent / "static"
_LLM_CACHE_S = 3.0
_SSE_PING_S = 15.0


def validate(knob: knobs.Knob, value) -> str | None:
    """Return an error message, or None if the value is acceptable."""
    if value is None:
        return None if knob.nullable else f"{knob.path}: null not allowed"
    if knob.kind == "bool":
        return None if isinstance(value, bool) else f"{knob.path}: expected true/false"
    if knob.kind in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{knob.path}: expected a number"
        if knob.kind == "int" and not float(value).is_integer():
            return f"{knob.path}: expected an integer"
        if not (knob.min <= value <= knob.max):
            return f"{knob.path}: {value} outside [{knob.min}, {knob.max}]"
        return None
    if knob.kind == "select":
        return None if value in knob.choices else \
            f"{knob.path}: {value!r} not one of {list(knob.choices)}"
    if knob.kind in ("text", "textarea"):
        return None if isinstance(value, str) else f"{knob.path}: expected a string"
    return f"{knob.path}: unknown kind {knob.kind}"


def _coerce(knob: knobs.Knob, value):
    """JSON numbers arrive as int OR float; land them as the knob's kind."""
    if value is None or knob.kind not in ("float", "int"):
        return value
    return int(value) if knob.kind == "int" else float(value)


class TuningServer:
    def __init__(self, config_path: str, kproc, host: str = "127.0.0.1",
                 port: int = 8765,
                 llm_url: str = "http://127.0.0.1:8080/v1/models"):
        self.config_path = os.path.abspath(config_path)
        self.kproc = kproc
        self.llm_url = llm_url
        self._llm_cache = (0.0, False)
        self._save_lock = threading.Lock()
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet: the log pane is the product
                pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._safely(self._route_get)

            def do_POST(self):
                self._safely(self._route_post)

            def _safely(self, route):
                try:
                    route()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client went away; nothing to answer
                except Exception as e:  # noqa: BLE001 — last-resort 500
                    try:
                        self._json(500, {
                            "error": f"internal error: {type(e).__name__}: {e}"})
                    except Exception:
                        pass  # headers already sent (e.g. mid-SSE); connection is lost anyway

            def _route_get(self):
                if self.path == "/":
                    return self._index()
                if self.path == "/api/state":
                    return self._json(200, server_ref.state())
                if self.path == "/api/logs":
                    return self._sse()
                self._json(404, {"error": f"no route: {self.path}"})

            def _route_post(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"null")
                except json.JSONDecodeError as e:
                    return self._json(400, {"error": f"bad JSON: {e}"})
                if self.path == "/api/save":
                    status, payload = server_ref.save(body)
                    return self._json(status, payload)
                if self.path == "/api/kiosk/start":
                    return self._kiosk(lambda: server_ref.kproc.start(
                        diag=bool((body or {}).get("diag", True))))
                if self.path == "/api/kiosk/stop":
                    return self._kiosk(server_ref.kproc.stop)
                if self.path == "/api/kiosk/restart":
                    return self._kiosk(server_ref.kproc.restart)
                self._json(404, {"error": f"no route: {self.path}"})

            def _kiosk(self, action):
                try:
                    action()
                except KioskProcError as e:
                    return self._json(409, {"error": str(e)})
                self._json(200, server_ref.kproc.status())

            def _index(self):
                page = _STATIC / "index.html"
                if not page.exists():
                    return self._json(404, {"error": "index.html not built yet"})
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse(self):
                snapshot, q = server_ref.kproc.attach()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for line in snapshot:
                        self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            line = q.get(timeout=_SSE_PING_S)
                            self.wfile.write(f"data: {line}\n\n".encode())
                        except Exception:  # queue.Empty -> keep-alive
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server_ref.kproc.detach(q)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    # ---- app logic (handler-independent, easy to test) ----

    def state(self) -> dict:
        data = yaml.safe_load(Path(self.config_path).read_text())
        rows = knobs.as_json()
        for row in rows:
            row["value"] = config_edit.get_path(data, row["path"])
        return {"knobs": rows, "kiosk": self.kproc.status(),
                "llm": {"reachable": self._llm_reachable()},
                "config_path": self.config_path}

    def save(self, body) -> tuple[int, dict]:
        changes = (body or {}).get("changes")
        if not isinstance(changes, dict) or not changes:
            return 400, {"error": "body must be {\"changes\": {path: value}}"}
        coerced = {}
        for path, value in changes.items():
            knob = knobs.BY_PATH.get(path)
            if knob is None:
                return 400, {"error": f"not a registered knob: {path}"}
            err = validate(knob, value)   # validate the RAW value first —
            if err:                       # coercing 2.5 -> 2 would hide the error
                return 400, {"error": err}
            coerced[path] = _coerce(knob, value)
        with self._save_lock:
            text = Path(self.config_path).read_text()
            try:
                edited = config_edit.set_values(text, coerced)
            except ConfigEditError as e:
                return 409, {"error": str(e)}
            self._write_atomic(edited)
        return 200, {"saved": sorted(coerced)}

    def _write_atomic(self, text: str) -> None:
        d = os.path.dirname(self.config_path)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".config-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.chmod(tmp, os.stat(self.config_path).st_mode & 0o777)
            os.replace(tmp, self.config_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _llm_reachable(self) -> bool:
        ts, val = self._llm_cache
        if time.monotonic() - ts < _LLM_CACHE_S:
            return val
        try:
            with urllib.request.urlopen(self.llm_url, timeout=0.5):
                val = True
        except Exception:
            val = False
        self._llm_cache = (time.monotonic(), val)
        return val

    def serve_forever(self):
        self.httpd.serve_forever()

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
