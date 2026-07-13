"""Child-process manager for kiosk.py.

The tuning server owns exactly one kiosk child: Start/Stop/Restart with
TERM-then-KILL semantics (mirrors kiosk-stack.sh term_then_kill), stdout+stderr
pumped by a reader thread into a bounded ring and fanned out to SSE
subscribers. Never starts over a foreign `kiosk.py --talkback` (same
pgrep + /proc/<pid>/comm guard as the stack script); never orphans the
child — the server calls stop() on shutdown."""

from __future__ import annotations

import collections
import os
import queue
import re
import subprocess
import sys
import threading

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class KioskProcError(Exception):
    pass


def default_foreign_pids() -> list[int]:
    """PIDs of python kiosk.py --talkback processes we did not start."""
    out = subprocess.run(["pgrep", "-f", r"kiosk\.py --talkback"],
                         capture_output=True, text=True)
    pids = []
    for tok in out.stdout.split():
        try:
            pid = int(tok)
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
        except (ValueError, OSError):
            continue
        if comm.startswith("python"):
            pids.append(pid)
    return pids


class KioskProcess:
    def __init__(self, cmd=None, cwd=None, term_grace_s=5.0, ring_size=2000,
                 foreign_pids=None):
        self._cmd = list(cmd) if cmd else [sys.executable, "kiosk.py", "--talkback"]
        self._cwd = cwd
        self._grace = term_grace_s
        self._foreign_pids = foreign_pids or default_foreign_pids
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._ring: collections.deque[str] = collections.deque(maxlen=ring_size)
        self._subs: list[queue.Queue] = []
        self._diag = False
        self._pump_started: threading.Event | None = None

    # ---- lifecycle ----

    def start(self, diag: bool) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise KioskProcError(
                    f"kiosk already running (pid {self._proc.pid})")
            own = self._proc.pid if self._proc is not None else None
            foreign = [p for p in self._foreign_pids() if p != own]
            if foreign:
                raise KioskProcError(
                    "a kiosk.py --talkback this server did not start is running "
                    f"(pid {foreign[0]}); stop it first (kiosk-stack.sh stop)")
            env = dict(os.environ,
                       PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
            env.pop("TVAD_DIAG", None)
            if diag:
                env["TVAD_DIAG"] = "1"
            self._proc = subprocess.Popen(
                self._cmd, cwd=self._cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1)
            self._diag = diag
            self._pump_started = threading.Event()
            self._reader = threading.Thread(
                target=self._pump, args=(self._proc,),
                name="kiosk-pump", daemon=True)
            self._reader.start()
        # Wait for pump to start reading (outside lock to avoid blocking other threads)
        self._pump_started.wait(timeout=5.0)

    def stop(self) -> None:
        with self._lock:
            proc, reader = self._proc, self._reader
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._proc = None
            return
        proc.terminate()
        try:
            proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if reader is not None:
            reader.join(timeout=5.0)
        with self._lock:
            self._proc = None

    def restart(self) -> None:
        diag = self._diag
        self.stop()
        self.start(diag=diag)

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {"running": running,
                    "pid": self._proc.pid if running else None,
                    "diag": self._diag if running else False}

    # ---- log fan-out ----

    def attach(self) -> tuple[list[str], queue.Queue]:
        """Snapshot of the ring + a live queue, atomically (no gap/dup)."""
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            snapshot = list(self._ring)
            self._subs.append(q)
        return snapshot, q

    def detach(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _emit(self, line: str) -> None:
        with self._lock:
            self._ring.append(line)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass  # slow client: it still has the ring on reconnect

    def _pump(self, proc: subprocess.Popen) -> None:
        first_line = True
        for raw in proc.stdout:
            self._emit(_ANSI_RE.sub("", raw.rstrip("\n")))
            if first_line and self._pump_started:
                self._pump_started.set()
                first_line = False
        code = proc.wait()
        self._emit(f"[tune] kiosk exited (code {code})")
