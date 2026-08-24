"""Start training runs, watch them, and write down what they say.

The manager owns subprocesses, not training. It launches a worker, reads its
event stream on a thread, records everything in the store, and forwards each
event to whoever is listening -- the studio, the CLI, or nobody.

A run that outlives the session that started it is normal. A manager built on an
existing store finds those runs still marked running and, since their process is
gone with the session, marks them stopped rather than lying about them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ntb.runs.config import RunConfig
from ntb.runs.store import Run, RunStore, Status

#: Called with every event a worker emits, on the reader thread.
Listener = Callable[[str, dict[str, Any]], None]

DEFAULT_ROOT = Path("runs")

#: How long to let a reader thread finish before the store goes away under it.
THREAD_JOIN_SECONDS = 15.0


class RunError(Exception):
    """A run cannot be started, or cannot be acted on."""


class RunManager:
    """Every training run this session knows about."""

    def __init__(self, root: Path | None = None, *, listener: Listener | None = None) -> None:
        self.root = root or DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = RunStore(self.root / "runs.db")
        self.listener = listener
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._threads: dict[str, threading.Thread] = {}
        #: Set while shutting down, so a reader thread stops touching the store.
        self._closing = False

        for run in self.store.unfinished():
            # Its process died with the session that owned it.
            self.store.finish(run.id, Status.STOPPED, "the session that started it ended")

    # -- lifecycle ----------------------------------------------------------

    def start(self, config: RunConfig) -> Run:
        """Launch a worker for ``config`` and return the run it created."""
        if not config.document.is_file():
            raise RunError(f"{config.document}: no such document")

        run_id = uuid.uuid4().hex[:12]
        workdir = self.root / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        config = config.model_copy(update={"workdir": workdir})

        payload = workdir / "config.json"
        payload.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        run = self.store.create(run_id, str(config.document), json.loads(config.model_dump_json()))
        # Our own worker, our own argv: no shell, nothing from the document.
        process = subprocess.Popen(
            [sys.executable, "-m", "ntb.runs.worker", str(payload)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._processes[run_id] = process

        thread = threading.Thread(target=self._read, args=(run_id, process), daemon=True)
        self._threads[run_id] = thread
        thread.start()
        return run

    def stop(self, run_id: str) -> Run:
        """Ask a run to stop. It is a process, so this is a signal, not a request."""
        process = self._processes.get(run_id)
        if process is None or process.poll() is not None:
            raise RunError(f"run {run_id!r} is not running")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged worker
            process.kill()
        self.store.finish(run_id, Status.STOPPED)
        return self._require(run_id)

    def wait(self, run_id: str, timeout: float | None = None) -> Run:
        """Block until a run finishes. For the CLI and for tests, not the server."""
        thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout)
        return self._require(run_id)

    def resume(self, run_id: str, **overrides: Any) -> Run:
        """Start a new run from a finished one's last checkpoint."""
        previous = self._require(run_id)
        if previous.checkpoint is None:
            raise RunError(f"run {run_id!r} left no checkpoint to resume from")
        config = RunConfig.model_validate(previous.config).model_copy(
            update={"resume_from": Path(previous.checkpoint), "workdir": None, **overrides}
        )
        return self.start(config)

    def close(self) -> None:
        """Stop everything still running, then let the readers finish writing.

        Closing the store under a live reader thread is a use-after-close, and
        the thread is the one holding the last words of a run.
        """
        for run_id, process in list(self._processes.items()):
            if process.poll() is None:
                process.terminate()
                self.store.finish(run_id, Status.STOPPED)
        for thread in list(self._threads.values()):
            thread.join(timeout=THREAD_JOIN_SECONDS)
        self._closing = True
        self.store.close()

    # -- reading ------------------------------------------------------------

    def _read(self, run_id: str, process: subprocess.Popen[str]) -> None:
        """Drain the worker's event stream until it ends."""
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A print() from someone's data script. Not fatal, not silent.
                self._notify(run_id, "output", {"text": line})
                continue
            self._handle(run_id, event)

        code = process.wait()
        stderr = (process.stderr.read() if process.stderr else "") or ""
        if self._closing:
            return
        current = self.store.get(run_id)
        if current is not None and current.status is Status.RUNNING:
            if code == 0:
                self.store.finish(run_id, Status.DONE)
            else:
                self.store.finish(run_id, Status.FAILED, stderr.strip()[-2000:] or f"exit {code}")
            self._notify(run_id, "closed", {"code": code})

    def _handle(self, run_id: str, event: dict[str, Any]) -> None:
        if self._closing:
            return
        kind = str(event.get("event", ""))
        if kind == "started":
            self.store.update(
                run_id,
                parameters=event.get("parameters"),
                total_steps=event.get("total_steps"),
            )
        elif kind == "metric":
            self.store.record(
                run_id,
                int(event["step"]),
                int(event.get("epoch", 0)),
                str(event.get("name", "loss")),
                float(event["value"]),
                float(event.get("seconds", 0.0)),
            )
        elif kind == "checkpoint":
            self.store.update(run_id, checkpoint=event.get("path"))
        elif kind == "failed":
            self.store.finish(run_id, Status.FAILED, str(event.get("error", "")))
        elif kind == "finished":
            self.store.finish(run_id, Status.DONE)
        self._notify(run_id, kind, event)

    def _notify(self, run_id: str, kind: str, event: dict[str, Any]) -> None:
        if self.listener is None:
            return
        try:
            self.listener(run_id, {"event": kind, **event})
        except Exception:
            # A broken listener must not take the run down with it.
            pass

    def _require(self, run_id: str) -> Run:
        run = self.store.get(run_id)
        if run is None:
            raise RunError(f"no run {run_id!r}")
        return run

    # -- reading the record -------------------------------------------------

    def recent(self, limit: int = 50) -> list[Run]:
        return self.store.recent(limit)

    def get(self, run_id: str) -> Run | None:
        return self.store.get(run_id)

    def metrics(self, run_id: str, name: str = "loss") -> list[dict[str, Any]]:
        return self.store.metrics(run_id, name)
