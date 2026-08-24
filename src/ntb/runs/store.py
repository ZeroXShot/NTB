"""Where a run's history lives.

SQLite, one file, no server. A run outlives the studio session that started it,
so its metrics have to be somewhere a later session can read them -- and a
metric written per step is the one thing here that has to be cheap.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    document    TEXT NOT NULL,
    status      TEXT NOT NULL,
    config      TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    error       TEXT,
    parameters  INTEGER,
    total_steps INTEGER,
    last_step   INTEGER NOT NULL DEFAULT 0,
    checkpoint  TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id  TEXT NOT NULL,
    step    INTEGER NOT NULL,
    epoch   INTEGER NOT NULL,
    name    TEXT NOT NULL,
    value   REAL NOT NULL,
    seconds REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_by_run ON metrics (run_id, name, step);
"""


class Status(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"

    @property
    def finished(self) -> bool:
        return self is not Status.RUNNING


@dataclass(frozen=True, slots=True)
class Run:
    """One training run, as the studio and the CLI see it."""

    id: str
    document: str
    status: Status
    config: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    error: str | None = None
    parameters: int | None = None
    total_steps: int | None = None
    last_step: int = 0
    checkpoint: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document": self.document,
            "status": self.status.value,
            "config": self.config,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "error": self.error,
            "parameters": self.parameters,
            "totalSteps": self.total_steps,
            "lastStep": self.last_step,
            "checkpoint": self.checkpoint,
        }


class RunStore:
    """The run table and its metrics. Safe to use from the thread that reads a worker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # A worker's reader thread writes while a request reads.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- runs ---------------------------------------------------------------

    def create(self, run_id: str, document: str, config: dict[str, Any]) -> Run:
        started = time.time()
        self._db.execute(
            "INSERT INTO runs (id, document, status, config, started_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, document, Status.RUNNING.value, json.dumps(config, default=str), started),
        )
        self._db.commit()
        return Run(
            id=run_id,
            document=document,
            status=Status.RUNNING,
            config=config,
            started_at=started,
        )

    def update(self, run_id: str, **fields: Any) -> None:
        if not fields:  # pragma: no cover - callers always pass something
            return
        columns = ", ".join(f"{name} = ?" for name in fields)
        self._db.execute(f"UPDATE runs SET {columns} WHERE id = ?", (*fields.values(), run_id))
        self._db.commit()

    def finish(self, run_id: str, status: Status, error: str | None = None) -> None:
        self.update(run_id, status=status.value, ended_at=time.time(), error=error)

    def get(self, run_id: str) -> Run | None:
        row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row is not None else None

    def recent(self, limit: int = 50) -> list[Run]:
        rows = self._db.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_run(row) for row in rows]

    def unfinished(self) -> list[Run]:
        """Runs the database still thinks are going. A crash leaves these behind."""
        rows = self._db.execute(
            "SELECT * FROM runs WHERE status = ?", (Status.RUNNING.value,)
        ).fetchall()
        return [_run(row) for row in rows]

    # -- metrics ------------------------------------------------------------

    def record(self, run_id: str, step: int, epoch: int, name: str, value: float, seconds: float) -> None:
        self._db.execute(
            "INSERT INTO metrics (run_id, step, epoch, name, value, seconds) VALUES (?,?,?,?,?,?)",
            (run_id, step, epoch, name, value, seconds),
        )
        self._db.execute("UPDATE runs SET last_step = ? WHERE id = ?", (step, run_id))
        self._db.commit()

    def metrics(self, run_id: str, name: str = "loss", limit: int = 5000) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT step, epoch, value, seconds FROM metrics "
            "WHERE run_id = ? AND name = ? ORDER BY step LIMIT ?",
            (run_id, name, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def names(self, run_id: str) -> list[str]:
        rows = self._db.execute(
            "SELECT DISTINCT name FROM metrics WHERE run_id = ? ORDER BY name", (run_id,)
        ).fetchall()
        return [row["name"] for row in rows]

    def __iter__(self) -> Iterator[Run]:
        return iter(self.recent())


def _run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        document=row["document"],
        status=Status(row["status"]),
        config=json.loads(row["config"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        error=row["error"],
        parameters=row["parameters"],
        total_steps=row["total_steps"],
        last_step=row["last_step"],
        checkpoint=row["checkpoint"],
    )
