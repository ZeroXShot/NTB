"""The document an agent is editing, and the runs it started.

The workspace wraps the same `Session` the studio uses, so an agent's edits go
through the same command bus, the same validation and the same undo history as
a person's. Nothing here reimplements editing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ntb.commands import AnyCommand
from ntb.emit import EmitError, emit_keras_document, emit_torch_document
from ntb.ir import Document, io
from ntb.server.session import Session, blank_document
from ntb.shapes import infer_shapes
from ntb.spatial import NotResolvable, ResolveError, resolve

if TYPE_CHECKING:
    from ntb.runs import RunManager

DEFAULT_RUNS_ROOT = Path("runs")

#: Diagnostics carried back after an edit. Beyond this, read `ntb://diagnostics`.
MAX_REPORTED = 20


class WorkspaceError(Exception):
    """Something an agent asked for cannot be done to this document."""


class Workspace:
    """One document, its history, and the runs launched from it."""

    def __init__(self, path: Path | None = None, *, runs_root: Path | None = None) -> None:
        self.session = Session()
        if path is not None:
            self.session.open(Path(path))
        self.runs_root = runs_root or DEFAULT_RUNS_ROOT
        self._runs: RunManager | None = None

    @property
    def document(self) -> Document:
        return self.session.document

    # -- editing ------------------------------------------------------------

    def apply(self, command: AnyCommand) -> dict[str, Any]:
        self.session.apply(command)
        return self.report()

    def new(self, name: str) -> dict[str, Any]:
        self.session.open_document(blank_document(name))
        return self.report()

    def open(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise WorkspaceError(f"{path}: no such file")
        self.session.open(path)
        return self.report()

    def save(self, path: Path | None = None) -> Path:
        try:
            return self.session.save(path)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc

    # -- reading ------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """What an agent needs after an edit: whether it broke anything.

        Not the document. The studio gets a whole snapshot because it draws one;
        an agent that wants the document reads `ntb://doc`.
        """
        derived = self.session.derived()
        errors = [d["text"] for d in derived.diagnostics if d["severity"] == "error"]
        warnings = [d["text"] for d in derived.diagnostics if d["severity"] == "warning"]
        root = self.document.root_module
        return {
            "revision": self.session.revision,
            "path": str(self.session.path) if self.session.path else None,
            "dirty": self.session.dirty,
            "name": self.document.name,
            "root": self.document.root,
            "nodes": len(root.nodes),
            "edges": len(root.edges),
            "valid": not errors,
            "errors": errors[:MAX_REPORTED],
            "warnings": warnings[:MAX_REPORTED],
            "canUndo": self.session.snapshot()["canUndo"],
        }

    def source(self, backend: str) -> str:
        """Generated code for a backend that produces text."""
        if backend not in {"torch", "keras"}:
            raise WorkspaceError(f"backend {backend!r} emits no source; use torch or keras")
        emit = emit_torch_document if backend == "torch" else emit_keras_document
        try:
            return emit(self.document).source
        except (EmitError, NotResolvable, ResolveError) as exc:
            raise WorkspaceError(str(exc)) from exc

    def lowered(self) -> dict[str, Any]:
        """The flat graph the backends see, and where each edge came from."""
        try:
            graph = resolve(self.document)
        except (NotResolvable, ResolveError) as exc:
            raise WorkspaceError(str(exc)) from exc
        types = infer_shapes(graph).types
        return {
            "name": graph.name,
            "nodes": [
                {
                    "id": n.id,
                    "op": n.op,
                    "from": n.origin,
                    "type": str(types.get((n.id, "out"), "")),
                }
                for n in graph.nodes
            ],
            "edges": [
                {
                    "src": str(e.src),
                    "dst": str(e.dst),
                    # An edge nobody drew is the point of a spatial rule, and
                    # also the thing to debug, so say which rule made it.
                    "derivedFrom": e.origin.rule or e.origin.generator,
                }
                for e in graph.edges
            ],
        }

    def dump(self) -> str:
        return io.dumps(self.document)

    # -- runs ---------------------------------------------------------------

    def runs(self) -> RunManager:
        """The run manager, built on first use so importing costs nothing."""
        if self._runs is None:
            from ntb.runs import RunManager

            self._runs = RunManager(self.runs_root)
        return self._runs

    def close(self) -> None:
        if self._runs is not None:
            self._runs.close()
            self._runs = None
