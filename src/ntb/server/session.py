"""One editing session: the authoritative document and what follows from it.

The server owns the document; clients send commands and get snapshots back.
A snapshot carries the whole document rather than a patch. For v1 that keeps a
single implementation of what a command means -- the frontend never has to
reimplement the bus in TypeScript to stay in sync.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ntb.commands import AnyCommand, History
from ntb.emit import EmitError, emit_torch_document
from ntb.ir import Document, Module, io
from ntb.shapes import infer_shapes
from ntb.spatial import NotResolvable, ResolveError, resolve
from ntb.validate import Diagnostic, validate

EMPTY_MODULE = "model"


def blank_document(name: str = "untitled") -> Document:
    """The document a studio starts from: one empty root module."""
    return Document(name=name, root=EMPTY_MODULE, modules=(Module(id=EMPTY_MODULE),))


@dataclass(frozen=True, slots=True)
class Derived:
    """Everything computed from the document. Never authored, never saved."""

    diagnostics: tuple[dict[str, Any], ...] = ()
    types: dict[str, str] = field(default_factory=dict)
    code: str = ""
    code_error: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "diagnostics": list(self.diagnostics),
            "types": self.types,
            "code": self.code,
            "codeError": self.code_error,
        }


class Session:
    """A document being edited, plus its undo history and its file."""

    def __init__(self, document: Document | None = None, path: Path | None = None) -> None:
        self._history = History(document if document is not None else blank_document())
        self._path = path
        self._revision = 0
        self._saved_revision = 0
        self._derived: tuple[int, Derived] | None = None

    @property
    def document(self) -> Document:
        return self._history.document

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._revision != self._saved_revision

    def apply(self, command: AnyCommand) -> Document:
        document = self._history.do(command)
        self._revision += 1
        return document

    def undo(self) -> Document:
        document = self._history.undo()
        self._revision += 1
        return document

    def redo(self) -> Document:
        document = self._history.redo()
        self._revision += 1
        return document

    def open(self, path: Path) -> Document:
        return self.open_document(io.load(path), path)

    def open_document(self, document: Document, path: Path | None = None) -> Document:
        """Replace the session's document, e.g. with a blank one."""
        self._history.reset(document)
        self._path = path
        self._revision += 1
        self._saved_revision = self._revision if path is not None else -1
        return document

    def save(self, path: Path | None = None) -> Path:
        target = path or self._path
        if target is None:
            raise ValueError("this session has no file yet; pass a path to save it")
        io.save(self.document, target)
        self._path = target
        self._saved_revision = self._revision
        return target

    def derived(self) -> Derived:
        """Diagnostics, inferred types and generated code, computed once per edit."""
        if self._derived is not None and self._derived[0] == self._revision:
            return self._derived[1]
        derived = _derive(self.document)
        self._derived = (self._revision, derived)
        return derived

    def snapshot(self) -> dict[str, Any]:
        """The whole client-visible state."""
        return {
            "revision": self._revision,
            "document": json.loads(io.dumps(self.document)),
            "path": str(self._path) if self._path else None,
            "dirty": self.dirty,
            "canUndo": self._history.can_undo,
            "canRedo": self._history.can_redo,
            "derived": self.derived().as_json(),
        }


def _derive(document: Document) -> Derived:
    report = validate(document)
    diagnostics = tuple(_diagnostic(d) for d in report.diagnostics)

    types: dict[str, str] = {}
    try:
        graph = resolve(document)
    except (NotResolvable, ResolveError):
        graph = None
    if graph is not None:
        shapes = infer_shapes(graph)
        types = {f"{node}.{port}": str(tensor) for (node, port), tensor in shapes.types.items()}

    code, code_error = "", ""
    if report.ok:
        try:
            code = emit_torch_document(document).source
        except (EmitError, NotResolvable, ResolveError) as exc:
            code_error = str(exc)
    else:
        code_error = "fix the errors above to see the generated code"
    return Derived(diagnostics=diagnostics, types=types, code=code, code_error=code_error)


def _diagnostic(diagnostic: Diagnostic) -> dict[str, Any]:
    location = diagnostic.location
    return {
        "code": diagnostic.code.value,
        "severity": diagnostic.severity.value,
        "message": diagnostic.message,
        "module": location.module,
        "node": location.node,
        "port": location.port,
        "edge": location.edge,
        "text": str(diagnostic),
    }
