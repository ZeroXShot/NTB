"""Undo/redo as two stacks of commands.

Nothing here knows what a command does: applying a command hands back the one
that undoes it, and applying *that* hands back the one that redoes it.
"""

from __future__ import annotations

from ntb.commands.bus import apply_command
from ntb.commands.model import AnyCommand, CommandError
from ntb.ir.document import Document

#: Editing sessions are long; the stack is bounded so memory is too.
DEFAULT_LIMIT = 500


class History:
    """A document plus the edits that led to it."""

    def __init__(self, document: Document, *, limit: int = DEFAULT_LIMIT) -> None:
        if limit < 1:
            raise ValueError("history limit must be at least 1")
        self._document = document
        self._limit = limit
        self._undo: list[AnyCommand] = []
        self._redo: list[AnyCommand] = []

    @property
    def document(self) -> Document:
        return self._document

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def do(self, command: AnyCommand) -> Document:
        """Apply a command and make it undoable. A failed command changes nothing."""
        result = apply_command(self._document, command)
        self._document = result.document
        self._undo.append(result.inverse)
        del self._undo[: max(0, len(self._undo) - self._limit)]
        self._redo.clear()
        return self._document

    def undo(self) -> Document:
        return self._step(self._undo, self._redo, "undo")

    def redo(self) -> Document:
        return self._step(self._redo, self._undo, "redo")

    def reset(self, document: Document) -> Document:
        """Replace the document, e.g. on open. History does not survive it."""
        self._document = document
        self._undo.clear()
        self._redo.clear()
        return self._document

    def _step(self, source: list[AnyCommand], target: list[AnyCommand], what: str) -> Document:
        if not source:
            raise CommandError(f"nothing to {what}")
        result = apply_command(self._document, source.pop())
        self._document = result.document
        target.append(result.inverse)
        return self._document
