"""The one place a document changes.

UI, CLI and the future MCP server are all clients of ``apply_command``: undo,
broadcast and audit are then a single mechanism rather than three. See ADR 0005.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ntb.commands.model import AnyCommand, CommandError
from ntb.ir.document import Document

_ADAPTER: TypeAdapter[AnyCommand] = TypeAdapter(AnyCommand)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The document after the command, and the command that undoes it."""

    document: Document
    inverse: AnyCommand


def apply_command(document: Document, command: AnyCommand) -> CommandResult:
    """Apply one command. The document is never mutated; a new one comes back."""
    updated, inverse = command.apply(document)
    return CommandResult(document=updated, inverse=inverse)


def apply_all(document: Document, commands: tuple[AnyCommand, ...]) -> CommandResult:
    """Apply commands in order as one undo step."""
    from ntb.commands.model import Batch

    return apply_command(document, Batch(commands=commands))


def parse_command(payload: Any) -> AnyCommand:
    """Build a command from a wire payload, rejecting anything unrecognised."""
    try:
        return _ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise CommandError(f"not a valid command: {exc.errors()[0]['msg']}") from exc


def dump_command(command: AnyCommand) -> dict[str, Any]:
    """Serialise a command for the wire or a log."""
    payload: dict[str, Any] = _ADAPTER.dump_python(command, mode="json")
    return payload
