"""The studio wire protocol.

Client messages are commands and session verbs; the server answers with the
whole session state or with an error. Keeping the message set this small is what
lets the MCP server in phase 7 be another client rather than another API.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ntb.commands import AnyCommand, CommandError


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ApplyMessage(_Message):
    """Run a command through the bus."""

    type: Literal["command"] = "command"
    command: AnyCommand


class UndoMessage(_Message):
    type: Literal["undo"] = "undo"


class RedoMessage(_Message):
    type: Literal["redo"] = "redo"


class SaveMessage(_Message):
    type: Literal["save"] = "save"
    path: str | None = None


class OpenMessage(_Message):
    type: Literal["open"] = "open"
    path: str


class NewMessage(_Message):
    """Discard the session and start from an empty document."""

    type: Literal["new"] = "new"
    name: str = "untitled"


class RefreshMessage(_Message):
    """Ask for the current state, e.g. after reconnecting."""

    type: Literal["refresh"] = "refresh"


ClientMessage: TypeAlias = Annotated[
    ApplyMessage
    | UndoMessage
    | RedoMessage
    | SaveMessage
    | OpenMessage
    | NewMessage
    | RefreshMessage,
    Field(discriminator="type"),
]

_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_message(payload: Any) -> ClientMessage:
    try:
        return _ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise CommandError(f"not a valid message: {exc.errors()[0]['msg']}") from exc


def state_message(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {"type": "state", **snapshot}


def error_message(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}
