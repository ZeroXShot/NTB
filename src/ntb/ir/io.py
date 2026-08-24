"""Reading and writing `.ntb`: canonical JSON, atomic saves. See docs/adr/0008."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ntb.ir.document import SCHEMA_VERSION, Document
from ntb.ir.migrate import migrate

SUFFIX = ".ntb"


class DocumentError(Exception):
    """A `.ntb` file could not be read as a document."""


def dumps(document: Document) -> str:
    """Serialise to canonical JSON text."""
    payload = document.model_dump(mode="json", exclude_defaults=False)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def loads(text: str, *, source: str = "<string>") -> Document:
    """Parse canonical JSON text, migrating older schema versions forward."""
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentError(f"{source}: not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DocumentError(f"{source}: expected a JSON object at the top level")

    version = payload.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int):
        raise DocumentError(f"{source}: schema_version must be an integer, got {version!r}")
    if version > SCHEMA_VERSION:
        raise DocumentError(
            f"{source}: written by a newer NTB (schema v{version}, this build reads "
            f"up to v{SCHEMA_VERSION}). Upgrade with: pip install --upgrade ntb"
        )
    payload = migrate(payload, from_version=version)
    return Document.model_validate(payload)


def load(path: str | os.PathLike[str]) -> Document:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DocumentError(f"{path}: cannot read: {exc}") from exc
    return loads(text, source=str(path))


def save(document: Document, path: str | os.PathLike[str]) -> None:
    """Write atomically. The temp file shares the destination filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(document)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
