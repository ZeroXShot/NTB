"""JSON Schema generation for NTB-IR.

Single-source chain: pydantic models -> schema/ntb-ir-v1.json -> studio
TypeScript types. CI fails if the checked-in copy drifts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ntb.ir.document import SCHEMA_VERSION, Document

SCHEMA_ID = "https://ntb.dev/schema/ntb-ir-v{version}.json"


def schema_filename(version: int = SCHEMA_VERSION) -> str:
    return f"ntb-ir-v{version}.json"


def build_schema(version: int = SCHEMA_VERSION) -> dict[str, Any]:
    """Produce the JSON Schema describing a `.ntb` document."""
    schema = Document.model_json_schema(mode="validation")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID.format(version=version),
        "title": "NTB-IR document",
        "description": (
            "An NTB architecture. Generated from the pydantic models in ntb.ir; "
            "do not edit by hand -- run `ntb schema --write` instead."
        ),
        **schema,
    }


def dumps(version: int = SCHEMA_VERSION) -> str:
    """Canonical JSON text, matching the formatting used for `.ntb` files."""
    return json.dumps(build_schema(version), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write(directory: Path, version: int = SCHEMA_VERSION) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / schema_filename(version)
    path.write_text(dumps(version), encoding="utf-8", newline="\n")
    return path


def is_current(directory: Path, version: int = SCHEMA_VERSION) -> bool:
    """Whether the checked-in schema matches what the models generate."""
    path = directory / schema_filename(version)
    if not path.is_file():
        return False
    return path.read_text(encoding="utf-8") == dumps(version)
