"""Forward migrations between `.ntb` schema versions.

To add one: write ``_migrate_1_to_2(payload)``, register it below, bump
``SCHEMA_VERSION``. Migrations take raw dicts; the models describe only the
current version.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ntb.ir.document import SCHEMA_VERSION

Payload = dict[str, Any]

#: Maps a source version to the function producing version ``source + 1``.
_MIGRATIONS: dict[int, Callable[[Payload], Payload]] = {}


def migrate(payload: Payload, *, from_version: int) -> Payload:
    """Bring a raw document payload up to :data:`SCHEMA_VERSION`."""
    version = from_version
    while version < SCHEMA_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            raise ValueError(
                f"no migration from schema v{version} to v{version + 1}; "
                "this document cannot be opened by this build"
            )
        payload = step(payload)
        version += 1
        payload["schema_version"] = version
    return payload
