"""Ops contributed from outside the repo, found through entry points.

A distribution declares `[project.entry-points."ntb.ops"]`; importing the module
it names registers its ops. From then on a plugin op is an op: the studio
palette, the emitters, the parity harness and the MCP tools all read the same
registry (ADR 13).

Loading third-party code happens once, on first use of the registry, and a
plugin that fails is reported rather than raised -- a broken plugin must not
stop NTB opening a document that does not use it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from ntb.ops.registry import REGISTRY

GROUP = "ntb.ops"

#: Set to disable plugin loading entirely, e.g. to reproduce a bug without one.
DISABLE = "NTB_NO_PLUGINS"

#: Reserved for the ops this repo defines and guarantees the meaning of.
RESERVED = "ntb."


@dataclass(frozen=True, slots=True)
class Plugin:
    """One entry point that loaded, and what it registered."""

    name: str
    value: str
    distribution: str = ""
    ops: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Problem:
    """One entry point that did not load, and why."""

    name: str
    value: str
    reason: str


@dataclass(frozen=True, slots=True)
class Loaded:
    plugins: tuple[Plugin, ...] = ()
    problems: tuple[Problem, ...] = ()


_loaded: Loaded | None = None


def loaded() -> Loaded:
    """Load the plugins once, and answer with what happened."""
    global _loaded
    if _loaded is None:
        _loaded = load()
    return _loaded


def load(points: tuple[EntryPoint, ...] | None = None) -> Loaded:
    """Import every plugin entry point, keeping the failures rather than raising.

    A plugin registers into the one global registry, because that is what makes
    its op an op everywhere. There is no second registry to load it into.
    """
    if os.environ.get(DISABLE):
        return Loaded()
    if points is None:
        points = tuple(entry_points(group=GROUP))

    plugins: list[Plugin] = []
    problems: list[Problem] = []
    for point in points:
        before = set(REGISTRY.names())
        try:
            target = point.load()
        except Exception as exc:  # a plugin's own error, whatever it is
            problems.append(Problem(point.name, point.value, f"{type(exc).__name__}: {exc}"))
            continue
        if callable(target):
            try:
                target()
            except Exception as exc:
                problems.append(Problem(point.name, point.value, f"{type(exc).__name__}: {exc}"))
                continue

        added = tuple(sorted(set(REGISTRY.names()) - before))
        reserved = [name for name in added if name.startswith(RESERVED)]
        if reserved:
            for name in added:
                REGISTRY.discard(name)
            problems.append(
                Problem(
                    point.name,
                    point.value,
                    f"{RESERVED}* is reserved for built-in ops; {reserved[0]!r} is not yours",
                )
            )
            continue
        plugins.append(Plugin(point.name, point.value, _distribution(point), added))
    return Loaded(tuple(plugins), tuple(problems))


def reset() -> None:
    """Forget that loading happened. For tests."""
    global _loaded
    _loaded = None


def _distribution(point: EntryPoint) -> str:
    dist = getattr(point, "dist", None)
    return "" if dist is None else str(dist.name)
