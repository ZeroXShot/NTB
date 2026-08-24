"""Geometry to connectivity: the four spatial rules.

Pure functions over placed members, so the semantics of "vertical stack" can be
tested without a document, a registry or an emitter anywhere near it.

Every rule is deterministic and order-stable: the same placements produce the
same pairs in the same order, which is what makes generated topologies
reviewable in a diff. Ties in a coordinate are broken by member key.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from ntb.ir.spatial import Axis, SpatialRule, SpatialRuleKind

#: Placements are authored by dragging blocks, so coordinates are never exact.
TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Placed:
    """A member of a rule: what it is called and where it sits."""

    key: str
    pos: tuple[float, float, float]

    def coord(self, axis: Axis) -> float:
        return self.pos[axis.offset]


class RuleError(ValueError):
    """A rule cannot be applied to the members it was given."""


def derive_pairs(rule: SpatialRule, members: Sequence[Placed]) -> tuple[tuple[int, int], ...]:
    """The edges ``rule`` implies, as (source, target) indices into ``members``."""
    if rule.kind is SpatialRuleKind.VERTICAL_STACK:
        pairs = _stack(rule.axis, members)
    elif rule.kind is SpatialRuleKind.AXIS_PROJECTION:
        pairs = _projection(rule.axis, members)
    elif rule.kind is SpatialRuleKind.NEIGHBORHOOD:
        pairs = _neighborhood(rule, members)
    else:
        pairs = _lattice(rule, members)
    return tuple(pairs)


def _order(axis: Axis, members: Sequence[Placed]) -> list[int]:
    """Member indices sorted along ``axis``, ties broken by key."""
    return sorted(range(len(members)), key=lambda i: (members[i].coord(axis), members[i].key))


def _stack(axis: Axis, members: Sequence[Placed]) -> list[tuple[int, int]]:
    order = _order(axis, members)
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def _projection(axis: Axis, members: Sequence[Placed]) -> list[tuple[int, int]]:
    order = _order(axis, members)
    pairs: list[tuple[int, int]] = []
    for position, source in enumerate(order):
        for target in order[position + 1 :]:
            # Strictly ahead: members sharing a coordinate are side by side, not
            # downstream of one another.
            if members[target].coord(axis) - members[source].coord(axis) > TOLERANCE:
                pairs.append((source, target))
    return pairs


def _neighborhood(rule: SpatialRule, members: Sequence[Placed]) -> list[tuple[int, int]]:
    if rule.radius is None:  # pragma: no cover - the model validator requires it
        raise RuleError(f"rule {rule.id!r} is a neighborhood without a radius")
    order = _order(rule.axis, members)
    pairs: list[tuple[int, int]] = []
    for position, source in enumerate(order):
        for target in order[position + 1 :]:
            if _distance(members[source], members[target]) <= rule.radius + TOLERANCE:
                pairs.append((source, target))
                if rule.bidirectional:
                    pairs.append((target, source))
    return pairs


def _lattice(rule: SpatialRule, members: Sequence[Placed]) -> list[tuple[int, int]]:
    steps = tuple(_step(members, axis) for axis in Axis)
    if not any(step is not None for step in steps):
        raise RuleError(f"rule {rule.id!r} is a lattice but its members all sit at the same point")

    pairs: list[tuple[int, int]] = []
    order = _order(rule.axis, members)
    for position, source in enumerate(order):
        for target in order[position + 1 :]:
            axis = _adjacent_axis(members[source], members[target], steps)
            if axis is None:
                continue
            low, high = (source, target)
            if members[source].coord(axis) > members[target].coord(axis):
                low, high = target, source
            pairs.append((low, high))
            if rule.bidirectional:
                pairs.append((high, low))
    return pairs


def _adjacent_axis(left: Placed, right: Placed, steps: tuple[float | None, ...]) -> Axis | None:
    """The axis these two are one grid cell apart on, if there is exactly one."""
    found: Axis | None = None
    for axis in Axis:
        gap = abs(left.coord(axis) - right.coord(axis))
        step = steps[axis.offset]
        if gap <= TOLERANCE:
            continue
        if step is None or abs(gap - step) > TOLERANCE or found is not None:
            return None
        found = axis
    return found


def _step(members: Sequence[Placed], axis: Axis) -> float | None:
    """The grid spacing along ``axis``: the smallest gap between two members."""
    values = sorted(member.coord(axis) for member in members)
    gaps = [b - a for a, b in pairwise(values) if b - a > TOLERANCE]
    return min(gaps) if gaps else None


def _distance(left: Placed, right: Placed) -> float:
    return math.dist(left.pos, right.pos)
