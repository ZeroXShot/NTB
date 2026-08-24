"""Geometry as semantics: placement, and rules that derive edges from it.

``SpatialRule`` is a closed, versioned set, not an open DSL. See docs/adr/0002.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ntb.ir.types import Identifier


class Axis(StrEnum):
    """A world axis. ``Z`` is the vertical stacking axis by convention."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def offset(self) -> int:
        """Index into a 3-tuple. Not ``index``: that would shadow ``str.index``."""
        return {"x": 0, "y": 1, "z": 2}[self.value]


class Orientation(StrEnum):
    """Which way a block faces, i.e. along which axis it consumes and produces."""

    ALONG_X = "along_x"
    ALONG_Y = "along_y"
    ALONG_Z = "along_z"

    @property
    def axis(self) -> Axis:
        return Axis(self.value.removeprefix("along_"))


class Placement(BaseModel):
    """Where a node sits. ``pos`` is the block centre, ``extent`` its size."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0)
    orient: Orientation = Orientation.ALONG_X

    @model_validator(mode="after")
    def _validate_extent(self) -> Placement:
        if any(e <= 0.0 for e in self.extent):
            raise ValueError(f"extent must be positive on every axis, got {self.extent}")
        return self

    def coord(self, axis: Axis) -> float:
        return self.pos[axis.offset]

    def size(self, axis: Axis) -> float:
        return self.extent[axis.offset]


class SpatialRuleKind(StrEnum):
    """The closed set of ways geometry may generate connectivity."""

    #: Connect nodes in ascending order along one axis, each to the next.
    VERTICAL_STACK = "vertical_stack"
    #: Connect every node to every node strictly ahead of it along one axis.
    AXIS_PROJECTION = "axis_projection"
    #: Connect nodes whose centres are within ``radius`` of one another.
    NEIGHBORHOOD = "neighborhood"
    #: Connect nodes that are adjacent cells of a regular grid.
    LATTICE = "lattice"


class SpatialRule(BaseModel):
    """Derives edges from where its ``members`` sit. Never invents nodes.

    Resolution is deterministic: same placements, same edges, same order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    kind: SpatialRuleKind
    members: tuple[Identifier, ...] = Field(
        description="Ids of the nodes this rule wires, within the owning module."
    )
    axis: Axis = Axis.Z
    radius: float | None = Field(
        default=None,
        gt=0.0,
        description="Euclidean radius for NEIGHBORHOOD; ignored by other kinds.",
    )
    output_port: Identifier = Field(
        default="out",
        description="Port on the source node that generated edges leave from.",
    )
    input_port: Identifier = Field(
        default="in",
        description="Port on the target node that generated edges arrive at.",
    )
    bidirectional: bool = Field(
        default=False,
        description="NEIGHBORHOOD and LATTICE only; the ordered kinds would cycle.",
    )
    label: str = ""

    @model_validator(mode="after")
    def _validate_kind_arguments(self) -> SpatialRule:
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"spatial rule {self.id!r} lists a member twice")
        # One member is legal because a member may be a generator id, which
        # stands for all of its instances. Resolution rejects a rule that turns
        # out to cover fewer than two blocks.
        if not self.members:
            raise ValueError(f"spatial rule {self.id!r} lists no members")

        if self.kind is SpatialRuleKind.NEIGHBORHOOD:
            if self.radius is None:
                raise ValueError(f"spatial rule {self.id!r} of kind neighborhood needs a radius")
        elif self.radius is not None:
            raise ValueError(
                f"spatial rule {self.id!r} of kind {self.kind.value} does not take a radius"
            )

        ordered = {SpatialRuleKind.VERTICAL_STACK, SpatialRuleKind.AXIS_PROJECTION}
        if self.bidirectional and self.kind in ordered:
            raise ValueError(
                f"spatial rule {self.id!r} of kind {self.kind.value} cannot be bidirectional: "
                "it would introduce a cycle"
            )
        return self
