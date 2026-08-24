"""Nodes, ports, edges and generators: the authored graph."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ntb.ir.spatial import Axis, Placement
from ntb.ir.types import Identifier, TensorType


class PortDirection(StrEnum):
    IN = "in"
    OUT = "out"


class Port(BaseModel):
    """A named connection point. Ops declare ports; nodes only carry pinned ones."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Identifier
    direction: PortDirection
    type: TensorType | None = Field(
        default=None,
        description="Author-pinned type. None means 'infer me'.",
    )
    optional: bool = False


class Node(BaseModel):
    """One operation, or one instance of a module, placed in space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    op: Identifier = Field(description="Registry op name, or 'ntb.module' to instantiate a Module.")
    name: str = Field(default="", description="Human label; free text, may be empty.")
    attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Op attributes, validated against the op's attribute schema.",
    )
    ports: tuple[Port, ...] = ()
    placement: Placement = Placement()

    @model_validator(mode="after")
    def _validate_ports(self) -> Node:
        seen: set[tuple[str, str]] = set()
        for port in self.ports:
            key = (port.direction.value, port.name)
            if key in seen:
                raise ValueError(f"node {self.id!r} declares port {port.name!r} twice")
            seen.add(key)
        return self

    def port(self, name: str, direction: PortDirection) -> Port | None:
        for port in self.ports:
            if port.name == name and port.direction is direction:
                return port
        return None


class Endpoint(BaseModel):
    """One end of an edge: a port on a node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node: Identifier
    port: Identifier = "out"

    def __str__(self) -> str:
        return f"{self.node}.{self.port}"


class Edge(BaseModel):
    """An authored connection. Rule-derived edges appear only in the core IR."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    src: Endpoint
    dst: Endpoint

    @model_validator(mode="after")
    def _validate_not_self_loop(self) -> Edge:
        if self.src.node == self.dst.node:
            raise ValueError(f"edge {self.id!r} connects node {self.src.node!r} to itself")
        return self


class Generator(BaseModel):
    """Repeats a module along an axis, parameterising it by the 0-based index ``i``.

    Keeps "N stacked layers" a single object rather than N copied nodes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    module: Identifier = Field(description="Id of the Module to instantiate.")
    count: int = Field(gt=0, le=100_000)
    axis: Axis = Axis.Z
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    step: float = Field(
        default=1.0,
        description="Spacing between consecutive instances along `axis`.",
    )
    attr_bindings: dict[str, str] = Field(
        default_factory=dict,
        description="Attribute name -> expression in the loop variable `i`.",
    )
    chain: bool = Field(
        default=True,
        description="Wire instance i into i+1. False lays them out in parallel.",
    )
    label: str = ""
