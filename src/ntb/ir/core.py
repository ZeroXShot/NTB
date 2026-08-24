"""NTB-Core IR: the lowered, flat, geometry-free DAG.

Produced by ``ntb.spatial.resolve``; consumed by shape inference and the
emitters. ``Origin`` points every element back at what the user drew.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ntb.ir.graph import Endpoint
from ntb.ir.types import Identifier, TensorType


class Origin(BaseModel):
    """Where a lowered element came from in the authored document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: Identifier
    node: Identifier | None = None
    generator: Identifier | None = None
    instance: int | None = Field(default=None, ge=0)
    rule: Identifier | None = None

    def __str__(self) -> str:
        parts = [self.module]
        if self.generator is not None:
            parts.append(f"{self.generator}[{self.instance}]")
        if self.node is not None:
            parts.append(self.node)
        if self.rule is not None:
            parts.append(f"via {self.rule}")
        return "/".join(parts)


class CoreNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    op: Identifier
    attrs: dict[str, Any] = Field(default_factory=dict)
    origin: Origin


class CoreEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    src: Endpoint
    dst: Endpoint
    origin: Origin


class GraphInput(BaseModel):
    """A model input: the name the author gave it and where it lands."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Identifier
    endpoint: Endpoint
    type: TensorType


class GraphOutput(BaseModel):
    """A model output: the name the author gave it and where it comes from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Identifier
    endpoint: Endpoint


class CoreGraph(BaseModel):
    """A flat DAG ready for shape inference and emission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    nodes: tuple[CoreNode, ...] = ()
    edges: tuple[CoreEdge, ...] = ()
    inputs: tuple[GraphInput, ...] = ()
    outputs: tuple[GraphOutput, ...] = ()

    @model_validator(mode="after")
    def _validate_references(self) -> CoreGraph:
        known = {n.id for n in self.nodes}
        if len(known) != len(self.nodes):
            raise ValueError("core graph contains duplicate node ids")
        for edge in self.edges:
            for end, role in ((edge.src, "src"), (edge.dst, "dst")):
                if end.node not in known:
                    raise ValueError(
                        f"core edge {edge.id!r} {role} references unknown node {end.node!r}"
                    )
        return self

    def node(self, node_id: str) -> CoreNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def incoming(self, node_id: str) -> tuple[CoreEdge, ...]:
        return tuple(e for e in self.edges if e.dst.node == node_id)

    def outgoing(self, node_id: str) -> tuple[CoreEdge, ...]:
        return tuple(e for e in self.edges if e.src.node == node_id)

    def topological_order(self) -> tuple[CoreNode, ...]:
        """Nodes in dependency order; ties break by id. Raises CycleError."""
        indegree: dict[str, int] = {n.id: 0 for n in self.nodes}
        successors: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            # Parallel edges each count, or a node is released too early.
            successors[edge.src.node].append(edge.dst.node)
            indegree[edge.dst.node] += 1

        ready = deque(sorted(n for n, deg in indegree.items() if deg == 0))
        by_id = {n.id: n for n in self.nodes}
        ordered: list[CoreNode] = []
        while ready:
            current = ready.popleft()
            ordered.append(by_id[current])
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)

        if len(ordered) != len(self.nodes):
            remaining = frozenset(indegree) - {n.id for n in ordered}
            raise CycleError(remaining)
        return tuple(ordered)


class CycleError(Exception):
    """The core graph contains a cycle."""

    def __init__(self, nodes: frozenset[str]) -> None:
        self.nodes = nodes
        listed = ", ".join(sorted(nodes))
        super().__init__(f"graph contains a cycle involving: {listed}")
