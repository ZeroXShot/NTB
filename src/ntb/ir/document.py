"""Modules and documents: the top level of NTB-IR."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ntb.ir.graph import Edge, Endpoint, Generator, Node, Port, PortDirection
from ntb.ir.spatial import SpatialRule
from ntb.ir.types import Identifier

#: On-disk `.ntb` schema version. Bump with a migration in `ntb.ir.migrate`.
SCHEMA_VERSION = 1


class Module(BaseModel):
    """A reusable, parameterisable subgraph.

    Layers, transformer blocks and whole models are all Modules; NTB hardcodes
    no notion of "a layer".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    name: str = ""
    doc: str = ""
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Module-level parameters referenceable from node attributes.",
    )
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()
    input_bindings: dict[Identifier, Endpoint] = Field(
        default_factory=dict,
        description="Boundary port -> where it lands inside. Unbound ports bind by position.",
    )
    output_bindings: dict[Identifier, Endpoint] = Field(
        default_factory=dict,
        description="Boundary port -> where it comes from inside.",
    )
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    spatial_rules: tuple[SpatialRule, ...] = ()
    generators: tuple[Generator, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> Module:
        _reject_duplicates((n.id for n in self.nodes), "node", self.id)
        _reject_duplicates((e.id for e in self.edges), "edge", self.id)
        _reject_duplicates((r.id for r in self.spatial_rules), "spatial rule", self.id)
        _reject_duplicates((g.id for g in self.generators), "generator", self.id)

        for port in self.inputs:
            if port.direction is not PortDirection.IN:
                raise ValueError(f"module {self.id!r}: input port {port.name!r} is not an in port")
        for port in self.outputs:
            if port.direction is not PortDirection.OUT:
                raise ValueError(
                    f"module {self.id!r}: output port {port.name!r} is not an out port"
                )

        for bindings, ports, role in (
            (self.input_bindings, self.inputs, "input"),
            (self.output_bindings, self.outputs, "output"),
        ):
            declared = {port.name for port in ports}
            for name in bindings:
                if name not in declared:
                    raise ValueError(
                        f"module {self.id!r} binds {role} port {name!r}, which it does not declare"
                    )
        return self

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(n.id for n in self.nodes)

    def node(self, node_id: str) -> Node | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


class Document(BaseModel):
    """A complete `.ntb` file.

    Structural validation only; semantic validation lives in :mod:`ntb.validate`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    ntb_version: str = ""
    name: str = ""
    doc: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    root: Identifier = Field(description="Id of the module that is the model itself.")
    modules: tuple[Module, ...] = ()

    @model_validator(mode="after")
    def _validate_modules(self) -> Document:
        _reject_duplicates((m.id for m in self.modules), "module", self.name or "document")
        if self.root not in {m.id for m in self.modules}:
            raise ValueError(f"root module {self.root!r} is not defined in the document")
        return self

    def module(self, module_id: str) -> Module | None:
        for module in self.modules:
            if module.id == module_id:
                return module
        return None

    @property
    def root_module(self) -> Module:
        module = self.module(self.root)
        if module is None:  # pragma: no cover - guaranteed by the validator
            raise LookupError(f"root module {self.root!r} missing")
        return module


def _reject_duplicates(ids: Any, what: str, owner: str) -> None:
    seen: set[str] = set()
    for identifier in ids:
        if identifier in seen:
            raise ValueError(f"{owner!r} declares {what} {identifier!r} twice")
        seen.add(identifier)
