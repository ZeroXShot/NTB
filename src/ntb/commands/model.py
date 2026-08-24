"""Every mutation of a document, as data. See docs/adr/0005.

Each command knows how to apply itself and how to describe its own inverse, so
undo is not a second implementation of editing. Commands carry the index they
removed something from: restoring in place keeps `.ntb` files diff-stable.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ntb.ir.document import Document, Module
from ntb.ir.graph import Edge, Node, Port, PortDirection
from ntb.ir.spatial import Placement
from ntb.ir.types import Identifier

M = TypeVar("M", bound=BaseModel)


class CommandError(Exception):
    """A command that cannot apply to the document it was given."""


class Command(BaseModel):
    """Base for the command union. `kind` is the wire discriminator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        raise NotImplementedError  # pragma: no cover


class AddNode(Command):
    """Insert a node, by default at the end of the module."""

    kind: Literal["add_node"] = "add_node"
    module: Identifier
    node: Node
    index: int | None = None

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        if module.node(self.node.id) is not None:
            raise CommandError(f"module {self.module!r} already has a node {self.node.id!r}")
        nodes = _insert(module.nodes, self.node, self.index)
        inverse = RemoveNode(module=self.module, node=self.node.id)
        return _put(document, _rebuild(module, nodes=nodes)), inverse


class RemoveNode(Command):
    """Remove a node and every edge touching it."""

    kind: Literal["remove_node"] = "remove_node"
    module: Identifier
    node: Identifier

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        node = _node(module, self.node)
        index = module.nodes.index(node)
        touches = [self.node in (e.src.node, e.dst.node) for e in module.edges]
        cut = [(i, e) for i, e in enumerate(module.edges) if touches[i]]
        updated = _rebuild(
            module,
            nodes=tuple(n for n in module.nodes if n.id != self.node),
            edges=tuple(e for i, e in enumerate(module.edges) if not touches[i]),
        )
        inverse = Batch(
            commands=(
                AddNode(module=self.module, node=node, index=index),
                *(Connect(module=self.module, edge=e, index=i) for i, e in cut),
            )
        )
        return _put(document, updated), inverse


class MoveNode(Command):
    """Set a node's placement. Placement is semantic, so this edits the model."""

    kind: Literal["move_node"] = "move_node"
    module: Identifier
    node: Identifier
    placement: Placement

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        node = _node(module, self.node)
        updated = _swap(module, _rebuild(node, placement=self.placement))
        inverse = MoveNode(module=self.module, node=self.node, placement=node.placement)
        return _put(document, updated), inverse


class SetAttrs(Command):
    """Replace a node's attributes wholesale. Partial edits are the caller's job."""

    kind: Literal["set_attrs"] = "set_attrs"
    module: Identifier
    node: Identifier
    attrs: dict[str, Any] = Field(default_factory=dict)

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        node = _node(module, self.node)
        updated = _swap(module, _rebuild(node, attrs=dict(self.attrs)))
        inverse = SetAttrs(module=self.module, node=self.node, attrs=dict(node.attrs))
        return _put(document, updated), inverse


class RenameNode(Command):
    """Change a node's human label, never its id."""

    kind: Literal["rename_node"] = "rename_node"
    module: Identifier
    node: Identifier
    name: str

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        node = _node(module, self.node)
        updated = _swap(module, _rebuild(node, name=self.name))
        inverse = RenameNode(module=self.module, node=self.node, name=node.name)
        return _put(document, updated), inverse


class Connect(Command):
    """Add an edge between two nodes of the same module."""

    kind: Literal["connect"] = "connect"
    module: Identifier
    edge: Edge
    index: int | None = None

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        if any(e.id == self.edge.id for e in module.edges):
            raise CommandError(f"module {self.module!r} already has an edge {self.edge.id!r}")
        for endpoint in (self.edge.src, self.edge.dst):
            _node(module, endpoint.node)
        edges = _insert(module.edges, self.edge, self.index)
        inverse = Disconnect(module=self.module, edge=self.edge.id)
        return _put(document, _rebuild(module, edges=edges)), inverse


class Disconnect(Command):
    """Remove one edge by id."""

    kind: Literal["disconnect"] = "disconnect"
    module: Identifier
    edge: Identifier

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        edge = next((e for e in module.edges if e.id == self.edge), None)
        if edge is None:
            raise CommandError(f"module {self.module!r} has no edge {self.edge!r}")
        index = module.edges.index(edge)
        updated = _rebuild(module, edges=tuple(e for e in module.edges if e.id != self.edge))
        inverse = Connect(module=self.module, edge=edge, index=index)
        return _put(document, updated), inverse


class SetModulePorts(Command):
    """Declare a module's boundary: what the model takes in and gives back."""

    kind: Literal["set_module_ports"] = "set_module_ports"
    module: Identifier
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        for port in self.inputs:
            _require_direction(port, PortDirection.IN, self.module)
        for port in self.outputs:
            _require_direction(port, PortDirection.OUT, self.module)
        updated = _rebuild(module, inputs=self.inputs, outputs=self.outputs)
        inverse = SetModulePorts(module=self.module, inputs=module.inputs, outputs=module.outputs)
        return _put(document, updated), inverse


class AddModule(Command):
    """Add a module to the document."""

    kind: Literal["add_module"] = "add_module"
    module: Module
    index: int | None = None

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        if document.module(self.module.id) is not None:
            raise CommandError(f"document already has a module {self.module.id!r}")
        modules = _insert(document.modules, self.module, self.index)
        inverse = RemoveModule(module=self.module.id)
        return _rebuild(document, modules=modules), inverse


class RemoveModule(Command):
    """Remove a module. The root and any module a generator instantiates stay."""

    kind: Literal["remove_module"] = "remove_module"
    module: Identifier

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        module = _module(document, self.module)
        if module.id == document.root:
            raise CommandError(f"module {self.module!r} is the root; set another root first")
        for other in document.modules:
            for generator in other.generators:
                if generator.module == self.module:
                    raise CommandError(
                        f"module {self.module!r} is instantiated by generator "
                        f"{other.id}/{generator.id}"
                    )
        index = document.modules.index(module)
        updated = _rebuild(
            document, modules=tuple(m for m in document.modules if m.id != self.module)
        )
        return updated, AddModule(module=module, index=index)


class SetRoot(Command):
    """Point the document at the module that is the model itself."""

    kind: Literal["set_root"] = "set_root"
    root: Identifier

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        _module(document, self.root)
        return _rebuild(document, root=self.root), SetRoot(root=document.root)


class SetMetadata(Command):
    """Replace the document's name, doc and free-form metadata."""

    kind: Literal["set_metadata"] = "set_metadata"
    name: str = ""
    doc: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        updated = _rebuild(document, name=self.name, doc=self.doc, metadata=dict(self.metadata))
        inverse = SetMetadata(
            name=document.name, doc=document.doc, metadata=dict(document.metadata)
        )
        return updated, inverse


class Batch(Command):
    """Several commands as one undo step. Applies in order, undoes in reverse."""

    kind: Literal["batch"] = "batch"
    commands: tuple[AnyCommand, ...] = ()
    label: str = ""

    def apply(self, document: Document) -> tuple[Document, AnyCommand]:
        inverses: list[AnyCommand] = []
        for command in self.commands:
            document, inverse = command.apply(document)
            inverses.append(inverse)
        return document, Batch(commands=tuple(reversed(inverses)), label=self.label)


AnyCommand: TypeAlias = Annotated[
    AddNode
    | RemoveNode
    | MoveNode
    | SetAttrs
    | RenameNode
    | Connect
    | Disconnect
    | SetModulePorts
    | AddModule
    | RemoveModule
    | SetRoot
    | SetMetadata
    | Batch,
    Field(discriminator="kind"),
]

Batch.model_rebuild()


def _module(document: Document, module_id: str) -> Module:
    module = document.module(module_id)
    if module is None:
        raise CommandError(f"document has no module {module_id!r}")
    return module


def _node(module: Module, node_id: str) -> Node:
    node = module.node(node_id)
    if node is None:
        raise CommandError(f"module {module.id!r} has no node {node_id!r}")
    return node


def _require_direction(port: Port, direction: PortDirection, module_id: str) -> None:
    if port.direction is not direction:
        raise CommandError(f"module {module_id!r}: port {port.name!r} is not an {direction} port")


def _insert(items: tuple[Any, ...], item: Any, index: int | None) -> tuple[Any, ...]:
    if index is None or index >= len(items):
        return (*items, item)
    if index < 0:
        raise CommandError(f"insert index {index} is negative")
    return (*items[:index], item, *items[index:])


def _swap(module: Module, node: Node) -> Module:
    """Replace a node by id, keeping its position in the module."""
    return _rebuild(module, nodes=tuple(node if n.id == node.id else n for n in module.nodes))


def _put(document: Document, module: Module) -> Document:
    return _rebuild(
        document, modules=tuple(module if m.id == module.id else m for m in document.modules)
    )


def _rebuild(model: M, **changes: Any) -> M:
    """Rebuild a frozen model with changes, re-running validation.

    ``model_copy(update=...)`` would skip it and let a command produce a
    document the constructor would have rejected.
    """
    data = model.model_dump()
    data.update(changes)
    try:
        return type(model).model_validate(data)
    except ValueError as exc:
        raise CommandError(str(exc)) from exc
