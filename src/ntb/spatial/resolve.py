"""Lower a Document to a CoreGraph.

Phase 1 handles modules and explicit edges. Generators and spatial rules are
recognised but not expanded yet; they raise :class:`NotResolvable` naming the
phase they land in, rather than silently lowering half a model. See
docs/adr/0002.

A module's boundary ports bind to the free ports of its *terminal* nodes, in
declaration order: a node contributes inputs only if nothing feeds any of its
inputs, and outputs only if nothing consumes any of its outputs. Without the
terminal test an unconsumed secondary output -- attention weights, say -- would
capture the module output ahead of the node that actually ends the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ntb.ir.core import CoreEdge, CoreGraph, CoreNode, GraphInput, GraphOutput, Origin
from ntb.ir.document import Document, Module
from ntb.ir.graph import Endpoint, Node
from ntb.ops.registry import REGISTRY, OpRegistry

#: Op name marking a node that instantiates another Module.
MODULE_OP = "ntb.module"

SEPARATOR = "/"


class NotResolvable(Exception):
    """The document uses a feature this build cannot lower yet."""


class ResolveError(Exception):
    """The document is structurally inconsistent and cannot be lowered."""


@dataclass
class _Boundary:
    """A module's own ports, mapped to endpoints inside it."""

    inputs: dict[str, Endpoint] = field(default_factory=dict)
    outputs: dict[str, Endpoint] = field(default_factory=dict)


def resolve(document: Document, *, registry: OpRegistry = REGISTRY) -> CoreGraph:
    """Flatten ``document`` into a graph of primitive ops."""
    lowering = _Lowering(document, registry)
    root = lowering.expand(document.root_module, prefix="", stack=())

    inputs: list[GraphInput] = []
    for port in document.root_module.inputs:
        endpoint = root.inputs.get(port.name)
        if endpoint is not None and port.type is not None:
            inputs.append(GraphInput(name=port.name, endpoint=endpoint, type=port.type))

    outputs = tuple(
        GraphOutput(name=port.name, endpoint=endpoint)
        for port in document.root_module.outputs
        if (endpoint := root.outputs.get(port.name)) is not None
    )
    return CoreGraph(
        name=document.name or document.root,
        nodes=tuple(lowering.nodes),
        edges=tuple(lowering.edges),
        inputs=tuple(inputs),
        outputs=outputs,
    )


class _Lowering:
    def __init__(self, document: Document, registry: OpRegistry) -> None:
        self.document = document
        self.registry = registry
        self.nodes: list[CoreNode] = []
        self.edges: list[CoreEdge] = []
        self._serial = 0

    def expand(self, module: Module, *, prefix: str, stack: tuple[str, ...]) -> _Boundary:
        """Emit ``module`` under ``prefix`` and return how its ports bind."""
        if module.id in stack:
            raise ResolveError(
                "module instantiation is recursive: " + " -> ".join((*stack, module.id))
            )
        self._reject_unsupported(module)

        children: dict[str, _Boundary] = {}
        for node in module.nodes:
            qualified = _join(prefix, node.id)
            if node.op == MODULE_OP:
                children[node.id] = self.expand(
                    self._target_module(module, node),
                    prefix=qualified,
                    stack=(*stack, module.id),
                )
            else:
                self.nodes.append(
                    CoreNode(
                        id=qualified,
                        op=node.op,
                        attrs=dict(node.attrs),
                        origin=Origin(module=module.id, node=node.id),
                    )
                )

        for edge in module.edges:
            source = self._resolve_endpoint(module, edge.src, prefix, children, produces=True)
            target = self._resolve_endpoint(module, edge.dst, prefix, children, produces=False)
            self._connect(source, target, Origin(module=module.id, node=edge.id))

        return self._bind(module, prefix=prefix, children=children)

    def _reject_unsupported(self, module: Module) -> None:
        for feature, present in (
            ("Generator", module.generators),
            ("SpatialRule", module.spatial_rules),
        ):
            if present:
                raise NotResolvable(
                    f"module {module.id!r} uses a {feature}; expansion ships in phase 4"
                )

    def _target_module(self, module: Module, node: Node) -> Module:
        target = node.attrs.get("module")
        if not isinstance(target, str):
            raise ResolveError(
                f"node {node.id!r} in module {module.id!r} is a {MODULE_OP} but its "
                "'module' attribute is missing or not a string"
            )
        child = self.document.module(target)
        if child is None:
            raise ResolveError(
                f"node {node.id!r} in module {module.id!r} instantiates unknown module {target!r}"
            )
        return child

    def _resolve_endpoint(
        self,
        module: Module,
        endpoint: Endpoint,
        prefix: str,
        children: dict[str, _Boundary],
        *,
        produces: bool,
    ) -> Endpoint:
        node = module.node(endpoint.node)
        if node is None:
            raise ResolveError(f"module {module.id!r} references unknown node {endpoint.node!r}")
        if node.op != MODULE_OP:
            return Endpoint(node=_join(prefix, endpoint.node), port=endpoint.port)

        boundary = children[endpoint.node]
        side = boundary.outputs if produces else boundary.inputs
        inner = side.get(endpoint.port)
        if inner is None:
            role = "output" if produces else "input"
            raise ResolveError(
                f"module instance {endpoint.node!r} has no bound {role} port {endpoint.port!r}"
            )
        return inner

    def _bind(self, module: Module, *, prefix: str, children: dict[str, _Boundary]) -> _Boundary:
        consumed = {(e.dst.node, e.dst.port) for e in self.edges}
        produced = {(e.src.node, e.src.port) for e in self.edges}

        free_in: list[Endpoint] = []
        free_out: list[Endpoint] = []
        for node in module.nodes:
            if node.op == MODULE_OP:
                child = children[node.id]
                candidates_in = list(child.inputs.values())
                candidates_out = list(child.outputs.values())
            else:
                qualified = _join(prefix, node.id)
                spec = self.registry.get(node.op)
                in_names = [p.name for p in spec.inputs] if spec else ["in"]
                out_names = [p.name for p in spec.outputs] if spec else ["out"]
                candidates_in = [Endpoint(node=qualified, port=n) for n in in_names]
                candidates_out = [Endpoint(node=qualified, port=n) for n in out_names]

            if not any((e.node, e.port) in consumed for e in candidates_in):
                free_in.extend(candidates_in)
            if not any((e.node, e.port) in produced for e in candidates_out):
                free_out.extend(candidates_out)

        return _Boundary(
            inputs=_match(module, "input", [p.name for p in module.inputs], free_in),
            outputs=_match(module, "output", [p.name for p in module.outputs], free_out),
        )

    def _connect(self, source: Endpoint, target: Endpoint, origin: Origin) -> None:
        self._serial += 1
        self.edges.append(CoreEdge(id=f"e{self._serial}", src=source, dst=target, origin=origin))


def _match(
    module: Module, role: str, names: list[str], free: list[Endpoint]
) -> dict[str, Endpoint]:
    if not names:
        return {}
    if len(free) < len(names):
        raise ResolveError(
            f"module {module.id!r} declares {len(names)} {role} port(s) but only "
            f"{len(free)} free node port(s) are available to bind them"
        )
    return dict(zip(names, free[: len(names)], strict=True))


def _join(prefix: str, name: str) -> str:
    return f"{prefix}{SEPARATOR}{name}" if prefix else name
