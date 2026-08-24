"""Lower a Document to a CoreGraph.

This is where geometry stops being semantic. Generators expand into instances,
spatial rules become explicit edges, module instances are inlined, and what
comes out is a flat typed DAG with no notion of space at all -- so no backend
ever learns that NTB has three dimensions. See docs/adr/0002.

Two conventions the rest of the system depends on:

*Module boundaries.* A port bound in ``input_bindings`` or ``output_bindings``
lands exactly where the author said. Everything else falls back to position: the
free ports of the module's *terminal* members, in declaration order, where a
member contributes inputs only if nothing feeds any of its inputs and outputs
only if nothing consumes any of its outputs. Without the terminal test an
unconsumed secondary output -- attention weights, say -- would capture the module
output ahead of the node that ends the chain. Positional binding is convenient
and silent; write the binding down whenever the module has more than one
plausible answer.

*Parameters.* A node attribute written as ``"$expr"`` is evaluated against the
module's parameters (see :mod:`ntb.spatial.expr`). A generator adds the instance
index ``i`` to them, which is what makes a stack of blocks that widen with depth
a single object rather than twelve copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ntb.ir.core import CoreEdge, CoreGraph, CoreNode, GraphInput, GraphOutput, Origin
from ntb.ir.document import Document, Module
from ntb.ir.graph import Endpoint, Generator, Node, Port
from ntb.ir.spatial import SpatialRule
from ntb.ops.registry import REGISTRY, OpRegistry
from ntb.spatial.expr import ExpressionError, contains_expression, resolve_attrs
from ntb.spatial.rules import Placed, RuleError, derive_pairs

#: Op name marking a node that instantiates another Module.
MODULE_OP = "ntb.module"

SEPARATOR = "/"

#: Loop variable a generator exposes to the module it repeats.
INDEX_PARAM = "i"


class NotResolvable(Exception):
    """The document uses a feature this build cannot lower.

    Nothing raises it today -- every authored feature lowers. It stays as the
    reserved path for a document written by a newer NTB, and callers already
    handle it.
    """


class ResolveError(Exception):
    """The document is structurally inconsistent and cannot be lowered."""


@dataclass
class _Boundary:
    """A module's own ports, mapped to endpoints inside it."""

    inputs: dict[str, Endpoint] = field(default_factory=dict)
    outputs: dict[str, Endpoint] = field(default_factory=dict)


@dataclass
class _Member:
    """Something a module contains: a node, a module instance, or one instance
    of a generator. Rules and boundary binding treat all three alike."""

    key: str
    pos: tuple[float, float, float]
    boundary: _Boundary
    inlined: bool = False


def resolve(document: Document, *, registry: OpRegistry = REGISTRY) -> CoreGraph:
    """Flatten ``document`` into a graph of primitive ops."""
    lowering = _Lowering(document, registry)
    root_module = document.root_module
    root = lowering.expand(root_module, prefix="", stack=(), params=root_module.params)

    inputs: list[GraphInput] = []
    for port in root_module.inputs:
        endpoint = root.inputs.get(port.name)
        if endpoint is not None and port.type is not None:
            inputs.append(GraphInput(name=port.name, endpoint=endpoint, type=port.type))

    outputs = tuple(
        GraphOutput(name=port.name, endpoint=endpoint)
        for port in root_module.outputs
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

    def expand(
        self,
        module: Module,
        *,
        prefix: str,
        stack: tuple[str, ...],
        params: Mapping[str, Any],
    ) -> _Boundary:
        """Emit ``module`` under ``prefix`` and return how its ports bind."""
        if module.id in stack:
            raise ResolveError(
                "module instantiation is recursive: " + " -> ".join((*stack, module.id))
            )
        inner = (*stack, module.id)
        members: dict[str, _Member] = {}

        for node in module.nodes:
            members[node.id] = self._member(module, node, prefix, inner, params)
        for generator in module.generators:
            for key, member in self._expand_generator(module, generator, prefix, inner, params):
                members[key] = member

        for edge in module.edges:
            source = self._endpoint(module, edge.src, prefix, members, produces=True)
            target = self._endpoint(module, edge.dst, prefix, members, produces=False)
            self._connect(source, target, Origin(module=module.id, node=edge.id))

        for rule in module.spatial_rules:
            self._apply_rule(module, rule, members)

        return self._bind(module, members)

    # --- members -----------------------------------------------------------

    def _member(
        self,
        module: Module,
        node: Node,
        prefix: str,
        stack: tuple[str, ...],
        params: Mapping[str, Any],
    ) -> _Member:
        qualified = _join(prefix, node.id)
        attrs = self._attrs(module, node.id, node.attrs, params)
        self._check_resolved(module, node, attrs)
        pos = node.placement.pos

        if node.op == MODULE_OP:
            child = self._module_named(module, node.id, attrs.get("module"))
            child_params = {**child.params, **{k: v for k, v in attrs.items() if k != "module"}}
            boundary = self.expand(child, prefix=qualified, stack=stack, params=child_params)
            return _Member(key=node.id, pos=pos, boundary=boundary, inlined=True)

        self.nodes.append(
            CoreNode(
                id=qualified,
                op=node.op,
                attrs=attrs,
                origin=Origin(module=module.id, node=node.id),
            )
        )
        return _Member(key=node.id, pos=pos, boundary=self._ports(node.op, qualified))

    def _expand_generator(
        self,
        module: Module,
        generator: Generator,
        prefix: str,
        stack: tuple[str, ...],
        params: Mapping[str, Any],
    ) -> list[tuple[str, _Member]]:
        """Instantiate a generator, chaining the instances if it asks for it."""
        child = self._module_named(module, generator.id, generator.module)
        offset = generator.axis.offset
        produced: list[tuple[str, _Member]] = []

        for index in range(generator.count):
            key = f"{generator.id}-{index}"
            bindings = {**params, INDEX_PARAM: index}
            child_params: dict[str, Any] = {
                **child.params,
                INDEX_PARAM: index,
                **self._bindings(module, generator, bindings),
            }
            boundary = self.expand(
                child, prefix=_join(prefix, key), stack=stack, params=child_params
            )
            pos = list(generator.origin)
            pos[offset] += index * generator.step
            member = _Member(
                key=key,
                pos=(pos[0], pos[1], pos[2]),
                boundary=boundary,
                inlined=True,
            )
            produced.append((key, member))

        if generator.chain:
            for index in range(len(produced) - 1):
                origin = Origin(module=module.id, generator=generator.id, instance=index + 1)
                self._chain(module, generator, produced[index][1], produced[index + 1][1], origin)
        return produced

    def _bindings(
        self, module: Module, generator: Generator, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            return resolve_attrs(
                {name: _expression(value) for name, value in generator.attr_bindings.items()},
                params,
            )
        except ExpressionError as exc:
            raise ResolveError(
                f"generator {generator.id!r} in module {module.id!r}: {exc}"
            ) from exc

    def _attrs(
        self, module: Module, node_id: str, attrs: Mapping[str, Any], params: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            return resolve_attrs(attrs, params)
        except ExpressionError as exc:
            raise ResolveError(f"node {node_id!r} in module {module.id!r}: {exc}") from exc

    def _check_resolved(self, module: Module, node: Node, attrs: Mapping[str, Any]) -> None:
        """Type-check the attributes an expression produced.

        Everything else was checked by ``ntb.validate`` before lowering, but an
        expression's value only exists here, and a shape rule must never be
        handed a value of the wrong type.
        """
        spec = self.registry.get(node.op)
        if spec is None:
            return
        declared = spec.attr_specs
        for name, value in node.attrs.items():
            attr = declared.get(name)
            if attr is None or not contains_expression(value):
                continue
            problem = attr.validate(attrs[name])
            if problem is not None:
                raise ResolveError(
                    f"node {node.id!r} in module {module.id!r}: attribute {name!r} "
                    f"resolved to {attrs[name]!r}, which {problem}"
                )

    def _ports(self, op: str, qualified: str) -> _Boundary:
        """A primitive node's ports, taken from the registry."""
        spec = self.registry.get(op)
        in_names = [p.name for p in spec.inputs] if spec else ["in"]
        out_names = [p.name for p in spec.outputs] if spec else ["out"]
        return _Boundary(
            inputs={name: Endpoint(node=qualified, port=name) for name in in_names},
            outputs={name: Endpoint(node=qualified, port=name) for name in out_names},
        )

    def _module_named(self, module: Module, owner: str, target: Any) -> Module:
        if not isinstance(target, str):
            raise ResolveError(
                f"{owner!r} in module {module.id!r} does not name the module to instantiate"
            )
        child = self.document.module(target)
        if child is None:
            raise ResolveError(
                f"{owner!r} in module {module.id!r} instantiates unknown module {target!r}"
            )
        return child

    # --- wiring ------------------------------------------------------------

    def _endpoint(
        self,
        module: Module,
        endpoint: Endpoint,
        prefix: str,
        members: Mapping[str, _Member],
        *,
        produces: bool,
    ) -> Endpoint:
        member = members.get(endpoint.node)
        if member is None:
            raise ResolveError(f"module {module.id!r} references unknown node {endpoint.node!r}")
        if not member.inlined:
            return Endpoint(node=_join(prefix, endpoint.node), port=endpoint.port)
        return _port_of(member, endpoint.port, produces, module.id)

    def _chain(
        self, module: Module, generator: Generator, source: _Member, target: _Member, origin: Origin
    ) -> None:
        outputs = list(source.boundary.outputs.values())
        inputs = list(target.boundary.inputs.values())
        if not outputs or not inputs:
            raise ResolveError(
                f"generator {generator.id!r} in module {module.id!r} chains instances, but "
                f"module {generator.module!r} does not declare both an input and an output port"
            )
        self._connect(outputs[0], inputs[0], origin)

    def _apply_rule(
        self, module: Module, rule: SpatialRule, members: Mapping[str, _Member]
    ) -> None:
        expanded = _rule_members(module, rule, members)
        placed = [Placed(key=member.key, pos=member.pos) for member in expanded]
        try:
            pairs = derive_pairs(rule, placed)
        except RuleError as exc:
            raise ResolveError(f"module {module.id!r}: {exc}") from exc

        origin = Origin(module=module.id, rule=rule.id)
        for source, target in pairs:
            self._connect(
                _port_of(expanded[source], rule.output_port, True, module.id),
                _port_of(expanded[target], rule.input_port, False, module.id),
                origin,
            )

    def _bind(self, module: Module, members: Mapping[str, _Member]) -> _Boundary:
        consumed = {(e.dst.node, e.dst.port) for e in self.edges}
        produced = {(e.src.node, e.src.port) for e in self.edges}

        free_in: list[Endpoint] = []
        free_out: list[Endpoint] = []
        for member in members.values():
            candidates_in = list(member.boundary.inputs.values())
            candidates_out = list(member.boundary.outputs.values())
            if candidates_in and not any((e.node, e.port) in consumed for e in candidates_in):
                free_in.extend(candidates_in)
            if candidates_out and not any((e.node, e.port) in produced for e in candidates_out):
                free_out.extend(candidates_out)

        return _Boundary(
            inputs=self._side(
                module, "input", module.inputs, module.input_bindings, members, free_in
            ),
            outputs=self._side(
                module, "output", module.outputs, module.output_bindings, members, free_out
            ),
        )

    def _side(
        self,
        module: Module,
        role: str,
        ports: tuple[Port, ...],
        bindings: Mapping[str, Endpoint],
        members: Mapping[str, _Member],
        free: list[Endpoint],
    ) -> dict[str, Endpoint]:
        """Bound ports land where they were told; the rest bind by position."""
        bound: dict[str, Endpoint] = {}
        for name, endpoint in bindings.items():
            member = members.get(endpoint.node)
            if member is None:
                raise ResolveError(
                    f"module {module.id!r} binds {role} port {name!r} to unknown node "
                    f"{endpoint.node!r}"
                )
            bound[name] = _port_of(member, endpoint.port, role == "output", module.id)

        taken = {(e.node, e.port) for e in bound.values()}
        remaining = [port.name for port in ports if port.name not in bound]
        available = [e for e in free if (e.node, e.port) not in taken]
        return {**bound, **_match(module, role, remaining, available)}

    def _connect(self, source: Endpoint, target: Endpoint, origin: Origin) -> None:
        self._serial += 1
        self.edges.append(CoreEdge(id=f"e{self._serial}", src=source, dst=target, origin=origin))


def _rule_members(
    module: Module, rule: SpatialRule, members: Mapping[str, _Member]
) -> list[_Member]:
    """Resolve a rule's member list. A generator id stands for its instances."""
    expanded: list[_Member] = []
    for name in rule.members:
        if name in members:
            expanded.append(members[name])
            continue
        instances = [member for key, member in members.items() if key.startswith(f"{name}-")]
        if not instances:
            raise ResolveError(
                f"spatial rule {rule.id!r} in module {module.id!r} lists {name!r}, "
                "which is neither a node nor a generator here"
            )
        expanded.extend(instances)
    if len(expanded) < 2:
        raise ResolveError(
            f"spatial rule {rule.id!r} in module {module.id!r} resolved to "
            f"{len(expanded)} member(s); it needs at least two"
        )
    return expanded


def _port_of(member: _Member, port: str, produces: bool, module_id: str) -> Endpoint:
    side = member.boundary.outputs if produces else member.boundary.inputs
    endpoint = side.get(port)
    if endpoint is None:
        role = "output" if produces else "input"
        known = ", ".join(sorted(side)) or "none"
        raise ResolveError(
            f"{member.key!r} in module {module_id!r} has no {role} port {port!r}; has: {known}"
        )
    return endpoint


def _expression(value: str) -> str:
    """Generator bindings are expressions whether or not they carry the marker."""
    return value if value.startswith("$") else f"${value}"


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
