"""Shape and dtype propagation over a core graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from ntb.ir.core import CoreGraph, CoreNode, CycleError
from ntb.ir.types import TensorType
from ntb.ops.registry import REGISTRY, OpRegistry, UnknownOpError
from ntb.ops.spec import PortSpec, ShapeContext, ShapeRuleError

#: A value on a specific port: ``(node id, port name)``.
PortRef = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ShapeIssue:
    """One reason inference could not finish for a node."""

    node: str
    message: str
    port: str | None = None


@dataclass(frozen=True, slots=True)
class ShapeReport:
    """Types for every port inference reached, plus what stopped it elsewhere."""

    types: dict[PortRef, TensorType] = field(default_factory=dict)
    issues: tuple[ShapeIssue, ...] = ()
    #: Nodes whose type is unknown, either because they failed or because an
    #: input of theirs did. Downstream failures are reported once, at the source.
    unresolved: frozenset[str] = frozenset()

    @property
    def ok(self) -> bool:
        return not self.issues

    def type_of(self, node: str, port: str = "out") -> TensorType | None:
        return self.types.get((node, port))


def infer_shapes(graph: CoreGraph, *, registry: OpRegistry = REGISTRY) -> ShapeReport:
    """Propagate types through ``graph`` in dependency order.

    Never raises for a bad graph: a broken node is recorded and its dependants
    are marked unresolved rather than reported again, so one mistake produces
    one message instead of a cascade.
    """
    try:
        order = graph.topological_order()
    except CycleError as exc:
        return ShapeReport(
            issues=tuple(
                ShapeIssue(node=node, message="node takes part in a cycle")
                for node in sorted(exc.nodes)
            ),
            unresolved=exc.nodes,
        )

    types: dict[PortRef, TensorType] = {
        (item.endpoint.node, item.endpoint.port): item.type for item in graph.inputs
    }
    issues: list[ShapeIssue] = []
    unresolved: set[str] = set()

    for node in order:
        spec = registry.get(node.op)
        if spec is None:
            issues.append(
                ShapeIssue(node=node.id, message=str(UnknownOpError(node.op, registry.names())))
            )
            unresolved.add(node.id)
            continue

        declared = {port.name for port in spec.inputs}
        many = {port.name for port in spec.inputs if port.variadic}
        inputs: dict[str, TensorType] = {}
        variadic: dict[str, list[TensorType]] = {name: [] for name in many}
        blocked = False

        for edge in graph.incoming(node.id):
            if edge.dst.port not in declared:
                issues.append(
                    ShapeIssue(
                        node=node.id,
                        port=edge.dst.port,
                        message=f"{node.op} has no input port {edge.dst.port!r}",
                    )
                )
                blocked = True
                continue
            if edge.dst.port in inputs and edge.dst.port not in many:
                issues.append(
                    ShapeIssue(
                        node=node.id,
                        port=edge.dst.port,
                        message=f"input port {edge.dst.port!r} is connected more than once",
                    )
                )
                blocked = True
                continue
            if edge.src.node in unresolved:
                blocked = True
                continue
            source = types.get((edge.src.node, edge.src.port))
            if source is None:
                issues.append(
                    ShapeIssue(
                        node=node.id,
                        port=edge.dst.port,
                        message=f"{edge.src} produces no value",
                    )
                )
                blocked = True
                continue
            if edge.dst.port in many:
                # Edge order is the argument order a variadic port sees, and it
                # is deterministic, so a fan-in emits the same code every time.
                variadic[edge.dst.port].append(source)
            inputs.setdefault(edge.dst.port, source)

        # Graph-level pinned inputs fill ports that no edge feeds.
        for port in spec.inputs:
            pinned = types.get((node.id, port.name))
            if pinned is not None and port.name not in inputs:
                inputs[port.name] = pinned

        if blocked:
            unresolved.add(node.id)
            continue

        try:
            outputs = spec.shape_rule(
                ShapeContext(
                    op=spec.name,
                    attrs=spec.resolved_attrs(node.attrs),
                    inputs=inputs,
                    variadic={name: tuple(types_) for name, types_ in variadic.items() if types_},
                )
            )
        except ShapeRuleError as exc:
            issues.append(ShapeIssue(node=node.id, message=str(exc)))
            unresolved.add(node.id)
            continue
        except (TypeError, ValueError) as exc:
            # A shape rule that blows up is a bug, but the studio still has to
            # draw the document, so it is reported like any other problem.
            issues.append(ShapeIssue(node=node.id, message=f"{spec.name}: {exc}"))
            unresolved.add(node.id)
            continue

        _record(node, spec.outputs, outputs, types, issues, unresolved)

    return ShapeReport(types=types, issues=tuple(issues), unresolved=frozenset(unresolved))


def _record(
    node: CoreNode,
    declared: tuple[PortSpec, ...],
    produced: dict[str, TensorType],
    types: dict[PortRef, TensorType],
    issues: list[ShapeIssue],
    unresolved: set[str],
) -> None:
    expected = {port.name for port in declared}
    missing = expected - set(produced)
    if missing:
        issues.append(
            ShapeIssue(
                node=node.id,
                message=f"shape rule produced no type for {', '.join(sorted(missing))}",
            )
        )
        unresolved.add(node.id)
        return
    for name, tensor in produced.items():
        types[(node.id, name)] = tensor
