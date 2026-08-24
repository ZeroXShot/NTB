"""Semantic validation of a document."""

from __future__ import annotations

from ntb.ir.core import CoreGraph
from ntb.ir.document import Document, Module
from ntb.ir.graph import Node
from ntb.ops.registry import REGISTRY, OpRegistry, UnknownOpError
from ntb.shapes.infer import ShapeIssue, ShapeReport, infer_shapes
from ntb.spatial.expr import contains_expression
from ntb.spatial.resolve import MODULE_OP, NotResolvable, ResolveError, resolve
from ntb.validate.diagnostics import Code, Diagnostic, Location, Report, Severity


def validate(document: Document, *, registry: OpRegistry = REGISTRY) -> Report:
    """Check ops, attributes and shapes, and report everything found."""
    found: list[Diagnostic] = []
    known_modules = frozenset(m.id for m in document.modules)
    for module in document.modules:
        found.extend(_check_module(module, registry, known_modules))

    # Shape inference needs a lowered graph. Report authoring problems first;
    # a document with unknown ops has nothing useful to infer.
    if not any(d.severity is Severity.ERROR for d in found):
        found.extend(_check_shapes(document, registry))
    return Report(diagnostics=tuple(found))


def _check_module(
    module: Module, registry: OpRegistry, known_modules: frozenset[str]
) -> list[Diagnostic]:
    found: list[Diagnostic] = []
    known = module.node_ids

    for node in module.nodes:
        found.extend(_check_node(module, node, registry))

    for edge in module.edges:
        for endpoint, role in ((edge.src, "source"), (edge.dst, "target")):
            if endpoint.node not in known:
                found.append(
                    Diagnostic(
                        code=Code.STRUCTURE,
                        message=f"edge {role} references unknown node {endpoint.node!r}",
                        location=Location(module=module.id, edge=edge.id),
                    )
                )

    # A rule may name a generator, which stands for all of its instances.
    addressable = known | {generator.id for generator in module.generators}
    for rule in module.spatial_rules:
        for member in rule.members:
            if member not in addressable:
                found.append(
                    Diagnostic(
                        code=Code.STRUCTURE,
                        message=f"spatial rule {rule.id!r} lists unknown node {member!r}",
                        location=Location(module=module.id),
                    )
                )

    for generator in module.generators:
        if generator.module not in known_modules:
            found.append(
                Diagnostic(
                    code=Code.STRUCTURE,
                    message=f"generator {generator.id!r} repeats unknown module "
                    f"{generator.module!r}",
                    location=Location(module=module.id),
                )
            )
    return found


def _check_node(module: Module, node: Node, registry: OpRegistry) -> list[Diagnostic]:
    at = Location(module=module.id, node=node.id)
    if node.op == MODULE_OP:
        return []

    spec = registry.get(node.op)
    if spec is None:
        return [
            Diagnostic(
                code=Code.UNKNOWN_OP,
                message=str(UnknownOpError(node.op, registry.names())),
                location=at,
            )
        ]

    found: list[Diagnostic] = []
    declared = spec.attr_specs
    for name, value in node.attrs.items():
        attr = declared.get(name)
        if attr is None:
            found.append(
                Diagnostic(
                    code=Code.UNKNOWN_ATTR,
                    message=f"{node.op} has no attribute {name!r}",
                    location=at,
                )
            )
            continue
        # "$width * 2" is checked when it is evaluated against the parameters
        # in scope, which this module does not know yet.
        if contains_expression(value):
            continue
        problem = attr.validate(value)
        if problem is not None:
            found.append(
                Diagnostic(
                    code=Code.BAD_ATTR,
                    message=f"attribute {name!r}: {problem}",
                    location=at,
                )
            )

    for attr in spec.attrs:
        if attr.required and attr.name not in node.attrs:
            found.append(
                Diagnostic(
                    code=Code.MISSING_ATTR,
                    message=f"{node.op} requires attribute {attr.name!r}",
                    location=at,
                )
            )
    return found


def _check_shapes(document: Document, registry: OpRegistry) -> list[Diagnostic]:
    try:
        graph = resolve(document, registry=registry)
    except NotResolvable as exc:
        return [
            Diagnostic(
                code=Code.UNRESOLVABLE,
                message=str(exc),
                location=Location(module=document.root),
                severity=Severity.WARNING,
            )
        ]
    except ResolveError as exc:
        return [
            Diagnostic(
                code=Code.STRUCTURE,
                message=str(exc),
                location=Location(module=document.root),
            )
        ]

    report = infer_shapes(graph, registry=registry)
    found = [_from_shape_issue(graph, issue) for issue in report.issues]
    found.extend(_check_outputs(graph, report))
    return found


def _from_shape_issue(graph: CoreGraph, issue: ShapeIssue) -> Diagnostic:
    node = graph.node(issue.node)
    origin = node.origin if node is not None else None
    return Diagnostic(
        code=Code.CYCLE if "cycle" in issue.message else Code.SHAPE,
        message=issue.message,
        location=Location(
            module=origin.module if origin else None,
            node=origin.node if origin else issue.node,
            port=issue.port,
        ),
    )


def _check_outputs(graph: CoreGraph, report: ShapeReport) -> list[Diagnostic]:
    if graph.nodes and not graph.outputs:
        return [
            Diagnostic(
                code=Code.NO_OUTPUT,
                message="the root module declares no output port",
                location=Location(module=graph.name),
                severity=Severity.WARNING,
            )
        ]
    found: list[Diagnostic] = []
    for output in graph.outputs:
        endpoint = output.endpoint
        if report.type_of(endpoint.node, endpoint.port) is None:
            node = graph.node(endpoint.node)
            origin = node.origin if node is not None else None
            found.append(
                Diagnostic(
                    code=Code.UNCONNECTED,
                    message=f"model output {endpoint} has no inferred type",
                    location=Location(
                        module=origin.module if origin else None,
                        node=origin.node if origin else endpoint.node,
                        port=endpoint.port,
                    ),
                    severity=Severity.WARNING,
                )
            )
    return found
