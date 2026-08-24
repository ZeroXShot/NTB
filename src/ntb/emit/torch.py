"""Emit a readable ``torch.nn.Module`` from a core graph.

Everything op-specific comes from the registry (ADR 3). This file knows how a
torch model is *shaped* -- constructor versus forward, one variable per value --
and nothing about what any individual op means.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ntb.emit import pysrc
from ntb.ir.core import CoreGraph, CoreNode
from ntb.ir.document import Document
from ntb.ops.registry import REGISTRY, OpRegistry
from ntb.ops.spec import BackendMapping, CallKind, OpSpec
from ntb.shapes.infer import ShapeReport, infer_shapes
from ntb.spatial.resolve import resolve

BACKEND = "torch"


class EmitError(Exception):
    """The graph cannot be emitted for this backend."""


@dataclass(frozen=True, slots=True)
class EmittedModule:
    source: str
    class_name: str

    def __str__(self) -> str:
        return self.source


def emit_document(
    document: Document, *, registry: OpRegistry = REGISTRY, class_name: str | None = None
) -> EmittedModule:
    graph = resolve(document, registry=registry)
    return emit(graph, registry=registry, class_name=class_name)


def emit(
    graph: CoreGraph, *, registry: OpRegistry = REGISTRY, class_name: str | None = None
) -> EmittedModule:
    """Generate the source of an ``nn.Module`` implementing ``graph``."""
    report = infer_shapes(graph, registry=registry)
    if report.issues:
        first = report.issues[0]
        raise EmitError(
            f"cannot emit a graph that does not type-check: {first.node}: {first.message}"
        )

    builder = _Builder(graph, report, registry, class_name)
    return builder.build()


class _Builder:
    def __init__(
        self,
        graph: CoreGraph,
        report: ShapeReport,
        registry: OpRegistry,
        class_name: str | None,
    ) -> None:
        self.graph = graph
        self.report = report
        self.registry = registry
        self.class_name = class_name or _class_name(graph.name)

        self.values = pysrc.NameAllocator()
        self.members = pysrc.NameAllocator()
        for reserved in ("self", "forward", "torch", "nn", "F", self.class_name):
            self.values.reserve(reserved)
            self.members.reserve(reserved)

        self.init_body: list[ast.stmt] = []
        self.forward_body: list[ast.stmt] = []
        self.imports: set[str] = {"torch"}
        self._values_by_port: dict[tuple[str, str], str] = {}

    def build(self) -> EmittedModule:
        arguments = []
        for item in self.graph.inputs:
            argument = self.values.allocate(item.name, fallback="x")
            self._bind(item.endpoint.node, item.endpoint.port, argument)
            arguments.append(argument)

        for node in self.graph.topological_order():
            self._emit_node(node)

        module = ast.Module(
            body=[*self._import_statements(), self._class(arguments)], type_ignores=[]
        )
        return EmittedModule(source=pysrc.unparse(module), class_name=self.class_name)

    # -- graph plumbing ----------------------------------------------------

    def _bind(self, node: str, port: str, variable: str) -> None:
        self._values_by_port[(node, port)] = variable

    def _emit_node(self, node: CoreNode) -> None:
        spec = self.registry.require(node.op)
        mapping = spec.torch
        if mapping is None:
            raise EmitError(f"op {node.op!r} has no torch mapping")

        attrs = spec.resolved_attrs(node.attrs)
        rank = self._input_rank(node, spec)
        arguments, keywords = self._call_inputs(node, spec, mapping)
        kwargs = _backend_kwargs(mapping, attrs)

        if mapping.kind is CallKind.MODULE:
            member = self.members.allocate(node.id, fallback="layer")
            self.init_body.append(
                pysrc.assign_attribute(
                    "self", member, pysrc.call(mapping.target_for(rank), [], kwargs)
                )
            )
            callee: ast.expr = pysrc.attribute("self", member)
            expression = pysrc.call(callee, arguments, keywords)
        else:
            expression = pysrc.call(mapping.target_for(rank), arguments, {**kwargs, **keywords})

        self.imports.update(mapping.imports)
        targets = []
        for port in spec.outputs:
            if self._is_used(node.id, port.name):
                target = self.values.allocate(
                    _value_name(node.id, port.name, len(spec.outputs)), fallback="y"
                )
                self._bind(node.id, port.name, target)
            else:
                target = "_"
            targets.append(target)
        self.forward_body.append(pysrc.assign_many(targets, expression))

    def _call_inputs(
        self, node: CoreNode, spec: OpSpec, mapping: BackendMapping
    ) -> tuple[list[ast.expr], dict[str, ast.expr]]:
        connected: dict[str, str] = {}
        for edge in self.graph.incoming(node.id):
            variable = self._values_by_port.get((edge.src.node, edge.src.port))
            if variable is None:
                raise EmitError(f"{edge.src} has no emitted value")
            connected[edge.dst.port] = variable
        for port in spec.inputs:
            pinned = self._values_by_port.get((node.id, port.name))
            if pinned is not None:
                connected.setdefault(port.name, pinned)

        # A backend may fill an unconnected port from another one, which is how
        # self-attention passes the same tensor as query, key and value.
        for target, source in mapping.default_inputs.items():
            if target not in connected and source in connected:
                connected[target] = connected[source]

        positional: list[ast.expr] = []
        keywords: dict[str, ast.expr] = {}
        for port in spec.inputs:
            variable = connected.get(port.name)
            if variable is None:
                if port.optional:
                    continue
                raise EmitError(f"node {node.id!r} has nothing connected to {port.name!r}")
            if port.name in mapping.input_kwargs:
                keywords[mapping.input_kwargs[port.name]] = pysrc.name(variable)
            else:
                positional.append(pysrc.name(variable))

        if mapping.pack_inputs:
            positional = [ast.List(elts=positional, ctx=ast.Load())]
        return positional, keywords

    def _is_used(self, node: str, port: str) -> bool:
        if any(e.src.node == node and e.src.port == port for e in self.graph.edges):
            return True
        return any(
            out.endpoint.node == node and out.endpoint.port == port for out in self.graph.outputs
        )

    def _input_rank(self, node: CoreNode, spec: OpSpec) -> int | None:
        if not spec.inputs:
            return None
        first = spec.inputs[0].name
        for edge in self.graph.incoming(node.id):
            if edge.dst.port == first:
                tensor = self.report.type_of(edge.src.node, edge.src.port)
                return tensor.rank if tensor else None
        tensor = self.report.type_of(node.id, first)
        return tensor.rank if tensor else None

    # -- rendering ---------------------------------------------------------

    def _import_statements(self) -> list[ast.stmt]:
        statements: list[ast.stmt] = []
        for module in sorted(self.imports):
            statements.append(ast.Import(names=[ast.alias(name=module)]))
        return statements

    def _class(self, arguments: list[str]) -> ast.ClassDef:
        init = ast.FunctionDef(
            name="__init__",
            args=_arguments(["self"]),
            body=[ast.Expr(value=_super_init()), *(self.init_body or [ast.Pass()])],
            decorator_list=[],
            returns=ast.Constant(value=None),
        )
        forward = ast.FunctionDef(
            name="forward",
            args=_arguments(["self", *arguments]),
            body=[*self.forward_body, self._return()],
            decorator_list=[],
            returns=None,
        )
        return ast.ClassDef(
            name=self.class_name,
            bases=[pysrc.dotted("torch.nn.Module")],
            keywords=[],
            body=[ast.Expr(value=ast.Constant(value=self._docstring())), init, forward],
            decorator_list=[],
        )

    def _docstring(self) -> str:
        return f"Generated by NTB from {self.graph.name or 'an NTB document'}. Do not edit by hand."

    def _return(self) -> ast.Return:
        outputs: list[ast.expr] = [
            pysrc.name(self._values_by_port[(out.endpoint.node, out.endpoint.port)])
            for out in self.graph.outputs
            if (out.endpoint.node, out.endpoint.port) in self._values_by_port
        ]
        if not outputs:
            return ast.Return(value=ast.Constant(value=None))
        if len(outputs) == 1:
            return ast.Return(value=outputs[0])
        return ast.Return(value=ast.Tuple(elts=outputs, ctx=ast.Load()))


def _super_init() -> ast.Call:
    """``super().__init__()``"""
    super_call = ast.Call(func=ast.Name(id="super", ctx=ast.Load()), args=[], keywords=[])
    bound = ast.Attribute(value=super_call, attr="__init__", ctx=ast.Load())
    return ast.Call(func=bound, args=[], keywords=[])


def _arguments(names: list[str]) -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=n) for n in names],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )


def _backend_kwargs(mapping: BackendMapping, attrs: dict[str, object]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for ntb_name, backend_name in mapping.attr_map.items():
        if ntb_name in attrs and attrs[ntb_name] is not None:
            kwargs[backend_name] = attrs[ntb_name]
    kwargs.update(mapping.constants)
    return kwargs


def _value_name(node_id: str, port: str, output_count: int) -> str:
    """A single-output node names its value after itself; others suffix the port."""
    tail = node_id.rsplit("/", 1)[-1]
    return tail if output_count == 1 else f"{tail}_{port}"


def _class_name(name: str) -> str:
    cleaned = pysrc.identifier(name or "model", fallback="model")
    return "".join(part.capitalize() or "_" for part in cleaned.split("_")) or "Model"
