"""Emit a Keras 3 model from a core graph.

Keras 3 runs on TensorFlow, JAX or torch, so one emitter covers all three
(ADR 7). The output uses the functional API, which is what an arbitrary DAG maps
onto: a layer is constructed and called in the same expression, and the model is
the inputs and outputs it was built from.

Everything op-specific comes from the registry (ADR 3). Two Keras facts of life
are knobs there rather than special cases here: integer padding becomes its own
layer (`pad_target`), and an op Keras spells as a reshape gets its shape from
inference (`shape_arg`).

NTB tensors are channels-first, like torch. The registry pins
``data_format="channels_first"`` on every layer that cares, so a document means
the same model in both backends.
"""

from __future__ import annotations

import ast

from ntb.emit import pysrc
from ntb.emit.torch import EmitError, EmittedModule
from ntb.ir.core import CoreGraph, CoreNode
from ntb.ir.document import Document
from ntb.ir.types import Dim, TensorType
from ntb.ops.registry import REGISTRY, OpRegistry
from ntb.ops.spec import BackendMapping, CallKind, OpSpec
from ntb.shapes.infer import ShapeReport, infer_shapes
from ntb.spatial.expr import ExpressionError, evaluate
from ntb.spatial.resolve import resolve

BACKEND = "keras"


def emit_document(
    document: Document, *, registry: OpRegistry = REGISTRY, function_name: str | None = None
) -> EmittedModule:
    graph = resolve(document, registry=registry)
    return emit(graph, registry=registry, function_name=function_name)


def emit(
    graph: CoreGraph, *, registry: OpRegistry = REGISTRY, function_name: str | None = None
) -> EmittedModule:
    """Generate the source of a function that builds ``graph`` as a Keras model."""
    report = infer_shapes(graph, registry=registry)
    if report.issues:
        first = report.issues[0]
        raise EmitError(
            f"cannot emit a graph that does not type-check: {first.node}: {first.message}"
        )
    return _Builder(graph, report, registry, function_name).build()


class _Builder:
    def __init__(
        self,
        graph: CoreGraph,
        report: ShapeReport,
        registry: OpRegistry,
        function_name: str | None,
    ) -> None:
        self.graph = graph
        self.report = report
        self.registry = registry
        self.function_name = function_name or _function_name(graph.name)
        self.model_name = pysrc.identifier(graph.name or "model", fallback="model")

        self.values = pysrc.NameAllocator()
        for reserved in ("keras", "inputs", "outputs", self.function_name):
            self.values.reserve(reserved)

        self.body: list[ast.stmt] = []
        self.imports: set[str] = {"keras"}
        self._values_by_port: dict[tuple[str, str], str] = {}

    def build(self) -> EmittedModule:
        arguments: list[str] = []
        for item in self.graph.inputs:
            variable = self.values.allocate(item.name, fallback="x")
            self.body.append(pysrc.assign(variable, self._input_layer(item.name, item.type)))
            self._values_by_port[(item.endpoint.node, item.endpoint.port)] = variable
            arguments.append(variable)

        for node in self.graph.topological_order():
            self._emit_node(node)

        outputs: list[str] = []
        for output in self.graph.outputs:
            produced = self._values_by_port.get((output.endpoint.node, output.endpoint.port))
            if produced is None:
                raise EmitError(f"model output {output.endpoint} has no emitted value")
            outputs.append(produced)

        self.body.append(
            ast.Return(
                value=pysrc.call(
                    "keras.Model",
                    [],
                    {
                        "inputs": _bundle(arguments),
                        "outputs": _bundle(outputs),
                        "name": self.model_name,
                    },
                )
            )
        )
        module = ast.Module(body=[*self._import_statements(), self._function()], type_ignores=[])
        return EmittedModule(source=pysrc.unparse(module), class_name=self.function_name)

    # -- nodes --------------------------------------------------------------

    def _emit_node(self, node: CoreNode) -> None:
        spec = self.registry.require(node.op)
        mapping = spec.keras
        if mapping is None:
            raise EmitError(f"op {node.op!r} has no keras mapping")

        attrs = spec.resolved_attrs(node.attrs)
        self._check_guard(node, mapping, attrs)
        rank = self._input_rank(node, spec)
        arguments, keywords = self._call_inputs(node, spec, mapping, attrs, rank)
        kwargs = self._backend_kwargs(node, mapping, attrs)
        if mapping.shape_arg:
            kwargs[mapping.shape_arg] = _static_shape(self._output_type(node, spec))

        if mapping.kind is CallKind.MODULE:
            layer = pysrc.call(mapping.target_for(rank), [], kwargs)
            expression = pysrc.call(layer, arguments, keywords)
        else:
            expression = pysrc.call(mapping.target_for(rank), arguments, {**kwargs, **keywords})

        self.imports.update(mapping.imports)
        targets = []
        for port in spec.outputs:
            if self._is_used(node.id, port.name):
                target = self.values.allocate(
                    _value_name(node.id, port.name, len(spec.outputs)), fallback="y"
                )
                self._values_by_port[(node.id, port.name)] = target
            else:
                target = "_"
            targets.append(target)
        self.body.append(pysrc.assign_many(targets, expression))

    def _check_guard(
        self, node: CoreNode, mapping: BackendMapping, attrs: dict[str, object]
    ) -> None:
        """Refuse a mapping whose preconditions the attributes do not meet."""
        if not mapping.guard:
            return
        try:
            holds = evaluate(mapping.guard, attrs)
        except ExpressionError as exc:  # pragma: no cover - a malformed guard is a bug
            raise EmitError(f"node {node.id!r}: guard {mapping.guard!r}: {exc}") from exc
        if not holds:
            raise EmitError(f"node {node.id!r}: {mapping.guard_message or mapping.guard}")

    def _call_inputs(
        self,
        node: CoreNode,
        spec: OpSpec,
        mapping: BackendMapping,
        attrs: dict[str, object],
        rank: int | None,
    ) -> tuple[list[ast.expr], dict[str, ast.expr]]:
        connected: dict[str, list[str]] = {}
        for edge in self.graph.incoming(node.id):
            variable = self._values_by_port.get((edge.src.node, edge.src.port))
            if variable is None:
                raise EmitError(f"{edge.src} has no emitted value")
            connected.setdefault(edge.dst.port, []).append(variable)
        for port in spec.inputs:
            pinned = self._values_by_port.get((node.id, port.name))
            if pinned is not None:
                connected.setdefault(port.name, [pinned])
        for target, source in mapping.default_inputs.items():
            if target not in connected and source in connected:
                connected[target] = connected[source]

        positional: list[ast.expr] = []
        keywords: dict[str, ast.expr] = {}
        for index, port in enumerate(spec.inputs):
            variables = connected.get(port.name)
            if not variables:
                if port.optional:
                    continue
                raise EmitError(f"node {node.id!r} has nothing connected to {port.name!r}")
            # Integer padding is not expressible on a Keras layer, so it becomes
            # a layer of its own in front of the one that wanted it.
            if index == 0 and mapping.pad_target:
                variables = [self._pad(node, mapping, attrs, rank, variables[0])]
            if port.variadic:
                positional.append(_names(variables))
            elif port.name in mapping.input_kwargs:
                keywords[mapping.input_kwargs[port.name]] = pysrc.name(variables[0])
            else:
                positional.append(pysrc.name(variables[0]))

        if mapping.pack_inputs and not any(p.variadic for p in spec.inputs):
            positional = [ast.List(elts=positional, ctx=ast.Load())]
        for name, value in mapping.call_constants.items():
            keywords[name] = pysrc.constant(value)
        return positional, keywords

    def _pad(
        self,
        node: CoreNode,
        mapping: BackendMapping,
        attrs: dict[str, object],
        rank: int | None,
        variable: str,
    ) -> str:
        padding = attrs.get("padding")
        values = padding if isinstance(padding, (list, tuple)) else [padding]
        if not any(values):
            return variable

        target = self.values.allocate(f"{node.id}_pad", fallback="pad")
        # ZeroPadding1D takes one (before, after) pair; 2D and 3D take one per axis.
        pairs = tuple((v, v) for v in values)
        layer = pysrc.call(
            mapping.pad_target,
            [],
            {
                "padding": pairs[0] if len(pairs) == 1 else pairs,
                "data_format": "channels_first",
            },
        )
        self.body.append(pysrc.assign(target, pysrc.call(layer, [pysrc.name(variable)], {})))
        return target

    def _backend_kwargs(
        self, node: CoreNode, mapping: BackendMapping, attrs: dict[str, object]
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        for ntb_name, backend_name in mapping.attr_map.items():
            if ntb_name in attrs and attrs[ntb_name] is not None:
                kwargs[backend_name] = attrs[ntb_name]
        for backend_name, expression in mapping.derived.items():
            try:
                kwargs[backend_name] = evaluate(expression, attrs)
            except ExpressionError as exc:
                raise EmitError(
                    f"node {node.id!r}: {backend_name} = {expression!r}: {exc}"
                ) from exc
        kwargs.update(mapping.constants)
        return kwargs

    # -- plumbing -----------------------------------------------------------

    def _input_layer(self, name: str, tensor: TensorType) -> ast.expr:
        return pysrc.call(
            "keras.Input",
            [],
            {
                "batch_shape": tuple(_dim(d) for d in tensor.shape),
                "dtype": tensor.dtype.value,
                "name": name,
            },
        )

    def _output_type(self, node: CoreNode, spec: OpSpec) -> TensorType:
        tensor = self.report.type_of(node.id, spec.outputs[0].name)
        if tensor is None:  # pragma: no cover - inference ran clean before this
            raise EmitError(f"node {node.id!r} has no inferred output type")
        return tensor

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

    def _import_statements(self) -> list[ast.stmt]:
        return [ast.Import(names=[ast.alias(name=module)]) for module in sorted(self.imports)]

    def _function(self) -> ast.stmt:
        docstring = ast.Expr(
            value=ast.Constant(
                value=f"Generated by NTB from {self.graph.name}. Do not edit by hand."
            )
        )
        return ast.FunctionDef(
            name=self.function_name,
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[docstring, *self.body],
            decorator_list=[],
            returns=pysrc.dotted("keras.Model"),
        )


def _dim(dim: Dim) -> int | None:
    """Keras spells an open dimension ``None``."""
    return dim if isinstance(dim, int) else None


def _static_shape(tensor: TensorType) -> tuple[int, ...]:
    """The inferred shape with every open dimension as -1."""
    return tuple(d if isinstance(d, int) else -1 for d in tensor.shape)


def _names(variables: list[str]) -> ast.expr:
    return ast.List(elts=[pysrc.name(v) for v in variables], ctx=ast.Load())


def _bundle(variables: list[str]) -> ast.expr:
    """A model with one input takes a tensor, not a list of one."""
    return pysrc.name(variables[0]) if len(variables) == 1 else _names(variables)


def _value_name(node_id: str, port: str, output_count: int) -> str:
    return node_id if output_count == 1 else f"{node_id}_{port}"


def _function_name(name: str) -> str:
    return f"build_{pysrc.identifier(name or 'model', fallback='model')}"
