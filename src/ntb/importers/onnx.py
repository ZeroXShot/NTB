"""Read an ONNX model into NTB-IR.

Best effort, and deliberately one-way: NTB-IR is the source of truth (ADR 1), so
this exists to *seed* a document from a model you already have, not to keep the
two in sync. What comes back is an architecture you can open in the studio, not
a trained model -- NTB-IR stores no weights, and the importer says so rather
than pretending.

Two things it will not do quietly: it never invents an op it does not know, and
it never guesses an attribute it could not read. Both end up in
``ImportResult.problems`` with the node that caused them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ntb.ir.document import Document, Module
from ntb.ir.graph import Edge, Endpoint, Node, Port, PortDirection
from ntb.ir.spatial import Placement
from ntb.ir.types import DType, Shape, TensorType
from ntb.ops.registry import REGISTRY, OpRegistry
from ntb.ops.spec import OpSpec

#: Horizontal gap between two dependency levels, and vertical gap between peers.
STEP_X = 3.0
STEP_Y = 2.0

_DTYPES = {
    1: DType.FLOAT32,
    2: DType.UINT8,
    3: DType.INT8,
    5: DType.INT16,
    6: DType.INT32,
    7: DType.INT64,
    9: DType.BOOL,
    10: DType.FLOAT16,
    11: DType.FLOAT64,
    14: DType.COMPLEX64,
    16: DType.BFLOAT16,
}


class OnnxImportError(Exception):
    """The file is not a model this importer can read at all."""


@dataclass(frozen=True, slots=True)
class ImportResult:
    """The document, and everything that did not survive the trip."""

    document: Document
    problems: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.problems


def import_onnx(
    source: Any, *, name: str | None = None, registry: OpRegistry = REGISTRY
) -> ImportResult:
    """Build a document from an ONNX model, a path, or a ModelProto."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - the extra is declared
        raise OnnxImportError("reading ONNX needs the onnx extra: pip install 'ntb[onnx]'") from exc

    model = source if hasattr(source, "graph") else onnx.load(str(source))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    stem = (
        name
        or model.graph.name
        or (Path(str(source)).stem if not hasattr(source, "graph") else "imported")
    )
    return _Importer(model, stem, registry).run()


class _Importer:
    def __init__(self, model: Any, name: str, registry: OpRegistry) -> None:
        self.graph = model.graph
        self.name = _identifier(name)
        self.registry = registry
        self.problems: list[str] = []

        self.initialisers = {tensor.name: tuple(tensor.dims) for tensor in self.graph.initializer}
        self.types = _value_types(self.graph)
        #: ONNX tensor name -> the NTB endpoint that produces it.
        self.produced: dict[str, Endpoint] = {}
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.level: dict[str, int] = {}

    def run(self) -> ImportResult:
        for index, node in enumerate(self.graph.node):
            self._node(node, index)

        module = Module(
            id=self.name,
            name=self.graph.name or "",
            doc="Imported from ONNX. Weights are not carried over.",
            inputs=self._inputs(),
            outputs=self._outputs(),
            input_bindings=self._input_bindings(),
            output_bindings=self._output_bindings(),
            nodes=tuple(self._placed()),
            edges=tuple(self.edges),
        )
        document = Document(
            name=self.name,
            doc="Imported from ONNX.",
            root=self.name,
            modules=(module,),
            metadata={"imported_from": "onnx", "producer": self.graph.name or ""},
        )
        return ImportResult(document=document, problems=tuple(self.problems))

    # -- nodes --------------------------------------------------------------

    def _node(self, node: Any, index: int) -> None:
        spec = self._spec(node)
        if spec is None:
            self.problems.append(
                f"{node.op_type} ({node.name or f'node {index}'}) has no NTB op; it was dropped"
            )
            return

        attrs = self._attrs(node, spec)
        node_id = _identifier(node.name or f"{spec.name.split('.')[-1]}_{index}")
        node_id = _unique(node_id, {n.id for n in self.nodes})

        sources = [name for name in node.input if name and name not in self.initialisers]
        self.level[node_id] = 1 + max(
            (self.level.get(self.produced[s].node, 0) for s in sources if s in self.produced),
            default=-1,
        )
        self.nodes.append(Node(id=node_id, op=spec.name, name=node.name or "", attrs=attrs))

        ports = [port.name for port in spec.inputs]
        for position, tensor in enumerate(sources):
            port = ports[min(position, len(ports) - 1)] if ports else "in"
            origin = self.produced.get(tensor)
            if origin is None:
                continue  # a graph input; it binds through the module boundary
            self.edges.append(
                Edge(
                    id=_unique(f"{origin.node}__{node_id}", {e.id for e in self.edges}),
                    src=origin,
                    dst=Endpoint(node=node_id, port=port),
                )
            )
        for position, tensor in enumerate(node.output):
            if not tensor:
                continue
            port = spec.outputs[min(position, len(spec.outputs) - 1)].name
            self.produced[tensor] = Endpoint(node=node_id, port=port)

    def _spec(self, node: Any) -> OpSpec | None:
        """Which NTB op an ONNX node is. Rank decides between conv1d/2d/3d."""
        candidates = [
            spec
            for spec in self.registry
            if spec.onnx is not None and spec.onnx.op_type == node.op_type
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        spatial = len(_attribute(node, "kernel_shape") or ())
        if spatial:
            for spec in candidates:
                if spec.name.endswith(f"{spatial}d"):
                    return spec
        rank = self._rank(node.input[0] if node.input else "")
        if rank is not None:
            for spec in candidates:
                if spec.name.endswith(f"{rank - 2}d"):
                    return spec
        self.problems.append(
            f"{node.op_type} ({node.name or 'unnamed'}) is ambiguous without a shape; "
            f"read as {candidates[0].name}"
        )
        return candidates[0]

    def _attrs(self, node: Any, spec: OpSpec) -> dict[str, Any]:
        assert spec.onnx is not None
        attrs: dict[str, Any] = {}
        reverse = {backend: ntb for ntb, backend in spec.onnx.attr_map.items()}
        for attribute in node.attribute:
            ntb_name = reverse.get(attribute.name)
            if ntb_name is not None:
                attrs[ntb_name] = _attribute_value(attribute)

        if spec.onnx.pad_attr:
            # ONNX writes begins then ends. NTB carries one number per axis, so
            # asymmetric padding is the one form that cannot come back.
            pads = _attribute(node, spec.onnx.pad_attr) or []
            half = len(pads) // 2
            if half and pads[:half] == pads[half:]:
                attrs["padding"] = list(pads[:half])
            elif half:
                self.problems.append(
                    f"{node.op_type} ({node.name or 'unnamed'}): padding {pads} is "
                    "asymmetric, which NTB cannot express; it was dropped"
                )

        # ONNX carries an op's learned tensors as inputs, and their shapes say
        # what the NTB attributes were: a Gemm whose B is (10, 128) came from a
        # linear with those features.
        shapes = [self.initialisers.get(name) for name in node.input]
        for param, shape in zip(
            spec.onnx.params, [s for s in shapes if s is not None], strict=False
        ):
            self._from_param(spec, param.shape, shape, attrs)
        _fix_up(spec, attrs)

        missing = [a.name for a in spec.attrs if a.required and a.name not in attrs]
        if missing:
            self.problems.append(
                f"{node.op_type} ({node.name or 'unnamed'}): could not read "
                f"{', '.join(missing)}; fill it in before validating"
            )
        return attrs

    def _from_param(
        self,
        spec: OpSpec,
        declared: tuple[str | int, ...],
        actual: tuple[int, ...],
        attrs: dict[str, Any],
    ) -> None:
        """Read attributes back out of a parameter's shape.

        The registry already says what a parameter's shape is made of, so most
        entries invert on sight. Anything computed (``in_channels // groups``)
        does not, and is left to :func:`_fix_up`.
        """
        for position, entry in enumerate(declared):
            if not isinstance(entry, str) or position >= len(actual):
                continue
            if entry.startswith("*"):
                attrs.setdefault(entry[1:], list(actual[position:]))
                return
            if entry.isidentifier():
                attrs.setdefault(entry, actual[position])
                continue
            # The one computed form the registry uses: `in_channels // groups`,
            # which inverts as long as the divisor is an attribute already read.
            head, _, tail = entry.partition(" // ")
            if head.isidentifier() and tail.isidentifier() and tail in attrs:
                attrs.setdefault(head, actual[position] * int(attrs[tail]))

    # -- boundary and layout ------------------------------------------------

    def _graph_inputs(self) -> list[Any]:
        return [item for item in self.graph.input if item.name not in self.initialisers]

    def _inputs(self) -> tuple[Port, ...]:
        return tuple(
            Port(
                name=_identifier(item.name),
                direction=PortDirection.IN,
                type=self.types.get(item.name) or TensorType(),
            )
            for item in self._graph_inputs()
        )

    def _outputs(self) -> tuple[Port, ...]:
        return tuple(
            Port(name=_identifier(item.name), direction=PortDirection.OUT)
            for item in self.graph.output
        )

    def _input_bindings(self) -> dict[str, Endpoint]:
        """Where each model input lands. The ONNX graph says exactly, so say it."""
        bindings: dict[str, Endpoint] = {}
        for item in self._graph_inputs():
            for node in self.graph.node:
                if item.name not in node.input:
                    continue
                target = next((n for n in self.nodes if n.name == node.name), None)
                if target is None:
                    continue
                spec = self.registry.get(target.op)
                port = spec.inputs[0].name if spec and spec.inputs else "in"
                bindings[_identifier(item.name)] = Endpoint(node=target.id, port=port)
                break
        return bindings

    def _output_bindings(self) -> dict[str, Endpoint]:
        return {
            _identifier(item.name): self.produced[item.name]
            for item in self.graph.output
            if item.name in self.produced
        }

    def _placed(self) -> list[Node]:
        """Lay the graph out left to right, one column per dependency level."""
        peers: dict[int, int] = {}
        placed: list[Node] = []
        for node in self.nodes:
            level = self.level.get(node.id, 0)
            row = peers.get(level, 0)
            peers[level] = row + 1
            placed.append(
                node.model_copy(
                    update={
                        "placement": Placement(
                            pos=(level * STEP_X, -row * STEP_Y, 0.0), extent=(2.0, 1.0, 1.0)
                        )
                    }
                )
            )
        return placed

    def _rank(self, tensor: str) -> int | None:
        found = self.types.get(tensor)
        return found.rank if found else None


def _fix_up(spec: OpSpec, attrs: dict[str, Any]) -> None:
    """Attributes an ONNX node carries in a shape rather than an attribute."""
    if spec.name == "ntb.layernorm" and "normalized_shape" in attrs:
        attrs["normalized_shape"] = list(attrs["normalized_shape"])


def _value_types(graph: Any) -> dict[str, TensorType]:
    types: dict[str, TensorType] = {}
    for collection in (graph.input, graph.value_info, graph.output):
        for item in collection:
            tensor = _tensor_type(item)
            if tensor is not None:
                types[item.name] = tensor
    return types


def _tensor_type(item: Any) -> TensorType | None:
    field = item.type.tensor_type
    if not field.elem_type:
        return None
    shape: list[int | str] = []
    for dim in field.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        elif dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append("batch" if not shape else f"dim{len(shape)}")
    return TensorType(dtype=_DTYPES.get(field.elem_type, DType.FLOAT32), shape=Shape(shape))


def _attribute(node: Any, name: str) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return _attribute_value(attribute)
    return None


def _attribute_value(attribute: Any) -> Any:
    if attribute.type == 1:  # FLOAT
        return float(attribute.f)
    if attribute.type == 2:  # INT
        return int(attribute.i)
    if attribute.type == 3:  # STRING
        return attribute.s.decode("utf-8")
    if attribute.type == 6:  # FLOATS
        return [float(v) for v in attribute.floats]
    if attribute.type == 7:  # INTS
        return [int(v) for v in attribute.ints]
    if attribute.type == 8:  # STRINGS
        return [v.decode("utf-8") for v in attribute.strings]
    return None


def _identifier(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "_-." else "_" for c in name).strip("_")
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"n_{cleaned}" if cleaned else "imported"
    return cleaned


def _unique(candidate: str, taken: set[str]) -> str:
    if candidate not in taken:
        return candidate
    for index in range(2, 10_000):  # pragma: no branch - a graph is never that dense
        if f"{candidate}_{index}" not in taken:
            return f"{candidate}_{index}"
    raise OnnxImportError(f"cannot find a free name for {candidate!r}")  # pragma: no cover
