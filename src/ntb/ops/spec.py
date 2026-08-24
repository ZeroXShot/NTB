"""Op declarations: ports, attributes, shape rule, backend mappings.

Validation, inference, the emitters and the parity harness are all derived from
these. See docs/adr/0003.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NoReturn, TypeAlias

from ntb.ir.types import DType, TensorType


class AttrType(StrEnum):
    """The closed set of attribute types an op may declare."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    INTS = "ints"
    FLOATS = "floats"
    DTYPE = "dtype"
    SHAPE = "shape"

    def check(self, value: Any) -> str | None:
        """Return an error message, or None when ``value`` is acceptable."""
        ok: bool
        match self:
            case AttrType.INT:
                ok = isinstance(value, int) and not isinstance(value, bool)
            case AttrType.FLOAT:
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            case AttrType.BOOL:
                ok = isinstance(value, bool)
            case AttrType.STRING:
                ok = isinstance(value, str)
            case AttrType.INTS:
                ok = isinstance(value, (list, tuple)) and all(
                    isinstance(v, int) and not isinstance(v, bool) for v in value
                )
            case AttrType.FLOATS:
                ok = isinstance(value, (list, tuple)) and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
                )
            case AttrType.DTYPE:
                ok = isinstance(value, DType) or (isinstance(value, str) and value in set(DType))
            case AttrType.SHAPE:
                ok = isinstance(value, (list, tuple)) and all(
                    isinstance(v, (int, str)) and not isinstance(v, bool) for v in value
                )
        return None if ok else f"expected {self.value}, got {type(value).__name__}"


@dataclass(frozen=True, slots=True)
class PortSpec:
    """A port an op always has."""

    name: str
    doc: str = ""
    optional: bool = False
    variadic: bool = False


@dataclass(frozen=True, slots=True)
class AttrSpec:
    """An attribute an op takes."""

    name: str
    type: AttrType
    doc: str = ""
    default: Any = None
    required: bool = False
    minimum: float | None = None
    choices: tuple[Any, ...] | None = None
    #: Falls back to another attribute when absent (pooling stride defaults to
    #: kernel_size). Resolving it here means every backend sees the same value;
    #: leaving it implicit let torch and ONNX disagree, since ONNX defaults
    #: strides to 1.
    default_from: str = ""

    def validate(self, value: Any) -> str | None:
        """Return an error message, or None when ``value`` is acceptable."""
        if (message := self.type.check(value)) is not None:
            return message
        if self.choices is not None and value not in self.choices:
            allowed = ", ".join(repr(c) for c in self.choices)
            return f"must be one of {allowed}, got {value!r}"
        if self.minimum is not None:
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if isinstance(item, (int, float)) and item < self.minimum:
                    return f"must be >= {self.minimum}, got {item}"
        return None


class ShapeRuleError(Exception):
    """A shape rule rejected its inputs."""


@dataclass(frozen=True, slots=True)
class ShapeContext:
    """Everything a shape rule is allowed to look at."""

    op: str
    attrs: Mapping[str, Any]
    inputs: Mapping[str, TensorType]

    def input(self, name: str) -> TensorType:
        try:
            return self.inputs[name]
        except KeyError:
            raise ShapeRuleError(f"{self.op}: required input {name!r} is not connected") from None

    def attr(self, name: str) -> Any:
        try:
            return self.attrs[name]
        except KeyError:  # pragma: no cover - registry defaults fill these in
            raise ShapeRuleError(f"{self.op}: missing attribute {name!r}") from None

    def fail(self, message: str) -> NoReturn:
        raise ShapeRuleError(f"{self.op}: {message}")


#: Computes output port types from attributes and connected input types.
ShapeRule: TypeAlias = Callable[[ShapeContext], dict[str, TensorType]]


class CallKind(StrEnum):
    """How a backend materialises an op."""

    #: Constructed once, then called (``torch.nn.Linear``).
    MODULE = "module"
    #: Called inline (``torch.nn.functional.relu``).
    FUNCTION = "function"


@dataclass(frozen=True, slots=True)
class BackendMapping:
    """How one op is expressed in one Python backend.

    ``attr_map`` renames NTB attributes to backend kwargs; anything absent is
    not passed. ``constants`` are kwargs NTB does not model.
    """

    target: str
    kind: CallKind = CallKind.MODULE
    attr_map: Mapping[str, str] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    imports: tuple[str, ...] = ()
    #: Target chosen by the rank of the first input, when the backend splits one
    #: NTB op across several names (BatchNorm1d/2d/3d).
    rank_targets: Mapping[int, str] = field(default_factory=dict)
    #: Pass all inputs as one list argument (``torch.cat([a, b], dim=1)``).
    pack_inputs: bool = False
    #: Fall an unconnected input back to another port (attention key -> query).
    default_inputs: Mapping[str, str] = field(default_factory=dict)
    #: Pass a named input as a keyword rather than positionally.
    input_kwargs: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def target_for(self, rank: int | None) -> str:
        """The callable to emit, given the rank of the first input."""
        if rank is not None and rank in self.rank_targets:
            return self.rank_targets[rank]
        return self.target


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """A learned tensor an ONNX node needs as an input.

    torch and Keras layers own their weights; ONNX takes them as graph inputs.
    ``shape`` entries are attribute expressions -- ``"out_features"``,
    ``"in_channels // groups"`` -- or ``"*kernel_size"`` to splat a list
    attribute. ``when`` names an attribute that must be true for the parameter
    to exist at all, which is how an optional bias is expressed.
    """

    name: str
    shape: tuple[str | int, ...]
    when: str | None = None
    #: Fill with ones rather than a random draw (normalisation scales).
    ones: bool = False
    #: Fill with zeros (biases, running means).
    zeros: bool = False
    #: The matching entry in the torch module state dict, which is what lets
    #: the parity harness give both backends identical weights.
    torch_name: str = ""


@dataclass(frozen=True, slots=True)
class OnnxMapping:
    """How one op is expressed as an ONNX node."""

    op_type: str
    domain: str = ""
    since_opset: int = 13
    attr_map: Mapping[str, str] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    #: Learned tensors this node takes, in the order ONNX expects them.
    params: tuple[ParamSpec, ...] = ()
    #: Final input order as tokens naming a port or a parameter. Defaults to
    #: every port in declaration order followed by every parameter; Gather
    #: needs it because the weight comes before the indices.
    input_order: tuple[str, ...] = ()
    #: NTB-defined op: emits into the ``ntb.ops`` domain with a reference impl.
    custom: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ParityCase:
    """A concrete example the cross-backend numeric test is generated from.

    Declaring one is part of declaring an op: without it the harness has no
    shapes to feed and the op ships unverified.
    """

    inputs: Mapping[str, tuple[int, ...]]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    #: Ports that take integer indices rather than a random float draw.
    integer_inputs: tuple[str, ...] = ()
    #: Upper bound for those indices.
    index_limit: int = 8


@dataclass(frozen=True, slots=True)
class OpSpec:
    """The complete declaration of one canonical op."""

    name: str
    category: str
    doc: str
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    shape_rule: ShapeRule
    attrs: tuple[AttrSpec, ...] = ()
    torch: BackendMapping | None = None
    keras: BackendMapping | None = None
    onnx: OnnxMapping | None = None
    #: Absolute tolerance used by the generated numeric-parity tests.
    parity_atol: float = 1e-5
    #: The example the parity harness runs. None means the op is unverified.
    parity: ParityCase | None = None

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError(f"op {self.name!r} declares no outputs")
        for group, label in ((self.inputs, "input"), (self.outputs, "output")):
            names = [p.name for p in group]
            if len(set(names)) != len(names):
                raise ValueError(f"op {self.name!r} declares a duplicate {label} port")
        attr_names = [a.name for a in self.attrs]
        if len(set(attr_names)) != len(attr_names):
            raise ValueError(f"op {self.name!r} declares a duplicate attribute")

    @property
    def attr_specs(self) -> dict[str, AttrSpec]:
        return {a.name: a for a in self.attrs}

    def defaults(self) -> dict[str, Any]:
        return {a.name: a.default for a in self.attrs if a.default is not None}

    def resolved_attrs(self, attrs: Mapping[str, Any]) -> dict[str, Any]:
        """Author-supplied attributes over the declared defaults, fallbacks filled."""
        resolved = {**self.defaults(), **attrs}
        for attr in self.attrs:
            if attr.default_from and resolved.get(attr.name) is None:
                fallback = resolved.get(attr.default_from)
                if fallback is not None:
                    resolved[attr.name] = fallback
        return resolved

    def backends(self) -> tuple[str, ...]:
        """Which backends can emit this op."""
        available = []
        if self.torch is not None:
            available.append("torch")
        if self.keras is not None:
            available.append("keras")
        if self.onnx is not None:
            available.append("onnx")
        return tuple(available)
