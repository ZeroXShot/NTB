"""Binary elementwise arithmetic with broadcasting."""

from __future__ import annotations

from ntb.ir.types import TensorType
from ntb.ops.registry import register
from ntb.ops.spec import (
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    ParityCase,
    PortSpec,
    ShapeContext,
)
from ntb.shapes.symbolic import ShapeError, broadcast


def _binary(ctx: ShapeContext) -> dict[str, TensorType]:
    a, b = ctx.input("a"), ctx.input("b")
    if a.dtype is not b.dtype:
        ctx.fail(f"operands must share a dtype, got {a.dtype.value} and {b.dtype.value}")
    try:
        shape = broadcast(a.shape, b.shape)
    except ShapeError as exc:
        ctx.fail(str(exc))
    layout = a.layout if a.rank >= b.rank else b.layout
    return {"out": TensorType(dtype=a.dtype, shape=shape, layout=layout)}


def _fan_in(ctx: ShapeContext) -> dict[str, TensorType]:
    """Sum everything arriving at one port, whatever arrives."""
    arriving = ctx.many("in")
    dtypes = {tensor.dtype for tensor in arriving}
    if len(dtypes) > 1:
        ctx.fail(
            "every operand must share a dtype, got " + ", ".join(sorted(d.value for d in dtypes))
        )
    shape = arriving[0].shape
    try:
        for tensor in arriving[1:]:
            shape = broadcast(shape, tensor.shape)
    except ShapeError as exc:
        ctx.fail(str(exc))
    return {"out": TensorType(dtype=arriving[0].dtype, shape=shape, layout=arriving[0].layout)}


def _op(name: str, doc: str, *, torch_fn: str, keras_fn: str, onnx_op: str) -> OpSpec:
    return register(
        OpSpec(
            name=name,
            category="elementwise",
            doc=doc,
            inputs=(PortSpec("a"), PortSpec("b")),
            outputs=(PortSpec("out"),),
            shape_rule=_binary,
            torch=BackendMapping(target=torch_fn, kind=CallKind.FUNCTION, imports=("torch",)),
            keras=BackendMapping(target=keras_fn, kind=CallKind.FUNCTION, imports=("keras",)),
            onnx=OnnxMapping(op_type=onnx_op),
            parity=ParityCase(inputs={"a": (2, 4), "b": (2, 4)}),
        )
    )


ADD = _op(
    "ntb.add",
    "Elementwise sum. This is what a residual connection is made of.",
    torch_fn="torch.add",
    keras_fn="keras.ops.add",
    onnx_op="Add",
)
SUB = _op(
    "ntb.sub",
    "Elementwise difference.",
    torch_fn="torch.sub",
    keras_fn="keras.ops.subtract",
    onnx_op="Sub",
)
MUL = _op(
    "ntb.mul",
    "Elementwise product.",
    torch_fn="torch.mul",
    keras_fn="keras.ops.multiply",
    onnx_op="Mul",
)
DIV = _op(
    "ntb.div",
    "Elementwise quotient.",
    torch_fn="torch.div",
    keras_fn="keras.ops.divide",
    onnx_op="Div",
)

SUM = register(
    OpSpec(
        name="ntb.sum",
        category="elementwise",
        doc=(
            "Sums every tensor wired into it. The port is variadic, which is what "
            "lets a spatial rule fan several blocks into one."
        ),
        inputs=(PortSpec("in", variadic=True),),
        outputs=(PortSpec("out"),),
        shape_rule=_fan_in,
        torch=BackendMapping(target="sum", kind=CallKind.FUNCTION),
        keras=BackendMapping(target="keras.layers.Add", imports=("keras",)),
        onnx=OnnxMapping(op_type="Sum"),
        parity=ParityCase(inputs={}, fan_in={"in": ((2, 4), (2, 4), (2, 4))}),
    )
)

OPS = (ADD, SUB, MUL, DIV, SUM)
