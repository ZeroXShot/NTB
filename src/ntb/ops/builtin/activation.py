"""Elementwise activations, plus softmax."""

from __future__ import annotations

from ntb.ir.types import TensorType
from ntb.ops.builtin._common import elementwise, require_float
from ntb.ops.registry import register
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    PortSpec,
    ShapeContext,
)
from ntb.shapes.symbolic import ShapeError, normalise_axis


def _simple(
    name: str,
    doc: str,
    *,
    torch_target: str,
    keras_target: str,
    onnx_op: str,
    since_opset: int = 13,
) -> OpSpec:
    return register(
        OpSpec(
            name=name,
            category="activation",
            doc=doc,
            inputs=(PortSpec("in"),),
            outputs=(PortSpec("out"),),
            shape_rule=elementwise,
            torch=BackendMapping(
                target=torch_target,
                kind=CallKind.FUNCTION,
                imports=("torch", "torch.nn.functional"),
            ),
            keras=BackendMapping(target=keras_target, kind=CallKind.FUNCTION, imports=("keras",)),
            onnx=OnnxMapping(op_type=onnx_op, since_opset=since_opset),
        )
    )


RELU = _simple(
    "ntb.relu",
    "Elementwise ``max(x, 0)``.",
    torch_target="torch.nn.functional.relu",
    keras_target="keras.activations.relu",
    onnx_op="Relu",
    since_opset=14,
)
SIGMOID = _simple(
    "ntb.sigmoid",
    "Elementwise logistic sigmoid.",
    torch_target="torch.sigmoid",
    keras_target="keras.activations.sigmoid",
    onnx_op="Sigmoid",
)
TANH = _simple(
    "ntb.tanh",
    "Elementwise hyperbolic tangent.",
    torch_target="torch.tanh",
    keras_target="keras.activations.tanh",
    onnx_op="Tanh",
)
SILU = register(
    OpSpec(
        name="ntb.silu",
        category="activation",
        doc="Elementwise ``x * sigmoid(x)``, also known as swish.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_rule=elementwise,
        torch=BackendMapping(
            target="torch.nn.functional.silu",
            kind=CallKind.FUNCTION,
            imports=("torch", "torch.nn.functional"),
        ),
        keras=BackendMapping(
            target="keras.activations.silu", kind=CallKind.FUNCTION, imports=("keras",)
        ),
        onnx=OnnxMapping(
            op_type="Silu",
            custom=True,
            notes="No standard ONNX op; emitted as Mul(x, Sigmoid(x)).",
        ),
    )
)

GELU = register(
    OpSpec(
        name="ntb.gelu",
        category="activation",
        doc="Gaussian error linear unit.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec(
                "approximate",
                AttrType.STRING,
                default="none",
                choices=("none", "tanh"),
                doc="The tanh approximation is what most transformers use.",
            ),
        ),
        shape_rule=elementwise,
        torch=BackendMapping(
            target="torch.nn.functional.gelu",
            kind=CallKind.FUNCTION,
            attr_map={"approximate": "approximate"},
            imports=("torch", "torch.nn.functional"),
        ),
        keras=BackendMapping(
            target="keras.activations.gelu",
            kind=CallKind.FUNCTION,
            imports=("keras",),
            notes="Takes approximate as a bool; the emitter maps 'tanh' to True.",
        ),
        onnx=OnnxMapping(op_type="Gelu", since_opset=20, attr_map={"approximate": "approximate"}),
    )
)


def _softmax(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    require_float(ctx, x)
    try:
        normalise_axis(ctx.attr("axis"), x.rank)
    except ShapeError as exc:
        ctx.fail(str(exc))
    return {"out": x}


SOFTMAX = register(
    OpSpec(
        name="ntb.softmax",
        category="activation",
        doc="Normalises one axis to sum to one.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("axis", AttrType.INT, default=-1),),
        shape_rule=_softmax,
        torch=BackendMapping(
            target="torch.nn.functional.softmax",
            kind=CallKind.FUNCTION,
            attr_map={"axis": "dim"},
            imports=("torch", "torch.nn.functional"),
        ),
        keras=BackendMapping(
            target="keras.ops.softmax",
            kind=CallKind.FUNCTION,
            attr_map={"axis": "axis"},
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="Softmax", attr_map={"axis": "axis"}),
    )
)

OPS = (RELU, SIGMOID, TANH, SILU, GELU, SOFTMAX)
