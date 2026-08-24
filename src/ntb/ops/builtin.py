"""The built-in op set.

An op needing code outside this file means the registry is missing a knob.
"""

from __future__ import annotations

from ntb.ir.types import TensorType
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
from ntb.shapes.symbolic import ShapeError, conv_out_dim, dims_equal


def _linear_shape(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    if x.rank < 1:
        ctx.fail("input must have at least one dimension")
    in_features = ctx.attr("in_features")
    if dims_equal(x.shape[-1], in_features) is False:
        ctx.fail(f"last input dimension is {x.shape[-1]}, but in_features is {in_features}")
    if not x.dtype.is_floating:
        ctx.fail(f"expected a floating dtype, got {x.dtype.value}")
    shape = (*x.shape[:-1], ctx.attr("out_features"))
    return {"out": TensorType(dtype=x.dtype, shape=shape)}


LINEAR = register(
    OpSpec(
        name="ntb.linear",
        category="dense",
        doc="Affine map over the last dimension: ``y = x @ W.T + b``.",
        inputs=(PortSpec("in", doc="Tensor whose last dimension is `in_features`."),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("in_features", AttrType.INT, required=True, minimum=1),
            AttrSpec("out_features", AttrType.INT, required=True, minimum=1),
            AttrSpec("bias", AttrType.BOOL, default=True, doc="Learn an additive bias."),
        ),
        shape_rule=_linear_shape,
        torch=BackendMapping(
            target="torch.nn.Linear",
            kind=CallKind.MODULE,
            attr_map={"in_features": "in_features", "out_features": "out_features", "bias": "bias"},
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.Dense",
            kind=CallKind.MODULE,
            attr_map={"out_features": "units", "bias": "use_bias"},
            imports=("keras",),
            notes="Keras infers in_features on first call.",
        ),
        onnx=OnnxMapping(op_type="Gemm", since_opset=13, constants={"transB": 1}),
    )
)


def _relu_shape(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    if not x.dtype.is_floating:
        ctx.fail(f"expected a floating dtype, got {x.dtype.value}")
    return {"out": x}


RELU = register(
    OpSpec(
        name="ntb.relu",
        category="activation",
        doc="Elementwise ``max(x, 0)``.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_rule=_relu_shape,
        torch=BackendMapping(
            target="torch.nn.functional.relu",
            kind=CallKind.FUNCTION,
            imports=("torch", "torch.nn.functional"),
        ),
        keras=BackendMapping(
            target="keras.activations.relu",
            kind=CallKind.FUNCTION,
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="Relu", since_opset=14),
    )
)


def _conv2d_shape(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    if x.rank != 4:
        ctx.fail(f"expected a rank-4 NCHW tensor, got rank {x.rank}")
    if not x.dtype.is_floating:
        ctx.fail(f"expected a floating dtype, got {x.dtype.value}")

    in_channels = ctx.attr("in_channels")
    if dims_equal(x.shape[1], in_channels) is False:
        ctx.fail(f"input has {x.shape[1]} channels, but in_channels is {in_channels}")

    groups = ctx.attr("groups")
    if in_channels % groups or ctx.attr("out_channels") % groups:
        ctx.fail(
            f"groups={groups} must divide both in_channels={in_channels} "
            f"and out_channels={ctx.attr('out_channels')}"
        )

    kernel = _pair(ctx, "kernel_size")
    stride = _pair(ctx, "stride")
    padding = _pair(ctx, "padding")
    dilation = _pair(ctx, "dilation")

    # Callers of a shape rule should only have to catch ShapeRuleError.
    try:
        spatial = tuple(
            conv_out_dim(
                x.shape[2 + axis],
                kernel=kernel[axis],
                stride=stride[axis],
                padding=padding[axis],
                dilation=dilation[axis],
            )
            for axis in (0, 1)
        )
    except ShapeError as exc:
        ctx.fail(str(exc))
        raise AssertionError from exc  # unreachable: ctx.fail always raises
    shape = (x.shape[0], ctx.attr("out_channels"), *spatial)
    return {"out": TensorType(dtype=x.dtype, shape=shape, layout="NCHW")}


def _pair(ctx: ShapeContext, name: str) -> tuple[int, int]:
    """Read an attribute that is either one int or two, as (rows, cols)."""
    value = ctx.attr(name)
    if isinstance(value, int):
        return (value, value)
    if len(value) == 1:
        return (value[0], value[0])
    if len(value) == 2:
        return (value[0], value[1])
    ctx.fail(f"{name} must be 1 or 2 integers, got {len(value)}")
    raise AssertionError  # unreachable: ctx.fail always raises


CONV2D = register(
    OpSpec(
        name="ntb.conv2d",
        category="convolution",
        doc="2D cross-correlation over an NCHW tensor.",
        inputs=(PortSpec("in", doc="NCHW tensor."),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("in_channels", AttrType.INT, required=True, minimum=1),
            AttrSpec("out_channels", AttrType.INT, required=True, minimum=1),
            AttrSpec("kernel_size", AttrType.INTS, default=[3, 3], minimum=1),
            AttrSpec("stride", AttrType.INTS, default=[1, 1], minimum=1),
            AttrSpec("padding", AttrType.INTS, default=[0, 0], minimum=0),
            AttrSpec("dilation", AttrType.INTS, default=[1, 1], minimum=1),
            AttrSpec("groups", AttrType.INT, default=1, minimum=1),
            AttrSpec("bias", AttrType.BOOL, default=True),
        ),
        shape_rule=_conv2d_shape,
        torch=BackendMapping(
            target="torch.nn.Conv2d",
            kind=CallKind.MODULE,
            attr_map={
                "in_channels": "in_channels",
                "out_channels": "out_channels",
                "kernel_size": "kernel_size",
                "stride": "stride",
                "padding": "padding",
                "dilation": "dilation",
                "groups": "groups",
                "bias": "bias",
            },
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.Conv2D",
            kind=CallKind.MODULE,
            attr_map={
                "out_channels": "filters",
                "kernel_size": "kernel_size",
                "stride": "strides",
                "dilation": "dilation_rate",
                "groups": "groups",
                "bias": "use_bias",
            },
            constants={"data_format": "channels_first", "padding": "valid"},
            imports=("keras",),
            notes="No integer padding in Keras; non-zero lowers to ZeroPadding2D.",
        ),
        onnx=OnnxMapping(
            op_type="Conv",
            since_opset=13,
            attr_map={
                "kernel_size": "kernel_shape",
                "stride": "strides",
                "dilation": "dilations",
                "groups": "group",
            },
            notes="Pads are [top, left, bottom, right]; the emitter expands them.",
        ),
    )
)

#: Every op declared in this module, in registration order.
BUILTIN_OPS = (LINEAR, RELU, CONV2D)

__all__ = ["BUILTIN_OPS", "CONV2D", "LINEAR", "RELU"]
