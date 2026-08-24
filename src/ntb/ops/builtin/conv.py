"""Convolutions in 1, 2 and 3 dimensions.

All three share one shape rule parameterised by spatial rank, so the 3D case
cannot drift from the 2D one.
"""

from __future__ import annotations

from ntb.ir.types import TensorType
from ntb.ops.builtin._common import ntuple, require_float, require_rank
from ntb.ops.registry import register
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    OnnxMapping,
    OpSpec,
    PortSpec,
    ShapeContext,
    ShapeRule,
)
from ntb.shapes.symbolic import ShapeError, conv_out_dim, dims_equal

_LAYOUTS = {1: "NCL", 2: "NCHW", 3: "NCDHW"}


def _conv_rule(spatial: int) -> ShapeRule:
    def rule(ctx: ShapeContext) -> dict[str, TensorType]:
        x = ctx.input("in")
        require_rank(ctx, x, spatial + 2)
        require_float(ctx, x)

        in_channels = ctx.attr("in_channels")
        out_channels = ctx.attr("out_channels")
        if dims_equal(x.shape[1], in_channels) is False:
            ctx.fail(f"input has {x.shape[1]} channels, but in_channels is {in_channels}")

        groups = ctx.attr("groups")
        if in_channels % groups or out_channels % groups:
            ctx.fail(
                f"groups={groups} must divide both in_channels={in_channels} "
                f"and out_channels={out_channels}"
            )

        kernel = ntuple(ctx, "kernel_size", spatial)
        stride = ntuple(ctx, "stride", spatial)
        padding = ntuple(ctx, "padding", spatial)
        dilation = ntuple(ctx, "dilation", spatial)

        # Callers of a shape rule should only have to catch ShapeRuleError.
        try:
            sizes = tuple(
                conv_out_dim(
                    x.shape[2 + axis],
                    kernel=kernel[axis],
                    stride=stride[axis],
                    padding=padding[axis],
                    dilation=dilation[axis],
                )
                for axis in range(spatial)
            )
        except ShapeError as exc:
            ctx.fail(str(exc))

        return {
            "out": TensorType(
                dtype=x.dtype,
                shape=(x.shape[0], out_channels, *sizes),
                layout=_LAYOUTS[spatial],
            )
        }

    return rule


def _conv(spatial: int) -> OpSpec:
    default = [3] * spatial
    return register(
        OpSpec(
            name=f"ntb.conv{spatial}d",
            category="convolution",
            doc=f"{spatial}D cross-correlation over an {_LAYOUTS[spatial]} tensor.",
            inputs=(PortSpec("in", doc=f"{_LAYOUTS[spatial]} tensor."),),
            outputs=(PortSpec("out"),),
            attrs=(
                AttrSpec("in_channels", AttrType.INT, required=True, minimum=1),
                AttrSpec("out_channels", AttrType.INT, required=True, minimum=1),
                AttrSpec("kernel_size", AttrType.INTS, default=default, minimum=1),
                AttrSpec("stride", AttrType.INTS, default=[1] * spatial, minimum=1),
                AttrSpec("padding", AttrType.INTS, default=[0] * spatial, minimum=0),
                AttrSpec("dilation", AttrType.INTS, default=[1] * spatial, minimum=1),
                AttrSpec("groups", AttrType.INT, default=1, minimum=1),
                AttrSpec("bias", AttrType.BOOL, default=True),
            ),
            shape_rule=_conv_rule(spatial),
            torch=BackendMapping(
                target=f"torch.nn.Conv{spatial}d",
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
                target=f"keras.layers.Conv{spatial}D",
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
                notes="No integer padding in Keras; non-zero lowers to ZeroPadding.",
            ),
            onnx=OnnxMapping(
                op_type="Conv",
                attr_map={
                    "kernel_size": "kernel_shape",
                    "stride": "strides",
                    "dilation": "dilations",
                    "groups": "group",
                },
                notes="Pads are begins then ends per axis; the emitter expands them.",
            ),
        )
    )


CONV1D = _conv(1)
CONV2D = _conv(2)
CONV3D = _conv(3)

OPS = (CONV1D, CONV2D, CONV3D)
