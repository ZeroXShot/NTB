"""Pooling over channels-first tensors."""

from __future__ import annotations

from ntb.ir.types import TensorType
from ntb.ops.builtin._common import ntuple, require_float
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
from ntb.shapes.symbolic import ShapeError, conv_out_dim


def _pool_rule(spatial: int) -> ShapeRule:
    def rule(ctx: ShapeContext) -> dict[str, TensorType]:
        x = ctx.input("in")
        require_float(ctx, x)
        if x.rank != spatial + 2:
            ctx.fail(f"input must be rank {spatial + 2}, got rank {x.rank}")

        kernel = ntuple(ctx, "kernel_size", spatial)
        stride = ntuple(ctx, "stride", spatial) if ctx.attrs.get("stride") else kernel
        padding = ntuple(ctx, "padding", spatial)

        try:
            sizes = tuple(
                conv_out_dim(
                    x.shape[2 + axis],
                    kernel=kernel[axis],
                    stride=stride[axis],
                    padding=padding[axis],
                )
                for axis in range(spatial)
            )
        except ShapeError as exc:
            ctx.fail(str(exc))

        return {"out": TensorType(dtype=x.dtype, shape=(*x.shape[:2], *sizes), layout=x.layout)}

    return rule


def _pool(kind: str, spatial: int, *, torch_cls: str, keras_cls: str, onnx_op: str) -> OpSpec:
    return register(
        OpSpec(
            name=f"ntb.{kind}pool{spatial}d",
            category="pooling",
            doc=f"{kind.capitalize()} pooling over {spatial} spatial dimensions.",
            inputs=(PortSpec("in"),),
            outputs=(PortSpec("out"),),
            attrs=(
                AttrSpec("kernel_size", AttrType.INTS, required=True, minimum=1),
                AttrSpec(
                    "stride",
                    AttrType.INTS,
                    minimum=1,
                    doc="Defaults to kernel_size, matching torch and Keras.",
                ),
                AttrSpec("padding", AttrType.INTS, default=[0] * spatial, minimum=0),
            ),
            shape_rule=_pool_rule(spatial),
            torch=BackendMapping(
                target=torch_cls,
                attr_map={
                    "kernel_size": "kernel_size",
                    "stride": "stride",
                    "padding": "padding",
                },
                imports=("torch", "torch.nn"),
            ),
            keras=BackendMapping(
                target=keras_cls,
                attr_map={"kernel_size": "pool_size", "stride": "strides"},
                constants={"data_format": "channels_first", "padding": "valid"},
                imports=("keras",),
            ),
            onnx=OnnxMapping(
                op_type=onnx_op,
                attr_map={"kernel_size": "kernel_shape", "stride": "strides"},
            ),
        )
    )


MAXPOOL1D = _pool(
    "max",
    1,
    torch_cls="torch.nn.MaxPool1d",
    keras_cls="keras.layers.MaxPooling1D",
    onnx_op="MaxPool",
)
MAXPOOL2D = _pool(
    "max",
    2,
    torch_cls="torch.nn.MaxPool2d",
    keras_cls="keras.layers.MaxPooling2D",
    onnx_op="MaxPool",
)
MAXPOOL3D = _pool(
    "max",
    3,
    torch_cls="torch.nn.MaxPool3d",
    keras_cls="keras.layers.MaxPooling3D",
    onnx_op="MaxPool",
)
AVGPOOL2D = _pool(
    "avg",
    2,
    torch_cls="torch.nn.AvgPool2d",
    keras_cls="keras.layers.AveragePooling2D",
    onnx_op="AveragePool",
)


def _global_avgpool(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    require_float(ctx, x)
    if x.rank < 3:
        ctx.fail(f"input needs at least one spatial axis, got rank {x.rank}")
    keep = ctx.attr("keepdims")
    spatial: tuple[int, ...] = (1,) * (x.rank - 2) if keep else ()
    return {"out": TensorType(dtype=x.dtype, shape=(*x.shape[:2], *spatial), layout=x.layout)}


GLOBAL_AVGPOOL = register(
    OpSpec(
        name="ntb.global_avgpool",
        category="pooling",
        doc="Averages every spatial axis away, leaving batch and channels.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("keepdims", AttrType.BOOL, default=False),),
        shape_rule=_global_avgpool,
        torch=BackendMapping(
            target="torch.nn.AdaptiveAvgPool2d",
            constants={"output_size": 1},
            imports=("torch", "torch.nn"),
            notes="The emitter picks the 1d/2d/3d variant and squeezes if needed.",
        ),
        keras=BackendMapping(
            target="keras.layers.GlobalAveragePooling2D",
            attr_map={"keepdims": "keepdims"},
            constants={"data_format": "channels_first"},
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="GlobalAveragePool"),
    )
)

OPS = (MAXPOOL1D, MAXPOOL2D, MAXPOOL3D, AVGPOOL2D, GLOBAL_AVGPOOL)
