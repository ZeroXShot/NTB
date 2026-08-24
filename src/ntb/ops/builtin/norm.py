"""Normalisation and regularisation."""

from __future__ import annotations

from ntb.ir.types import TensorType
from ntb.ops.builtin._common import elementwise, require_float
from ntb.ops.registry import register
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    OnnxMapping,
    OpSpec,
    ParamSpec,
    ParityCase,
    PortSpec,
    ShapeContext,
)
from ntb.shapes.symbolic import dims_equal


def _layernorm(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    require_float(ctx, x)
    shape = ctx.attr("normalized_shape")
    if len(shape) > x.rank:
        ctx.fail(f"normalized_shape has {len(shape)} dimensions but the input is rank {x.rank}")
    for offset, expected in enumerate(shape, start=x.rank - len(shape)):
        if dims_equal(x.shape[offset], expected) is False:
            ctx.fail(f"axis {offset} is {x.shape[offset]}, but normalized_shape says {expected}")
    return {"out": x}


LAYERNORM = register(
    OpSpec(
        name="ntb.layernorm",
        category="normalisation",
        doc="Normalises over the trailing axes named by `normalized_shape`.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("normalized_shape", AttrType.INTS, required=True, minimum=1),
            AttrSpec("eps", AttrType.FLOAT, default=1e-5, minimum=0.0),
            AttrSpec("affine", AttrType.BOOL, default=True),
        ),
        shape_rule=_layernorm,
        torch=BackendMapping(
            target="torch.nn.LayerNorm",
            attr_map={
                "normalized_shape": "normalized_shape",
                "eps": "eps",
                "affine": "elementwise_affine",
            },
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.LayerNormalization",
            attr_map={"eps": "epsilon", "affine": "scale"},
            imports=("keras",),
            notes="Axes are derived from normalized_shape by the emitter.",
        ),
        onnx=OnnxMapping(
            op_type="LayerNormalization",
            since_opset=17,
            attr_map={"eps": "epsilon"},
            params=(
                ParamSpec("scale", ("*normalized_shape",), ones=True, torch_name="weight"),
                ParamSpec(
                    "bias",
                    ("*normalized_shape",),
                    when="affine",
                    zeros=True,
                    torch_name="bias",
                ),
            ),
        ),
        parity=ParityCase(inputs={"in": (2, 3, 6)}, attrs={"normalized_shape": [6]}),
    )
)


def _rmsnorm(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    require_float(ctx, x)
    size = ctx.attr("normalized_size")
    if x.rank and dims_equal(x.shape[-1], size) is False:
        ctx.fail(f"last dimension is {x.shape[-1]}, but normalized_size is {size}")
    return {"out": x}


RMSNORM = register(
    OpSpec(
        name="ntb.rmsnorm",
        category="normalisation",
        doc="Root-mean-square normalisation over the last dimension.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("normalized_size", AttrType.INT, required=True, minimum=1),
            AttrSpec("eps", AttrType.FLOAT, default=1e-6, minimum=0.0),
        ),
        shape_rule=_rmsnorm,
        torch=BackendMapping(
            target="torch.nn.RMSNorm",
            attr_map={"normalized_size": "normalized_shape", "eps": "eps"},
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.RMSNormalization",
            attr_map={"eps": "epsilon"},
            imports=("keras",),
        ),
        onnx=OnnxMapping(
            op_type="RMSNormalization",
            since_opset=23,
            attr_map={"eps": "epsilon"},
            params=(ParamSpec("scale", ("normalized_size",), ones=True),),
        ),
    )
)


def _batchnorm(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    require_float(ctx, x)
    if x.rank < 2:
        ctx.fail(f"input must be rank >= 2 to have a channel axis, got rank {x.rank}")
    features = ctx.attr("num_features")
    if dims_equal(x.shape[1], features) is False:
        ctx.fail(f"channel axis is {x.shape[1]}, but num_features is {features}")
    return {"out": x}


BATCHNORM = register(
    OpSpec(
        name="ntb.batchnorm",
        category="normalisation",
        doc="Normalises each channel over the batch and spatial axes.",
        inputs=(PortSpec("in", doc="Channels-first tensor."),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("num_features", AttrType.INT, required=True, minimum=1),
            AttrSpec("eps", AttrType.FLOAT, default=1e-5, minimum=0.0),
            AttrSpec("momentum", AttrType.FLOAT, default=0.1, minimum=0.0),
            AttrSpec("affine", AttrType.BOOL, default=True),
        ),
        shape_rule=_batchnorm,
        torch=BackendMapping(
            target="torch.nn.BatchNorm2d",
            attr_map={
                "num_features": "num_features",
                "eps": "eps",
                "momentum": "momentum",
                "affine": "affine",
            },
            rank_targets={
                2: "torch.nn.BatchNorm1d",
                3: "torch.nn.BatchNorm1d",
                4: "torch.nn.BatchNorm2d",
                5: "torch.nn.BatchNorm3d",
            },
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.BatchNormalization",
            attr_map={"eps": "epsilon", "momentum": "momentum", "affine": "scale"},
            constants={"axis": 1},
            imports=("keras",),
        ),
        onnx=OnnxMapping(
            op_type="BatchNormalization",
            since_opset=15,
            attr_map={"eps": "epsilon", "momentum": "momentum"},
            params=(
                ParamSpec("scale", ("num_features",), ones=True, torch_name="weight"),
                ParamSpec("bias", ("num_features",), zeros=True, torch_name="bias"),
                ParamSpec("mean", ("num_features",), zeros=True, torch_name="running_mean"),
                ParamSpec("var", ("num_features",), ones=True, torch_name="running_var"),
            ),
        ),
        parity=ParityCase(inputs={"in": (2, 4, 5, 5)}, attrs={"num_features": 4}),
    )
)

DROPOUT = register(
    OpSpec(
        name="ntb.dropout",
        category="normalisation",
        doc="Zeroes elements at random during training; identity at inference.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("p", AttrType.FLOAT, default=0.5, minimum=0.0),),
        shape_rule=elementwise,
        torch=BackendMapping(
            target="torch.nn.Dropout",
            attr_map={"p": "p"},
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.Dropout", attr_map={"p": "rate"}, imports=("keras",)
        ),
        onnx=OnnxMapping(op_type="Dropout", since_opset=13),
    )
)

OPS = (LAYERNORM, RMSNORM, BATCHNORM, DROPOUT)
