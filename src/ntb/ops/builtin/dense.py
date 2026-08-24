"""Dense and lookup ops."""

from __future__ import annotations

from ntb.ir.types import DType, TensorType
from ntb.ops.builtin._common import require_float
from ntb.ops.registry import register
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    ParamSpec,
    ParityCase,
    PortSpec,
    ShapeContext,
)
from ntb.shapes.symbolic import ShapeError, broadcast, dims_equal


def _linear(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    require_float(ctx, x)
    if x.rank < 1:
        ctx.fail("input must have at least one dimension")
    in_features = ctx.attr("in_features")
    if dims_equal(x.shape[-1], in_features) is False:
        ctx.fail(f"last input dimension is {x.shape[-1]}, but in_features is {in_features}")
    return {"out": TensorType(dtype=x.dtype, shape=(*x.shape[:-1], ctx.attr("out_features")))}


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
            AttrSpec("bias", AttrType.BOOL, default=True),
        ),
        shape_rule=_linear,
        torch=BackendMapping(
            target="torch.nn.Linear",
            attr_map={
                "in_features": "in_features",
                "out_features": "out_features",
                "bias": "bias",
            },
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.Dense",
            attr_map={"out_features": "units", "bias": "use_bias"},
            imports=("keras",),
            notes="Keras infers in_features on first call.",
        ),
        onnx=OnnxMapping(
            op_type="Gemm",
            constants={"transB": 1},
            params=(
                ParamSpec("W", ("out_features", "in_features"), torch_name="weight"),
                ParamSpec("B", ("out_features",), when="bias", zeros=True, torch_name="bias"),
            ),
        ),
        parity=ParityCase(inputs={"in": (3, 8)}, attrs={"in_features": 8, "out_features": 5}),
    )
)


def _embedding(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    if not x.dtype.is_integer:
        ctx.fail(f"indices must be an integer dtype, got {x.dtype.value}")
    return {"out": TensorType(dtype=DType.FLOAT32, shape=(*x.shape, ctx.attr("embedding_dim")))}


EMBEDDING = register(
    OpSpec(
        name="ntb.embedding",
        category="dense",
        doc="Row lookup into a learned table; appends `embedding_dim`.",
        inputs=(PortSpec("in", doc="Integer index tensor."),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("num_embeddings", AttrType.INT, required=True, minimum=1),
            AttrSpec("embedding_dim", AttrType.INT, required=True, minimum=1),
            AttrSpec("padding_idx", AttrType.INT, minimum=0),
        ),
        shape_rule=_embedding,
        torch=BackendMapping(
            target="torch.nn.Embedding",
            attr_map={
                "num_embeddings": "num_embeddings",
                "embedding_dim": "embedding_dim",
                "padding_idx": "padding_idx",
            },
            imports=("torch", "torch.nn"),
        ),
        keras=BackendMapping(
            target="keras.layers.Embedding",
            attr_map={"num_embeddings": "input_dim", "embedding_dim": "output_dim"},
            imports=("keras",),
            notes="No padding_idx; mask_zero only covers index 0.",
        ),
        onnx=OnnxMapping(
            op_type="Gather",
            params=(ParamSpec("W", ("num_embeddings", "embedding_dim"), torch_name="weight"),),
            input_order=("W", "in"),
        ),
        parity=ParityCase(
            inputs={"in": (2, 4)},
            attrs={"num_embeddings": 8, "embedding_dim": 5},
            integer_inputs=("in",),
        ),
    )
)


def _matmul(ctx: ShapeContext) -> dict[str, TensorType]:
    a, b = ctx.input("a"), ctx.input("b")
    require_float(ctx, a, "a")
    require_float(ctx, b, "b")
    if a.rank < 2 or b.rank < 2:
        ctx.fail(f"both operands must be rank >= 2, got {a.rank} and {b.rank}")
    if dims_equal(a.shape[-1], b.shape[-2]) is False:
        ctx.fail(f"inner dimensions disagree: {a.shape[-1]} against {b.shape[-2]}")
    try:
        batch = broadcast(a.shape[:-2], b.shape[:-2])
    except ShapeError as exc:
        ctx.fail(f"batch dimensions do not broadcast: {exc}")
    return {"out": TensorType(dtype=a.dtype, shape=(*batch, a.shape[-2], b.shape[-1]))}


MATMUL = register(
    OpSpec(
        name="ntb.matmul",
        category="dense",
        doc="Batched matrix product with broadcast batch dimensions.",
        inputs=(PortSpec("a"), PortSpec("b")),
        outputs=(PortSpec("out"),),
        shape_rule=_matmul,
        torch=BackendMapping(target="torch.matmul", kind=CallKind.FUNCTION, imports=("torch",)),
        keras=BackendMapping(target="keras.ops.matmul", kind=CallKind.FUNCTION, imports=("keras",)),
        onnx=OnnxMapping(op_type="MatMul"),
        parity=ParityCase(inputs={"a": (2, 3, 4), "b": (2, 4, 5)}),
    )
)

OPS = (LINEAR, EMBEDDING, MATMUL)
