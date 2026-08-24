"""Multi-head attention."""

from __future__ import annotations

from ntb.ir.types import TensorType
from ntb.ops.builtin._common import require_float
from ntb.ops.registry import register
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    OnnxMapping,
    OpSpec,
    PortSpec,
    ShapeContext,
)
from ntb.shapes.symbolic import dims_equal


def _attention(ctx: ShapeContext) -> dict[str, TensorType]:
    query = ctx.input("query")
    require_float(ctx, query, "query")
    if query.rank != 3:
        ctx.fail(f"query must be rank 3 (batch, seq, embed), got rank {query.rank}")

    embed_dim = ctx.attr("embed_dim")
    num_heads = ctx.attr("num_heads")
    if embed_dim % num_heads:
        ctx.fail(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
    if dims_equal(query.shape[2], embed_dim) is False:
        ctx.fail(f"query embedding is {query.shape[2]}, but embed_dim is {embed_dim}")

    for port in ("key", "value"):
        other = ctx.inputs.get(port)
        if other is None:
            continue
        if other.rank != 3:
            ctx.fail(f"{port} must be rank 3, got rank {other.rank}")
        if dims_equal(other.shape[2], embed_dim) is False:
            ctx.fail(f"{port} embedding is {other.shape[2]}, but embed_dim is {embed_dim}")

    key = ctx.inputs.get("key", query)
    out = TensorType(dtype=query.dtype, shape=(*query.shape[:2], embed_dim), layout="NLC")
    weights = TensorType(
        dtype=query.dtype, shape=(query.shape[0], num_heads, query.shape[1], key.shape[1])
    )
    return {"out": out, "weights": weights}


ATTENTION = register(
    OpSpec(
        name="ntb.attention",
        category="attention",
        doc="Scaled dot-product attention over `num_heads` heads.",
        inputs=(
            PortSpec("query", doc="(batch, seq, embed_dim)."),
            PortSpec("key", optional=True, doc="Defaults to query (self-attention)."),
            PortSpec("value", optional=True, doc="Defaults to key."),
            PortSpec("mask", optional=True),
        ),
        outputs=(PortSpec("out"), PortSpec("weights", doc="Per-head attention weights.")),
        attrs=(
            AttrSpec("embed_dim", AttrType.INT, required=True, minimum=1),
            AttrSpec("num_heads", AttrType.INT, required=True, minimum=1),
            AttrSpec("dropout", AttrType.FLOAT, default=0.0, minimum=0.0),
            AttrSpec("causal", AttrType.BOOL, default=False),
            AttrSpec("bias", AttrType.BOOL, default=True),
        ),
        shape_rule=_attention,
        torch=BackendMapping(
            target="torch.nn.MultiheadAttention",
            attr_map={
                "embed_dim": "embed_dim",
                "num_heads": "num_heads",
                "dropout": "dropout",
                "bias": "bias",
            },
            constants={"batch_first": True},
            default_inputs={"key": "query", "value": "key"},
            input_kwargs={"mask": "attn_mask"},
            imports=("torch", "torch.nn"),
            notes="causal maps to is_causal on the call, not the constructor.",
        ),
        keras=BackendMapping(
            target="keras.layers.MultiHeadAttention",
            attr_map={"num_heads": "num_heads", "dropout": "dropout", "bias": "use_bias"},
            derived={"key_dim": "embed_dim // num_heads"},
            call_constants={"return_attention_scores": True},
            default_inputs={"key": "query", "value": "key"},
            input_kwargs={"mask": "attention_mask"},
            imports=("keras",),
            notes="key_dim is embed_dim // num_heads; scores are returned so the "
            "second NTB output exists.",
        ),
        onnx=OnnxMapping(
            op_type="Attention",
            since_opset=23,
            attr_map={"causal": "is_causal"},
            notes="Below opset 23, lowers to MatMul/Softmax/MatMul.",
        ),
        parity_atol=1e-4,
    )
)

OPS = (ATTENTION,)
