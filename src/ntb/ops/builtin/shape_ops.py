"""Ops that move data around without doing arithmetic."""

from __future__ import annotations

from ntb.ir.types import Shape, TensorType
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
from ntb.shapes.symbolic import (
    ShapeError,
    dim,
    dims_equal,
    normalise_axis,
    product,
    render,
    resolve_reshape,
)


def _reshape(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    target: Shape = tuple(ctx.attr("shape"))
    try:
        shape = resolve_reshape(x.shape, target)
    except ShapeError as exc:
        ctx.fail(str(exc))
    if x.is_static and all(isinstance(d, int) for d in shape):
        if product(x.shape) != product(shape):
            ctx.fail(f"cannot reshape {product(x.shape)} elements into {list(shape)}")
    return {"out": TensorType(dtype=x.dtype, shape=shape)}


RESHAPE = register(
    OpSpec(
        name="ntb.reshape",
        category="shape",
        doc="Reinterprets the shape. At most one entry may be -1.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("shape", AttrType.SHAPE, required=True),),
        shape_rule=_reshape,
        torch=BackendMapping(
            target="torch.reshape",
            kind=CallKind.FUNCTION,
            attr_map={"shape": "shape"},
            imports=("torch",),
        ),
        keras=BackendMapping(
            target="keras.ops.reshape",
            kind=CallKind.FUNCTION,
            attr_map={"shape": "newshape"},
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="Reshape", notes="Target shape is an input, not an attribute."),
    )
)


def _flatten(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    try:
        start = normalise_axis(ctx.attr("start_axis"), x.rank)
        end = normalise_axis(ctx.attr("end_axis"), x.rank)
    except ShapeError as exc:
        ctx.fail(str(exc))
    if start > end:
        ctx.fail(f"start_axis {start} is after end_axis {end}")
    merged = product(x.shape[start : end + 1])
    return {"out": TensorType(dtype=x.dtype, shape=(*x.shape[:start], merged, *x.shape[end + 1 :]))}


FLATTEN = register(
    OpSpec(
        name="ntb.flatten",
        category="shape",
        doc="Merges a contiguous run of axes into one.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(
            AttrSpec("start_axis", AttrType.INT, default=1),
            AttrSpec("end_axis", AttrType.INT, default=-1),
        ),
        shape_rule=_flatten,
        torch=BackendMapping(
            target="torch.flatten",
            kind=CallKind.FUNCTION,
            attr_map={"start_axis": "start_dim", "end_axis": "end_dim"},
            imports=("torch",),
        ),
        keras=BackendMapping(
            target="keras.ops.reshape",
            kind=CallKind.FUNCTION,
            imports=("keras",),
            notes="Keras has no axis-range flatten; the emitter computes the shape.",
        ),
        onnx=OnnxMapping(
            op_type="Flatten",
            attr_map={"start_axis": "axis"},
            notes="ONNX Flatten always yields rank 2; other ranges lower to Reshape.",
        ),
    )
)


def _permute(ctx: ShapeContext) -> dict[str, TensorType]:
    x = ctx.input("in")
    order = tuple(ctx.attr("order"))
    if len(order) != x.rank:
        ctx.fail(f"order has {len(order)} entries but the input is rank {x.rank}")
    try:
        resolved = tuple(normalise_axis(axis, x.rank) for axis in order)
    except ShapeError as exc:
        ctx.fail(str(exc))
    if len(set(resolved)) != len(resolved):
        ctx.fail(f"order repeats an axis: {list(order)}")
    return {"out": TensorType(dtype=x.dtype, shape=tuple(x.shape[i] for i in resolved))}


PERMUTE = register(
    OpSpec(
        name="ntb.permute",
        category="shape",
        doc="Reorders axes.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("order", AttrType.INTS, required=True),),
        shape_rule=_permute,
        torch=BackendMapping(
            target="torch.permute",
            kind=CallKind.FUNCTION,
            attr_map={"order": "dims"},
            imports=("torch",),
        ),
        keras=BackendMapping(
            target="keras.ops.transpose",
            kind=CallKind.FUNCTION,
            attr_map={"order": "axes"},
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="Transpose", attr_map={"order": "perm"}),
    )
)


def _concat(ctx: ShapeContext) -> dict[str, TensorType]:
    a, b = ctx.input("a"), ctx.input("b")
    if a.rank != b.rank:
        ctx.fail(f"operands must have equal rank, got {a.rank} and {b.rank}")
    if a.dtype is not b.dtype:
        ctx.fail(f"operands must share a dtype, got {a.dtype.value} and {b.dtype.value}")
    try:
        axis = normalise_axis(ctx.attr("axis"), a.rank)
    except ShapeError as exc:
        ctx.fail(str(exc))

    for i, (left, right) in enumerate(zip(a.shape, b.shape, strict=True)):
        if i != axis and dims_equal(left, right) is False:
            ctx.fail(f"axis {i} must match outside the concat axis: {left} against {right}")

    size = render(dim(a.shape[axis]) + dim(b.shape[axis]))
    return {
        "out": TensorType(
            dtype=a.dtype,
            shape=(*a.shape[:axis], size, *a.shape[axis + 1 :]),
            layout=a.layout,
        )
    }


CONCAT = register(
    OpSpec(
        name="ntb.concat",
        category="shape",
        doc="Joins two tensors along one axis.",
        inputs=(PortSpec("a"), PortSpec("b")),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("axis", AttrType.INT, default=1),),
        shape_rule=_concat,
        torch=BackendMapping(
            target="torch.cat",
            kind=CallKind.FUNCTION,
            attr_map={"axis": "dim"},
            imports=("torch",),
        ),
        keras=BackendMapping(
            target="keras.ops.concatenate",
            kind=CallKind.FUNCTION,
            attr_map={"axis": "axis"},
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="Concat", attr_map={"axis": "axis"}),
    )
)

OPS = (RESHAPE, FLATTEN, PERMUTE, CONCAT)
