"""Symbolic shape and dtype inference."""

from ntb.shapes.infer import ShapeIssue, ShapeReport, infer_shapes
from ntb.shapes.symbolic import (
    DimExpr,
    ShapeError,
    broadcast,
    conv_out_dim,
    dim,
    dims_equal,
    normalise_axis,
    product,
    render,
    resolve_reshape,
    simplify,
)

__all__ = [
    "DimExpr",
    "ShapeError",
    "ShapeIssue",
    "ShapeReport",
    "broadcast",
    "conv_out_dim",
    "dim",
    "dims_equal",
    "infer_shapes",
    "normalise_axis",
    "product",
    "render",
    "resolve_reshape",
    "simplify",
]
