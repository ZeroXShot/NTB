"""Symbolic shape and dtype inference over the core IR."""

from ntb.shapes.symbolic import (
    DimExpr,
    ShapeError,
    conv_out_dim,
    dim,
    dims_equal,
    render,
    simplify,
)

__all__ = [
    "DimExpr",
    "ShapeError",
    "conv_out_dim",
    "dim",
    "dims_equal",
    "render",
    "simplify",
]
