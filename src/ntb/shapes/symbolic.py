"""Dimension algebra over sympy.

Knows nothing about ops or graphs, so the op registry can import it.
"""

from __future__ import annotations

from typing import TypeAlias

import sympy

from ntb.ir.types import Dim

DimExpr: TypeAlias = sympy.Expr


class ShapeError(Exception):
    """A shape could not be computed or two shapes are incompatible."""


def dim(value: Dim) -> DimExpr:
    """Lift an IR dimension into sympy, as a non-negative integer symbol.

    Without those assumptions sympy will not simplify floor divisions.
    """
    if isinstance(value, int):
        if value < 0:
            raise ShapeError(f"negative dimension: {value}")
        return sympy.Integer(value)
    try:
        expr = sympy.sympify(value, locals={}, evaluate=True)
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ShapeError(f"cannot parse symbolic dimension {value!r}: {exc}") from exc
    if not isinstance(expr, sympy.Expr):
        raise ShapeError(f"symbolic dimension {value!r} is not an expression")
    substitutions = {
        symbol: sympy.Symbol(symbol.name, integer=True, nonnegative=True)
        for symbol in expr.free_symbols
        if isinstance(symbol, sympy.Symbol)
    }
    return expr.subs(substitutions) if substitutions else expr


def simplify(expr: DimExpr) -> DimExpr:
    return sympy.simplify(expr)


def render(expr: DimExpr) -> Dim:
    """Lower back to an IR dimension. Concrete results come back as ``int``."""
    simplified = sympy.simplify(expr)
    if simplified.is_Integer:
        return int(simplified)
    if simplified.is_number:
        raise ShapeError(f"dimension is a non-integer constant: {simplified}")
    return str(simplified)


def dims_equal(left: Dim, right: Dim) -> bool | None:
    """Whether two dimensions are provably equal.

    ``None`` means undecidable (``batch`` vs ``n``). Callers must warn, not
    fail: refusing those would make dynamic shapes unusable.
    """
    difference = sympy.simplify(dim(left) - dim(right))
    if difference.is_zero:
        return True
    if difference.is_zero is False:
        return False
    return None


def conv_out_dim(
    size: Dim,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Dim:
    """Output size of a convolution or pooling window along one axis.

    ``floor((size + 2*padding - dilation*(kernel-1) - 1) / stride) + 1``
    """
    if kernel < 1:
        raise ShapeError(f"kernel must be >= 1, got {kernel}")
    if stride < 1:
        raise ShapeError(f"stride must be >= 1, got {stride}")
    if dilation < 1:
        raise ShapeError(f"dilation must be >= 1, got {dilation}")
    if padding < 0:
        raise ShapeError(f"padding must be >= 0, got {padding}")

    effective = dilation * (kernel - 1) + 1
    numerator = dim(size) + 2 * padding - effective
    if numerator.is_number and numerator < 0:
        raise ShapeError(
            f"window of effective size {effective} does not fit in input of size {size} "
            f"with padding {padding}"
        )
    return render(sympy.floor(numerator / stride) + 1)
