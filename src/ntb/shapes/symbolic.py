"""Dimension algebra over sympy.

Knows nothing about ops or graphs, so the op registry can import it.
"""

from __future__ import annotations

from typing import TypeAlias

import sympy

from ntb.ir.types import Dim, Shape

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


def broadcast(left: Shape, right: Shape) -> Shape:
    """NumPy broadcasting over possibly symbolic dimensions.

    A symbolic dimension against a concrete one is assumed compatible; only a
    provable mismatch is an error.
    """
    rank = max(len(left), len(right))
    padded_left = (1,) * (rank - len(left)) + tuple(left)
    padded_right = (1,) * (rank - len(right)) + tuple(right)

    result: list[Dim] = []
    for axis, (a, b) in enumerate(zip(padded_left, padded_right, strict=True)):
        if a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        elif dims_equal(a, b) is False:
            raise ShapeError(
                f"cannot broadcast {a} against {b} at axis {axis} "
                f"(shapes {list(left)} and {list(right)})"
            )
        else:
            result.append(a if isinstance(a, int) else b)
    return tuple(result)


def normalise_axis(axis: int, rank: int) -> int:
    """Turn a possibly negative axis into a non-negative one."""
    resolved = axis + rank if axis < 0 else axis
    if not 0 <= resolved < rank:
        raise ShapeError(f"axis {axis} is out of range for rank {rank}")
    return resolved


def product(shape: Shape) -> Dim:
    """The number of elements in a shape, symbolically if need be."""
    total: DimExpr = sympy.Integer(1)
    for value in shape:
        total = total * dim(value)
    return render(total)


def resolve_reshape(shape: Shape, target: Shape) -> Shape:
    """Apply a reshape target, inferring at most one ``-1`` entry."""
    wildcards = [i for i, d in enumerate(target) if d == -1]
    if len(wildcards) > 1:
        raise ShapeError("reshape target may contain at most one -1")
    if not wildcards:
        return tuple(target)

    known: DimExpr = sympy.Integer(1)
    for i, value in enumerate(target):
        if i != wildcards[0]:
            known = known * dim(value)
    if known == 0:
        raise ShapeError("reshape target has a zero dimension alongside -1")
    inferred = render(sympy.simplify(dim(product(shape)) / known))
    return tuple(inferred if i == wildcards[0] else d for i, d in enumerate(target))
