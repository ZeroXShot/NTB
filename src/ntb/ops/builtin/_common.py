"""Helpers shared by the built-in op declarations."""

from __future__ import annotations

from collections.abc import Sequence

from ntb.ir.types import TensorType
from ntb.ops.spec import ShapeContext


def ntuple(ctx: ShapeContext, name: str, rank: int) -> tuple[int, ...]:
    """Read an attribute that is one int or exactly ``rank`` of them."""
    value = ctx.attr(name)
    if isinstance(value, int):
        return (value,) * rank
    if len(value) == 1:
        return (value[0],) * rank
    if len(value) == rank:
        return tuple(value)
    ctx.fail(f"{name} must be 1 or {rank} integers, got {len(value)}")


def require_float(ctx: ShapeContext, t: TensorType, port: str = "in") -> None:
    if not t.dtype.is_floating:
        ctx.fail(f"{port} must be a floating dtype, got {t.dtype.value}")


def require_rank(ctx: ShapeContext, t: TensorType, rank: int, port: str = "in") -> None:
    if t.rank != rank:
        ctx.fail(f"{port} must be rank {rank}, got rank {t.rank}")


def elementwise(ctx: ShapeContext) -> dict[str, TensorType]:
    """Shape rule for float ops that change nothing."""
    x = ctx.input("in")
    require_float(ctx, x)
    return {"out": x}


def inputs_of(ctx: ShapeContext, names: Sequence[str]) -> tuple[TensorType, ...]:
    return tuple(ctx.input(name) for name in names)
