"""Arithmetic over module parameters.

A generator repeats a module with attributes that depend on the instance index,
and a parametric module needs attributes that depend on its params. Both are
written as ``"$expr"`` in an attribute slot:

    {"out_features": "$width * 2", "num_heads": "$heads"}

`.ntb` files are documents people will download from each other, so this is not
``eval``: the expression is parsed and walked over a closed set of nodes. What
is not on the list does not run.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from typing import Any

#: Attribute values starting with this are expressions over the parameters.
MARKER = "$"

#: Guards against a document that computes 2**10000000 while the studio waits.
MAX_EXPONENT = 64

Number = int | float | bool

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "int": int,
    "float": float,
    "round": round,
    "len": len,
    "sum": sum,
}

_COMPARISONS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class ExpressionError(ValueError):
    """An expression is malformed, unsupported, or names something unknown."""


def is_expression(value: Any) -> bool:
    """True for an attribute value that has to be evaluated before use."""
    return isinstance(value, str) and value.startswith(MARKER)


def contains_expression(value: Any) -> bool:
    """True for a value, or a list holding a value, that needs evaluating."""
    if isinstance(value, list):
        return any(contains_expression(item) for item in value)
    return is_expression(value)


def evaluate(text: str, params: Mapping[str, Any]) -> Number:
    """Evaluate ``$expr`` against ``params``."""
    source = text[len(MARKER) :] if text.startswith(MARKER) else text
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"{text!r} is not an expression: {exc.msg}") from exc
    value = _eval(tree.body, params, text)
    if not isinstance(value, (int, float)):
        raise ExpressionError(f"{text!r} evaluated to {type(value).__name__}, not a number")
    return value


def resolve_attrs(attrs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``attrs`` with every expression evaluated. Lists are walked too."""
    return {name: _resolve_value(value, params) for name, value in attrs.items()}


def _resolve_value(value: Any, params: Mapping[str, Any]) -> Any:
    if is_expression(value):
        return evaluate(value, params)
    if isinstance(value, list):
        return [_resolve_value(item, params) for item in value]
    return value


def _eval(node: ast.expr, params: Mapping[str, Any], text: str) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ExpressionError(f"{text!r} contains a {type(node.value).__name__} literal")

    if isinstance(node, ast.Name):
        if node.id not in params:
            known = ", ".join(sorted(params)) or "none"
            raise ExpressionError(f"{text!r} uses unknown parameter {node.id!r}; known: {known}")
        value = params[node.id]
        # Lists are allowed only as something to measure: `len(kernel_size)`.
        if not isinstance(value, (int, float, list, tuple)):
            raise ExpressionError(f"parameter {node.id!r} is not a number")
        return value

    if isinstance(node, ast.BinOp):
        function = _BINARY.get(type(node.op))
        if function is None:
            raise ExpressionError(f"{text!r} uses an operator that is not allowed")
        left = _eval(node.left, params, text)
        right = _eval(node.right, params, text)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ExpressionError(f"{text!r} raises to {right}, over the limit of {MAX_EXPONENT}")
        try:
            return function(left, right)
        except ZeroDivisionError as exc:
            raise ExpressionError(f"{text!r} divides by zero") from exc
        except TypeError as exc:
            raise ExpressionError(f"{text!r}: {exc}") from exc

    if isinstance(node, ast.UnaryOp):
        unary = _UNARY.get(type(node.op))
        if unary is None:
            raise ExpressionError(f"{text!r} uses a unary operator that is not allowed")
        return unary(_eval(node.operand, params, text))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ExpressionError(f"{text!r} chains comparisons, which is not allowed")
        compare = _COMPARISONS.get(type(node.ops[0]))
        if compare is None:
            raise ExpressionError(f"{text!r} uses a comparison that is not allowed")
        return compare(_eval(node.left, params, text), _eval(node.comparators[0], params, text))

    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name not in _FUNCTIONS or node.keywords:
            allowed = ", ".join(sorted(_FUNCTIONS))
            raise ExpressionError(f"{text!r} calls {name or '?'}; only {allowed} are available")
        return _FUNCTIONS[name](*(_eval(argument, params, text) for argument in node.args))

    raise ExpressionError(f"{text!r} contains {type(node).__name__}, which is not allowed")
