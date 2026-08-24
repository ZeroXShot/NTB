"""Lowering NTB-IR to the core IR: modules, generators and spatial rules."""

from ntb.spatial.expr import ExpressionError, contains_expression, evaluate, resolve_attrs
from ntb.spatial.resolve import MODULE_OP, NotResolvable, ResolveError, resolve
from ntb.spatial.rules import Placed, RuleError, derive_pairs

__all__ = [
    "MODULE_OP",
    "ExpressionError",
    "NotResolvable",
    "Placed",
    "ResolveError",
    "RuleError",
    "contains_expression",
    "derive_pairs",
    "evaluate",
    "resolve",
    "resolve_attrs",
]
