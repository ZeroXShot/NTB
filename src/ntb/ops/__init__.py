"""Canonical op definitions. Importing this package registers the built-ins."""

from ntb.ops import builtin as _builtin  # noqa: F401  (import registers the ops)
from ntb.ops.registry import REGISTRY, OpRegistry, UnknownOpError, get, register, require
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    PortSpec,
    ShapeContext,
    ShapeRule,
    ShapeRuleError,
)

__all__ = [
    "REGISTRY",
    "AttrSpec",
    "AttrType",
    "BackendMapping",
    "CallKind",
    "OnnxMapping",
    "OpRegistry",
    "OpSpec",
    "PortSpec",
    "ShapeContext",
    "ShapeRule",
    "ShapeRuleError",
    "UnknownOpError",
    "get",
    "register",
    "require",
]
