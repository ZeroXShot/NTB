"""Canonical op definitions. Importing this package registers the built-ins."""

from ntb.ops import builtin as _builtin  # noqa: F401  (import registers the ops)
from ntb.ops import plugins as _plugins
from ntb.ops.registry import REGISTRY, OpRegistry, UnknownOpError, get, register, require
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    ParamSpec,
    ParityCase,
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
    "ParamSpec",
    "ParityCase",
    "PortSpec",
    "ShapeContext",
    "ShapeRule",
    "ShapeRuleError",
    "UnknownOpError",
    "get",
    "register",
    "require",
]

# Last, so a plugin importing `ntb.ops` finds a package that is already built.
# Third-party ops become ops here: everything downstream reads one registry.
_plugins.loaded()
