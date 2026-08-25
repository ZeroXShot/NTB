"""The public surface for contributing an op from outside this repo.

One import path, and a promise that it keeps working: everything a plugin needs
is re-exported here, so a plugin never reaches into `ntb.ops.spec` or any other
module whose layout is the project's business rather than yours. See
docs/plugins.md, and ADR 13 for why plugins are entry points.

    from ntb.sdk import BackendMapping, CallKind, OpSpec, PortSpec, register

An op is a declaration, not code (ADR 3): ports, attributes, one shape rule and
one mapping per backend. Validation, all three emitters, the studio palette, the
MCP tools and the numeric parity test are derived from that declaration.
"""

from ntb.ir.types import DType, TensorType
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
    WeightSpec,
)
from ntb.shapes.symbolic import ShapeError, normalise_axis

__all__ = [
    "REGISTRY",
    "AttrSpec",
    "AttrType",
    "BackendMapping",
    "CallKind",
    "DType",
    "OnnxMapping",
    "OpRegistry",
    "OpSpec",
    "ParamSpec",
    "ParityCase",
    "PortSpec",
    "ShapeContext",
    "ShapeError",
    "ShapeRule",
    "ShapeRuleError",
    "TensorType",
    "UnknownOpError",
    "WeightSpec",
    "get",
    "normalise_axis",
    "register",
    "require",
]
