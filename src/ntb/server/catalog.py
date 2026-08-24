"""The op registry, as JSON for the palette.

The studio must not carry its own idea of what an op is: everything the palette
and the inspector show comes from here, so a new op appears in the UI the moment
it is registered. See docs/adr/0003.
"""

from __future__ import annotations

from typing import Any

from ntb.ops import REGISTRY, OpSpec
from ntb.ops.registry import OpRegistry
from ntb.ops.spec import AttrSpec, PortSpec


def op_catalog(registry: OpRegistry = REGISTRY) -> list[dict[str, Any]]:
    """Every op, grouped-ready and ordered by category then name."""
    return [describe_op(spec) for spec in sorted(registry, key=lambda s: (s.category, s.name))]


def describe_op(spec: OpSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "category": spec.category,
        "summary": spec.doc.strip().splitlines()[0],
        "doc": spec.doc.strip(),
        "inputs": [_port(p) for p in spec.inputs],
        "outputs": [_port(p) for p in spec.outputs],
        "attrs": [_attr(a) for a in spec.attrs],
        "backends": list(spec.backends()),
    }


def _port(port: PortSpec) -> dict[str, Any]:
    return {
        "name": port.name,
        "doc": port.doc,
        "optional": port.optional,
        "variadic": port.variadic,
    }


def _attr(attr: AttrSpec) -> dict[str, Any]:
    return {
        "name": attr.name,
        "type": attr.type.value,
        "doc": attr.doc,
        "default": attr.default,
        "required": attr.required,
        "minimum": attr.minimum,
        "choices": list(attr.choices) if attr.choices is not None else None,
        "default_from": attr.default_from or None,
    }
