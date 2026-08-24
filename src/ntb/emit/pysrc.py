"""Build Python source through ``ast`` rather than string templates.

An AST cannot emit code that does not parse, and it makes the formatting
somebody else's problem: ``ruff format`` is applied at the end when available.
Generated code has to be code a user would keep in their own repo (ADR 1), so
readability is a requirement, not a nicety.
"""

from __future__ import annotations

import ast
import keyword
import re
import shutil
import subprocess
from typing import Any

_UNSAFE = re.compile(r"[^0-9a-zA-Z_]+")


def identifier(name: str, *, fallback: str = "node") -> str:
    """Turn an NTB id into a valid, non-colliding Python identifier."""
    cleaned = _UNSAFE.sub("_", name).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    if keyword.iskeyword(cleaned) or cleaned in {"self", "forward"}:
        cleaned = f"{cleaned}_"
    return cleaned


def dotted(path: str) -> ast.expr:
    """``torch.nn.Linear`` as an attribute chain."""
    head, *rest = path.split(".")
    node: ast.expr = ast.Name(id=head, ctx=ast.Load())
    for part in rest:
        node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
    return node


def constant(value: Any) -> ast.expr:
    """A literal, preserving tuples and lists as written."""
    if isinstance(value, (list, tuple)):
        elements = [constant(v) for v in value]
        return ast.Tuple(elts=elements, ctx=ast.Load())
    if isinstance(value, dict):
        return ast.Dict(
            keys=[constant(k) for k in value], values=[constant(v) for v in value.values()]
        )
    return ast.Constant(value=value)


def call(func: str | ast.expr, args: list[ast.expr], kwargs: dict[str, Any]) -> ast.Call:
    target = dotted(func) if isinstance(func, str) else func
    keywords = [
        ast.keyword(arg=name, value=value if isinstance(value, ast.expr) else constant(value))
        for name, value in kwargs.items()
    ]
    return ast.Call(func=target, args=args, keywords=keywords)


def name(value: str) -> ast.Name:
    return ast.Name(id=value, ctx=ast.Load())


def assign(target: str, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=value)


def assign_many(targets: list[str], value: ast.expr) -> ast.Assign:
    """``a, b = expr`` for ops with more than one output."""
    if len(targets) == 1:
        return assign(targets[0], value)
    tuple_target = ast.Tuple(
        elts=[ast.Name(id=t, ctx=ast.Store()) for t in targets], ctx=ast.Store()
    )
    return ast.Assign(targets=[tuple_target], value=value)


def assign_attribute(owner: str, attribute: str, value: ast.expr) -> ast.Assign:
    """``self.fc = torch.nn.Linear(...)``"""
    target = ast.Attribute(
        value=ast.Name(id=owner, ctx=ast.Load()), attr=attribute, ctx=ast.Store()
    )
    return ast.Assign(targets=[target], value=value)


def attribute(owner: str, attr: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id=owner, ctx=ast.Load()), attr=attr, ctx=ast.Load())


def unparse(module: ast.Module, *, format_source: bool = True) -> str:
    """Render a module, formatted if ruff is on PATH."""
    ast.fix_missing_locations(module)
    source = ast.unparse(module)
    if not source.endswith("\n"):
        source += "\n"
    return _format(source) if format_source else source


def _format(source: str) -> str:
    """Run ``ruff format`` over the source, falling back to it unformatted.

    Formatting must never be load-bearing: NTB has no runtime dependency on
    ruff, and generated code that only parses when a dev tool is installed
    would be a trap.
    """
    executable = shutil.which("ruff")
    if executable is None:  # pragma: no cover - depends on the environment
        return source
    try:
        result = subprocess.run(
            [executable, "format", "--stdin-filename", "model.py", "-"],
            input=source,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return source
    return result.stdout if result.returncode == 0 and result.stdout else source


class NameAllocator:
    """Hands out unique Python identifiers for NTB ids."""

    def __init__(self) -> None:
        self._taken: set[str] = set()
        self._by_source: dict[str, str] = {}

    def allocate(self, source: str, *, fallback: str = "node") -> str:
        existing = self._by_source.get(source)
        if existing is not None:
            return existing

        base = identifier(source, fallback=fallback)
        candidate = base
        counter = 2
        while candidate in self._taken:
            candidate = f"{base}_{counter}"
            counter += 1
        self._taken.add(candidate)
        self._by_source[source] = candidate
        return candidate

    def reserve(self, value: str) -> str:
        self._taken.add(value)
        return value
