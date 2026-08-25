"""The op registry.

Append-only: re-registering a name raises, since shadowing an op would change
what existing `.ntb` files mean.
"""

from __future__ import annotations

from collections.abc import Iterator

from ntb.ops.spec import OpSpec


class OpRegistry:
    """A collection of :class:`OpSpec` keyed by canonical name."""

    def __init__(self) -> None:
        self._specs: dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> OpSpec:
        if spec.name in self._specs:
            raise ValueError(f"op {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> OpSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> OpSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise UnknownOpError(name, tuple(sorted(self._specs)))
        return spec

    def discard(self, name: str) -> None:
        """Take an op back out. Only for rejecting a plugin that misbehaved.

        The registry is otherwise append-only: removing an op would change what
        existing `.ntb` files mean.
        """
        self._specs.pop(name, None)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def by_category(self) -> dict[str, tuple[OpSpec, ...]]:
        grouped: dict[str, list[OpSpec]] = {}
        for spec in self:
            grouped.setdefault(spec.category, []).append(spec)
        return {k: tuple(v) for k, v in sorted(grouped.items())}

    def __iter__(self) -> Iterator[OpSpec]:
        return iter(self._specs[name] for name in sorted(self._specs))

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs


class UnknownOpError(LookupError):
    """An op name is not in the registry."""

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        self.name = name
        self.known = known
        suggestion = _closest(name, known)
        hint = f"; did you mean {suggestion!r}?" if suggestion else ""
        super().__init__(f"unknown op {name!r}{hint}")


def _closest(name: str, known: tuple[str, ...]) -> str | None:
    from difflib import get_close_matches

    matches = get_close_matches(name, known, n=1, cutoff=0.7)
    return matches[0] if matches else None


#: The registry every other module uses.
REGISTRY = OpRegistry()


def register(spec: OpSpec) -> OpSpec:
    """Add ``spec`` to the global registry."""
    return REGISTRY.register(spec)


def get(name: str) -> OpSpec | None:
    return REGISTRY.get(name)


def require(name: str) -> OpSpec:
    return REGISTRY.require(name)
