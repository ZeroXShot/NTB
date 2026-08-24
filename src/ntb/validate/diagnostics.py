"""Located diagnostics.

Validation reports rather than raises: the studio needs every problem in a
document at once, attached to the block the user drew, not the first exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Code(StrEnum):
    """Stable identifiers, so the studio and tests can match on them."""

    UNKNOWN_OP = "unknown-op"
    UNKNOWN_ATTR = "unknown-attr"
    BAD_ATTR = "bad-attr"
    MISSING_ATTR = "missing-attr"
    SHAPE = "shape"
    CYCLE = "cycle"
    UNRESOLVABLE = "unresolvable"
    STRUCTURE = "structure"
    UNCONNECTED = "unconnected"
    NO_OUTPUT = "no-output"


@dataclass(frozen=True, slots=True)
class Location:
    """Where a diagnostic belongs in the authored document."""

    module: str | None = None
    node: str | None = None
    port: str | None = None
    edge: str | None = None
    #: The block of the root module this sits inside, which for a generated
    #: instance is the only way to say *which* repetition went wrong.
    block: str | None = None
    #: The full path of the lowered node, when the problem was found after
    #: lowering rather than in the authored document.
    path: str | None = None

    def __str__(self) -> str:
        parts = [p for p in (self.module, self.node) if p]
        text = "/".join(parts) if parts else "document"
        if self.path and self.path != self.node:
            text = f"{text} ({self.path})"
        if self.port:
            text = f"{text}.{self.port}"
        if self.edge:
            text = f"{text} (edge {self.edge})"
        return text


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: Code
    message: str
    location: Location = Location()
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        return f"{self.severity.value}: {self.location}: {self.message} [{self.code.value}]"


@dataclass(frozen=True, slots=True)
class Report:
    """Everything validation found, in the order it found it."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.WARNING)

    def codes(self) -> tuple[Code, ...]:
        return tuple(d.code for d in self.diagnostics)

    def __iter__(self) -> object:
        return iter(self.diagnostics)

    def __len__(self) -> int:
        return len(self.diagnostics)
