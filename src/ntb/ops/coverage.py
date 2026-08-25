"""How much of torch, Keras and ONNX the registry actually reaches.

"Are all the layers there?" is a question that deserves a number, not an
opinion. This introspects the three backends' own surfaces and compares them
against what the registry maps, so the gap is measured rather than guessed.

Matching is by the last segment of a target, lowercased: NTB's
``torch.nn.functional.relu`` covers ``torch.nn.ReLU`` because both end in
``relu``. That is loose on purpose -- an op reached through a backend's
functional form is still reached -- and the few cases it gets wrong are in
:data:`ALIASES`.

Nothing here decides what NTB *should* have. What is deliberately out of scope
is declared in :data:`EXCLUDED` with the reason, so an exclusion is an argument
somebody can disagree with rather than a silence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ntb.ops.plugins import RESERVED
from ntb.ops.registry import REGISTRY, OpRegistry
from ntb.ops.spec import OpSpec

#: Backend symbols whose name does not match the NTB target that covers them.
ALIASES: dict[str, str] = {
    # NTB reshapes rather than using an axis-range flatten, which is what
    # Keras and ONNX give it.
    "Flatten": "flatten",
    "Reshape": "reshape",
    # One NTB op with rank_targets stands for the whole family.
    "GlobalAveragePooling1D": "adaptiveavgpool2d",
    "GlobalAveragePooling2D": "adaptiveavgpool2d",
    "GlobalAveragePooling3D": "adaptiveavgpool2d",
    "GlobalAveragePool": "adaptiveavgpool2d",
    "AdaptiveAvgPool1d": "adaptiveavgpool2d",
    "AdaptiveAvgPool3d": "adaptiveavgpool2d",
    "BatchNormalization": "batchnorm2d",
    "BatchNorm1d": "batchnorm2d",
    "BatchNorm3d": "batchnorm2d",
    # Named for the operation in torch, for the layer in Keras.
    "Add": "add",
    "Subtract": "sub",
    "Multiply": "mul",
    "Dense": "linear",
    "Gemm": "linear",
    "MatMul": "matmul",
    "Concatenate": "concat",
    "Concat": "concat",
    "Transpose": "permute",
    "Gather": "embedding",
}

#: Out of scope, and why. An exclusion is an argument, so it carries one.
EXCLUDED: dict[str, str] = {
    # Composition
    "Module": "the base class, not an op",
    "Sequential": "a container; NTB composes with Modules and edges",
    "ModuleList": "a container",
    "ModuleDict": "a container",
    "ParameterList": "a container",
    "ParameterDict": "a container",
    "Container": "a deprecated container",
    "Layer": "the base class, not an op",
    "InputLayer": "NTB declares inputs on the module boundary",
    "Input": "NTB declares inputs on the module boundary",
    "Wrapper": "the base class for wrappers",
    "Lambda": "arbitrary Python in a layer; not portable to three backends",
    # Shape inference NTB does itself
    "LazyLinear": "a shape-inferring variant; NTB infers shapes symbolically",
    "LazyConv1d": "a shape-inferring variant",
    "LazyConv2d": "a shape-inferring variant",
    "LazyConv3d": "a shape-inferring variant",
    "LazyConvTranspose1d": "a shape-inferring variant",
    "LazyConvTranspose2d": "a shape-inferring variant",
    "LazyConvTranspose3d": "a shape-inferring variant",
    "LazyBatchNorm1d": "a shape-inferring variant",
    "LazyBatchNorm2d": "a shape-inferring variant",
    "LazyBatchNorm3d": "a shape-inferring variant",
    "LazyInstanceNorm1d": "a shape-inferring variant",
    "LazyInstanceNorm2d": "a shape-inferring variant",
    "LazyInstanceNorm3d": "a shape-inferring variant",
    # Execution, not architecture
    "DataParallel": "distributed execution, not an architecture",
    "DistributedDataParallel": "distributed execution",
    "If": "control flow; NTB-IR is a DAG",
    "Loop": "control flow; NTB-IR is a DAG",
    "Scan": "control flow; NTB-IR is a DAG",
    # Base classes with no operation of their own
    "RNNBase": "a base class",
    "RNNCellBase": "a base class",
}

#: Prefixes of whole families that are out of scope, with the reason.
EXCLUDED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Random", "data augmentation, not architecture"),
    ("Sequence", "ONNX sequence types; NTB tensors are not sequences"),
    ("Optional", "ONNX optional types; NTB has optional ports instead"),
    ("Quantize", "quantisation is a deployment concern, not an architecture"),
    ("Dequantize", "quantisation is a deployment concern"),
)

#: Losses get a surface of their own. Substring, not suffix: NLLLoss2d.
LOSS_MARKER = "Loss"

#: What we think is worth having next. A judgement, reviewed by pull request,
#: not a measurement -- which is why it is here and says so.
WANTED: tuple[str, ...] = (
    "LSTM",
    "GRU",
    "ConvTranspose2d",
    "GroupNorm",
    "InstanceNorm2d",
    "LeakyReLU",
    "ELU",
    "PReLU",
    "Softplus",
    "Upsample",
    "Pad",
    "Slice",
    "Split",
    "Squeeze",
    "Unsqueeze",
    "Cast",
    "Where",
    "ReduceMean",
    "ReduceMax",
    "ArgMax",
    "TopK",
    "Einsum",
    "Clip",
    "Exp",
    "Log",
    "Sqrt",
)

_TAIL = re.compile(r"[^.]+$")


@dataclass(frozen=True)
class Surface:
    """One backend's own vocabulary, and how much of it NTB reaches."""

    name: str
    version: str = ""
    available: bool = True
    covered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.covered) + len(self.missing)

    @property
    def percent(self) -> float:
        return 100.0 * len(self.covered) / self.total if self.total else 0.0

    def wanted(self) -> tuple[str, ...]:
        """The missing symbols we have said out loud that we want."""
        return tuple(name for name in WANTED if name in self.missing)


@dataclass(frozen=True)
class Report:
    surfaces: tuple[Surface, ...] = ()
    #: Built-in ops measured. Plugin ops are counted but never measured.
    ops: int = 0
    plugin_ops: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def surface(self, name: str) -> Surface | None:
        return next((s for s in self.surfaces if s.name == name), None)


def normalise(symbol: str) -> str:
    """The comparable form of a target or a class name."""
    tail = _TAIL.search(symbol)
    return (tail.group(0) if tail else symbol).replace("_", "").lower()


def excluded_reason(name: str) -> str | None:
    if name in EXCLUDED:
        return EXCLUDED[name]
    for prefix, reason in EXCLUDED_PREFIXES:
        if name.startswith(prefix):
            return reason
    return None


def builtins(registry: OpRegistry = REGISTRY) -> tuple[OpSpec, ...]:
    """The ops this repo ships.

    A plugin's op is its author's coverage, not NTB's, and counting it would
    make the published numbers depend on what happens to be installed.
    """
    return tuple(spec for spec in registry if spec.name.startswith(RESERVED))


def reached(specs: Sequence[OpSpec]) -> dict[str, set[str]]:
    """Every backend symbol these ops name, per backend."""
    out: dict[str, set[str]] = {"torch": set(), "keras": set(), "onnx": set()}
    for spec in specs:
        for backend in ("torch", "keras"):
            mapping = getattr(spec, backend)
            if mapping is None:
                continue
            targets = [mapping.target, *mapping.rank_targets.values()]
            if mapping.pad_target:
                targets.append(mapping.pad_target)
            out[backend].update(normalise(target) for target in targets)
        if spec.onnx is not None and not spec.onnx.custom:
            out["onnx"].add(normalise(spec.onnx.op_type))
    return out


def _classify(names: list[str], covered: set[str]) -> tuple[list[str], list[str], list[str]]:
    hit, miss, skip = [], [], []
    for name in sorted(set(names)):
        if excluded_reason(name) is not None:
            skip.append(name)
        elif normalise(name) in covered or ALIASES.get(name, "") in covered:
            hit.append(name)
        else:
            miss.append(name)
    return hit, miss, skip


def _torch_surfaces(covered: set[str]) -> list[Surface]:
    try:
        import torch
    except ImportError:
        return [
            Surface("torch.nn", available=False),
            Surface("torch.optim", available=False),
            Surface("torch losses", available=False),
        ]

    version = torch.__version__
    layers: list[str] = []
    losses: list[str] = []
    for name, obj in vars(torch.nn).items():
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if not issubclass(obj, torch.nn.Module):
            continue
        (losses if LOSS_MARKER in name else layers).append(name)

    optimisers = [
        name
        for name, obj in vars(torch.optim).items()
        if not name.startswith("_") and isinstance(obj, type) and name != "Optimizer"
    ]

    hit, miss, skip = _classify(layers, covered)
    return [
        Surface("torch.nn", version, True, tuple(hit), tuple(miss), tuple(skip)),
        Surface("torch.optim", version, True, (), tuple(sorted(optimisers)), ()),
        Surface("torch losses", version, True, (), tuple(sorted(losses)), ()),
    ]


def _keras_surface(covered: set[str]) -> Surface:
    try:
        import keras
    except Exception as exc:  # a missing backend raises, it does not ImportError
        return Surface("keras.layers", str(exc)[:60], available=False)

    names = [
        name
        for name, obj in vars(keras.layers).items()
        if not name.startswith("_") and isinstance(obj, type)
    ]
    hit, miss, skip = _classify(names, covered)
    return Surface("keras.layers", keras.__version__, True, tuple(hit), tuple(miss), tuple(skip))


def _onnx_surface(covered: set[str]) -> Surface:
    try:
        import onnx
        import onnx.defs
    except ImportError:
        return Surface("onnx", available=False)

    names = [
        schema.name
        for schema in onnx.defs.get_all_schemas()
        if schema.domain == "" and not schema.deprecated
    ]
    hit, miss, skip = _classify(names, covered)
    return Surface("onnx", onnx.__version__, True, tuple(hit), tuple(miss), tuple(skip))


def audit(registry: OpRegistry = REGISTRY) -> Report:
    """Measure the built-in ops against every backend that is installed."""
    specs = builtins(registry)
    covered = reached(specs)
    surfaces = [
        *_torch_surfaces(covered["torch"]),
        _keras_surface(covered["keras"]),
        _onnx_surface(covered["onnx"]),
    ]
    notes = (
        "Optimisers and losses are not registry ops yet, so nothing is counted "
        "as covered; they are a closed enum in `ntb.runs.config`.",
        "Only the ops this repo ships are measured. A plugin's op is its "
        "author's coverage, not NTB's.",
    )
    return Report(tuple(surfaces), len(specs), len(registry) - len(specs), notes)


def as_markdown(report: Report) -> str:
    """The report as `docs/coverage.md`, generated rather than maintained."""
    lines = [
        "# Coverage",
        "",
        "<!-- Generated by `ntb coverage --write`. Do not edit. -->",
        "",
        "How much of each backend's own vocabulary the op registry reaches.",
        "Matching is by the last segment of a target, lowercased, so an op",
        "reached through a backend's functional form counts as reached.",
        "",
        f"NTB registers **{report.ops} ops**.",
        "",
        "| Surface | Version | Covered | Of | | Excluded |",
        "|---|---|---:|---:|---|---:|",
    ]
    for surface in report.surfaces:
        if not surface.available:
            lines.append(f"| `{surface.name}` | not installed | | | | |")
            continue
        lines.append(
            f"| `{surface.name}` | {surface.version} | {len(surface.covered)} | "
            f"{surface.total} | {surface.percent:.0f}% | {len(surface.excluded)} |"
        )
    lines += ["", *(f"> {note}" for note in report.notes), ""]

    for surface in report.surfaces:
        if not surface.available or not surface.missing:
            continue
        lines += [f"## Not covered: `{surface.name}`", ""]
        wanted = surface.wanted()
        if wanted:
            lines += [
                "Wanted next (a judgement, reviewed by pull request, not a measurement):",
                "",
                "".join(f"`{name}` " for name in wanted).strip(),
                "",
            ]
        lines += [
            "<details><summary>Everything not covered</summary>",
            "",
            "".join(f"`{name}` " for name in surface.missing).strip(),
            "",
            "</details>",
            "",
        ]

    lines += ["## Deliberately out of scope", "", "| Symbol | Why |", "|---|---|"]
    for name, reason in sorted(EXCLUDED.items()):
        lines.append(f"| `{name}` | {reason} |")
    for prefix, reason in EXCLUDED_PREFIXES:
        lines.append(f"| `{prefix}*` | {reason} |")
    return "\n".join(lines) + "\n"
