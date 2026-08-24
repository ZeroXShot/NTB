"""The `ntb` command line.

Unimplemented commands are declared as explicit "not yet" errors so `--help`
shows the shape of the finished tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ntb import __version__
from ntb.ir import io, schema
from ntb.ir.document import Document
from ntb.ops import REGISTRY
from ntb.shapes import infer_shapes
from ntb.spatial import NotResolvable, ResolveError, resolve
from ntb.validate import Severity
from ntb.validate import validate as run_validation

app = typer.Typer(
    name="ntb",
    help="Neural Tensor Builder - build AI architectures graphically in 2D and 3D.",
    no_args_is_help=True,
    add_completion=False,
)

REPO_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schema"


@app.command()
def version() -> None:
    """Print the NTB version."""
    typer.echo(__version__)


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="Path to a .ntb file.")],
) -> None:
    """Summarise a document: modules, nodes, edges, rules and generators."""
    document = _load(path)
    typer.echo(f"{path.name}  schema v{document.schema_version}")
    if document.name:
        typer.echo(f"  name: {document.name}")
    typer.echo(f"  root: {document.root}")
    for module in document.modules:
        marker = "*" if module.id == document.root else " "
        typer.echo(
            f"  {marker} {module.id}: {len(module.nodes)} nodes, {len(module.edges)} edges, "
            f"{len(module.spatial_rules)} spatial rules, {len(module.generators)} generators"
        )


@app.command()
def ops(
    category: Annotated[str | None, typer.Option(help="Only show one category.")] = None,
) -> None:
    """List the canonical ops in the registry and the backends they reach."""
    for group, specs in REGISTRY.by_category().items():
        if category is not None and group != category:
            continue
        typer.echo(f"{group}:")
        for spec in specs:
            backends = ", ".join(spec.backends()) or "none"
            typer.echo(f"  {spec.name:<20} [{backends}]  {spec.doc.splitlines()[0]}")


@app.command("schema")
def schema_command(
    write: Annotated[bool, typer.Option("--write", help="Rewrite the checked-in schema.")] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Fail if the checked-in schema is stale.")
    ] = False,
    directory: Annotated[
        Path | None, typer.Option(help="Schema directory (defaults to ./schema).")
    ] = None,
) -> None:
    """Generate the NTB-IR JSON Schema from the pydantic models."""
    target = directory or REPO_SCHEMA_DIR
    if write:
        path = schema.write(target)
        typer.echo(f"wrote {path}")
        return
    if check:
        if schema.is_current(target):
            typer.echo("schema is up to date")
            return
        typer.secho(
            f"schema in {target} is stale; run `ntb schema --write`",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(schema.dumps(), nl=False)


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Path to a .ntb file.")],
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as failures.")] = False,
) -> None:
    """Check ops, attributes, shapes and dtypes."""
    report = run_validation(_load(path))
    colours = {
        Severity.ERROR: typer.colors.RED,
        Severity.WARNING: typer.colors.YELLOW,
        Severity.INFO: typer.colors.BLUE,
    }
    for diagnostic in report.diagnostics:
        typer.secho(str(diagnostic), fg=colours[diagnostic.severity], err=True)

    if report.errors or (strict and report.warnings):
        typer.secho(
            f"{path}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.secho(f"{path}: valid", fg=typer.colors.GREEN)


@app.command()
def shapes(
    path: Annotated[Path, typer.Argument(help="Path to a .ntb file.")],
) -> None:
    """Print the inferred type on every port."""
    document = _load(path)
    try:
        graph = resolve(document)
    except (NotResolvable, ResolveError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    report = infer_shapes(graph)
    for (node, port), tensor in sorted(report.types.items()):
        typer.echo(f"{node}.{port}  {tensor}")
    for issue in report.issues:
        typer.secho(f"{issue.node}: {issue.message}", fg=typer.colors.RED, err=True)
    if report.issues:
        raise typer.Exit(code=1)


@app.command()
def emit() -> None:
    """Generate torch / Keras 3 / ONNX from a document. (Phase 2)"""
    _not_yet("emit", phase=2)


@app.command()
def studio() -> None:
    """Open the graphical editor in a browser. (Phase 3)"""
    _not_yet("studio", phase=3)


@app.command()
def run() -> None:
    """Train a model from a document and stream metrics. (Phase 6)"""
    _not_yet("run", phase=6)


def _load(path: Path) -> Document:
    try:
        return io.load(path)
    except io.DocumentError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _not_yet(command: str, *, phase: int) -> None:
    typer.secho(
        f"`ntb {command}` is not implemented yet; it ships in phase {phase}. See docs/roadmap.md.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":  # pragma: no cover
    app()
