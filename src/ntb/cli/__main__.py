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
) -> None:
    """Check a document: structure now, shapes and ops from phase 1."""
    _load(path)
    typer.secho(f"{path}: structurally valid", fg=typer.colors.GREEN)
    typer.secho(
        "note: semantic validation (ops, shapes, dtypes) lands with ntb.validate",
        fg=typer.colors.YELLOW,
        err=True,
    )


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
