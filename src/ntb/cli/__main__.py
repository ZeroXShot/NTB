"""The `ntb` command line.

Unimplemented commands are declared as explicit "not yet" errors so `--help`
shows the shape of the finished tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ntb import __version__
from ntb.emit import (
    EmitError,
    OnnxEmitError,
    emit_keras_document,
    emit_torch_document,
    export_onnx_document,
)
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


@app.command("resolve")
def resolve_command(
    path: Annotated[Path, typer.Argument(help="Path to a .ntb file.")],
    edges: Annotated[
        bool, typer.Option("--edges/--no-edges", help="List the lowered edges too.")
    ] = True,
) -> None:
    """Lower a document and show what generators and spatial rules produced."""
    document = _load(path)
    try:
        graph = resolve(document)
    except (NotResolvable, ResolveError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{graph.name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    for node in graph.nodes:
        typer.echo(f"  {node.id:<28} {node.op:<16} from {node.origin}")
    if not edges:
        return
    for edge in graph.edges:
        # Where an edge came from is the thing worth seeing: an edge nobody drew
        # is the whole point of a spatial rule, and also the thing to debug.
        derived = edge.origin.rule or edge.origin.generator
        marker = typer.style(f"  via {derived}", fg=typer.colors.CYAN) if derived else ""
        typer.echo(f"  {edge.src!s:<28} -> {edge.dst!s:<28}{marker}")


@app.command()
def emit(
    path: Annotated[Path, typer.Argument(help="Path to a .ntb file.")],
    backend: Annotated[str, typer.Option(help="torch, keras or onnx.")] = "torch",
    out: Annotated[Path | None, typer.Option(help="Write here instead of standard output.")] = None,
    class_name: Annotated[
        str | None, typer.Option(help="Name the generated class, or the keras builder.")
    ] = None,
    opset: Annotated[int, typer.Option(help="ONNX opset to target.")] = 20,
) -> None:
    """Generate framework code from a document."""
    if backend not in {"torch", "keras", "onnx"}:
        typer.secho(
            f"backend {backend!r} is not available; choose torch, keras or onnx",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)

    document = _load(path)
    try:
        if backend == "onnx":
            _emit_onnx(document, out, opset)
            return
        emitted = (
            emit_keras_document(document, function_name=class_name)
            if backend == "keras"
            else emit_torch_document(document, class_name=class_name)
        )
    except (EmitError, OnnxEmitError, NotResolvable, ResolveError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if out is None:
        typer.echo(emitted.source, nl=False)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emitted.source, encoding="utf-8", newline="\n")
    typer.secho(f"wrote {out} ({emitted.class_name})", fg=typer.colors.GREEN)


def _emit_onnx(document: Document, out: Path | None, opset: int) -> None:
    if out is None:
        typer.secho(
            "ONNX is binary; pass --out to write a .onnx file", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)
    exported = export_onnx_document(document, opset=opset)
    out.parent.mkdir(parents=True, exist_ok=True)
    exported.save(out)
    nodes = len(exported.model.graph.node)
    typer.secho(f"wrote {out} (opset {exported.opset}, {nodes} nodes)", fg=typer.colors.GREEN)


@app.command("import")
def import_command(
    path: Annotated[Path, typer.Argument(help="Path to a .onnx file.")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the .ntb.")],
    name: Annotated[str | None, typer.Option(help="Name the imported model.")] = None,
) -> None:
    """Seed a document from an existing ONNX model. Weights are not carried."""
    try:
        from ntb.importers import OnnxImportError, import_onnx
    except ImportError as exc:  # pragma: no cover - the extra is declared
        typer.secho("reading ONNX needs: pip install 'ntb[onnx]'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    try:
        result = import_onnx(path, name=name)
    except (OnnxImportError, OSError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    io.save(result.document, out)
    module = result.document.root_module
    typer.secho(
        f"wrote {out} ({len(module.nodes)} nodes, {len(module.edges)} edges)",
        fg=typer.colors.GREEN,
    )
    for problem in result.problems:
        typer.secho(f"  {problem}", fg=typer.colors.YELLOW, err=True)
    typer.echo("weights are not imported; the architecture is.")


@app.command()
def studio(
    path: Annotated[Path | None, typer.Argument(help="Open this .ntb file.")] = None,
    host: Annotated[
        str, typer.Option(help="Interface to bind. Loopback by default.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8756,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open a browser on start.")
    ] = True,
) -> None:
    """Open the graphical editor in a browser."""
    try:
        from ntb.server.app import serve
    except ImportError as exc:
        typer.secho(
            "the studio needs the server extra: pip install 'ntb[server]'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    if path is not None and not path.is_file():
        typer.secho(f"{path}: no such file", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"NTB Studio on http://{host}:{port}/  (ctrl-c to stop)", fg=typer.colors.GREEN)
    serve(path, host=host, port=port, open_browser=open_browser)


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
