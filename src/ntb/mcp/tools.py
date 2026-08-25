"""The tools an agent gets, generated from the command union where possible.

Every editing tool is derived from a member of `AnyCommand`: its name is the
command's own discriminator, its schema is the command's own fields, and its
body is `Workspace.apply`. Adding a command to the bus adds a tool, and there is
no second implementation of editing to keep in step (ADR 12).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, get_args

# The SDK's own error: raised deliberately, its message reaches the agent.
# Anything else is a crash, whose text the SDK keeps on the server.
from mcp.server.mcpserver.exceptions import ToolError

from ntb.commands import Batch, Command, CommandError, parse_command
from ntb.commands.model import AnyCommand
from ntb.mcp.workspace import Workspace, WorkspaceError
from ntb.ops import REGISTRY
from ntb.server.catalog import describe_op

#: Batch's schema would embed every other command's. An agent that wants one
#: undo step gets `apply_commands` instead.
EXCLUDED: frozenset[type[Command]] = frozenset({Batch})

_EDIT_NOTE = "Edits the document through the command bus; undoable with `undo`."

__all__ = ["ToolError", "command_name", "command_tool", "command_types", "general_tools"]

T = TypeVar("T")


def command_types() -> tuple[type[Command], ...]:
    """The commands the union is made of, in the order it lists them."""
    members: tuple[Any, ...] = get_args(get_args(AnyCommand)[0])
    return tuple(t for t in members if t not in EXCLUDED)


def command_name(command_type: type[Command]) -> str:
    return str(command_type.model_fields["kind"].default)


def command_tool(workspace: Workspace, command_type: type[Command]) -> Callable[..., Any]:
    """A tool that builds one command from keyword arguments and applies it."""
    fields = {n: f for n, f in command_type.model_fields.items() if n != "kind"}
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if field.is_required() else field.default,
            annotation=field.annotation,
        )
        for name, field in fields.items()
    ]

    def tool(**kwargs: Any) -> dict[str, Any]:
        try:
            return workspace.apply(command_type(**kwargs))  # type: ignore[arg-type]
        except (CommandError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    tool.__name__ = command_name(command_type)
    tool.__doc__ = f"{(command_type.__doc__ or '').strip()}\n\n{_EDIT_NOTE}"
    tool.__signature__ = inspect.Signature(parameters, return_annotation=dict)  # type: ignore[attr-defined]
    return tool


def general_tools(workspace: Workspace) -> list[Callable[..., Any]]:
    """Everything that is not one command: session verbs, reading, and runs."""

    def apply_commands(commands: list[dict[str, Any]], label: str = "") -> dict[str, Any]:
        """Apply several commands as a single undo step.

        Each entry is one editing tool's arguments plus its `kind`, for example
        `{"kind": "connect", "module": "model", "edge": {...}}`.
        """
        try:
            parsed = tuple(parse_command(payload) for payload in commands)
            return workspace.apply(Batch(commands=parsed, label=label))
        except (CommandError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    def undo() -> dict[str, Any]:
        """Undo the last edit."""
        return _guard(lambda: (workspace.session.undo(), workspace.report())[1])

    def redo() -> dict[str, Any]:
        """Redo the edit last undone."""
        return _guard(lambda: (workspace.session.redo(), workspace.report())[1])

    def new_document(name: str = "untitled") -> dict[str, Any]:
        """Discard the document and start from an empty one."""
        return workspace.new(name)

    def open_document(path: str) -> dict[str, Any]:
        """Open a `.ntb` file, replacing the document in the workspace."""
        return _guard(lambda: workspace.open(Path(path)))

    def save_document(path: str | None = None) -> dict[str, Any]:
        """Write the document to disk. A run trains the file, so save before one."""
        saved = _guard(lambda: workspace.save(Path(path) if path else None))
        return {"path": str(saved), **workspace.report()}

    def describe_document() -> dict[str, Any]:
        """The state of the document: size, validity and every diagnostic."""
        return workspace.report()

    def inspect_module(module: str | None = None) -> dict[str, Any]:
        """One module in full: nodes, edges, boundary ports, rules, generators.

        The root module by default.
        """
        document = workspace.document
        found = document.module(module) if module else document.root_module
        if found is None:
            raise ToolError(f"document has no module {module!r}")
        return {
            "id": found.id,
            "nodes": [
                {"id": n.id, "op": n.op, "attrs": n.attrs, "pos": list(n.placement.pos)}
                for n in found.nodes
            ],
            "edges": [{"id": e.id, "src": str(e.src), "dst": str(e.dst)} for e in found.edges],
            "inputs": [{"name": p.name, "type": str(p.type or "")} for p in found.inputs],
            "outputs": [{"name": p.name, "type": str(p.type or "")} for p in found.outputs],
            "rules": [
                {"id": r.id, "kind": r.kind.value, "axis": r.axis.value, "members": list(r.members)}
                for r in found.spatial_rules
            ],
            "generators": [
                {"id": g.id, "module": g.module, "count": g.count, "axis": g.axis.value}
                for g in found.generators
            ],
        }

    def list_ops(category: str | None = None) -> list[dict[str, str]]:
        """The canonical ops, one line each. `op_details` gives the full spec."""
        return [
            {"name": s.name, "category": s.category, "summary": s.doc.strip().splitlines()[0]}
            for s in sorted(REGISTRY, key=lambda s: (s.category, s.name))
            if category is None or s.category == category
        ]

    def op_details(name: str) -> dict[str, Any]:
        """One op's ports, attributes and the backends it reaches."""
        spec = REGISTRY.get(name)
        if spec is None:
            raise ToolError(f"no op named {name!r}; call list_ops to see them")
        return describe_op(spec)

    def generate_code(backend: str = "torch") -> str:
        """The source this document emits, for `torch` or `keras`."""
        return _guard(lambda: workspace.source(backend))

    def resolved_graph() -> dict[str, Any]:
        """The flat graph the backends see, with the origin of every edge.

        This is how to check what a generator or a spatial rule actually built.
        """
        return _guard(workspace.lowered)

    def start_run(
        epochs: int = 1,
        steps_per_epoch: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        optimiser: str = "adam",
        loss: str = "mse",
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Train the saved document in a process of its own.

        Data is synthetic: random tensors shaped like the model's own inputs. It
        answers whether the architecture trains, not whether it is any good.
        """
        from ntb.runs import RunConfig, RunError

        path = workspace.session.path
        if path is None or workspace.session.dirty:
            raise ToolError("save the document first: a run trains the file on disk")
        try:
            config = RunConfig(
                document=path,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimiser=optimiser,
                loss=loss,
                device=device,
            )
            return workspace.runs().start(config).as_json()
        except (RunError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    def list_runs(limit: int = 20) -> list[dict[str, Any]]:
        """Training runs, most recent first."""
        return [run.as_json() for run in workspace.runs().recent(limit)]

    def run_status(run_id: str) -> dict[str, Any]:
        """One run, with the loss it has recorded so far."""
        runs = workspace.runs()
        run = runs.get(run_id)
        if run is None:
            raise ToolError(f"no run {run_id!r}")
        return {**run.as_json(), "metrics": runs.metrics(run_id)}

    def stop_run(run_id: str) -> dict[str, Any]:
        """Terminate a running training process."""
        from ntb.runs import RunError

        try:
            return workspace.runs().stop(run_id).as_json()
        except RunError as exc:
            raise ToolError(str(exc)) from exc

    return [
        apply_commands,
        undo,
        redo,
        new_document,
        open_document,
        save_document,
        describe_document,
        inspect_module,
        list_ops,
        op_details,
        generate_code,
        resolved_graph,
        start_run,
        list_runs,
        run_status,
        stop_run,
    ]


def _guard(call: Callable[[], T]) -> T:
    try:
        return call()
    except (CommandError, WorkspaceError) as exc:
        raise ToolError(str(exc)) from exc
