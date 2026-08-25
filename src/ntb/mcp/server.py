"""The NTB MCP server: an agent edits a document without touching the UI.

Tools are the command bus (ADR 12). Resources are the things worth reading in
full rather than summarising into a tool result: the document, the generated
code, the diagnostics, the op registry and the IR schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from ntb import __version__
from ntb.ir import schema
from ntb.mcp.tools import command_tool, command_types, general_tools
from ntb.mcp.workspace import Workspace
from ntb.server.catalog import op_catalog

INSTRUCTIONS = """\
NTB authors AI architectures. A document is a typed graph whose node positions
are semantic: generators repeat a module through space and spatial rules derive
edges from where blocks sit, so a model can be built by placing rather than by
wiring.

Start with `describe_document` and `list_ops`. Edit with the command tools --
`add_node`, `connect`, `move_node` -- and check `describe_document` after each
one: it reports every diagnostic. `resolved_graph` shows what generators and
rules actually built. `generate_code` gives the torch or Keras source, and
`save_document` writes the `.ntb` file.
"""


def build_server(workspace: Workspace) -> MCPServer:
    """An MCP server over one workspace."""
    server: MCPServer = MCPServer(
        name="ntb",
        title="Neural Tensor Builder",
        version=__version__,
        instructions=INSTRUCTIONS,
        website_url="https://github.com/ZeroXShot/NTB",
    )

    for command_type in command_types():
        tool = command_tool(workspace, command_type)
        server.add_tool(tool, name=tool.__name__)
    for tool in general_tools(workspace):
        server.add_tool(tool)

    @server.resource("ntb://doc", mime_type="application/json")
    def document() -> str:
        """The document being edited, as it would be saved."""
        return workspace.dump()

    @server.resource("ntb://code/{backend}", mime_type="text/x-python")
    def code(backend: str) -> str:
        """Generated source for a backend: torch or keras."""
        return workspace.source(backend)

    @server.resource("ntb://diagnostics", mime_type="application/json")
    def diagnostics() -> str:
        """Every diagnostic, with the node, port or edge each one is about."""
        return json.dumps(list(workspace.session.derived().diagnostics), indent=2)

    @server.resource("ntb://ops", mime_type="application/json")
    def ops() -> str:
        """The whole op registry: ports, attributes and backend coverage."""
        return json.dumps(op_catalog(), indent=2)

    @server.resource("ntb://schema", mime_type="application/json")
    def ir_schema() -> str:
        """The NTB-IR JSON Schema, the shape of everything a command carries."""
        return schema.dumps()

    return server


def serve(
    path: Path | None = None,
    *,
    http: bool = False,
    host: str = "127.0.0.1",
    port: int = 8757,
    runs_root: Path | None = None,
) -> None:
    """Run the server on stdio, or as streamable HTTP."""
    workspace = Workspace(path, runs_root=runs_root)
    server = build_server(workspace)
    try:
        if http:
            server.run(transport="streamable-http", host=host, port=port)
        else:
            server.run(transport="stdio")
    finally:
        workspace.close()
