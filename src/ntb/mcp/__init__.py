"""The MCP server: the command bus, exposed to agents. See docs/adr/0012."""

from ntb.mcp.server import build_server, serve
from ntb.mcp.tools import ToolError, command_tool, command_types, general_tools
from ntb.mcp.workspace import Workspace, WorkspaceError

__all__ = [
    "ToolError",
    "Workspace",
    "WorkspaceError",
    "build_server",
    "command_tool",
    "command_types",
    "general_tools",
    "serve",
]
