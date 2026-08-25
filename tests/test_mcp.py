"""The MCP server: an agent builds and validates an architecture, no UI.

The tools are generated from the command union (ADR 12), so most of what is
worth testing here is that the generation is faithful -- every command reachable,
every schema the command's own -- and that the exit criterion of phase 7 holds
end to end.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp", reason="the MCP server needs the mcp extra")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from ntb.commands.model import AnyCommand  # noqa: E402
from ntb.mcp import Workspace, build_server, command_types  # noqa: E402
from ntb.mcp.tools import command_name  # noqa: E402
from tests.conftest import EXAMPLES  # noqa: E402


def call(server: Any, tool: str, /, **arguments: Any) -> Any:
    """Invoke a tool the way a client would, and unwrap what it said.

    Structured output is what a client reads; MCP wraps a non-object return in
    `{"result": ...}`, so unwrap that and fall back to the text blocks.
    """
    result = asyncio.run(server.call_tool(tool, arguments))
    structured = result.structured_content
    if structured is not None:
        return structured["result"] if set(structured) == {"result"} else structured
    return result.content[0].text if result.content else ""


def fails(server: Any, tool: str, /, **arguments: Any) -> str:
    """What the agent is told when a tool refuses."""
    with pytest.raises(ToolError) as caught:
        asyncio.run(server.call_tool(tool, arguments))
    return str(caught.value)


@pytest.fixture
def server() -> Any:
    return build_server(Workspace())


class TestGeneration:
    def test_every_command_but_batch_is_a_tool(self, server: Any) -> None:
        tools = {tool.name for tool in asyncio.run(server.list_tools())}
        for command_type in command_types():
            assert command_name(command_type) in tools
        # Batch would embed every other command's schema in its own.
        assert "batch" not in tools
        assert "apply_commands" in tools

    def test_a_tool_schema_is_the_command_schema(self, server: Any) -> None:
        tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
        schema = tools["add_node"].input_schema
        assert set(schema["required"]) == {"module", "node"}
        assert "kind" not in schema["properties"]
        # The node is the IR's own Node, not a hand-written echo of it.
        assert "Node" in schema["$defs"]

    def test_adding_a_command_would_add_a_tool(self) -> None:
        """The union is the list. Nothing enumerates the commands twice."""
        from typing import get_args

        assert len(command_types()) == len(get_args(get_args(AnyCommand)[0])) - 1


class TestEditing:
    def test_an_agent_builds_an_mlp_that_validates_and_emits(self, server: Any) -> None:
        call(server, "new_document", name="mlp")
        for node_id, op, attrs, x in (
            ("fc1", "ntb.linear", {"in_features": 784, "out_features": 128}, 0.0),
            ("act1", "ntb.relu", {}, 2.0),
            ("fc2", "ntb.linear", {"in_features": 128, "out_features": 10}, 4.0),
        ):
            call(
                server,
                "add_node",
                module="model",
                node={"id": node_id, "op": op, "attrs": attrs, "placement": {"pos": [x, 0, 0]}},
            )
        call(
            server,
            "set_module_ports",
            module="model",
            inputs=[
                {
                    "name": "x",
                    "direction": "in",
                    "type": {"dtype": "float32", "shape": ["batch", 784]},
                }
            ],
            outputs=[{"name": "y", "direction": "out"}],
        )
        for edge_id, src, dst in (("e1", "fc1", "act1"), ("e2", "act1", "fc2")):
            call(
                server,
                "connect",
                module="model",
                edge={"id": edge_id, "src": {"node": src}, "dst": {"node": dst, "port": "in"}},
            )

        report = call(server, "describe_document")
        assert report["valid"], report["errors"]
        assert report["nodes"] == 3 and report["edges"] == 2

        source = call(server, "generate_code", backend="torch")
        assert "class Mlp(torch.nn.Module)" in source
        assert "784" in source and "128" in source

    def test_a_bad_edit_is_refused_and_changes_nothing(self, server: Any) -> None:
        call(server, "new_document", name="m")
        call(server, "add_node", module="model", node={"id": "a", "op": "ntb.relu"})
        message = fails(server, "add_node", module="model", node={"id": "a", "op": "ntb.relu"})
        assert "already has a node" in message
        assert call(server, "describe_document")["nodes"] == 1

    def test_undo_puts_it_back(self, server: Any) -> None:
        call(server, "new_document", name="m")
        call(server, "add_node", module="model", node={"id": "a", "op": "ntb.relu"})
        assert call(server, "undo")["nodes"] == 0
        assert call(server, "redo")["nodes"] == 1

    def test_a_batch_is_one_undo_step(self, server: Any) -> None:
        call(server, "new_document", name="m")
        call(
            server,
            "apply_commands",
            label="two nodes",
            commands=[
                {"kind": "add_node", "module": "model", "node": {"id": "a", "op": "ntb.relu"}},
                {"kind": "add_node", "module": "model", "node": {"id": "b", "op": "ntb.gelu"}},
            ],
        )
        assert call(server, "describe_document")["nodes"] == 2
        assert call(server, "undo")["nodes"] == 0

    def test_a_broken_document_reports_why_instead_of_emitting(self, server: Any) -> None:
        call(server, "new_document", name="m")
        call(
            server,
            "add_node",
            module="model",
            node={"id": "fc", "op": "ntb.linear", "attrs": {"in_features": 8}},
        )
        report = call(server, "describe_document")
        assert not report["valid"]
        assert any("out_features" in error for error in report["errors"])


class TestReading:
    def test_the_geometry_of_a_lattice_is_visible_as_derived_edges(self, server: Any) -> None:
        call(server, "open_document", path=str(EXAMPLES / "lattice_3d.ntb"))
        graph = call(server, "resolved_graph")
        assert len(graph["nodes"]) == 64
        derived = [edge for edge in graph["edges"] if edge["derivedFrom"]]
        assert len(derived) == 28

    def test_the_op_registry_is_the_one_the_studio_shows(self, server: Any) -> None:
        ops = call(server, "list_ops")
        assert {"name", "category", "summary"} == set(ops[0])
        details = call(server, "op_details", name="ntb.conv2d")
        assert "in_channels" in {attr["name"] for attr in details["attrs"]}
        assert "torch" in details["backends"]

    def test_an_unknown_op_says_so(self, server: Any) -> None:
        assert "no op named" in fails(server, "op_details", name="ntb.nonsense")

    def test_inspect_module_shows_what_an_edit_needs(self, server: Any) -> None:
        call(server, "open_document", path=str(EXAMPLES / "vertical_tower.ntb"))
        module = call(server, "inspect_module")
        assert module["generators"], "the tower is built by a generator"


class TestResources:
    def test_the_document_reads_back_as_the_file_would_be_saved(self, server: Any) -> None:
        call(server, "open_document", path=str(EXAMPLES / "mlp.ntb"))
        contents = list(asyncio.run(server.read_resource("ntb://doc")))
        assert json.loads(contents[0].content)["root"] == "mlp"

    def test_the_schema_is_served_so_an_agent_can_read_the_shapes(self, server: Any) -> None:
        contents = list(asyncio.run(server.read_resource("ntb://schema")))
        assert "$defs" in json.loads(contents[0].content)

    def test_generated_code_is_a_resource_per_backend(self, server: Any) -> None:
        call(server, "open_document", path=str(EXAMPLES / "mlp.ntb"))
        torch_source = next(iter(asyncio.run(server.read_resource("ntb://code/torch")))).content
        keras_source = next(iter(asyncio.run(server.read_resource("ntb://code/keras")))).content
        assert "torch.nn.Module" in torch_source
        assert "keras" in keras_source


class TestSaving:
    def test_a_document_is_written_where_it_was_asked(self, server: Any, tmp_path: Path) -> None:
        call(server, "new_document", name="m")
        call(server, "add_node", module="model", node={"id": "a", "op": "ntb.relu"})
        target = tmp_path / "out.ntb"
        assert call(server, "save_document", path=str(target))["path"] == str(target)
        assert json.loads(target.read_text(encoding="utf-8"))["name"] == "m"

    def test_training_an_unsaved_document_is_refused(self, server: Any) -> None:
        call(server, "new_document", name="m")
        assert "save the document first" in fails(server, "start_run")
