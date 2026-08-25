# MCP

An agent builds, validates and trains a model without touching the UI.

```bash
pip install "ntb[mcp]"
ntb mcp                      # stdio, the usual transport
ntb mcp model.ntb            # open a document first
ntb mcp --http --port 8757   # streamable HTTP on /mcp
```

Register it with any MCP client. For Claude Code:

```bash
claude mcp add ntb -- ntb mcp
```

## What it exposes

**Tools are the command bus.** Every editing tool is generated from a member of
the command union ([ADR 12](adr/0012-mcp-tools-are-generated-from-the-command-bus.md)):
`add_node`, `connect`, `move_node`, `add_generator`, `add_rule`, `bind_port` and
the rest, each with the IR's own schema. An agent's edit is validated by the
same code as a person's, and `undo` undoes it.

| | |
|---|---|
| editing | one tool per command, plus `apply_commands` for one undo step |
| history | `undo`, `redo` |
| session | `new_document`, `open_document`, `save_document` |
| reading | `describe_document`, `inspect_module`, `resolved_graph`, `generate_code` |
| registry | `list_ops`, `op_details` |
| training | `start_run`, `list_runs`, `run_status`, `stop_run` |

**Resources** are the things worth reading whole:

| URI | |
|---|---|
| `ntb://doc` | the document, exactly as it would be saved |
| `ntb://code/{backend}` | generated `torch` or `keras` source |
| `ntb://diagnostics` | every diagnostic, with the node, port or edge it is about |
| `ntb://ops` | the whole op registry |
| `ntb://schema` | the NTB-IR JSON Schema |

## What a tool answers

A summary, not the document:

```json
{"revision": 7, "path": null, "dirty": true, "name": "agent-mlp",
 "root": "model", "nodes": 3, "edges": 2, "valid": true,
 "errors": [], "warnings": [], "canUndo": true}
```

The studio gets a whole snapshot after every edit because it draws one. An agent
pays for what it reads, and most edits only need to know whether they broke
something — so the document is a resource it asks for when it wants it.

`resolved_graph` is the one to reach for after placing a generator or a spatial
rule: it shows the flat graph the backends see and marks which rule derived each
edge. Geometry that produced nothing produces no edges, and this is where that
shows up.

## Two writers, one document

The MCP server holds its own workspace; it does not attach to a running
`ntb studio`. Sharing one document between an agent and a person is a design
question of its own — whose undo stack, whose save — and answering it badly now
would be worse than not answering it. Today, an agent edits a file and a person
opens it.
