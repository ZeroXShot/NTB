# 12. MCP tools are generated from the command bus

**Status:** Accepted · 2026-08-25

## Context

Phase 7 exposes NTB to agents. An agent needs to do what a person does in the
studio: add a node, connect two ports, place a generator, check the diagnostics,
read the generated code, save the file.

Every one of those already exists as a command on the bus ([ADR 5](0005-single-command-bus.md)),
validated, undoable and tested. The temptation is to write an MCP tool for each
one by hand, because tool descriptions are prose and prose feels like something
a human should write. That is how the third implementation of editing gets
written, and how it starts drifting from the other two.

The alternative — one generic `apply(command)` tool taking the whole union — is
cheap to write and bad to use. An agent choosing between twenty commands inside
one blob of JSON schema does worse than one choosing between twenty tools.

## Decision

The editing tools are **generated from the command union**. For each member of
`AnyCommand`: the tool's name is the command's own `kind` discriminator, its
input schema is the command's own fields with `kind` removed, its description is
the command's own docstring, and its body is `Workspace.apply`.

Adding a command to the bus therefore adds a tool. Nothing enumerates the
commands twice, and a test asserts that.

`Batch` is the one exclusion: its schema would embed every other command's
inside itself. Multi-step edits get `apply_commands`, which takes the same
payloads as a list and applies them as one undo step.

Everything that is not a single command — session verbs, reading, runs — is
written by hand, because there is nothing to generate it from.

## Consequences

* One implementation of editing serves the studio, the CLI and agents. A
  validation rule fixed once is fixed for all three.
* Tool schemas are the IR's own pydantic models, so an agent sees the real
  `Node`, the real `Placement`, the real `SpatialRule` — including the
  descriptions already written on those fields.
* Tool results are a **summary**, not the document: revision, size, validity and
  the diagnostics. The studio gets a whole snapshot because it draws one; an
  agent that wants the document reads the `ntb://doc` resource. Tokens are the
  agent's scarce resource, and most edits do not need the graph read back.
* Tool descriptions are only as good as the command docstrings, which is a
  reason to keep those good rather than a reason to write them twice.
* The MCP server holds its own workspace rather than attaching to a running
  studio. Two writers on one document is a design question of its own, and
  answering it badly now would be worse than not answering it.
