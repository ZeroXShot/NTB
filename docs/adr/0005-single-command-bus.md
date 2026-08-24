# 5. One command bus for UI, CLI and MCP

**Status:** Accepted · 2026-08-24

## Context

NTB will be driven from three directions: a person in the studio, a script on
the command line, and an agent over MCP. Each needs to mutate a document. Given
separately, three mutation paths mean three sets of invariants to keep and three
chances to corrupt a model -- and undo/redo would only ever work in one of them.

## Decision

Every mutation goes through `apply_command(doc, cmd)`, which returns a patch and
its inverse. The studio, the CLI and the MCP server are all *clients* of that one
function. The server holds the authoritative document; clients send commands and
receive patches. Undo/redo is a stack of inverse commands.

## Consequences

* The MCP server of phase 7 is mostly a thin exposure of commands that already
  exist and are already tested, rather than a second implementation of editing.
* Undo, audit, and eventual multi-user collaboration are one mechanism.
* Every command needs a correct inverse. This is tested as a property: apply
  then invert must return the original document.
* The client is never authoritative. Optimistic UI updates must reconcile
  against the server's patch.
