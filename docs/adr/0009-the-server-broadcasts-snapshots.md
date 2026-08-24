# 9. The studio server broadcasts snapshots, not patches

**Status:** Accepted · 2026-08-24 · Amends [ADR 5](0005-single-command-bus.md)

## Context

ADR 5 says clients send commands and receive patches. Applying a patch on the
client means the client must know what a command does -- a second implementation
of the bus, in TypeScript, that has to agree with the Python one edit for edit.
Any disagreement shows up as a canvas that no longer matches the file that would
be saved, which is the worst class of bug this project can have.

The other half of the state is derived anyway: diagnostics, inferred types and
the generated code are computed from the whole document, not from the patch.

## Decision

The server applies the command and broadcasts the resulting session state: the
document, the derived state, and whether undo and redo are available. The client
redraws from it and never mutates a document of its own.

## Consequences

* One implementation of what an edit means, in Python, where it is tested.
* Each edit sends the whole document. A 10k-node document is roughly a megabyte
  of JSON, which is fine over loopback and not fine over a network. When NTB
  grows remote sessions or collaboration this decision has to be revisited, and
  the command is already the patch that a future protocol would send.
* The client's undo is the server's undo, so two windows on one document cannot
  disagree about history.
