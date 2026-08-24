# 8. `.ntb` is canonical JSON

**Status:** Accepted · 2026-08-24

## Context

Model files will live in git -- in this repo's `examples/`, and in users' own
repositories. A binary format (protobuf, msgpack) is smaller and faster, but it
turns every change into an opaque diff and makes review impossible.

Formats also outlive the code that wrote them. Once users have `.ntb` files
committed, the format cannot be broken.

## Decision

`.ntb` is JSON with sorted keys, two-space indent, UTF-8, and one trailing
newline. Defaults are written out rather than omitted, so a file reads as a
complete description of the model and a later change to a default cannot
silently change what an existing file means.

Saving is atomic: write a temp file in the destination directory, fsync, rename.
An interrupted save must never truncate a model.

Documents carry `schema_version`, and `ntb.ir.migrate` exists from day one, with
its "no migration available" path tested. Reading a file from a *newer* NTB fails
with an explicit upgrade instruction rather than a validation error.

## Consequences

* Changing one attribute produces a one-line diff. Model changes are reviewable.
* Files are larger than they need to be. For the sizes involved this does not
  matter; a compact format can be added later as an export, never as the
  authoring format.
* Every schema change owes a migration and a fixture of the old version.
