# 1. NTB-IR is the single source of truth

**Status:** Accepted · 2026-08-24

## Context

A visual model builder can relate its graph to code in three ways: generate code
from the graph, parse code back into the graph, or both. Full round-tripping is
the feature users ask for first, and it is where tools of this kind historically
die -- parsing arbitrary Python well enough to reconstruct intent is an
open-ended problem, and every partial answer produces a graph that disagrees
with the code it came from.

NTB also has a requirement that settles the question: it must express
architectures that do not exist in any framework yet. Those have no source code
to parse.

## Decision

`NTB-IR` is the authoritative representation. Code generation is one-way:
NTB-IR to torch / Keras 3 / ONNX. Import from ONNX or from source exists, but as
a best-effort seeding step that produces a document the user then owns; it is
never a continuous sync.

Everything that touches a model -- the studio, the CLI, the MCP server,
validation, shape inference, every emitter -- reads and writes NTB-IR. A feature
that needs a second representation is a design error, not a special case.

## Consequences

* Generated code must be good enough to *keep*, because the user cannot edit it
  and feed it back. This raises the bar on the emitters and is why they build a
  Python AST rather than filling in string templates (see ADR 3).
* Importing a model is a one-time conversion. The docs must say so plainly, or
  users will expect a sync that does not exist.
* Novel architectures are first-class rather than an afterthought.
