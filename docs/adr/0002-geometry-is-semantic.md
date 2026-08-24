# 2. Geometry is semantic

**Status:** Accepted · 2026-08-24

## Context

Existing tools treat 3D as presentation: Netron, VisualTorch and TensorSpace all
draw a graph that was fully determined before rendering. NTB exists to support
research into networks and transformers that stack vertically through three
dimensions. If position were only a drawing hint, that research would have no
representation to work in, and NTB would be a nicer viewer.

## Decision

A node's `Placement` -- position, extent and orientation -- is part of the
model. Connectivity may be *derived* from geometry through a `SpatialRule`, and
repetition through space is expressed by a `Generator` rather than by copied
nodes.

To keep this from leaking everywhere, the IR has two levels. `resolve()` expands
generators and evaluates spatial rules into explicit edges, producing
**NTB-Core IR**: a flat, typed DAG with no geometry at all. Shape inference,
validation and all three backends only ever see the core level.

`SpatialRule` is a **closed, versioned set** of kinds, not an open DSL. Each
kind resolves deterministically, is covered by tests, and is extended only
through a PR carrying its own ADR.

## Consequences

* The research surface can evolve quickly without touching the emitters.
* Resolution must be deterministic, or `.ntb` files stop being reproducible and
  golden-file tests become flaky.
* The studio has to visualise *derived* edges distinctly from authored ones. A
  user who cannot see what a rule generated cannot debug it.
* The closed-set rule will feel restrictive. That is the trade: an open spatial
  DSL would be an under-specified language nobody could reason about.
