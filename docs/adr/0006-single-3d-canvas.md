# 6. One 3D canvas; 2D is an orthographic camera

**Status:** Accepted · 2026-08-24 · amended by
[ADR 14](0014-rendering-layers-and-id-buffer-picking.md)

> **Amendment.** Two sentences below describe a system that was planned and not
> built: the renderer is `WebGLRenderer` with no WebGPU path, and picking
> raycasts the instanced mesh rather than using an id buffer. ADR 14 records
> what is true, and why the id-buffer half is now being built rather than
> dropped. The central decision here -- one scene, one selection, one set of
> interactions, with 2D as an orthographic camera -- stands unchanged.

## Context

NTB needs a 2D editing mode and a 3D one. The cheap route is a mature 2D node
library (React Flow) plus a separate three.js scene for 3D. It would get the 2D
editor working in days.

It also creates two editors with two selection models, two undo paths and two
renderings of the same document -- which diverge. Worse, given ADR 2, a React
Flow canvas structurally cannot represent a node's Z coordinate, so the 2D mode
would be unable to show part of the model's meaning.

## Decision

One three.js scene (`WebGPURenderer`, WebGL2 fallback). "2D mode" is the same
scene under an orthographic camera with Z locked and grid snapping; "3D mode" is
a perspective camera. Same picking, same undo, same IR.

Picking uses a GPU id-buffer pass rather than per-object raycasting: it is what
scales to large graphs and it behaves identically under both cameras.

## Consequences

* The editor chrome -- ports, edge routing, snapping, marquee selection -- has to
  be built rather than inherited. This is the main cost, paid in phase 3.
* There is exactly one place where a rendering or interaction bug can live.
* Instanced rendering and id-buffer picking are in from the start, because
  retrofitting them after the interaction code exists is far more expensive.
