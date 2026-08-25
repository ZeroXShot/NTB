# 14. Rendering layers, id-buffer picking, and a render budget

**Status:** Accepted · 2026-08-25 · amends [ADR 6](0006-single-3d-canvas.md)

## Context

[ADR 6](0006-single-3d-canvas.md) said two things about how the canvas would be
built that turned out not to be true of what got built:

* it named `WebGPURenderer` with a WebGL2 fallback; the scene is `WebGLRenderer`
  and nothing else;
* it said picking uses a GPU id-buffer pass rather than per-object raycasting;
  picking raycasts an `InstancedMesh`, and resolves a hit by using the
  `instanceId` directly as an index into the node array.

Neither was a lie at the time — they were the plan, and `docs/roadmap.md` has
recorded the picking one as a known limit since phase 3. But an Accepted ADR
reads as a description of the system, and two of its sentences describe a system
nobody wrote. That is worse than having no ADR: it is a decision record that
cannot be trusted, in the directory whose whole purpose is to be trusted.

Meanwhile the work ahead makes both questions live again. Giving each op its own
glyph means nodes stop sharing one `InstancedMesh` — and the moment they do,
`instanceId` stops being an index into the node array, silently. Hover means
picking on every pointer move, which raycasting twenty thousand instances cannot
carry. Imported meshes mean raycasting arbitrary triangles. And a bundled edge
whose width is computed in a vertex shader has no CPU-side geometry to raycast
against at all.

## Decision

**One renderer, `WebGLRenderer`, behind a factory.** `three` vendors
`three/webgpu`, so this is not about availability. It is about dialect: every
shader in the design ahead — the edge ribbons, the id material, the activity
modulation — would have to be written again in TSL for a WebGPU path, or written
twice. That is the whole rendering surface, duplicated, to accelerate a scene of
boxes that WebGL2 draws at frame rate. The renderer is constructed in one place
so the question can be reopened cheaply, and it will be reopened when TSL is the
only way to express something needed, not before.

**Picking is a GPU id-buffer pass.** The scene renders one pixel under the
cursor into a 1×1 target with an id material that shares each layer's vertex
shader, and decodes an RGBA8 pixel back into a kind tag and an index. This is
what ADR 6 promised, and it is now the reason the rest of the design is
possible rather than a performance nicety. A raycasting implementation stays
behind the same interface, as the fallback and as the oracle the picking test
compares against.

**Instancing is per glyph geometry, with an explicit slot map.** Nothing infers
a node id from an instance index. The map exists whichever picker is in use.

**The render budget is visible.** The scene draws at most a fixed number of
blocks, and says how many it left out. Drawing part of a model without saying so
makes a truncated document look like a document with pieces missing.

## Consequences

* ADR 6's claims about the renderer and about picking are superseded by this
  one. Its central decision — one scene, one selection, one set of interactions,
  with 2D as an orthographic camera — stands unchanged, and this ADR keeps it:
  level of detail bands on **projected pixel size** work identically under both
  cameras, which is what stops the two modes drifting apart.
* Hover, imported meshes and per-glyph geometry all become possible at once,
  because they all needed the same thing.
* Picking gains a failure mode that only shows up at a device pixel ratio other
  than 1, if the pointer-to-pixel conversion and the render offset disagree. It
  gets a test of its own for that reason.
* A click cannot wait for an asynchronous read-back, so `pointerdown` reads the
  target synchronously when it has no fresh hover result. One stall per click.
* We are now committed to patched shader chunks, so `three` is pinned to an
  exact version rather than a caret range.
