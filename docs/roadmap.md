# Roadmap

Each phase has an exit criterion that can be checked, not argued about. Phases
do not overlap: an unfinished foundation makes every later phase more expensive.

| Phase | What ships | Exit criterion | Status |
|---|---|---|---|
| 0 | Repo, licence, CI matrix, ADRs | CI green on 9 os/python combinations | **done** |
| 1 | NTB-IR, op registry, symbolic shapes, `ntb validate` | A transformer and a 3D CNN authored by hand validate; symbolic dims propagate end to end | **done** |
| 2 | Python AST emitter, torch backend, ONNX export, parity harness | Every example compiles to torch, trains a step, exports to ONNX and passes numeric parity | **done** |
| 3 | Studio v1: server, command bus, single 3D canvas in 2D mode | `pip install ntb && ntb studio` on Windows, macOS arm64 and Linux; build an MLP graphically and train it. **First PyPI release (0.1.0)** | **done** |
| 4 | Spatial semantics: perspective editing, `Generator`, `SpatialRule` | A vertically stacked 3D architecture resolves to a DAG, validates, emits torch and trains | |
| 5 | Keras 3 backend; best-effort ONNX import | One `.ntb` emits torch and Keras; all three backends agree numerically | |
| 6 | Training inside NTB: isolated run subprocess, metrics, curves | Launch, monitor and resume a training run from the studio | |
| 7 | MCP server (spec 2026-07-28), stdio + streamable HTTP | An agent builds and validates an architecture without touching the UI | |
| 8 | Plugin SDK for third-party ops, docs site, example gallery | An op contributed from outside the repo loads and emits | |

## Phase 3 in detail

- [x] Command bus: every mutation is an invertible command, and undo is a stack
      of the inverses the bus hands back
- [x] Session server: FastAPI, one authoritative document, WebSocket broadcast
      of the whole session state ([ADR 9](adr/0009-the-server-broadcasts-snapshots.md))
- [x] Same-origin policy on a server that can write files
      ([ADR 10](adr/0010-the-studio-server-is-local-only.md))
- [x] Op palette generated from the registry, so a new op needs no UI change
- [x] One three.js scene, orthographic camera, instanced blocks, drag to place
- [x] Inspector with attribute editors built from the registry's declarations
- [x] Live diagnostics and live torch source, both computed by the server
- [x] TypeScript IR types generated from the JSON Schema, drift-checked in CI
- [x] `ntb studio`, with the bundle packaged inside the wheel

Known limits, to revisit when they start to hurt:

- Picking raycasts the instanced mesh rather than using a GPU id buffer. That is
  the plan's approach for 10k nodes and is phase 4 work, when there are that many.
- Connections are made by picking two blocks (`c`, then the target), not by
  dragging port to port. Ports are drawn from the registry but not yet hit-tested.
- A drag sends one `move_node` per release; there is no coalescing of a long
  gesture into a single history entry beyond that.

## Phase 2 in detail

- [x] `ntb.emit.pysrc`: Python source built through `ast`, formatted with ruff
- [x] torch emitter producing a readable `nn.Module`, with backend adaptations
      declared in the registry rather than special-cased in the emitter
- [x] Direct ONNX export, including weight initialisers and dynamic axes
- [x] Numeric parity harness generated from the registry: 24 of 30 ops verified
      identical between torch and onnxruntime, with weights transferred so the
      comparison means something
- [x] Golden files for the generated torch source
- [x] `ntb emit --backend torch|onnx`

Not yet verified across backends: `attention`, `rmsnorm` (both need opset 23+),
`silu` (no standard ONNX op), `dropout` (stochastic), and `reshape`/`flatten`
(ONNX takes the target shape as an input tensor, which the exporter does not
build yet).

## Phase 1 in detail

Phase 1 is where the shape of everything else gets fixed, so it is worth doing
slowly.

- [x] `NTB-IR` types: documents, modules, nodes, ports, edges
- [x] Semantic placement, `SpatialRule` and `Generator` **declarations**
      (resolution itself is phase 4)
- [x] `NTB-Core IR`: flat DAG, topological order, cycle reporting
- [x] Canonical `.ntb` JSON, atomic saves, migration machinery
- [x] JSON Schema generated from the models, drift-checked in CI
- [x] Op registry, declarative `OpSpec`, first three ops
- [x] Symbolic dimension algebra (sympy)
- [x] 30 canonical ops across dense, activation, convolution, normalisation,
      pooling, shape, elementwise and attention
- [x] `ntb.spatial.resolve`: modules and explicit edges lower to the core IR
      (generators and spatial rules are recognised and deferred to phase 4)
- [x] Graph-level shape inference with one message per mistake, not a cascade
- [x] `ntb.validate`: located diagnostics with codes and severities
- [x] `ntb validate` and `ntb shapes` on the command line

Deferred to when there is a UI to need it:

- [ ] Incremental invalidation of inference on edit (phase 3, with the command bus)
- [ ] More ops as the emitters and real models ask for them

## What is deliberately *not* planned

* **Round-tripping code back into the graph.** See
  [ADR 1](adr/0001-ntb-ir-is-the-single-source-of-truth.md).
* **A raw TensorFlow backend.** Keras 3 covers it. See
  [ADR 7](adr/0007-keras3-covers-tensorflow.md).
* **A bundled desktop app for the first release.** See
  [ADR 4](adr/0004-pure-python-wheel.md). A Tauri wrapper around the same server
  stays possible later.
