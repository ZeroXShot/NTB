# Roadmap

Each phase has an exit criterion that can be checked, not argued about. Phases
do not overlap: an unfinished foundation makes every later phase more expensive.

**Nothing is published to PyPI until every phase is done.** The release workflow
and the packaging are ready and tested; the tag is deliberately withheld, so that
the first `.ntb` files in circulation come from a format that has already met
every feature that was going to bend it. Until then, install from a checkout.

| Phase | What ships | Exit criterion | Status |
|---|---|---|---|
| 0 | Repo, licence, CI matrix, ADRs | CI green on 9 os/python combinations | **done** |
| 1 | NTB-IR, op registry, symbolic shapes, `ntb validate` | A transformer and a 3D CNN authored by hand validate; symbolic dims propagate end to end | **done** |
| 2 | Python AST emitter, torch backend, ONNX export, parity harness | Every example compiles to torch, trains a step, exports to ONNX and passes numeric parity | **done** |
| 3 | Studio v1: server, command bus, single 3D canvas in 2D mode | `pip install ntb && ntb studio` on Windows, macOS arm64 and Linux; build an MLP graphically and train it | **done** |
| 4 | Spatial semantics: perspective editing, `Generator`, `SpatialRule` | A vertically stacked 3D architecture resolves to a DAG, validates, emits torch and trains | **done** |
| 5 | Keras 3 backend; best-effort ONNX import | One `.ntb` emits torch and Keras; all three backends agree numerically | **done** |
| 6 | Training inside NTB: isolated run subprocess, metrics, curves | Launch, monitor and resume a training run from the studio | **done** |
| 7 | MCP server (spec 2026-07-28), stdio + streamable HTTP | An agent builds and validates an architecture without touching the UI | |
| 8 | Plugin SDK for third-party ops, docs site, example gallery | An op contributed from outside the repo loads and emits | |

## Phase 6 in detail

See [docs/training.md](training.md).

- [x] A run is a subprocess that speaks one JSON object per line
      ([ADR 11](adr/0011-training-runs-in-their-own-process.md)); the studio
      survives the model that does not, and *stop* actually stops
- [x] `RunManager`: launches, reads the stream on a thread, records, forwards
- [x] SQLite store, so a run outlives the session that started it, and one left
      running by a session that ended is marked stopped rather than lied about
- [x] Checkpoints, and resume — which starts a *new* run continuing the step
      count, leaving the original record alone
- [x] Synthetic data shaped from the model's own inputs, for the question you
      have while drawing it, plus a `dataloaders(batch_size)` script for real data
- [x] `ntb run` and `ntb runs`, and a Train tab in the studio with a live loss
      curve fed by the WebSocket
- [x] The studio saves before starting a run: the worker reads the file, and
      training something other than what is on screen would be worse

Deliberately absent: validation loops, schedulers, early stopping, distributed
training, and any metric but the loss. When "does this train" is answered,
`ntb emit` hands over source for a real training setup.

## Phase 5 in detail

- [x] Keras 3 emitter, functional API, channels-first so a document means the
      same model as in torch
- [x] `ntb emit --backend keras`, with golden files for all five examples
- [x] Three-way numeric parity: 25 of 25 verifiable ops agree between torch and
      Keras with the weights transferred, and 25 of 25 between torch and
      onnxruntime. The weight layouts are declared in the registry
      (`WeightSpec`), not hand-written per op
- [x] Declarative knobs instead of special cases: `pad_target` (Keras has no
      integer padding), `shape_arg` (no axis-range flatten), `derived`
      (`key_dim = embed_dim // num_heads`), `call_constants`, and `guard`
- [x] `ntb import model.onnx --out model.ntb`, with automatic left-to-right
      layout and explicit boundary bindings, since ONNX says exactly where the
      model's input goes
- [x] Import reads attributes back out of the weight shapes the registry already
      declares, and reports what it could not read instead of guessing

Two bugs the phase found in earlier work, both now pinned by tests:

* The **ONNX exporter dropped `padding` entirely**. Every padded convolution
  exported as a smaller one. The parity cases were unpadded, and `cnn3d` ends in
  a global pool, so the output shape matched anyway. The parity cases are padded
  now.
* `avgpool` disagreed between torch and ONNX: torch averages over the padded
  window, ONNX leaves the padding out of the divisor unless told otherwise.

What Keras cannot express, and now refuses rather than approximating:

* A **padded max pool**. Keras pads with zeros; a max pool needs -inf, so the
  model would be a different one. Convolutions and average pooling are fine.
* `layernorm` over more than one axis.

Not imported: weights. NTB-IR stores an architecture, not a checkpoint. Carrying
tensors across is phase 6 work.

## Phase 4 in detail

The phase the project exists for. See [docs/spatial.md](spatial.md).

- [x] `Generator` expansion: one object stands for N repetitions through space,
      chained or left to a rule
- [x] All four `SpatialRule` kinds, as pure deterministic functions over
      placements (`ntb.spatial.rules`)
- [x] Module parameters and `"$expr"` attributes, parsed and walked rather than
      evaluated, because `.ntb` files travel between people
- [x] `ntb.sum` and variadic ports: without fan-in, three of the four rules
      produce documents that can never validate
- [x] `ntb resolve`, which marks every edge geometry produced rather than a person
- [x] A spatial preview the studio draws from, computed by the same code that
      lowers the document
- [x] Perspective camera, orbit, alt-drag to lift a block along Z, and a Space
      panel that edits generators and rules
- [x] Diagnostics that name the repetition (`col3-0/mix`), not just the module
      it was stamped from
- [x] `examples/lattice_3d.ntb`: 16 cells on a 2x2x4 grid, 64 lowered nodes,
      28 edges nobody drew, and every parameter receives a gradient

Found along the way, and worth recording: the torch emitter named values after
the last segment of a node path, so two repetitions of one module shared a
variable and the generated `forward` silently dropped most of the model. Only 24
of 64 parameters saw a gradient. It is pinned by a test now.

Known limits:

- Picking still raycasts the instanced mesh rather than using a GPU id buffer.
- A generated block cannot be selected into the inspector for editing, by
  design: the generator is the object, not its repetitions.
- `resolve()` re-runs from scratch on every edit. Fine at this size, and the
  place to start if a 10k-node document ever feels slow.

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
