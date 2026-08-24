# Roadmap

Each phase has an exit criterion that can be checked, not argued about. Phases
do not overlap: an unfinished foundation makes every later phase more expensive.

| Phase | What ships | Exit criterion | Status |
|---|---|---|---|
| 0 | Repo, licence, CI matrix, ADRs | CI green on 9 os/python combinations | **done** |
| 1 | NTB-IR, op registry, symbolic shapes, `ntb validate` | A transformer and a 3D CNN authored by hand validate; symbolic dims propagate end to end | in progress |
| 2 | Python AST emitter, torch backend, ONNX export, parity harness | Every example compiles to torch, trains a step, exports to ONNX and passes numeric parity | |
| 3 | Studio v1: server, command bus, single 3D canvas in 2D mode | `pip install ntb && ntb studio` on Windows, macOS arm64 and Linux; build an MLP graphically and train it. **First PyPI release (0.1.0)** | |
| 4 | Spatial semantics: perspective editing, `Generator`, `SpatialRule` | A vertically stacked 3D architecture resolves to a DAG, validates, emits torch and trains | |
| 5 | Keras 3 backend; best-effort ONNX import | One `.ntb` emits torch and Keras; all three backends agree numerically | |
| 6 | Training inside NTB: isolated run subprocess, metrics, curves | Launch, monitor and resume a training run from the studio | |
| 7 | MCP server (spec 2026-07-28), stdio + streamable HTTP | An agent builds and validates an architecture without touching the UI | |
| 8 | Plugin SDK for third-party ops, docs site, example gallery | An op contributed from outside the repo loads and emits | |

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
- [ ] The remaining ~37 canonical ops: norms, activations, attention, embedding,
      pooling, reshape, elementwise, conv1d/conv3d
- [ ] Graph-level shape inference: walk the core graph, cache, invalidate
      incrementally on edit
- [ ] `ntb.validate`: located diagnostics with severities, replacing the
      exceptions shape rules raise today
- [ ] `ntb validate` reporting real semantic errors

## What is deliberately *not* planned

* **Round-tripping code back into the graph.** See
  [ADR 1](adr/0001-ntb-ir-is-the-single-source-of-truth.md).
* **A raw TensorFlow backend.** Keras 3 covers it. See
  [ADR 7](adr/0007-keras3-covers-tensorflow.md).
* **A bundled desktop app for the first release.** See
  [ADR 4](adr/0004-pure-python-wheel.md). A Tauri wrapper around the same server
  stays possible later.
