# Architecture Decision Records

Every decision that constrains the rest of the project lives here as a numbered
record. A record states the decision, why it was taken, and what it costs, so
that a contributor arriving in a year can tell a deliberate constraint from an
accident.

Change a decision by writing a new ADR that supersedes the old one. Do not edit
an accepted record: the point is the trail.

| # | Decision | Status |
|---|---|---|
| [0001](0001-ntb-ir-is-the-single-source-of-truth.md) | NTB-IR is the single source of truth | Accepted |
| [0002](0002-geometry-is-semantic.md) | Geometry is semantic | Accepted |
| [0003](0003-op-registry-is-declarative.md) | The op registry is declarative data | Accepted |
| [0004](0004-pure-python-wheel.md) | Ship a pure-Python wheel | Accepted |
| [0005](0005-single-command-bus.md) | One command bus for UI, CLI and MCP | Accepted |
| [0006](0006-single-3d-canvas.md) | One 3D canvas; 2D is an orthographic camera | Accepted |
| [0007](0007-keras3-covers-tensorflow.md) | Keras 3 covers TensorFlow and JAX | Accepted |
| [0008](0008-canonical-json-on-disk.md) | .ntb is canonical JSON | Accepted |
| [0009](0009-the-server-broadcasts-snapshots.md) | The studio server broadcasts snapshots, not patches | Accepted |
| [0010](0010-the-studio-server-is-local-only.md) | The studio server answers only same-origin requests | Accepted |
| [0011](0011-training-runs-in-their-own-process.md) | A training run gets its own process | Accepted |
