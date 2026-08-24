# NTB — Neural Tensor Builder

Build AI architectures graphically, in 2D **and** in 3D, then emit them to
PyTorch, Keras 3 (TensorFlow / JAX) or ONNX.

> **Status: pre-alpha.** The IR, a 30-op registry, symbolic shape inference and
> validation work today. The graphical studio lands in phase 3 — see
> [the roadmap](docs/roadmap.md).

## Why another model builder

Netron shows you a model. VisualTorch draws one. NTB is for *authoring* models,
including ones no framework has a name for yet.

The difference is that **geometry is semantic**. Where a block sits in space is
part of the model, not a hint for the renderer. Connectivity can be derived from
spatial rules — vertical stacking, axis projection, neighbourhood coupling —
and a lowering pass compiles all of that down to an ordinary DAG that the
backends understand. That is what makes networks stacked vertically through
three dimensions expressible at all.

## Install

```bash
pip install ntb            # core: author, validate, inspect
pip install "ntb[torch]"   # + PyTorch emission
pip install "ntb[all]"     # + Keras 3, ONNX and the studio server
```

The wheel is pure Python (`py3-none-any`), so Linux, macOS and Windows on both
x86_64 and arm64 are supported by construction. It installs into the same
environment as your existing torch or TensorFlow, so it sees your actual
versions and hardware.

## Try it

```bash
ntb ops                                  # the canonical op registry
ntb validate examples/transformer_block.ntb
ntb shapes examples/cnn3d.ntb            # the inferred type on every port
ntb info examples/vertical_tower.ntb     # a 12-high stack built by a Generator
```

`ntb shapes examples/cnn3d.ntb` prints, among others:

```
conv1.in   float32[batch, 1, 32, 64, 64]
pool2.out  float32[batch, 32, 8, 16, 16]
head.out   float32[batch, 10]
```

Shapes are symbolic, so a dimension you left open stays open:

```python
from ntb.ir.types import TensorType
from ntb.ops import REGISTRY
from ntb.ops.spec import ShapeContext

conv = REGISTRY.require("ntb.conv2d")
attrs = conv.resolved_attrs({"in_channels": 3, "out_channels": 16, "stride": [2, 2]})
ctx = ShapeContext(
    op=conv.name, attrs=attrs, inputs={"in": TensorType(shape=("batch", 3, "h", 224))}
)
print(conv.shape_rule(ctx)["out"])
# float32[batch, 16, floor(h/2 - 3/2) + 1, 111]
```

## How it fits together

```
NTB-IR              authored, spatial: placements, generators, spatial rules
  │
  │  resolve()      expand generators; spatial rules become explicit edges
  ▼
NTB-Core IR         flat typed DAG, no geometry
  │
  ├─ infer_shapes() symbolic propagation (sympy)
  ├─ validate()     diagnostics located on the block you drew
  │
  └─ emit()  ──┬──  torch    readable nn.Module source
               ├──  keras3   covers TensorFlow and JAX
               └──  onnx     GraphProto, direct
```

Four rules hold the project together, and are written up in
[docs/adr](docs/adr/):

1. **One IR.** Studio, CLI, MCP, emitters — everything reads NTB-IR.
2. **One command bus.** Every mutation goes through `apply_command`, so undo,
   audit and the future MCP server are the same mechanism.
3. **The op registry is data.** Ports, attributes, shape rules and all three
   backend mappings are declared once per op; validation, emission and the
   parity tests are derived from that declaration.
4. **Geometry is semantic.** See [ADR 2](docs/adr/0002-geometry-is-semantic.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most useful contribution right now
is a new canonical op: it is one declaration in `src/ntb/ops/builtin.py`, and
the test harness generates its cross-backend checks.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
