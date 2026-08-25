# NTB — Neural Tensor Builder

Build AI architectures graphically, in 2D **and** in 3D, then emit them to
PyTorch, Keras 3 (TensorFlow / JAX) or ONNX.

```bash
git clone https://github.com/ZeroXShot/NTB && cd NTB
pip install -e ".[all]"
cd apps/studio && npm install && npm run build && cd ../..
ntb studio examples/lattice_3d.ntb
```

## Why another model builder

Netron shows you a model. VisualTorch draws one. NTB is for *authoring* models,
including ones no framework has a name for yet.

The difference is that **geometry is semantic**. Where a block sits in space is
part of the model, not a hint for the renderer. Connectivity can be derived from
spatial rules — vertical stacking, axis projection, neighbourhood coupling — and
a lowering pass compiles all of that down to an ordinary DAG the backends
understand. That is what makes networks stacked vertically through three
dimensions expressible at all.

```
$ ntb resolve examples/lattice_3d.ntb
lattice-3d: 64 nodes, 76 edges
  col0-0/act.out  -> col1-0/mix.in    via grid
  col0-0/act.out  -> col0-1/mix.in    via grid
```

Sixteen cells, four generators, one rule, and not one edge drawn by hand. See
[Geometry](spatial.md) and the [examples](examples.md).

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

## Four rules

They are what keeps the project one system rather than several, and each is
written up in [Decisions](adr/README.md):

1. **One IR.** Studio, CLI, agents, emitters — everything reads NTB-IR.
2. **One command bus.** Every mutation goes through `apply_command`, so undo,
   audit and the [MCP tools](mcp.md) an agent uses are one mechanism.
3. **The op registry is data.** Ports, attributes, shape rules and all three
   backend mappings are declared once per op; validation, emission and the
   parity tests are derived from that declaration — including for an op that
   [arrives from outside the repo](plugins.md).
4. **Geometry is semantic.** [ADR 2](adr/0002-geometry-is-semantic.md).

## Where to go next

| | |
|---|---|
| [Examples](examples.md) | five documents, smallest first |
| [Geometry](spatial.md) | generators, the four spatial rules, expressions |
| [The studio](studio.md) | the editor |
| [Training](training.md) | runs, curves, checkpoints |
| [Agents](mcp.md) | the MCP server |
| [Writing an op](plugins.md) | contributing one from your own package |
| [Roadmap](roadmap.md) | what is done and what is not |
