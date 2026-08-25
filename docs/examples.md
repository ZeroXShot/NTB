# Examples

Five documents in [`examples/`](https://github.com/ZeroXShot/NTB/tree/main/examples),
ordered so that each one needs something the last did not. Every one of them
validates, emits to all three backends and trains.

```bash
ntb studio examples/lattice_3d.ntb   # open one
ntb info examples/lattice_3d.ntb     # what is in it
ntb resolve examples/lattice_3d.ntb  # what it lowers to
ntb emit examples/lattice_3d.ntb     # the torch it becomes
```

## mlp.ntb — the floor

A two-layer perceptron over flattened 28×28 images. Three nodes, two edges, no
geometry in play at all: a plain 2D DAG, which is what every other visual model
builder stops at.

**101,770 parameters.** Start here to read what a `.ntb` file *is*; it is the
only one small enough to take in at a glance.

## cnn3d.ntb — shapes that do arithmetic

An 11-node 3D convolutional stack, `[batch, 1, 32, 64, 64]` in, ten classes out.

**14,730 parameters.** This is the one to look at with `ntb shapes`: padding,
stride and pooling all have to agree for eleven layers running, and the shape
inference does that arithmetic symbolically while `batch` stays a symbol.

## transformer_block.ntb — a real block, no special case

A pre-norm encoder block: multi-head attention, residual, layer norm, MLP,
residual. Ten nodes and eleven edges — the extra edge is a residual, which is
the first thing here that is not a straight line.

**3,152,384 parameters,** identical in the torch and the Keras emission. NTB has
no concept of "a transformer block": it is nodes and edges like everything else,
which is the point.

## vertical_tower.ntb — one object, twelve layers

A twelve-high stack expressed as a single `Generator`. The root module contains
*no nodes* — one generator, and a module to repeat.

```
tower: 0 nodes, 0 edges, 0 spatial rules, 1 generators
  → resolves to 24 nodes, 23 edges
```

**789,504 parameters.** Change `count` from 12 to 24 and the network doubles;
nothing is copied and nothing is edited twice. The `lateral` module alongside it
carries a spatial rule, so this is also the smallest example of the two
mechanisms together.

## lattice_3d.ntb — geometry *is* the architecture

Sixteen cells on a 2×2×4 grid, as four generators. **Nobody drew an edge in the
root module.** One `lattice` rule reads where the cells sit and wires each to its
neighbours, up and sideways.

```
lattice: 0 nodes, 0 edges, 1 spatial rules, 4 generators
  → resolves to 64 nodes and 76 edges, 28 of them derived from that one rule
```

**68,608 parameters, and all of them receive gradients** — there is a test that
says so, because an earlier emitter bug silently starved forty of them.

Move a column and the topology changes, because in NTB position *is* the model.
This is the example the project exists for; [docs/spatial.md](spatial.md) is the
reference for the four rules and the parameter expressions that let a stack
widen with depth.

## examples/plugin — an op from outside

Not a document but a distribution: one op, `example.softsign`, contributed the
way anyone would contribute one. See [docs/plugins.md](plugins.md).
