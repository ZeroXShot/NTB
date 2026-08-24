# Geometry as semantics

In NTB, where a block sits is part of the model. This page is the reference for
what that means: how positions become connections, how one object stands for a
hundred, and what the guarantees are.

The rule behind all of it is [ADR 2](adr/0002-geometry-is-semantic.md). Move a
block and you have edited the model, exactly as if you had retyped a layer.

## The pipeline

```
NTB-IR          nodes with placements, generators, spatial rules
  │
  │  resolve()  repetitions are stamped out, rules become explicit edges
  ▼
NTB-Core IR     a flat typed DAG with no geometry in it at all
```

Everything on this page happens in `resolve()`. No backend ever learns that NTB
has three dimensions: by the time torch or ONNX sees the model, it is an
ordinary graph. That separation is what lets the spatial layer keep moving
without destabilising the emitters.

`ntb resolve model.ntb` prints the result, marking every edge that geometry
produced rather than a person.

## Spatial rules

A rule reads where its members sit and produces edges. It never invents blocks.
The set of kinds is **closed and versioned** — a new kind needs its own ADR —
because an open-ended rule language would be a second, worse programming
language nobody specified.

| Kind | Connects | Takes |
|---|---|---|
| `vertical_stack` | each member to the next one along the axis | `axis` |
| `axis_projection` | each member to every member strictly ahead of it | `axis` |
| `neighborhood` | members within `radius` of one another (Euclidean) | `radius` |
| `lattice` | members exactly one grid step apart on a single axis | — |

Every rule also carries `output_port` and `input_port`: which port of the source
an edge leaves from, and which port of the target it arrives at. That is how one
set of blocks can carry two rules — one feeding port `a`, another feeding `b`.

`neighborhood` and `lattice` may be `bidirectional`. The ordered kinds may not:
they would produce a cycle, and the model validator says so rather than letting
you find out at emit time.

### Determinism

Same placements, same edges, same order. Ties in a coordinate are broken by
member name. This is not an implementation detail: it is what makes a generated
topology reviewable in a `git diff` rather than a surprise on every save.

### Members

A member is a node id, or a **generator id**, which stands for all of that
generator's repetitions in index order. That is how a rule wires a stack it did
not itself create.

### Fan-in

Three of the four kinds naturally send several edges into one block. An ordinary
port accepts one source and validation rejects the rest, so a block meant to
receive them needs a **variadic** port. `ntb.sum` has one:

```
mix = ntb.sum      # any number of sources
 └─ fc = ntb.linear
```

The `lattice_3d` example is built from exactly this cell, which is why it does
not care how many neighbours a given cell turns out to have.

## Generators

A generator is *one object* that means N repetitions of a module through space:

```json
{
  "id": "col0", "module": "cell", "count": 4,
  "axis": "z", "origin": [0, 0, 0], "step": 1.0, "chain": false
}
```

Repetition `i` sits at `origin + i * step` along `axis`, and is named
`col0-0`, `col0-1`, … Widening a stack from 4 to 40 is a one-character diff, and
the studio's Space panel edits it with a number field.

`chain: true` wires each repetition's first output into the next one's first
input — the common case, a tower. `chain: false` lays them out in parallel and
leaves the wiring to a spatial rule, which is how a lattice is built.

## Parameters

An attribute written as `"$expr"` is evaluated against the module's parameters:

```json
"params":  {"width": 64},
"attrs":   {"in_features": "$width", "out_features": "$width * 2"}
```

A generator adds the index `i`, and `attr_bindings` sets parameters per
repetition, so a stack whose width doubles with depth is still one object:

```json
"attr_bindings": {"width": "64 * 2 ** i"}
```

Expressions take `+ - * / // % **`, comparisons of nothing, and
`min max abs int float round`. They are **parsed and walked, never `eval`ed** —
`.ntb` files travel between people, and opening one must not run their code.
An expression that resolves to the wrong type is refused at lowering, naming the
node and the value.

## What this costs

* A cycle is only detectable after lowering. A bidirectional rule can produce
  one; you get a `cycle` diagnostic pointing at the blocks involved.
* A rule that wires several sources into a non-variadic port is an error, not a
  silent merge. Use `ntb.sum`, or narrow the rule.
* `count` is capped at 100,000. A generator is not a loop.
* Positions are floats, compared with a tolerance of 1e-6. Blocks meant to be on
  a grid should be placed on a grid; the studio snaps to 0.25 for that reason.

## Worked example

`examples/lattice_3d.ntb`: sixteen cells on a 2×2×4 grid, four generators, one
lattice rule, and **not one edge drawn by a person**.

```bash
ntb resolve examples/lattice_3d.ntb   # 64 nodes, 76 edges, 28 of them derived
ntb emit examples/lattice_3d.ntb      # the torch module it becomes
```

Signal enters the bottom of one column and leaves the top of another, flowing up
each column and across between them, because that is where the blocks are. Move
a column and the architecture changes.
