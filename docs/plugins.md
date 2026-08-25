# Writing an op

NTB is for architectures no framework has a name for yet, so sooner or later it
will not have the op you need. Adding one does not require forking NTB: put it
in a distribution of your own and NTB will find it.

A working example lives in [`examples/plugin`](https://github.com/ZeroXShot/NTB/tree/main/examples/plugin).
It is three files.

```bash
pip install -e examples/plugin
ntb plugins        # example -> ntb_example_op from ntb-example-op
ntb ops | grep softsign
```

## The two halves

**Declare the entry point.** This is the whole mechanism:

```toml
# pyproject.toml
[project.entry-points."ntb.ops"]
example = "ntb_example_op"
```

NTB reads that group on startup and imports what it names. The module may
register at import time, or the entry point may name a callable
(`"mypkg:setup"`) which NTB calls.

**Declare the op.** An op is data, not code — you are not implementing softsign,
you are saying where softsign already lives in each backend and what it does to
a shape:

```python
from ntb.sdk import (
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    ParityCase,
    PortSpec,
    register,
)


def _same_shape(ctx):
    tensor = ctx.input("in")
    if not tensor.dtype.is_floating:
        ctx.fail(f"in must be a floating dtype, got {tensor.dtype.value}")
    return {"out": tensor}


register(
    OpSpec(
        name="example.softsign",
        category="activation",
        doc="Elementwise ``x / (1 + |x|)``, a bounded alternative to tanh.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_rule=_same_shape,
        torch=BackendMapping(
            target="torch.nn.functional.softsign",
            kind=CallKind.FUNCTION,
            imports=("torch", "torch.nn.functional"),
        ),
        keras=BackendMapping(
            target="keras.activations.softsign", kind=CallKind.FUNCTION, imports=("keras",)
        ),
        onnx=OnnxMapping(op_type="Softsign", since_opset=1),
        parity=ParityCase(inputs={"in": (2, 6)}),
    )
)
```

Import from `ntb.sdk` and nowhere else. It is the surface that is promised to
keep working; everything under `ntb.ops.*` is NTB's own business and moves.

## What you get for it

Nothing has to be told about your op. Every one of these reads the same registry:

| | |
|---|---|
| `ntb ops`, `ntb validate` | the op and its attribute rules |
| the studio palette and inspector | its category, ports and attributes |
| the MCP tools | `list_ops`, `op_details`, and agents can place it |
| torch, Keras and ONNX emitters | the three mappings |
| the numeric parity harness | your `ParityCase` |

That last one is the one to notice. The harness is **generated from the
registry**, so declaring a parity case is enough to have your op fed the same
random tensor in all three backends and compared:

```
$ pytest -q -k softsign
4 passed
```

An op without a `ParityCase` still works and still emits — it just ships
unverified, and `ntb ops` is where that shows.

## The rules

**`ntb.*` is reserved.** A plugin that registers into it has its ops taken back
out and is reported as a problem. Those are the names this repo guarantees the
meaning of, and `ntb.conv2d` has to mean the same thing on every machine.

**A broken plugin does not break NTB.** An import that raises is reported, never
propagated: `ntb plugins` lists it and exits non-zero, and a document that does
not use the missing op still opens, validates and emits.

**Plugin ops are not portable.** A `.ntb` naming `example.softsign` needs your
plugin installed. Without it, `ntb validate` says

```
error: m/a: unknown op 'example.softsign' [unknown-op]
```

which is a diagnostic, not a crash. Reserving `ntb.*` is what keeps the portable
subset portable.

**To reproduce a bug without plugins in the way**, set `NTB_NO_PLUGINS=1`.

## Contributing an op to NTB itself

If the op is standard — something torch, Keras and ONNX all already have — it
probably belongs in the registry rather than in a plugin. That is the same
declaration, dropped in `src/ntb/ops/builtin/`, plus a parity case.
[CONTRIBUTING.md](https://github.com/ZeroXShot/NTB/blob/main/CONTRIBUTING.md)
has the details. See [ADR 13](adr/0013-third-party-ops-are-entry-points.md) for
why plugins work the way they do.
