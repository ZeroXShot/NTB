# 13. Third-party ops arrive through entry points

**Status:** Accepted · 2026-08-25

## Context

NTB exists so that people can build architectures no framework has a name for
yet. Sooner or later one of those needs an op the registry does not have, and
the answer cannot be "open a pull request and wait". A researcher has to be able
to add an op to their own NTB without forking it.

The op registry is already declarative data ([ADR 3](0003-op-registry-is-declarative.md)):
ports, attributes, a shape rule and one mapping per backend, from which
validation, all three emitters, the studio palette, the MCP tools and the
numeric parity harness are derived. So the question is not *how do we support
custom ops* — the registry already can — but *how does an op that lives in
someone else's distribution get into it*.

## Decision

A distribution declares `[project.entry-points."ntb.ops"]`. NTB reads that group
once, on first use of the registry, and imports what it names. Importing the
module registers the ops, exactly as the built-ins do; an entry point may also
name a callable, which is then called.

The authoring surface is one module, `ntb.sdk`. A plugin imports from there and
nowhere else, so the layout of `ntb.ops.spec` stays the project's business.

`ntb.*` is reserved. A plugin that registers into it has its ops taken back out
and is reported as a problem — those names are the ones this repo guarantees the
meaning of, and a `.ntb` file naming `ntb.conv2d` must mean the same thing on
every machine.

A plugin that fails to import is **reported, not raised**. `ntb plugins` lists
the failures and exits non-zero; everything else keeps working.

## Consequences

* A plugin op is an op. It appears in `ntb ops`, the studio palette, the MCP
  tools, and it emits to torch, Keras and ONNX — because every one of those
  reads the same registry rather than a list of its own.
* It is verified by machinery it did not write: the parity harness is generated
  from the registry, so declaring a `ParityCase` is enough to have the op
  checked numerically against onnxruntime and Keras. An op without one ships
  unverified, and that is visible.
* NTB imports third-party code at startup. That is the cost of the op appearing
  everywhere without anything being told about it. `NTB_NO_PLUGINS=1` turns it
  off, which is what to reach for when reproducing a bug.
* A broken plugin degrades NTB instead of breaking it. A document that does not
  use the missing op still opens, validates and emits.
* Plugin ops are not portable. A `.ntb` naming `example.softsign` needs that
  plugin installed to resolve, and says so as a validation error rather than a
  crash. Reserving `ntb.*` is what keeps the portable subset portable.
