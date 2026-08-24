# 3. The op registry is declarative data

**Status:** Accepted · 2026-08-24

## Context

NTB targets three backends. The naive structure -- a torch emitter, a Keras
emitter and an ONNX emitter, each with its own knowledge of what `conv2d` means
-- gives three places to update per op and three places for them to disagree.
With ~40 canonical ops planned for phase 1 and third-party ops from phase 8,
that divergence is a certainty, not a risk.

## Decision

One `OpSpec` per canonical op declares, in one place: ports, attribute schema,
the shape rule, and the mapping to each backend. Validation, shape inference,
all three emitters and the numeric-parity test harness are *derived* from that
declaration.

A backend quirk belongs in that backend's mapping, not in the emitter. If a
quirk cannot be expressed, the fix is a new knob on the mapping -- not a special
case in emitter code.

Registration is append-only within a process: re-registering a name raises. A
plugin silently shadowing `ntb.conv2d` would change what every existing `.ntb`
file means.

## Consequences

* Adding an op is a declaration plus a parity test that is generated for it.
* Emitters stay small and uniform; they walk the core graph and consult the
  registry.
* The registry has to be expressive enough for genuinely awkward mappings.
  Keras has no integer padding; ONNX pads are top/left/bottom/right. Expect the
  registry to grow knobs. That growth is the design working, not failing.
