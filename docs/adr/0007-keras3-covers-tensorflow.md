# 7. Keras 3 covers TensorFlow and JAX

**Status:** Accepted · 2026-08-24

## Context

The requirement is TensorFlow support. The direct reading is a TensorFlow emitter
alongside the torch one: a second full backend, with its own op mappings, its own
parity tests and its own maintenance.

Keras 3 is multi-backend -- the same Keras code runs on TensorFlow, JAX or torch,
chosen by configuration.

## Decision

Emit Keras 3 rather than raw TensorFlow. One emitter reaches TensorFlow and JAX,
and reaches torch a second way, which incidentally gives the parity harness a
useful cross-check.

## Consequences

* Roughly one backend's worth of work covers two required frameworks and a bonus
  one.
* NTB inherits Keras's abstractions and its gaps. Some ops map awkwardly: Keras
  has no integer padding, so non-zero padding lowers to an explicit
  `ZeroPadding2D`. Those adaptations live in the op registry (ADR 3).
* Users wanting hand-written TensorFlow without Keras are not served. If that
  demand turns out to be real, a native TF emitter can be added later against the
  same registry.
