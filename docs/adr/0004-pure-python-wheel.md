# 4. Ship a pure-Python wheel

**Status:** Accepted · 2026-08-24

## Context

NTB must run on Linux, macOS and Windows, on x86_64 and arm64. Two shapes were
considered: a desktop app (Tauri or Electron) bundling a Python sidecar, or a
Python package that serves its UI to the browser.

Bundling loses badly here. The runtimes NTB talks to -- torch, TensorFlow -- are
multi-gigabyte, architecture-specific, and often already installed in the user's
environment. A bundled sidecar would ship a *second* copy that cannot see the
user's models or GPU, and the arm64 packaging story (universal binaries, CUDA
wheels) is a permanent tax.

## Decision

Distribute as `pip install ntb`, and run the studio with `ntb studio`, which
starts a local server and opens the browser. The wheel is `py3-none-any`: no
native extensions in the core, ever. The compiled frontend is built in CI and
packaged inside the wheel, so end users never need Node.js.

Frameworks are optional extras: `ntb[torch]`, `ntb[keras]`, `ntb[onnx]`,
`ntb[all]`. NTB with none of them installed must still open, edit and validate a
document.

## Consequences

* Every platform and architecture is supported by construction, with no per-arch
  build matrix. CI enforces this by failing if the built wheel is not
  `py3-none-any`.
* NTB installs into the same environment as the user's torch, so it sees their
  actual versions and hardware.
* Performance-critical work must stay expressible in Python, or move into an
  *optional* accelerator that NTB works without.
* A native desktop wrapper stays possible later; it would wrap this server
  rather than replace it.
