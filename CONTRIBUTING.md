# Contributing to NTB

Thanks for looking. NTB is a long project with a small number of load-bearing
decisions; the fastest way to have a change accepted is to work with them rather
than around them.

## The rules that hold the project together

Read [docs/adr](docs/adr/) before proposing anything structural. In short:

1. **NTB-IR is the only representation of a model.** If a change needs a second
   one, it is a design error.
2. **Every mutation goes through the command bus** (`ntb.commands`), with a
   correct inverse. No direct document edits.
3. **Op semantics are declared once**, in `src/ntb/ops`. A backend quirk lives in
   that backend's mapping, never as a special case inside an emitter.
4. **Geometry is semantic.** `SpatialRule` is a closed set; adding a kind needs
   its own ADR.
5. **The core wheel stays pure Python.** No native extensions.

## Setting up

```bash
python -m pip install -e ".[all]" pytest ruff mypy
pytest -q
ruff check . && ruff format --check .
mypy
```

Python 3.11+ on any of Linux, macOS or Windows. CI runs all three across 3.11,
3.12 and 3.13.

## Adding a canonical op

This is the most useful contribution right now, and it is a declaration:

1. Add an `OpSpec` to `src/ntb/ops/builtin.py` with its ports, attributes, shape
   rule and **all three** backend mappings. An op that only reaches one backend
   makes documents non-portable, and the test suite rejects it.
2. Add shape-rule cases to `tests/test_ops.py`: at least one concrete shape, one
   symbolic shape, and one rejection with a message that says what is wrong.
3. The cross-backend numeric parity test is generated from your declaration — you
   do not write it.

Error messages are part of the feature. `input has 3 channels, but in_channels is
64` is the standard; `invalid shape` is not.

## Before opening a PR

* `pytest -q`, `ruff check .`, `ruff format .`, `mypy` all clean.
* If you touched `src/ntb/ir/`, run `ntb schema --write` and commit the result —
  CI fails on a stale schema.
* If you touched `examples/*.ntb`, re-save them through `ntb.ir.io` rather than
  editing by hand; the canonical-format test will catch it otherwise.
* One logical change per PR. A refactor and a feature in one diff is two PRs.

## Commits, issues and language

English, everywhere: code, comments, docstrings, commit messages, issues and
PRs. This is a public repo aiming at outside contributors.

Commit subjects in the imperative mood, under ~70 characters: `add ntb.layernorm
to the op registry`.

## Reporting a bug

Include the `.ntb` file (or a minimal one that reproduces it), your platform and
Python version, `ntb version`, and what you expected instead. A model file is
plain JSON, so pasting it into an issue is fine.
