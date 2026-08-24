## What this changes

<!-- One paragraph. Link the issue if there is one. -->

## Checklist

- [ ] `pytest -q`, `ruff check .`, `ruff format --check .` and `mypy` are clean
- [ ] Touched `src/ntb/ir/`? Ran `ntb schema --write` and committed the result
- [ ] Touched `examples/*.ntb`? Re-saved them through `ntb.ir.io`
- [ ] Structural change? Added or updated an ADR in `docs/adr/`
- [ ] New op? All three backend mappings declared, shape-rule tests added
