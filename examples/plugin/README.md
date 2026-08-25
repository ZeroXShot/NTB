# ntb-example-op

A complete NTB plugin: one op, `example.softsign`, contributed from a
distribution of its own.

```bash
pip install -e examples/plugin
ntb plugins                  # example -> ntb_example_op: example.softsign
ntb ops | grep softsign
```

After that the op is an op. It appears in the studio palette, `ntb ops`, the MCP
tools, and it emits to torch, Keras and ONNX — because all of those read one
registry, and the plugin registered into it.

Three files is the whole thing:

| | |
|---|---|
| `pyproject.toml` | declares `[project.entry-points."ntb.ops"]` |
| `src/ntb_example_op/__init__.py` | the op declaration |
| this file | |

See [docs/plugins.md](../../docs/plugins.md) for what goes in a declaration and
what NTB does with it.
