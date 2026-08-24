# The studio

```bash
pip install "ntb[server]"
ntb studio                      # empty model
ntb studio examples/mlp.ntb     # open a file
```

The command starts a local server and opens a browser. Nothing leaves your
machine, and nothing but your machine can talk to it
([ADR 10](adr/0010-the-studio-server-is-local-only.md)).

| Option | Meaning |
|---|---|
| `--port 8756` | Port to listen on |
| `--host 127.0.0.1` | Interface to bind. Widening this is your decision |
| `--no-open` | Do not launch a browser |

## The window

* **Ops** (left) — the registry. Every op NTB knows, with the backends it can
  reach. Clicking one adds a block.
* **Canvas** (centre) — one three.js scene. In this release you are looking at
  it through an orthographic camera with the Z axis locked; phase 4 unlocks the
  perspective camera and edits in three dimensions.
* **Inspector** (right) — id, label, attributes and the *inferred* type on every
  port. The attribute editors are built from the op's declaration, so they are
  never out of date with the registry.
* **Problems / torch** (bottom) — validation and the generated PyTorch source.
  Both are recomputed by the server after every edit, and the source is exactly
  what `ntb emit` would write to a file.

## Editing

| Action | How |
|---|---|
| Add a block | Click an op in the palette |
| Select | Click a block; shift-click to add to the selection |
| Move | Drag. Position is part of the model, so this is an edit |
| Connect | Select a block, press `c`, click the target |
| Cancel a connection | `Escape` |
| Delete | `Delete` or `Backspace` |
| Undo / redo | `Ctrl`/`Cmd` `Z`, add `Shift` to redo |
| Save | `Ctrl`/`Cmd` `S` |
| Pan / zoom | Drag the background / scroll |

A connection joins the source's first output to the target's first free input.
Which ports those are comes from the registry.

## Declaring the model's inputs

A model needs to know what flows in. The studio edits blocks; the boundary is
still declared on the module, which for now is done in the file or through the
command bus:

```python
from ntb.commands import SetModulePorts, apply_command
from ntb.ir import Port, PortDirection, TensorType, io

document = io.load("model.ntb")
result = apply_command(
    document,
    SetModulePorts(
        module=document.root,
        inputs=(
            Port(
                name="x",
                direction=PortDirection.IN,
                type=TensorType(shape=("batch", 784)),
            ),
        ),
    ),
)
io.save(result.document, "model.ntb")
```

A boundary editor is part of phase 4.

## What it does not do yet

* No editing in 3D. The scene is ready for it; the camera and the tools are not.
* Connections are made by picking two blocks, not by dragging port to port.
* No training from the UI — that is phase 6. Emit the source and run it.
* One document per server. Opening a second file replaces the session.

## If something goes wrong

The window says `offline` when the socket is down; it reconnects on its own. If
the page says the studio is not built, you are running from a source checkout
without a frontend bundle — see [CONTRIBUTING](../CONTRIBUTING.md).
