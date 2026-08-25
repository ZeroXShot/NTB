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
* **Canvas** (centre) — one three.js scene, seen through an orthographic camera
  in 2D and a perspective one in 3D. The toolbar switches between them; it is
  the same scene, the same selection and the same picking either way.
* **Inspector** (right) — id, label, attributes and the *inferred* type on every
  port. The attribute editors are built from the op's declaration, so they are
  never out of date with the registry.
* **Problems / torch / Space / Train** (bottom) — validation, the generated
  PyTorch source, the generators and spatial rules of the current module, and
  training runs with their loss curves ([docs/training.md](training.md)). The first
  two are recomputed by the server after every edit, and the source is exactly
  what `ntb emit` would write to a file. The third is where 3D architectures are
  actually authored: see [docs/spatial.md](spatial.md).

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
| Orbit (3D) | Drag the background; shift-drag pans |
| Lift along Z | Alt-drag a block |

A connection joins the source's first output to the target's first free input.
Which ports those are comes from the registry.

## Declaring the model's inputs

A model needs to know what flows in, and where its ports land. The **Space**
panel binds a declared port to a specific `node.port`; `auto` leaves it to the
positional rule (see [docs/spatial.md](spatial.md)).

Declaring the ports themselves, with their types, is still done in the file or
through the command bus:

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

## Generators and rules

Blocks a generator produced are drawn in violet and cannot be moved one at a
time: the generator is the object, so edit it in the **Space** panel and every
repetition follows at once. Edges are coloured by where they came from — grey
for the ones you drew, cyan for a spatial rule, blue for a generator's chain —
and the corner of the canvas counts how many edges nobody drew.

Alt-drag lifts a block along Z, which is the only way to author a Z coordinate
by hand; the inspector has exact number fields for all three axes.

## What it does not do yet

* No editor for declaring a port and its type; the Space panel only binds ports
  that already exist. Use the snippet above.
* Connections are made by picking two blocks, not by dragging port to port.
* Training uses synthetic data unless you launch it from the command line with
  a data script; the panel has no file picker yet.
* One document per server. Opening a second file replaces the session.

## If something goes wrong

The window says `offline` when the socket is down; it reconnects on its own. If
the page says the studio is not built, you are running from a source checkout
without a frontend bundle — see [CONTRIBUTING](https://github.com/ZeroXShot/NTB/blob/main/CONTRIBUTING.md).
