# Training

```bash
ntb run examples/mlp.ntb --epochs 3 --loss cross_entropy
ntb runs                     # what has run
ntb runs --show <id>         # one run's metrics
```

Or from the studio: the **Train** tab starts a run and draws its loss as it
arrives.

A run happens in a process of its own
([ADR 11](adr/0011-training-runs-in-their-own-process.md)). An out-of-memory
kills the run, not the editor, and *stop* actually stops it.

## What it trains on

| `--data-script` | Data |
|---|---|
| not given | **Synthetic**: random tensors shaped like the model's own inputs |
| a Python file | **Yours** |

Synthetic data answers the question you have while you are still drawing the
model — *does this architecture train, and how fast is a step* — and nothing
else. The loss it produces is meaningless; it is noise fitted to noise.

For real data, write a file with one function:

```python
# data.py
import torch
from torch.utils.data import DataLoader, TensorDataset


def dataloaders(batch_size):
    """Return (train, validation). Each yields (inputs, target) per batch."""
    x = torch.randn(1024, 784)
    y = torch.randint(0, 10, (1024,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return loader, loader
```

```bash
ntb run model.ntb --data-script data.py --loss cross_entropy
```

It is imported in the worker's process, not the server's, so it is yours to
break. If a model takes several inputs, yield them as a tuple.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 1 | Passes over the data |
| `--steps` | 50 | Steps per epoch, synthetic data only |
| `--batch-size` | 32 | Examples per step |
| `--lr` | 0.001 | Learning rate |
| `--optimiser` | adam | `sgd`, `adam`, `adamw` |
| `--loss` | mse | `mse`, `cross_entropy`, `bce` |
| `--device` | cpu | `cpu`, `cuda`, `cuda:1` |
| `--checkpoint-every` | 0 | Steps between checkpoints; 0 writes one at the end |
| `--seed` | 0 | Seeds torch and the synthetic data |
| `--root` | ./runs | Where runs are kept |

The list is short on purpose. NTB is not a training framework, and a run
configuration that could express everything would be a worse one of those. When
the answer to "does this train" is yes, `ntb emit` gives you the model as source
you can drop into whatever you actually train with.

## What a run leaves behind

```
runs/
├─ runs.db                  every run, and every metric, in SQLite
└─ <run id>/
   ├─ config.json           exactly what the worker was asked to do
   └─ mlp-step300.pt        model and optimiser state
```

Metrics are in a database rather than in memory, so a run outlives the session
that started it, and a later session can read what happened. A run whose session
ended is marked *stopped* the next time a manager opens the store — its process
went with the session, and leaving it marked *running* forever would be a lie.

## Resuming

```bash
ntb runs                     # find the id
```

In the studio, a finished run with a checkpoint has a **resume** button; it
starts a *new* run from that checkpoint and continues the step count. The
original record stays, because the point of a record is that it does not change.

## What this is not

* No validation loop, no metrics but the loss, no schedulers, no early stopping.
* No distributed training.
* No weights in `.ntb`. A document is an architecture; a checkpoint is a
  checkpoint. Importing one back is not implemented.
* Keras runs are not launched from here yet. `ntb emit --backend keras` gives
  you the model; training it is `model.fit`.
