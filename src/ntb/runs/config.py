"""What a training run is: the model, the data, and how to optimise it.

Deliberately small and closed. NTB is not a training framework, and a run
configuration that can express anything would be a worse one of those. What it
has to express is enough to answer "does this architecture train, and how fast",
and to hand off to a real training script when the answer is yes.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Optimiser(StrEnum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"


class Loss(StrEnum):
    #: Regression against a float target.
    MSE = "mse"
    #: Classification against integer class indices.
    CROSS_ENTROPY = "cross_entropy"
    #: Binary classification against 0/1 targets.
    BCE = "bce"


class DataSource(StrEnum):
    #: Random tensors with a fixed seed. Answers "does it train", not "is it good".
    SYNTHETIC = "synthetic"
    #: A Python file exposing `dataloaders(batch_size) -> (train, validation)`.
    SCRIPT = "script"


class RunConfig(BaseModel):
    """Everything a run needs, and nothing a training framework would add."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: Path = Field(description="The .ntb to train.")
    epochs: int = Field(default=1, ge=1, le=100_000)
    batch_size: int = Field(default=32, ge=1, le=1_000_000)
    steps_per_epoch: int = Field(
        default=50,
        ge=1,
        description="Synthetic data only; a script's loader decides its own length.",
    )
    learning_rate: float = Field(default=1e-3, gt=0.0)
    optimiser: Optimiser = Optimiser.ADAM
    loss: Loss = Loss.MSE
    seed: int = 0
    device: str = Field(default="cpu", description="'cpu', 'cuda', or 'cuda:1'.")

    data: DataSource = DataSource.SYNTHETIC
    data_script: Path | None = None
    #: Classes for the synthetic target when the loss is a classification one.
    classes: int = Field(default=10, ge=2)

    #: Where checkpoints and the run's own files go. The manager sets it; a
    #: bare worker falls back to ./runs.
    workdir: Path | None = None
    checkpoint_every: int = Field(
        default=0,
        ge=0,
        description="Steps between checkpoints. 0 writes one at the end only.",
    )
    resume_from: Path | None = None

    def total_steps(self) -> int:
        """Only exact for synthetic data; a script's length is unknown up front."""
        return self.epochs * self.steps_per_epoch
