"""Shared test fixtures and paths."""

from __future__ import annotations

import os
from pathlib import Path

# Keras 3 defaults to TensorFlow, which NTB does not depend on. The torch
# backend is already installed wherever the keras tests can run at all, and it
# has to be chosen before keras is first imported.
os.environ.setdefault("KERAS_BACKEND", "torch")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"
