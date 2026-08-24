"""Rewrite the golden files after a deliberate change to an emitter.

Review the resulting diff: these files exist so that a change in generated code
is read by a human, not accepted silently.

    python tests/golden/regenerate.py
"""

from __future__ import annotations

from pathlib import Path

from ntb.emit import emit_keras_document, emit_torch_document
from ntb.ir import io

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
NAMES = ("mlp", "transformer_block", "cnn3d", "vertical_tower", "lattice_3d")


def main() -> None:
    root = Path(__file__).resolve().parent
    for backend, emit in (("torch", emit_torch_document), ("keras", emit_keras_document)):
        target = root / backend
        target.mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            emitted = emit(io.load(EXAMPLES / f"{name}.ntb"))
            (target / f"{name}.py").write_text(emitted.source, encoding="utf-8", newline=chr(10))
            print(f"wrote {backend}/{name}.py")


if __name__ == "__main__":
    main()
