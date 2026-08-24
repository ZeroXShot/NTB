"""Tensor types: dtypes and possibly symbolic shapes.

A dimension is a non-negative ``int`` or a ``str`` expression ("batch"). Turning
those into algebra is :mod:`ntb.shapes`, not the IR.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DType(StrEnum):
    """Element types. Closed set: each must exist in torch, Keras 3 and ONNX."""

    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"

    @property
    def is_floating(self) -> bool:
        return self in _FLOATING

    @property
    def is_integer(self) -> bool:
        return self in _INTEGER


_FLOATING = frozenset({DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32, DType.FLOAT64})
_INTEGER = frozenset({DType.INT8, DType.INT16, DType.INT32, DType.INT64, DType.UINT8})

#: A single dimension: a concrete size, or a symbolic expression as text.
Dim: TypeAlias = int | str

#: An ordered list of dimensions. The empty shape is a scalar.
Shape: TypeAlias = tuple[Dim, ...]


class TensorType(BaseModel):
    """The static type of a value flowing along an edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dtype: DType = DType.FLOAT32
    shape: Shape = ()
    layout: str | None = Field(
        default=None,
        description="Informational hint ('NCHW'); the op registry defines the truth.",
    )

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, shape: Shape) -> Shape:
        for i, dim in enumerate(shape):
            if isinstance(dim, int):
                if dim < 0:
                    raise ValueError(f"dimension {i} is negative: {dim}")
            elif not dim.strip():
                raise ValueError(f"dimension {i} is an empty symbolic expression")
        return shape

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_static(self) -> bool:
        """True when every dimension is a concrete integer."""
        return all(isinstance(d, int) for d in self.shape)

    def symbols(self) -> frozenset[str]:
        """Raw symbolic dimension expressions appearing in this shape."""
        return frozenset(d for d in self.shape if isinstance(d, str))

    def __str__(self) -> str:
        dims = ", ".join(str(d) for d in self.shape)
        return f"{self.dtype.value}[{dims}]"


#: Safe as a JSON key and as a Python attribute name after sanitisation.
Identifier: TypeAlias = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
]
