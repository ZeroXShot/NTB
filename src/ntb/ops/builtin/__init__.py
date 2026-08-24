"""The built-in op set.

An op needing code outside this package means the registry is missing a knob.
Importing a submodule registers its ops.
"""

from ntb.ops.builtin import (
    activation,
    attention,
    conv,
    dense,
    elementwise,
    norm,
    pooling,
    shape_ops,
)

BUILTIN_OPS = (
    *dense.OPS,
    *activation.OPS,
    *conv.OPS,
    *norm.OPS,
    *pooling.OPS,
    *shape_ops.OPS,
    *elementwise.OPS,
    *attention.OPS,
)

__all__ = ["BUILTIN_OPS"]
