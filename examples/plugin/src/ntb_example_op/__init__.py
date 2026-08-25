"""An op contributed from outside NTB, in full.

Importing this module registers `example.softsign` with NTB, because the
distribution's `[project.entry-points."ntb.ops"]` points here. From then on the
op is an op: it appears in the studio palette, `ntb ops`, the MCP tools, and the
generated torch, Keras and ONNX.

An op is a declaration, not code. Nothing below implements softsign; it says
where softsign lives in each backend and what it does to a shape.
"""

from ntb.sdk import (
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    ParityCase,
    PortSpec,
    ShapeContext,
    TensorType,
    register,
)


def _same_shape(ctx: ShapeContext) -> dict[str, TensorType]:
    """Elementwise: out is in. Fail loudly on anything that is not a float."""
    tensor = ctx.input("in")
    if not tensor.dtype.is_floating:
        ctx.fail(f"in must be a floating dtype, got {tensor.dtype.value}")
    return {"out": tensor}


SOFTSIGN = register(
    OpSpec(
        name="example.softsign",
        category="activation",
        doc="Elementwise ``x / (1 + |x|)``, a bounded alternative to tanh.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_rule=_same_shape,
        torch=BackendMapping(
            target="torch.nn.functional.softsign",
            kind=CallKind.FUNCTION,
            imports=("torch", "torch.nn.functional"),
        ),
        keras=BackendMapping(
            target="keras.activations.softsign",
            kind=CallKind.FUNCTION,
            imports=("keras",),
        ),
        onnx=OnnxMapping(op_type="Softsign", since_opset=1),
        # Without this the op would ship unverified: the harness reads it to
        # feed the same random tensor to all three backends and compare.
        parity=ParityCase(inputs={"in": (2, 6)}),
    )
)
