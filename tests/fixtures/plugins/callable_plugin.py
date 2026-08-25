"""A plugin whose entry point names a function rather than a module."""

from ntb.sdk import BackendMapping, CallKind, OpSpec, PortSpec, ShapeContext, TensorType, register


def _same(ctx: ShapeContext) -> dict[str, TensorType]:
    return {"out": ctx.input("in")}


def setup() -> None:
    register(
        OpSpec(
            name="test.called",
            category="activation",
            doc="Registered by a callable entry point.",
            inputs=(PortSpec("in"),),
            outputs=(PortSpec("out"),),
            shape_rule=_same,
            torch=BackendMapping(target="torch.nn.functional.relu", kind=CallKind.FUNCTION),
        )
    )
