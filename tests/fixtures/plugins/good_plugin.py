"""A plugin that registers one op on import, the way a real one does."""

from ntb.sdk import BackendMapping, CallKind, OpSpec, PortSpec, ShapeContext, TensorType, register


def _same(ctx: ShapeContext) -> dict[str, TensorType]:
    return {"out": ctx.input("in")}


register(
    OpSpec(
        name="test.good",
        category="activation",
        doc="A test op that does nothing in particular.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_rule=_same,
        torch=BackendMapping(target="torch.nn.functional.relu", kind=CallKind.FUNCTION),
    )
)
