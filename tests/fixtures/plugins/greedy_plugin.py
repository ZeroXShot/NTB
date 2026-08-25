"""A plugin claiming a name in the reserved namespace."""

from ntb.sdk import BackendMapping, CallKind, OpSpec, PortSpec, ShapeContext, TensorType, register


def _same(ctx: ShapeContext) -> dict[str, TensorType]:
    return {"out": ctx.input("in")}


register(
    OpSpec(
        name="ntb.greedy",
        category="activation",
        doc="An op pretending to be built in.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_rule=_same,
        torch=BackendMapping(target="torch.nn.functional.relu", kind=CallKind.FUNCTION),
    )
)
