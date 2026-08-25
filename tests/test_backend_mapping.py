"""The declarative knobs a backend mapping offers, and that they all work.

Every backend quirk NTB meets has to become a knob here rather than a branch in
an emitter (ADR 3). These tests are about the knobs themselves, so they declare
throwaway ops rather than leaning on whichever built-in happens to use one.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ntb.emit import EmitError, OnnxEmitError, emit_keras_document, emit_torch_document
from ntb.emit import export_onnx_document as export_onnx
from ntb.ir import Document, Module, Node, Port, PortDirection, TensorType
from ntb.ops import REGISTRY
from ntb.ops.spec import (
    AttrSpec,
    AttrType,
    BackendMapping,
    CallKind,
    OnnxMapping,
    OpSpec,
    PortSpec,
    ShapeContext,
)


def elementwise(ctx: ShapeContext) -> dict[str, TensorType]:
    return {"out": ctx.input("in")}


@pytest.fixture
def registered() -> Iterator[None]:
    """Ops declared by a test, taken back out again afterwards."""
    before = set(REGISTRY.names())
    yield
    for name in set(REGISTRY.names()) - before:
        REGISTRY.discard(name)


def document(op: str, **attrs: object) -> Document:
    return Document(
        name="probe",
        root="m",
        modules=(
            Module(
                id="m",
                nodes=(Node(id="a", op=op, attrs=attrs),),
                inputs=(
                    Port(
                        name="x",
                        direction=PortDirection.IN,
                        type=TensorType(shape=("batch", 4)),
                    ),
                ),
                outputs=(Port(name="y", direction=PortDirection.OUT),),
            ),
        ),
    )


def guarded(backend: str) -> OpSpec:
    """An op whose mapping for one backend refuses when `strict` is set."""
    guard = BackendMapping(
        target="torch.nn.functional.relu",
        kind=CallKind.FUNCTION,
        guard="strict == 0",
        guard_message="this mapping means something else when strict is set",
        imports=("torch", "torch.nn.functional"),
    )
    plain = BackendMapping(
        target="keras.activations.relu", kind=CallKind.FUNCTION, imports=("keras",)
    )
    onnx = OnnxMapping(op_type="Relu")
    if backend == "keras":
        plain = BackendMapping(
            target="keras.activations.relu",
            kind=CallKind.FUNCTION,
            guard=guard.guard,
            guard_message=guard.guard_message,
            imports=("keras",),
        )
        guard = BackendMapping(
            target="torch.nn.functional.relu",
            kind=CallKind.FUNCTION,
            imports=("torch", "torch.nn.functional"),
        )
    elif backend == "onnx":
        onnx = OnnxMapping(op_type="Relu", guard=guard.guard, guard_message=guard.guard_message)
        guard = BackendMapping(
            target="torch.nn.functional.relu",
            kind=CallKind.FUNCTION,
            imports=("torch", "torch.nn.functional"),
        )
    return OpSpec(
        name=f"test.guarded_{backend}",
        category="activation",
        doc="A test op with a guard on one backend.",
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        attrs=(AttrSpec("strict", AttrType.INT, default=0),),
        shape_rule=elementwise,
        torch=guard,
        keras=plain,
        onnx=onnx,
    )


class TestGuards:
    """A precondition has to mean the same thing whichever backend emits.

    Every guard NTB ships today happens to be on a Keras mapping, and the Keras
    emitter was the only one that looked at the field. So this was latent rather
    than live -- a guard declared on torch or ONNX was silently ignored.
    """

    def test_torch_refuses_a_guard_it_cannot_satisfy(self, registered: None) -> None:
        REGISTRY.register(guarded("torch"))
        with pytest.raises(EmitError, match="means something else"):
            emit_torch_document(document("test.guarded_torch", strict=1))

    def test_keras_refuses_a_guard_it_cannot_satisfy(self, registered: None) -> None:
        REGISTRY.register(guarded("keras"))
        with pytest.raises(EmitError, match="means something else"):
            emit_keras_document(document("test.guarded_keras", strict=1))

    def test_onnx_refuses_a_guard_it_cannot_satisfy(self, registered: None) -> None:
        pytest.importorskip("onnx")
        REGISTRY.register(guarded("onnx"))
        with pytest.raises(OnnxEmitError, match="means something else"):
            export_onnx(document("test.guarded_onnx", strict=1))

    def test_a_guard_that_holds_emits_normally(self, registered: None) -> None:
        REGISTRY.register(guarded("torch"))
        source = emit_torch_document(document("test.guarded_torch", strict=0)).source
        assert "relu" in source

    def test_a_backend_without_a_guard_is_not_affected(self, registered: None) -> None:
        REGISTRY.register(guarded("torch"))
        # The precondition is torch's; Keras has no reason to refuse.
        assert "relu" in emit_keras_document(document("test.guarded_torch", strict=1)).source


class TestValueMap:
    """A backend that spells the same choice differently."""

    def test_gelu_reaches_keras_as_a_bool_and_torch_as_a_string(self) -> None:
        keras = emit_keras_document(document("ntb.gelu", approximate="tanh")).source
        torch = emit_torch_document(document("ntb.gelu", approximate="tanh")).source
        assert "approximate=True" in keras
        assert 'approximate="tanh"' in torch

    def test_the_other_choice_maps_too(self) -> None:
        assert (
            "approximate=False"
            in emit_keras_document(document("ntb.gelu", approximate="none")).source
        )

    def test_a_value_with_no_entry_passes_through(self, registered: None) -> None:
        mapping = BackendMapping(target="x", value_map={"approximate": {"tanh": True}})
        assert mapping.translate("approximate", "tanh") is True
        assert mapping.translate("approximate", "none") == "none"
        assert mapping.translate("something_else", "none") == "none"


@pytest.mark.keras
@pytest.mark.torch
class TestValueMapNumerically:
    def test_the_tanh_approximation_is_the_same_model_in_both(self) -> None:
        """The bug this fixes: Keras silently used the exact form instead."""
        torch = pytest.importorskip("torch")
        keras = pytest.importorskip("keras")

        x = torch.randn(4, 16)
        exact = torch.nn.functional.gelu(x, approximate="none")
        tanh = torch.nn.functional.gelu(x, approximate="tanh")
        # The two really are different, or this test would prove nothing.
        assert not torch.allclose(exact, tanh, atol=1e-6)

        from_keras = keras.ops.convert_to_numpy(keras.activations.gelu(x, approximate=True))
        assert torch.allclose(tanh, torch.as_tensor(from_keras), atol=1e-5)
