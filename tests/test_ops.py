"""The op registry is the single source of truth, so it gets audited as one."""

from __future__ import annotations

import re

import pytest

from ntb.ir.types import DType, TensorType
from ntb.ops import REGISTRY, AttrSpec, AttrType, OpSpec, PortSpec, UnknownOpError
from ntb.ops.registry import OpRegistry
from ntb.ops.spec import ShapeContext, ShapeRuleError


def infer(op: str, inputs: dict[str, TensorType], **attrs: object) -> dict[str, TensorType]:
    spec = REGISTRY.require(op)
    ctx = ShapeContext(op=spec.name, attrs=spec.resolved_attrs(attrs), inputs=inputs)
    return spec.shape_rule(ctx)


class TestRegistry:
    def test_builtin_ops_are_registered(self) -> None:
        assert set(REGISTRY.names()) >= {"ntb.linear", "ntb.relu", "ntb.conv2d"}

    def test_unknown_op_suggests_a_close_name(self) -> None:
        with pytest.raises(UnknownOpError, match=re.escape("did you mean 'ntb.linear'")):
            REGISTRY.require("ntb.linaer")

    def test_registering_a_name_twice_is_refused(self) -> None:
        # A plugin quietly shadowing ntb.conv2d would change what every existing
        # .ntb file means, so shadowing is an error rather than a last-wins.
        registry = OpRegistry()
        spec = OpSpec(
            name="x",
            category="test",
            doc="d",
            inputs=(),
            outputs=(PortSpec("out"),),
            shape_rule=lambda ctx: {"out": TensorType()},
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_iteration_is_sorted_and_grouped(self) -> None:
        assert [s.name for s in REGISTRY] == sorted(REGISTRY.names())
        assert "activation" in REGISTRY.by_category()


class TestOpSpecInvariants:
    @pytest.mark.parametrize("spec", list(REGISTRY), ids=lambda s: s.name)
    def test_every_op_reaches_every_backend(self, spec: OpSpec) -> None:
        # An op that only one backend can emit silently makes documents
        # non-portable, which defeats the point of a framework-agnostic IR.
        assert set(spec.backends()) == {"torch", "keras", "onnx"}, spec.name

    @pytest.mark.parametrize("spec", list(REGISTRY), ids=lambda s: s.name)
    def test_every_op_is_documented(self, spec: OpSpec) -> None:
        assert spec.doc.strip(), spec.name
        assert spec.category.strip(), spec.name

    @pytest.mark.parametrize("spec", list(REGISTRY), ids=lambda s: s.name)
    def test_backend_attr_maps_only_mention_declared_attrs(self, spec: OpSpec) -> None:
        declared = set(spec.attr_specs)
        for label, mapping in (("torch", spec.torch), ("keras", spec.keras)):
            assert mapping is not None
            unknown = set(mapping.attr_map) - declared
            assert not unknown, f"{spec.name} {label} maps undeclared attrs: {unknown}"
        assert spec.onnx is not None
        assert not set(spec.onnx.attr_map) - declared, spec.name

    @pytest.mark.parametrize("spec", list(REGISTRY), ids=lambda s: s.name)
    def test_required_attrs_have_no_default(self, spec: OpSpec) -> None:
        for attr in spec.attrs:
            assert not (attr.required and attr.default is not None), f"{spec.name}.{attr.name}"

    def test_an_op_without_outputs_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="declares no outputs"):
            OpSpec(
                name="bad",
                category="test",
                doc="d",
                inputs=(),
                outputs=(),
                shape_rule=lambda ctx: {},
            )

    def test_duplicate_ports_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate input port"):
            OpSpec(
                name="bad",
                category="test",
                doc="d",
                inputs=(PortSpec("a"), PortSpec("a")),
                outputs=(PortSpec("out"),),
                shape_rule=lambda ctx: {},
            )


class TestAttrValidation:
    def test_bool_is_not_accepted_as_an_int(self) -> None:
        # Python says True == 1; an op that takes out_features=True is a bug.
        assert AttrSpec("n", AttrType.INT).validate(True) is not None

    def test_minimum_applies_elementwise_to_lists(self) -> None:
        spec = AttrSpec("k", AttrType.INTS, minimum=1)
        assert spec.validate([3, 3]) is None
        assert spec.validate([3, 0]) is not None

    def test_choices_are_enforced(self) -> None:
        spec = AttrSpec("mode", AttrType.STRING, choices=("same", "valid"))
        assert spec.validate("same") is None
        assert "must be one of" in (spec.validate("full") or "")

    def test_dtype_accepts_the_enum_and_its_value(self) -> None:
        spec = AttrSpec("t", AttrType.DTYPE)
        assert spec.validate(DType.FLOAT16) is None
        assert spec.validate("float16") is None
        assert spec.validate("float13") is not None


class TestShapeRules:
    def test_linear_rewrites_only_the_last_dimension(self) -> None:
        out = infer(
            "ntb.linear",
            {"in": TensorType(shape=("batch", "seq", 512))},
            in_features=512,
            out_features=10,
        )["out"]
        assert out.shape == ("batch", "seq", 10)

    def test_linear_catches_a_concrete_mismatch(self) -> None:
        with pytest.raises(ShapeRuleError, match="in_features is 256"):
            infer(
                "ntb.linear",
                {"in": TensorType(shape=("batch", 512))},
                in_features=256,
                out_features=10,
            )

    def test_linear_allows_an_undecidable_symbolic_match(self) -> None:
        # 'features' may or may not be 512 at runtime. Refusing the connection
        # would make dynamic shapes unusable, so it is allowed through.
        out = infer(
            "ntb.linear",
            {"in": TensorType(shape=("batch", "features"))},
            in_features=512,
            out_features=10,
        )["out"]
        assert out.shape == ("batch", 10)

    def test_linear_rejects_an_integer_input(self) -> None:
        with pytest.raises(ShapeRuleError, match="floating dtype"):
            infer(
                "ntb.linear",
                {"in": TensorType(dtype=DType.INT32, shape=(4, 8))},
                in_features=8,
                out_features=2,
            )

    def test_unconnected_input_names_the_port(self) -> None:
        with pytest.raises(ShapeRuleError, match="required input 'in' is not connected"):
            infer("ntb.relu", {})

    def test_relu_is_shape_and_dtype_preserving(self) -> None:
        t = TensorType(dtype=DType.FLOAT16, shape=("batch", 3, 224, 224))
        assert infer("ntb.relu", {"in": t})["out"] == t

    def test_conv2d_computes_concrete_spatial_dims(self) -> None:
        out = infer(
            "ntb.conv2d",
            {"in": TensorType(shape=(8, 3, 224, 224))},
            in_channels=3,
            out_channels=64,
            kernel_size=[7, 7],
            stride=[2, 2],
            padding=[3, 3],
        )["out"]
        assert out.shape == (8, 64, 112, 112)
        assert out.layout == "NCHW"

    def test_conv2d_keeps_symbolic_dims_symbolic(self) -> None:
        out = infer(
            "ntb.conv2d",
            {"in": TensorType(shape=("batch", 3, "h", "w"))},
            in_channels=3,
            out_channels=16,
        )["out"]
        assert out.shape[0] == "batch"
        assert out.shape[1] == 16
        assert isinstance(out.shape[2], str) and "h" in out.shape[2]

    def test_conv2d_accepts_a_scalar_kernel_size(self) -> None:
        out = infer(
            "ntb.conv2d",
            {"in": TensorType(shape=(1, 3, 8, 8))},
            in_channels=3,
            out_channels=3,
            kernel_size=3,
        )["out"]
        assert out.shape == (1, 3, 6, 6)

    def test_conv2d_rejects_a_window_larger_than_the_input(self) -> None:
        with pytest.raises(ShapeRuleError, match="does not fit"):
            infer(
                "ntb.conv2d",
                {"in": TensorType(shape=(1, 3, 2, 2))},
                in_channels=3,
                out_channels=3,
                kernel_size=[5, 5],
            )

    def test_conv2d_rejects_groups_that_do_not_divide_the_channels(self) -> None:
        with pytest.raises(ShapeRuleError, match="must divide"):
            infer(
                "ntb.conv2d",
                {"in": TensorType(shape=(1, 6, 8, 8))},
                in_channels=6,
                out_channels=8,
                groups=4,
            )

    def test_conv2d_rejects_the_wrong_rank(self) -> None:
        with pytest.raises(ShapeRuleError, match="rank-4"):
            infer(
                "ntb.conv2d",
                {"in": TensorType(shape=(3, 8, 8))},
                in_channels=3,
                out_channels=3,
            )
