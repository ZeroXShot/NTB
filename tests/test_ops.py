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
        with pytest.raises(ShapeRuleError, match="must be rank 4"):
            infer(
                "ntb.conv2d",
                {"in": TensorType(shape=(3, 8, 8))},
                in_channels=3,
                out_channels=3,
            )


class TestDenseAndAttention:
    def test_embedding_appends_the_embedding_dimension(self) -> None:
        out = infer(
            "ntb.embedding",
            {"in": TensorType(dtype=DType.INT64, shape=("batch", "seq"))},
            num_embeddings=50000,
            embedding_dim=768,
        )["out"]
        assert out.shape == ("batch", "seq", 768)
        assert out.dtype is DType.FLOAT32

    def test_embedding_rejects_float_indices(self) -> None:
        with pytest.raises(ShapeRuleError, match="integer dtype"):
            infer(
                "ntb.embedding",
                {"in": TensorType(shape=("batch", "seq"))},
                num_embeddings=10,
                embedding_dim=4,
            )

    def test_matmul_broadcasts_batch_dimensions(self) -> None:
        out = infer(
            "ntb.matmul",
            {
                "a": TensorType(shape=("batch", 1, 4, 8)),
                "b": TensorType(shape=(3, 8, 16)),
            },
        )["out"]
        assert out.shape == ("batch", 3, 4, 16)

    def test_matmul_catches_an_inner_mismatch(self) -> None:
        with pytest.raises(ShapeRuleError, match="inner dimensions disagree"):
            infer(
                "ntb.matmul",
                {"a": TensorType(shape=(4, 8)), "b": TensorType(shape=(16, 2))},
            )

    def test_attention_returns_output_and_weights(self) -> None:
        out = infer(
            "ntb.attention",
            {"query": TensorType(shape=("batch", "seq", 512))},
            embed_dim=512,
            num_heads=8,
        )
        assert out["out"].shape == ("batch", "seq", 512)
        assert out["weights"].shape == ("batch", 8, "seq", "seq")

    def test_attention_requires_heads_to_divide_the_embedding(self) -> None:
        with pytest.raises(ShapeRuleError, match="divisible by num_heads"):
            infer(
                "ntb.attention",
                {"query": TensorType(shape=(1, 4, 100))},
                embed_dim=100,
                num_heads=8,
            )

    def test_attention_cross_attends_to_a_different_key_length(self) -> None:
        out = infer(
            "ntb.attention",
            {
                "query": TensorType(shape=(2, "q", 64)),
                "key": TensorType(shape=(2, "k", 64)),
            },
            embed_dim=64,
            num_heads=4,
        )
        assert out["weights"].shape == (2, 4, "q", "k")


class TestConvFamily:
    def test_conv3d_reduces_every_spatial_axis(self) -> None:
        out = infer(
            "ntb.conv3d",
            {"in": TensorType(shape=(1, 3, 16, 32, 32))},
            in_channels=3,
            out_channels=8,
            kernel_size=[3, 3, 3],
            stride=[1, 2, 2],
        )["out"]
        assert out.shape == (1, 8, 14, 15, 15)
        assert out.layout == "NCDHW"

    def test_conv1d_needs_rank_3(self) -> None:
        with pytest.raises(ShapeRuleError, match="must be rank 3"):
            infer(
                "ntb.conv1d",
                {"in": TensorType(shape=(1, 3, 8, 8))},
                in_channels=3,
                out_channels=8,
            )


class TestNormalisation:
    def test_layernorm_preserves_the_input(self) -> None:
        t = TensorType(shape=("batch", "seq", 768))
        assert infer("ntb.layernorm", {"in": t}, normalized_shape=[768])["out"] == t

    def test_layernorm_catches_a_trailing_axis_mismatch(self) -> None:
        with pytest.raises(ShapeRuleError, match="normalized_shape says 512"):
            infer(
                "ntb.layernorm",
                {"in": TensorType(shape=("batch", 768))},
                normalized_shape=[512],
            )

    def test_batchnorm_checks_the_channel_axis(self) -> None:
        with pytest.raises(ShapeRuleError, match="num_features is 32"):
            infer(
                "ntb.batchnorm",
                {"in": TensorType(shape=(1, 64, 8, 8))},
                num_features=32,
            )


class TestPooling:
    def test_stride_defaults_to_the_kernel(self) -> None:
        out = infer("ntb.maxpool2d", {"in": TensorType(shape=(1, 3, 32, 32))}, kernel_size=[2, 2])[
            "out"
        ]
        assert out.shape == (1, 3, 16, 16)

    def test_global_avgpool_drops_every_spatial_axis(self) -> None:
        out = infer("ntb.global_avgpool", {"in": TensorType(shape=("batch", 512, 7, 7))})["out"]
        assert out.shape == ("batch", 512)

    def test_global_avgpool_can_keep_them(self) -> None:
        out = infer("ntb.global_avgpool", {"in": TensorType(shape=(1, 512, 7, 7))}, keepdims=True)[
            "out"
        ]
        assert out.shape == (1, 512, 1, 1)


class TestShapeOps:
    def test_flatten_merges_the_trailing_axes(self) -> None:
        out = infer("ntb.flatten", {"in": TensorType(shape=("batch", 64, 7, 7))})["out"]
        assert out.shape == ("batch", 3136)

    def test_reshape_infers_a_single_wildcard(self) -> None:
        out = infer("ntb.reshape", {"in": TensorType(shape=(2, 3, 4))}, shape=[-1, 12])["out"]
        assert out.shape == (2, 12)

    def test_reshape_rejects_an_impossible_target(self) -> None:
        with pytest.raises(ShapeRuleError, match="cannot reshape"):
            infer("ntb.reshape", {"in": TensorType(shape=(2, 3))}, shape=[5, 5])

    def test_permute_reorders_axes(self) -> None:
        out = infer("ntb.permute", {"in": TensorType(shape=("batch", "seq", 64))}, order=[0, 2, 1])[
            "out"
        ]
        assert out.shape == ("batch", 64, "seq")

    def test_permute_rejects_a_repeated_axis(self) -> None:
        with pytest.raises(ShapeRuleError, match="repeats an axis"):
            infer("ntb.permute", {"in": TensorType(shape=(1, 2, 3))}, order=[0, 0, 1])

    def test_concat_sums_the_axis_it_joins(self) -> None:
        out = infer(
            "ntb.concat",
            {
                "a": TensorType(shape=("batch", 32, 8, 8)),
                "b": TensorType(shape=("batch", 16, 8, 8)),
            },
        )["out"]
        assert out.shape == ("batch", 48, 8, 8)

    def test_concat_requires_the_other_axes_to_match(self) -> None:
        with pytest.raises(ShapeRuleError, match="outside the concat axis"):
            infer(
                "ntb.concat",
                {
                    "a": TensorType(shape=(1, 32, 8, 8)),
                    "b": TensorType(shape=(1, 16, 4, 8)),
                },
            )


class TestElementwise:
    def test_add_broadcasts(self) -> None:
        out = infer(
            "ntb.add",
            {"a": TensorType(shape=("batch", 1, 64)), "b": TensorType(shape=(8, 64))},
        )["out"]
        assert out.shape == ("batch", 8, 64)

    def test_add_rejects_a_provable_mismatch(self) -> None:
        with pytest.raises(ShapeRuleError, match="cannot broadcast"):
            infer(
                "ntb.add",
                {"a": TensorType(shape=(4, 3)), "b": TensorType(shape=(5, 3))},
            )

    def test_add_rejects_mixed_dtypes(self) -> None:
        with pytest.raises(ShapeRuleError, match="share a dtype"):
            infer(
                "ntb.add",
                {
                    "a": TensorType(dtype=DType.FLOAT32, shape=(4,)),
                    "b": TensorType(dtype=DType.FLOAT16, shape=(4,)),
                },
            )
