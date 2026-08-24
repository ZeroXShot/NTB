"""The Keras 3 emitter.

Keras 3 runs on TensorFlow, JAX and torch, so this one backend is how a `.ntb`
reaches all three (ADR 7). The tests that matter are the ones that build the
model and run it: generated code that merely parses proves nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ntb.emit.keras import emit, emit_document
from ntb.emit.torch import EmitError
from ntb.ir import (
    CoreGraph,
    CoreNode,
    Document,
    Endpoint,
    GraphInput,
    GraphOutput,
    Module,
    Node,
    Port,
    PortDirection,
    TensorType,
    io,
)
from ntb.ir.core import Origin
from tests.conftest import EXAMPLES

GOLDEN = Path(__file__).resolve().parent / "golden" / "keras"
EXAMPLE_NAMES = ["mlp", "transformer_block", "cnn3d", "vertical_tower", "lattice_3d"]
SHAPES = {
    "mlp": (4, 784),
    "transformer_block": (2, 16, 512),
    "cnn3d": (2, 1, 32, 64, 64),
    "vertical_tower": (4, 256),
    "lattice_3d": (8, 64),
}


class TestGoldenFiles:
    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_generated_source_matches_the_golden_file(self, name: str) -> None:
        emitted = emit_document(io.load(EXAMPLES / f"{name}.ntb"))
        expected = (GOLDEN / f"{name}.py").read_text(encoding="utf-8")
        assert emitted.source == expected, "regenerate with tests/golden/regenerate.py and review"

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_generated_source_parses(self, name: str) -> None:
        ast.parse((GOLDEN / f"{name}.py").read_text(encoding="utf-8"))


class TestShape:
    def test_a_single_input_is_a_tensor_not_a_list_of_one(self) -> None:
        source = emit_document(io.load(EXAMPLES / "mlp.ntb")).source
        assert "keras.Model(inputs=x, outputs=fc2" in source

    def test_symbolic_dimensions_become_none(self) -> None:
        source = emit_document(io.load(EXAMPLES / "mlp.ntb")).source
        assert "batch_shape=(None, 784)" in source

    def test_layers_are_channels_first_like_torch(self) -> None:
        # NTB tensors are NCHW. A Keras default of channels_last would make the
        # same document mean a different model in the two backends.
        source = emit_document(io.load(EXAMPLES / "cnn3d.ntb")).source
        assert source.count('data_format="channels_first"') >= 5

    def test_integer_padding_becomes_its_own_layer(self) -> None:
        source = emit_document(io.load(EXAMPLES / "cnn3d.ntb")).source
        assert "keras.layers.ZeroPadding3D(" in source
        assert "padding=((1, 1), (1, 1), (1, 1))" in source

    def test_an_op_keras_spells_as_a_reshape_gets_its_inferred_shape(self) -> None:
        source = emit_document(io.load(EXAMPLES / "cnn3d.ntb")).source
        assert "keras.ops.reshape(gap, newshape=(-1, 32))" in source

    def test_a_derived_argument_is_computed_from_the_attributes(self) -> None:
        # Keras wants key_dim per head; NTB stores the whole embedding width.
        source = emit_document(io.load(EXAMPLES / "transformer_block.ntb")).source
        assert "key_dim=64" in source and "num_heads=8" in source

    def test_two_instances_of_one_module_get_different_names(self) -> None:
        source = emit_document(io.load(EXAMPLES / "lattice_3d.ntb")).source
        assert "col0_0_mix" in source and "col1_0_mix" in source


class TestFailures:
    def test_a_graph_that_does_not_type_check_is_refused(self) -> None:
        module = Module(
            id="m",
            inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=(1, 512))),),
            nodes=(Node(id="fc", op="ntb.linear", attrs={"in_features": 8, "out_features": 4}),),
        )
        with pytest.raises(EmitError, match="does not type-check"):
            emit_document(Document(root="m", modules=(module,)))

    def test_a_guard_refuses_a_mapping_that_would_mean_something_else(self) -> None:
        # Keras normalises one axis; NTB was asked for two, and a silently
        # different model is the one outcome worth refusing.
        graph = CoreGraph(
            name="ln",
            nodes=(
                CoreNode(
                    id="n",
                    op="ntb.layernorm",
                    attrs={"normalized_shape": [4, 8]},
                    origin=Origin(module="m", node="n"),
                ),
            ),
            inputs=(
                GraphInput(
                    name="x",
                    endpoint=Endpoint(node="n", port="in"),
                    type=TensorType(shape=(2, 4, 8)),
                ),
            ),
            outputs=(GraphOutput(name="y", endpoint=Endpoint(node="n", port="out")),),
        )
        with pytest.raises(EmitError, match="normalises one axis"):
            emit(graph)


@pytest.mark.keras
class TestGeneratedCodeRuns:
    """The only test that proves the emitter works: build the model and call it."""

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_the_model_builds_and_matches_inferred_shapes(self, name: str) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("keras")

        from ntb.shapes import infer_shapes
        from ntb.spatial import resolve

        document = io.load(EXAMPLES / f"{name}.ntb")
        emitted = emit_document(document)
        namespace: dict[str, object] = {}
        exec(compile(emitted.source, f"{name}.py", "exec"), namespace)
        model = namespace[emitted.class_name]()  # type: ignore[operator]

        shape = SHAPES[name]
        output = model(np.random.randn(*shape).astype("float32"))

        graph = resolve(document)
        report = infer_shapes(graph)
        binding = {
            d: shape[i] for i, d in enumerate(graph.inputs[0].type.shape) if isinstance(d, str)
        }
        endpoint = graph.outputs[0].endpoint
        predicted = report.type_of(endpoint.node, endpoint.port)
        assert predicted is not None
        expected = tuple(binding.get(d, d) if isinstance(d, str) else d for d in predicted.shape)
        assert tuple(output.shape) == expected

    def test_the_whole_mlp_agrees_with_torch_layer_for_layer(self) -> None:
        # Per-op parity lives in test_parity.py. This is the same claim for a
        # whole model: one document, two frameworks, the same numbers.
        np = pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")
        keras = pytest.importorskip("keras")

        from ntb.emit.torch import emit_document as emit_torch_document

        document = io.load(EXAMPLES / "mlp.ntb")
        torch_emitted = emit_torch_document(document)
        namespace: dict[str, object] = {}
        exec(compile(torch_emitted.source, "mlp_torch.py", "exec"), namespace)
        torch_model = namespace[torch_emitted.class_name]().eval()  # type: ignore[operator]

        keras_emitted = emit_document(document)
        keras_namespace: dict[str, object] = {}
        exec(compile(keras_emitted.source, "mlp_keras.py", "exec"), keras_namespace)
        keras_model = keras_namespace[keras_emitted.class_name]()  # type: ignore[operator]

        state = dict(torch_model.named_parameters())
        for layer, prefix in zip(
            [ly for ly in keras_model.layers if ly.weights], ("fc1", "fc2"), strict=True
        ):
            layer.set_weights(
                [
                    state[f"{prefix}.weight"].detach().numpy().T,
                    state[f"{prefix}.bias"].detach().numpy(),
                ]
            )

        sample = np.random.randn(4, 784).astype("float32")
        with torch.no_grad():
            expected = torch_model(torch.from_numpy(sample)).numpy()
        produced = keras.ops.convert_to_numpy(keras_model(sample))
        np.testing.assert_allclose(expected, produced, atol=1e-5, rtol=1e-4)
