"""The torch emitter.

Golden files pin the generated source. They are reviewed by hand, because the
whole point is that the output is code a user would keep (ADR 1): a diff here
should be read, not blindly accepted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ntb.emit import EmitError, emit_torch, emit_torch_document
from ntb.emit.torch import _class_name
from ntb.ir import (
    CoreGraph,
    CoreNode,
    Document,
    Edge,
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

GOLDEN = Path(__file__).resolve().parent / "golden" / "torch"
EXAMPLE_NAMES = ["mlp", "transformer_block", "cnn3d"]


def core(op: str, **attrs: object) -> CoreNode:
    return CoreNode(id="n", op=op, attrs=attrs, origin=Origin(module="m", node="n"))


class TestGoldenFiles:
    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_generated_source_matches_the_golden_file(self, name: str) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / f"{name}.ntb"))
        expected = (GOLDEN / f"{name}.py").read_text(encoding="utf-8")
        assert emitted.source == expected, "regenerate with tests/golden/regenerate.py and review"

    @pytest.mark.parametrize("name", EXAMPLE_NAMES)
    def test_generated_source_parses(self, name: str) -> None:
        ast.parse((GOLDEN / f"{name}.py").read_text(encoding="utf-8"))

    def test_emission_is_deterministic(self) -> None:
        # Golden files are worthless if two runs disagree.
        document = io.load(EXAMPLES / "transformer_block.ntb")
        assert emit_torch_document(document).source == emit_torch_document(document).source


class TestShape:
    def test_class_name_comes_from_the_document(self) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / "cnn3d.ntb"))
        assert emitted.class_name == "Cnn3d"

    def test_class_name_can_be_overridden(self) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / "mlp.ntb"), class_name="MyNet")
        assert "class MyNet(torch.nn.Module):" in emitted.source

    def test_forward_arguments_use_the_authored_port_names(self) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / "cnn3d.ntb"))
        assert "def forward(self, volume):" in emitted.source

    def test_stateful_ops_go_in_init_and_pure_ones_inline(self) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / "mlp.ntb"))
        assert "self.fc1 = torch.nn.Linear(" in emitted.source
        assert "torch.nn.functional.relu(fc1)" in emitted.source

    def test_an_unused_output_is_discarded(self) -> None:
        # Attention returns weights nobody consumes; naming them would leave a
        # dead variable in the user's repo.
        emitted = emit_torch_document(io.load(EXAMPLES / "transformer_block.ntb"))
        assert "attn_out, _ = self.attn(" in emitted.source


class TestRegistryDrivenAdaptations:
    def test_rank_picks_the_backend_variant(self) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / "cnn3d.ntb"))
        assert "torch.nn.BatchNorm3d(" in emitted.source
        assert "torch.nn.AdaptiveAvgPool3d(" in emitted.source

    def test_default_inputs_fill_self_attention(self) -> None:
        emitted = emit_torch_document(io.load(EXAMPLES / "transformer_block.ntb"))
        assert "self.attn(norm1, norm1, norm1)" in emitted.source

    def test_packed_inputs_become_a_list(self) -> None:
        graph = CoreGraph(
            name="cat",
            nodes=(CoreNode(id="j", op="ntb.concat", origin=Origin(module="m", node="j")),),
            inputs=(
                GraphInput(
                    name="a",
                    endpoint=Endpoint(node="j", port="a"),
                    type=TensorType(shape=(1, 2, 4, 4)),
                ),
                GraphInput(
                    name="b",
                    endpoint=Endpoint(node="j", port="b"),
                    type=TensorType(shape=(1, 3, 4, 4)),
                ),
            ),
            outputs=(GraphOutput(name="y", endpoint=Endpoint(node="j", port="out")),),
        )
        assert "torch.cat([a, b], dim=1)" in emit_torch(graph).source


class TestFailures:
    def test_a_graph_that_does_not_type_check_is_refused(self) -> None:
        # Emitting a model the shape rules reject would produce code that
        # crashes at the first forward pass.
        module = Module(
            id="m",
            inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=(1, 512))),),
            nodes=(Node(id="fc", op="ntb.linear", attrs={"in_features": 8, "out_features": 4}),),
        )
        with pytest.raises(EmitError, match="does not type-check"):
            emit_torch_document(Document(root="m", modules=(module,)))

    def test_an_unconnected_required_input_is_refused(self) -> None:
        graph = CoreGraph(
            nodes=(CoreNode(id="a", op="ntb.relu", origin=Origin(module="m", node="a")),)
        )
        with pytest.raises(EmitError):
            emit_torch(graph)


class TestNaming:
    @pytest.mark.parametrize(
        ("document_name", "expected"),
        [("mlp", "Mlp"), ("vertical-tower", "VerticalTower"), ("", "Model")],
    )
    def test_class_names_are_pascal_case(self, document_name: str, expected: str) -> None:
        assert _class_name(document_name) == expected

    def test_colliding_node_names_are_disambiguated(self) -> None:
        module = Module(
            id="m",
            inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=(1, 8))),),
            outputs=(Port(name="y", direction=PortDirection.OUT),),
            nodes=(
                Node(id="a-b", op="ntb.relu"),
                Node(id="a.b", op="ntb.relu"),
            ),
            edges=(Edge(id="e", src=Endpoint(node="a-b"), dst=Endpoint(node="a.b", port="in")),),
        )
        source = emit_torch_document(Document(root="m", modules=(module,))).source
        assert "a_b = " in source
        assert "a_b_2 = " in source


class TestGeneratedCodeRuns:
    """The only test that proves the emitter works: run what it produced."""

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("name", "shape"),
        [
            ("mlp", (4, 784)),
            ("transformer_block", (2, 16, 512)),
            ("cnn3d", (2, 1, 32, 64, 64)),
        ],
    )
    def test_the_model_runs_and_matches_inferred_shapes(
        self, name: str, shape: tuple[int, ...]
    ) -> None:
        torch = pytest.importorskip("torch")

        from ntb.shapes import infer_shapes
        from ntb.spatial import resolve

        document = io.load(EXAMPLES / f"{name}.ntb")
        emitted = emit_torch_document(document)
        namespace: dict[str, object] = {}
        exec(compile(emitted.source, f"{name}.py", "exec"), namespace)
        model = namespace[emitted.class_name]().eval()  # type: ignore[operator]

        output = model(torch.randn(*shape))

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

    @pytest.mark.torch
    def test_a_training_step_produces_gradients(self) -> None:
        torch = pytest.importorskip("torch")

        emitted = emit_torch_document(io.load(EXAMPLES / "mlp.ntb"))
        namespace: dict[str, object] = {}
        exec(compile(emitted.source, "mlp.py", "exec"), namespace)
        model = namespace[emitted.class_name]()  # type: ignore[operator]

        model(torch.randn(4, 784)).sum().backward()
        assert all(p.grad is not None for p in model.parameters())
