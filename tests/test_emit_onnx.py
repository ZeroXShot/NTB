"""The ONNX exporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from ntb.emit.onnx import DEFAULT_OPSET, OnnxEmitError, export, export_document, resolve_param_shape
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
from ntb.ops.spec import ParamSpec
from tests.conftest import EXAMPLES

pytest.importorskip("onnx")
pytestmark = pytest.mark.onnx

EXPORTABLE = ["mlp", "cnn3d"]


class TestExport:
    @pytest.mark.parametrize("name", EXPORTABLE)
    def test_the_model_passes_the_onnx_checker(self, name: str) -> None:
        # export() runs onnx.checker itself, so reaching this line is the check.
        exported = export_document(io.load(EXAMPLES / f"{name}.ntb"))
        assert exported.opset == DEFAULT_OPSET
        assert exported.model.graph.node

    def test_graph_inputs_keep_the_authored_names(self) -> None:
        exported = export_document(io.load(EXAMPLES / "cnn3d.ntb"))
        assert [i.name for i in exported.model.graph.input] == ["volume"]

    def test_symbolic_dimensions_become_named_axes(self) -> None:
        # A dynamic batch has to survive export, or the model is only usable at
        # the one batch size it was drawn with.
        exported = export_document(io.load(EXAMPLES / "mlp.ntb"))
        first = exported.model.graph.input[0].type.tensor_type.shape.dim[0]
        assert first.dim_param == "batch"

    def test_parameterised_ops_get_initialisers(self) -> None:
        exported = export_document(io.load(EXAMPLES / "mlp.ntb"))
        names = {t.name for t in exported.model.graph.initializer}
        assert {"fc1_W", "fc1_B", "fc2_W", "fc2_B"} <= names

    def test_saving_writes_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "model.onnx"
        export_document(io.load(EXAMPLES / "mlp.ntb")).save(target)
        assert target.stat().st_size > 0


class TestOpsetGating:
    def test_an_op_needing_a_newer_opset_says_so(self) -> None:
        graph = CoreGraph(
            name="rms",
            nodes=(
                CoreNode(
                    id="n",
                    op="ntb.rmsnorm",
                    attrs={"normalized_size": 4},
                    origin=Origin(module="m", node="n"),
                ),
            ),
            inputs=(
                GraphInput(
                    name="x",
                    endpoint=Endpoint(node="n", port="in"),
                    type=TensorType(shape=(2, 4)),
                ),
            ),
            outputs=(GraphOutput(name="y", endpoint=Endpoint(node="n", port="out")),),
        )
        with pytest.raises(OnnxEmitError, match="needs opset 23"):
            export(graph, opset=20)


class TestFailures:
    def test_a_graph_that_does_not_type_check_is_refused(self) -> None:
        module = Module(
            id="m",
            inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=(1, 512))),),
            nodes=(Node(id="fc", op="ntb.linear", attrs={"in_features": 8, "out_features": 4}),),
        )
        with pytest.raises(OnnxEmitError, match="does not type-check"):
            export_document(Document(root="m", modules=(module,)))


class TestParameterShapes:
    def test_attribute_expressions_are_evaluated(self) -> None:
        param = ParamSpec("W", ("out_channels", "in_channels // groups", "*kernel_size"))
        attrs = {"out_channels": 8, "in_channels": 6, "groups": 2, "kernel_size": [3, 3]}
        assert resolve_param_shape(param, attrs) == (8, 3, 3, 3)

    def test_a_splat_needs_a_list_attribute(self) -> None:
        with pytest.raises(OnnxEmitError, match="does not name a list attribute"):
            resolve_param_shape(ParamSpec("W", ("*groups",)), {"groups": 2})

    def test_an_unresolved_expression_is_reported(self) -> None:
        with pytest.raises(OnnxEmitError, match="did not resolve to an integer"):
            resolve_param_shape(ParamSpec("W", ("mystery",)), {})


class TestRuns:
    @pytest.mark.parametrize("name", EXPORTABLE)
    def test_onnxruntime_runs_the_exported_model(self, name: str) -> None:
        np = pytest.importorskip("numpy")
        ort = pytest.importorskip("onnxruntime")

        shapes = {"mlp": (4, 784), "cnn3d": (2, 1, 32, 64, 64)}
        exported = export_document(io.load(EXAMPLES / f"{name}.ntb"))
        session = ort.InferenceSession(
            exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feed = {session.get_inputs()[0].name: np.random.randn(*shapes[name]).astype(np.float32)}
        assert session.run(None, feed)[0].shape[0] == shapes[name][0]
