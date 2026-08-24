"""Reading an ONNX model back into NTB-IR.

Import is best effort and one-way (ADR 1). The test that means something is the
round trip: a document NTB exported, read back, has to validate and infer the
same shapes -- including the ones a final global pool would otherwise hide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ntb.ir import io
from ntb.shapes import infer_shapes
from ntb.spatial import resolve
from ntb.validate import validate
from tests.conftest import EXAMPLES

pytest.importorskip("onnx")
pytestmark = pytest.mark.onnx

from ntb.emit import export_onnx_document  # noqa: E402
from ntb.importers import import_onnx  # noqa: E402

ROUND_TRIP = ["mlp", "cnn3d"]


def exported(name: str) -> object:
    return export_onnx_document(io.load(EXAMPLES / f"{name}.ntb")).model


class TestRoundTrip:
    @pytest.mark.parametrize("name", ROUND_TRIP)
    def test_an_exported_model_reads_back_and_validates(self, name: str) -> None:
        result = import_onnx(exported(name), name=name)
        assert result.problems == ()
        assert result.complete
        report = validate(result.document)
        assert report.ok, [str(d) for d in report.diagnostics]

    @pytest.mark.parametrize("name", ROUND_TRIP)
    def test_the_shapes_survive_the_trip(self, name: str) -> None:
        original = io.load(EXAMPLES / f"{name}.ntb")
        imported = import_onnx(exported(name), name=name).document

        before, after = resolve(original), resolve(imported)
        first = infer_shapes(before).type_of(*_output(before))
        second = infer_shapes(after).type_of(*_output(after))
        assert first is not None and str(first) == str(second)

    def test_padding_comes_back(self) -> None:
        # An intermediate shape, not the final one: cnn3d ends in a global pool,
        # which hides a convolution that lost its padding.
        imported = import_onnx(exported("cnn3d"), name="cnn3d").document
        conv = next(n for n in imported.root_module.nodes if n.op == "ntb.conv3d")
        assert conv.attrs["padding"] == [1, 1, 1]

        graph = resolve(imported)
        pooled = next(n for n in graph.nodes if n.op == "ntb.maxpool3d")
        assert (
            str(infer_shapes(graph).type_of(pooled.id, "out")) == "float32[batch, 16, 16, 32, 32]"
        )

    def test_attributes_are_read_out_of_the_weight_shapes(self) -> None:
        imported = import_onnx(exported("mlp"), name="mlp").document
        linear = imported.root_module.nodes[0]
        assert linear.attrs == {"in_features": 784, "out_features": 128}

    def test_the_boundary_is_bound_explicitly(self) -> None:
        # ONNX says exactly where the model's input goes, so the document says so
        # too rather than leaning on the positional rule.
        module = import_onnx(exported("mlp"), name="mlp").document.root_module
        assert str(module.input_bindings["x"]) == "fc1.in"
        assert module.output_bindings

    def test_nodes_are_laid_out_left_to_right(self) -> None:
        module = import_onnx(exported("mlp"), name="mlp").document.root_module
        assert [n.placement.pos[0] for n in module.nodes] == [0.0, 3.0, 6.0]

    def test_the_conv_rank_is_read_from_the_kernel(self) -> None:
        imported = import_onnx(exported("cnn3d"), name="cnn3d").document
        assert {n.op for n in imported.root_module.nodes} >= {"ntb.conv3d", "ntb.maxpool3d"}

    def test_a_file_path_works_too(self, tmp_path: Path) -> None:
        target = tmp_path / "mlp.onnx"
        export_onnx_document(io.load(EXAMPLES / "mlp.ntb")).save(target)
        assert import_onnx(target).document.root_module.nodes


class TestBestEffort:
    def test_an_unknown_op_is_reported_not_invented(self) -> None:
        import onnx
        from onnx import helper

        graph = helper.make_graph(
            [helper.make_node("Einsum", ["x"], ["y"], equation="ij->ji")],
            "odd",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 2])],
        )
        result = import_onnx(helper.make_model(graph), name="odd")
        assert not result.complete
        assert "Einsum" in result.problems[0] and "dropped" in result.problems[0]
        assert result.document.root_module.nodes == ()

    def test_asymmetric_padding_is_reported(self) -> None:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        weight = numpy_helper.from_array(np.zeros((4, 3, 3, 3), dtype=np.float32), name="W")
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Conv", ["x", "W"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 0, 0], group=1
                )
            ],
            "lopsided",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, 7, 7])],
            initializer=[weight],
        )
        result = import_onnx(helper.make_model(graph), name="lopsided")
        assert any("asymmetric" in problem for problem in result.problems)

    def test_a_missing_attribute_says_what_to_fill_in(self) -> None:
        import onnx
        from onnx import helper

        # A Gemm with no weight initialiser: the features cannot be recovered.
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["x", "b"], ["y"])],
            "bare",
            [
                helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 4]),
                helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [4, 4]),
            ],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 4])],
        )
        result = import_onnx(helper.make_model(graph), name="bare")
        assert any("could not read" in problem for problem in result.problems)


def _output(graph: object) -> tuple[str, str]:
    endpoint = graph.outputs[0].endpoint  # type: ignore[attr-defined]
    return endpoint.node, endpoint.port
