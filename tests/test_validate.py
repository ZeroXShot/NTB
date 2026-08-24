"""Semantic validation: located diagnostics."""

from __future__ import annotations

import pytest

from ntb.ir import Document, Edge, Endpoint, Module, Node, Port, PortDirection, TensorType, io
from ntb.validate import Code, Severity, validate
from tests.conftest import EXAMPLES


def document(*nodes: Node, edges: tuple[Edge, ...] = ()) -> Document:
    module = Module(
        id="m",
        inputs=(
            Port(
                name="x",
                direction=PortDirection.IN,
                type=TensorType(shape=("batch", 8)),
            ),
        ),
        outputs=(Port(name="y", direction=PortDirection.OUT),),
        nodes=nodes,
        edges=edges,
    )
    return Document(root="m", modules=(module,))


class TestShippedExamples:
    @pytest.mark.parametrize("name", ["mlp.ntb", "transformer_block.ntb", "cnn3d.ntb"])
    def test_example_is_clean(self, name: str) -> None:
        report = validate(io.load(EXAMPLES / name))
        assert report.ok, [str(d) for d in report.diagnostics]
        assert not report.warnings, [str(d) for d in report.diagnostics]

    def test_the_tower_example_validates(self) -> None:
        # Its twelve blocks come from a Generator, so this only passes once
        # generators expand.
        report = validate(io.load(EXAMPLES / "vertical_tower.ntb"))
        assert report.ok, [str(d) for d in report.diagnostics]
        assert report.diagnostics == ()


class TestOpAndAttributeChecks:
    def test_unknown_op(self) -> None:
        report = validate(document(Node(id="a", op="ntb.nope")))
        assert report.codes() == (Code.UNKNOWN_OP,)
        assert report.diagnostics[0].location.node == "a"

    def test_unknown_attribute(self) -> None:
        report = validate(
            document(
                Node(
                    id="a",
                    op="ntb.linear",
                    attrs={"in_features": 8, "out_features": 4, "nope": 1},
                )
            )
        )
        assert Code.UNKNOWN_ATTR in report.codes()

    def test_missing_required_attribute(self) -> None:
        report = validate(document(Node(id="a", op="ntb.linear")))
        assert Code.MISSING_ATTR in report.codes()
        assert "requires attribute 'in_features'" in report.diagnostics[0].message

    def test_attribute_of_the_wrong_type(self) -> None:
        report = validate(
            document(
                Node(
                    id="a",
                    op="ntb.linear",
                    attrs={"in_features": 8, "out_features": "four"},
                )
            )
        )
        assert Code.BAD_ATTR in report.codes()

    def test_a_bool_is_not_an_int(self) -> None:
        report = validate(
            document(Node(id="a", op="ntb.linear", attrs={"in_features": 8, "out_features": True}))
        )
        assert Code.BAD_ATTR in report.codes()


class TestStructureChecks:
    def test_edge_to_an_unknown_node(self) -> None:
        module = Module(
            id="m",
            nodes=(Node(id="a", op="ntb.relu"),),
            edges=(Edge(id="e", src=Endpoint(node="a"), dst=Endpoint(node="ghost", port="in")),),
        )
        report = validate(Document(root="m", modules=(module,)))
        assert Code.STRUCTURE in report.codes()
        assert report.diagnostics[0].location.edge == "e"


class TestShapeChecks:
    def test_a_mismatch_lands_on_the_node_the_user_drew(self) -> None:
        report = validate(
            document(
                Node(
                    id="fc",
                    op="ntb.linear",
                    attrs={"in_features": 256, "out_features": 4},
                )
            )
        )
        assert Code.SHAPE in report.codes()
        diagnostic = report.diagnostics[0]
        assert diagnostic.location.node == "fc"
        assert diagnostic.location.module == "m"
        assert diagnostic.severity is Severity.ERROR

    def test_a_valid_chain_reports_nothing(self) -> None:
        report = validate(
            document(
                Node(id="fc", op="ntb.linear", attrs={"in_features": 8, "out_features": 4}),
                Node(id="act", op="ntb.relu"),
                edges=(
                    Edge(
                        id="e",
                        src=Endpoint(node="fc"),
                        dst=Endpoint(node="act", port="in"),
                    ),
                ),
            )
        )
        assert report.ok and not report.warnings


class TestReportShape:
    def test_diagnostics_render_readably(self) -> None:
        report = validate(document(Node(id="a", op="ntb.nope")))
        assert str(report.diagnostics[0]).startswith("error: m/a: ")
        assert str(report.diagnostics[0]).endswith("[unknown-op]")
