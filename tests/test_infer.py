"""Graph-level shape propagation."""

from __future__ import annotations

from ntb.ir.core import CoreEdge, CoreGraph, CoreNode, Origin
from ntb.ir.graph import Endpoint
from ntb.ir.types import DType, TensorType
from ntb.shapes import infer_shapes

IMAGE = TensorType(dtype=DType.FLOAT32, shape=("batch", 3, 32, 32))


def node(node_id: str, op: str, **attrs: object) -> CoreNode:
    return CoreNode(id=node_id, op=op, attrs=attrs, origin=Origin(module="m", node=node_id))


def link(edge_id: str, src: str, dst: str, src_port: str = "out", dst_port: str = "in") -> CoreEdge:
    return CoreEdge(
        id=edge_id,
        src=Endpoint(node=src, port=src_port),
        dst=Endpoint(node=dst, port=dst_port),
        origin=Origin(module="m"),
    )


def chain() -> CoreGraph:
    return CoreGraph(
        nodes=(
            node("conv", "ntb.conv2d", in_channels=3, out_channels=16, padding=[1, 1]),
            node("act", "ntb.relu"),
            node("pool", "ntb.maxpool2d", kernel_size=[2, 2]),
        ),
        edges=(link("e1", "conv", "act"), link("e2", "act", "pool")),
        inputs=((Endpoint(node="conv", port="in"), IMAGE),),
        outputs=(Endpoint(node="pool", port="out"),),
    )


class TestHappyPath:
    def test_types_flow_to_the_end(self) -> None:
        report = infer_shapes(chain())
        assert report.ok
        assert report.type_of("pool") == TensorType(
            dtype=DType.FLOAT32, shape=("batch", 16, 16, 16), layout="NCHW"
        )

    def test_symbolic_dimensions_survive_the_whole_graph(self) -> None:
        report = infer_shapes(chain())
        assert report.type_of("conv") is not None
        assert report.type_of("conv").shape[0] == "batch"  # type: ignore[union-attr]

    def test_multi_output_ops_record_every_port(self) -> None:
        graph = CoreGraph(
            nodes=(node("attn", "ntb.attention", embed_dim=64, num_heads=4),),
            inputs=((Endpoint(node="attn", port="query"), TensorType(shape=(2, "seq", 64))),),
        )
        report = infer_shapes(graph)
        assert report.type_of("attn", "out") is not None
        assert report.type_of("attn", "weights") is not None


class TestFailures:
    def test_a_shape_error_is_reported_at_its_node(self) -> None:
        graph = CoreGraph(
            nodes=(node("fc", "ntb.linear", in_features=256, out_features=10),),
            inputs=((Endpoint(node="fc", port="in"), TensorType(shape=(1, 512))),),
        )
        report = infer_shapes(graph)
        assert not report.ok
        assert report.issues[0].node == "fc"
        assert "in_features is 256" in report.issues[0].message

    def test_one_broken_node_does_not_cascade(self) -> None:
        # Downstream nodes are marked unresolved, not re-reported: three
        # messages for one mistake is how a diagnostics panel becomes useless.
        graph = CoreGraph(
            nodes=(
                node("fc", "ntb.linear", in_features=256, out_features=10),
                node("act", "ntb.relu"),
                node("tail", "ntb.relu"),
            ),
            edges=(link("e1", "fc", "act"), link("e2", "act", "tail")),
            inputs=((Endpoint(node="fc", port="in"), TensorType(shape=(1, 512))),),
        )
        report = infer_shapes(graph)
        assert len(report.issues) == 1
        assert report.unresolved == {"fc", "act", "tail"}

    def test_unknown_op_is_reported_with_a_suggestion(self) -> None:
        report = infer_shapes(CoreGraph(nodes=(node("x", "ntb.relo"),)))
        assert "did you mean 'ntb.relu'" in report.issues[0].message

    def test_an_edge_into_an_undeclared_port_is_caught(self) -> None:
        graph = CoreGraph(
            nodes=(node("a", "ntb.relu"), node("b", "ntb.relu")),
            edges=(link("e1", "a", "b", dst_port="nope"),),
            inputs=((Endpoint(node="a", port="in"), IMAGE),),
        )
        report = infer_shapes(graph)
        assert "has no input port 'nope'" in report.issues[0].message

    def test_two_edges_into_one_port_is_caught(self) -> None:
        graph = CoreGraph(
            nodes=(node("a", "ntb.relu"), node("b", "ntb.relu"), node("c", "ntb.relu")),
            edges=(link("e1", "a", "c"), link("e2", "b", "c")),
            inputs=(
                (Endpoint(node="a", port="in"), IMAGE),
                (Endpoint(node="b", port="in"), IMAGE),
            ),
        )
        report = infer_shapes(graph)
        assert "connected more than once" in report.issues[0].message

    def test_a_cycle_names_every_node_in_it(self) -> None:
        graph = CoreGraph(
            nodes=(node("a", "ntb.relu"), node("b", "ntb.relu")),
            edges=(link("e1", "a", "b"), link("e2", "b", "a")),
        )
        report = infer_shapes(graph)
        assert {issue.node for issue in report.issues} == {"a", "b"}
        assert all("cycle" in issue.message for issue in report.issues)

    def test_an_unconnected_required_input_is_reported(self) -> None:
        report = infer_shapes(CoreGraph(nodes=(node("act", "ntb.relu"),)))
        assert "not connected" in report.issues[0].message
