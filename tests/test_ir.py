"""Structural guarantees of NTB-IR."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ntb.ir import (
    CoreEdge,
    CoreGraph,
    CoreNode,
    Document,
    Edge,
    Endpoint,
    Module,
    Node,
    Placement,
    Port,
    PortDirection,
    SpatialRule,
    SpatialRuleKind,
    TensorType,
)
from ntb.ir.core import CycleError, Origin
from ntb.ir.types import DType


def node(node_id: str, op: str = "ntb.relu") -> Node:
    return Node(id=node_id, op=op)


def core_node(node_id: str) -> CoreNode:
    return CoreNode(id=node_id, op="ntb.relu", origin=Origin(module="m", node=node_id))


def core_edge(edge_id: str, src: str, dst: str) -> CoreEdge:
    return CoreEdge(
        id=edge_id,
        src=Endpoint(node=src, port="out"),
        dst=Endpoint(node=dst, port="in"),
        origin=Origin(module="m"),
    )


class TestTensorType:
    def test_negative_dimension_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            TensorType(shape=(1, -2))

    def test_blank_symbolic_dimension_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty symbolic"):
            TensorType(shape=("batch", "  "))

    def test_symbolic_shape_is_not_static(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=("batch", 128))
        assert not t.is_static
        assert t.rank == 2
        assert t.symbols() == {"batch"}
        assert str(t) == "float32[batch, 128]"

    def test_scalar_is_static(self) -> None:
        assert TensorType().is_static

    def test_is_hashable_so_it_can_key_inference_caches(self) -> None:
        assert len({TensorType(shape=(1,)), TensorType(shape=(1,))}) == 1


class TestSpatialRule:
    def test_neighborhood_requires_a_radius(self) -> None:
        with pytest.raises(ValidationError, match="needs a radius"):
            SpatialRule(id="r", kind=SpatialRuleKind.NEIGHBORHOOD, members=("a", "b"))

    def test_ordered_kinds_reject_a_radius(self) -> None:
        with pytest.raises(ValidationError, match="does not take a radius"):
            SpatialRule(
                id="r",
                kind=SpatialRuleKind.VERTICAL_STACK,
                members=("a", "b"),
                radius=2.0,
            )

    def test_ordered_kinds_cannot_be_bidirectional(self) -> None:
        # An ordered rule wired both ways is a guaranteed cycle, so it is
        # rejected at authoring time rather than at lowering time.
        with pytest.raises(ValidationError, match="cannot be bidirectional"):
            SpatialRule(
                id="r",
                kind=SpatialRuleKind.AXIS_PROJECTION,
                members=("a", "b"),
                bidirectional=True,
            )

    def test_repeated_member_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lists a member twice"):
            SpatialRule(id="r", kind=SpatialRuleKind.LATTICE, members=("a", "a"))

    def test_no_members_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lists no members"):
            SpatialRule(id="r", kind=SpatialRuleKind.LATTICE, members=())

    def test_one_member_is_allowed_because_it_may_be_a_generator(self) -> None:
        # Resolution is where a rule that covers fewer than two blocks fails.
        assert SpatialRule(id="r", kind=SpatialRuleKind.LATTICE, members=("stack",)).members


class TestPlacement:
    def test_zero_extent_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extent must be positive"):
            Placement(extent=(1.0, 0.0, 1.0))

    def test_coordinates_are_read_by_axis(self) -> None:
        from ntb.ir import Axis, Orientation

        placement = Placement(pos=(1.0, 2.0, 3.0), extent=(4.0, 5.0, 6.0))
        assert placement.coord(Axis.Z) == 3.0
        assert placement.size(Axis.Y) == 5.0
        assert Orientation.ALONG_Z.axis is Axis.Z


class TestModule:
    def test_duplicate_node_ids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="declares node 'a' twice"):
            Module(id="m", nodes=(node("a"), node("a")))

    def test_output_port_must_be_an_out_port(self) -> None:
        with pytest.raises(ValidationError, match="is not an out port"):
            Module(id="m", outputs=(Port(name="y", direction=PortDirection.IN),))

    def test_self_loop_edge_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="to itself"):
            Edge(id="e", src=Endpoint(node="a"), dst=Endpoint(node="a", port="in"))


class TestDocument:
    def test_root_must_exist(self) -> None:
        with pytest.raises(ValidationError, match="root module 'nope' is not defined"):
            Document(root="nope", modules=(Module(id="m"),))

    def test_root_module_is_reachable(self) -> None:
        document = Document(root="m", modules=(Module(id="m"),))
        assert document.root_module.id == "m"
        assert document.module("absent") is None


class TestCoreGraph:
    def test_edge_to_unknown_node_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="references unknown node"):
            CoreGraph(nodes=(core_node("a"),), edges=(core_edge("e", "a", "ghost"),))

    def test_topological_order_respects_dependencies(self) -> None:
        graph = CoreGraph(
            nodes=(core_node("c"), core_node("a"), core_node("b")),
            edges=(core_edge("e1", "a", "b"), core_edge("e2", "b", "c")),
        )
        assert [n.id for n in graph.topological_order()] == ["a", "b", "c"]

    def test_topological_order_is_deterministic_across_independent_nodes(self) -> None:
        # Ties break by id. Emitted code must not reshuffle between runs, or
        # every golden-file test becomes flaky.
        graph = CoreGraph(nodes=(core_node("z"), core_node("a"), core_node("m")))
        assert [n.id for n in graph.topological_order()] == ["a", "m", "z"]

    def test_cycle_reports_the_nodes_involved(self) -> None:
        graph = CoreGraph(
            nodes=(core_node("a"), core_node("b")),
            edges=(core_edge("e1", "a", "b"), core_edge("e2", "b", "a")),
        )
        with pytest.raises(CycleError) as excinfo:
            graph.topological_order()
        assert excinfo.value.nodes == {"a", "b"}

    def test_parallel_edges_are_counted_separately(self) -> None:
        # Two edges into the same node from the same source: if indegree were
        # deduplicated, the target would be released one edge too early.
        graph = CoreGraph(
            nodes=(core_node("a"), core_node("b")),
            edges=(core_edge("e1", "a", "b"), core_edge("e2", "a", "b")),
        )
        assert [n.id for n in graph.topological_order()] == ["a", "b"]

    def test_incoming_and_outgoing(self) -> None:
        graph = CoreGraph(
            nodes=(core_node("a"), core_node("b")),
            edges=(core_edge("e1", "a", "b"),),
        )
        assert [e.id for e in graph.outgoing("a")] == ["e1"]
        assert [e.id for e in graph.incoming("b")] == ["e1"]
        assert graph.node("missing") is None


def test_origin_renders_a_readable_trail() -> None:
    origin = Origin(module="tower", generator="stack", instance=3, node="fc", rule="couple")
    assert str(origin) == "tower/stack[3]/fc/via couple"
