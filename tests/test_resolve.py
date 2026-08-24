"""Lowering a document to the core IR."""

from __future__ import annotations

import pytest

from ntb.ir import Document, Edge, Endpoint, Module, Node, Port, PortDirection, TensorType, io
from ntb.ir.types import DType
from ntb.spatial import MODULE_OP, NotResolvable, ResolveError, resolve
from tests.conftest import EXAMPLES


def linear(node_id: str, out_features: int = 8, in_features: int = 8) -> Node:
    return Node(
        id=node_id,
        op="ntb.linear",
        attrs={"in_features": in_features, "out_features": out_features},
    )


def wire(edge_id: str, src: str, dst: str) -> Edge:
    return Edge(id=edge_id, src=Endpoint(node=src), dst=Endpoint(node=dst, port="in"))


class TestFlatModule:
    def test_nodes_and_edges_carry_over(self) -> None:
        module = Module(
            id="m",
            nodes=(linear("a"), Node(id="b", op="ntb.relu")),
            edges=(wire("e", "a", "b"),),
        )
        graph = resolve(Document(root="m", modules=(module,)))
        assert [n.id for n in graph.nodes] == ["a", "b"]
        assert [(str(e.src), str(e.dst)) for e in graph.edges] == [("a.out", "b.in")]

    def test_every_node_keeps_its_origin(self) -> None:
        module = Module(id="m", nodes=(linear("a"),))
        graph = resolve(Document(root="m", modules=(module,)))
        assert graph.nodes[0].origin.module == "m"
        assert graph.nodes[0].origin.node == "a"

    def test_boundary_ports_bind_to_free_node_ports(self) -> None:
        module = Module(
            id="m",
            inputs=(
                Port(
                    name="x",
                    direction=PortDirection.IN,
                    type=TensorType(dtype=DType.FLOAT32, shape=("batch", 8)),
                ),
            ),
            outputs=(Port(name="y", direction=PortDirection.OUT),),
            nodes=(linear("a"), Node(id="b", op="ntb.relu")),
            edges=(wire("e", "a", "b"),),
        )
        graph = resolve(Document(root="m", modules=(module,)))
        assert [(i.name, str(i.endpoint)) for i in graph.inputs] == [("x", "a.in")]
        assert [(o.name, str(o.endpoint)) for o in graph.outputs] == [("y", "b.out")]

    def test_multi_port_ops_expose_the_right_free_ports(self) -> None:
        # ntb.add has ports a and b, not 'in'. Free-port detection reads the
        # registry rather than assuming a naming convention.
        module = Module(
            id="m",
            inputs=(
                Port(name="p", direction=PortDirection.IN, type=TensorType(shape=(4,))),
                Port(name="q", direction=PortDirection.IN, type=TensorType(shape=(4,))),
            ),
            nodes=(Node(id="sum", op="ntb.add"),),
        )
        graph = resolve(Document(root="m", modules=(module,)))
        assert sorted(str(i.endpoint) for i in graph.inputs) == ["sum.a", "sum.b"]


class TestNestedModules:
    def test_a_module_instance_is_inlined_under_a_path(self) -> None:
        inner = Module(
            id="inner",
            inputs=(Port(name="x", direction=PortDirection.IN),),
            outputs=(Port(name="y", direction=PortDirection.OUT),),
            nodes=(linear("fc"), Node(id="act", op="ntb.relu")),
            edges=(wire("e", "fc", "act"),),
        )
        outer = Module(
            id="outer",
            nodes=(
                Node(id="stage", op=MODULE_OP, attrs={"module": "inner"}),
                Node(id="tail", op="ntb.relu"),
            ),
            edges=(
                Edge(
                    id="e",
                    src=Endpoint(node="stage", port="y"),
                    dst=Endpoint(node="tail", port="in"),
                ),
            ),
        )
        graph = resolve(Document(root="outer", modules=(inner, outer)))
        assert [n.id for n in graph.nodes] == ["stage/fc", "stage/act", "tail"]
        assert ("stage/act.out", "tail.in") in [(str(e.src), str(e.dst)) for e in graph.edges]

    def test_recursive_instantiation_is_refused(self) -> None:
        module = Module(
            id="loop",
            nodes=(Node(id="self", op=MODULE_OP, attrs={"module": "loop"}),),
        )
        with pytest.raises(ResolveError, match="recursive"):
            resolve(Document(root="loop", modules=(module,)))

    def test_instantiating_an_unknown_module_is_refused(self) -> None:
        module = Module(id="m", nodes=(Node(id="x", op=MODULE_OP, attrs={"module": "ghost"}),))
        with pytest.raises(ResolveError, match="unknown module 'ghost'"):
            resolve(Document(root="m", modules=(module,)))

    def test_a_module_node_without_a_target_is_refused(self) -> None:
        module = Module(id="m", nodes=(Node(id="x", op=MODULE_OP),))
        with pytest.raises(ResolveError, match="'module' attribute is missing"):
            resolve(Document(root="m", modules=(module,)))


class TestPhaseFourFeatures:
    def test_a_generator_says_which_phase_expands_it(self) -> None:
        document = io.load(EXAMPLES / "vertical_tower.ntb")
        with pytest.raises(NotResolvable, match=r"Generator.*phase 4"):
            resolve(document)

    def test_a_spatial_rule_says_which_phase_resolves_it(self) -> None:
        document = io.load(EXAMPLES / "vertical_tower.ntb")
        lateral = document.module("lateral")
        assert lateral is not None
        with pytest.raises(NotResolvable, match=r"SpatialRule.*phase 4"):
            resolve(document.model_copy(update={"root": "lateral"}))


class TestShippedExamples:
    @pytest.mark.parametrize("name", ["mlp.ntb", "transformer_block.ntb", "cnn3d.ntb"])
    def test_example_lowers_to_a_connected_dag(self, name: str) -> None:
        graph = resolve(io.load(EXAMPLES / name))
        assert graph.nodes
        assert graph.inputs
        assert graph.outputs
        graph.topological_order()
