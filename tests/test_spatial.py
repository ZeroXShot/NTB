"""Spatial semantics: parameter expressions, the four rules, and generators.

This is the part of NTB that no other tool has, so it is the part that most
needs to be pinned down. Rule resolution is deterministic, which is what makes
a generated topology reviewable in a diff rather than a surprise.
"""

from __future__ import annotations

from typing import Any

import pytest

from ntb.ir import (
    Axis,
    Document,
    Edge,
    Endpoint,
    Generator,
    Module,
    Node,
    Placement,
    Port,
    PortDirection,
    SpatialRule,
    SpatialRuleKind,
    TensorType,
    io,
)
from ntb.spatial import ResolveError, resolve
from ntb.spatial.expr import ExpressionError, evaluate, is_expression, resolve_attrs
from ntb.spatial.preview import BlockKind, LinkKind, preview
from ntb.spatial.rules import Placed, RuleError, derive_pairs
from tests.conftest import EXAMPLES


def placed(*coords: tuple[str, float, float, float]) -> list[Placed]:
    return [Placed(key=key, pos=(x, y, z)) for key, x, y, z in coords]


def rule(kind: SpatialRuleKind, members: tuple[str, ...], **kwargs: Any) -> SpatialRule:
    return SpatialRule(id="r", kind=kind, members=members, **kwargs)


class TestExpressions:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("$1 + 1", 2),
            ("$width * 2", 128),
            ("$width // 3", 21),
            ("$2 ** 5", 32),
            ("$max(width, 8)", 64),
            ("$min(width, 8)", 8),
            ("$-i", -3),
            ("$64 * 2 ** i", 512),
            ("$round(width / 7)", 9),
        ],
    )
    def test_arithmetic_over_parameters(self, text: str, expected: float) -> None:
        assert evaluate(text, {"width": 64, "i": 3}) == expected

    def test_an_unknown_parameter_names_what_is_available(self) -> None:
        with pytest.raises(ExpressionError, match="unknown parameter 'depth'; known: width"):
            evaluate("$depth", {"width": 8})

    @pytest.mark.parametrize(
        "text",
        [
            "$__import__('os').system('echo')",
            "$open('/etc/passwd')",
            "$[1, 2, 3]",
            "$'text'",
            "$width if width else 0",
            "$lambda: 1",
        ],
    )
    def test_anything_but_arithmetic_is_refused(self, text: str) -> None:
        # `.ntb` files travel between people. This must never be an eval().
        with pytest.raises(ExpressionError):
            evaluate(text, {"width": 8})

    def test_a_runaway_exponent_is_refused(self) -> None:
        with pytest.raises(ExpressionError, match="over the limit"):
            evaluate("$2 ** 100000", {})

    def test_dividing_by_zero_is_reported_not_raised_raw(self) -> None:
        with pytest.raises(ExpressionError, match="divides by zero"):
            evaluate("$1 / 0", {})

    def test_a_syntax_error_says_so(self) -> None:
        with pytest.raises(ExpressionError, match="is not an expression"):
            evaluate("$1 +", {})

    def test_only_dollar_strings_are_expressions(self) -> None:
        assert is_expression("$i + 1")
        assert not is_expression("i + 1")
        assert not is_expression(3)

    def test_lists_are_walked(self) -> None:
        resolved = resolve_attrs({"kernel_size": ["$k", 3], "name": "keep"}, {"k": 5})
        assert resolved == {"kernel_size": [5, 3], "name": "keep"}


class TestVerticalStack:
    def test_it_connects_consecutive_blocks_along_the_axis(self) -> None:
        members = placed(("a", 0, 0, 2), ("b", 0, 0, 0), ("c", 0, 0, 1))
        pairs = derive_pairs(rule(SpatialRuleKind.VERTICAL_STACK, ("a", "b", "c")), members)
        assert [(members[i].key, members[j].key) for i, j in pairs] == [("b", "c"), ("c", "a")]

    def test_a_tie_is_broken_by_name_so_the_result_is_stable(self) -> None:
        members = placed(("b", 0, 0, 0), ("a", 0, 0, 0))
        pairs = derive_pairs(rule(SpatialRuleKind.VERTICAL_STACK, ("a", "b")), members)
        assert [(members[i].key, members[j].key) for i, j in pairs] == [("a", "b")]

    def test_another_axis_works_the_same_way(self) -> None:
        members = placed(("a", 3, 0, 0), ("b", 1, 0, 0))
        pairs = derive_pairs(rule(SpatialRuleKind.VERTICAL_STACK, ("a", "b"), axis=Axis.X), members)
        assert [(members[i].key, members[j].key) for i, j in pairs] == [("b", "a")]


class TestAxisProjection:
    def test_everything_reaches_everything_ahead_of_it(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 0, 0, 1), ("c", 0, 0, 2))
        pairs = derive_pairs(rule(SpatialRuleKind.AXIS_PROJECTION, ("a", "b", "c")), members)
        assert [(members[i].key, members[j].key) for i, j in pairs] == [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ]

    def test_blocks_side_by_side_are_not_ahead_of_each_other(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 5, 0, 0))
        assert derive_pairs(rule(SpatialRuleKind.AXIS_PROJECTION, ("a", "b")), members) == ()


class TestNeighborhood:
    def test_only_blocks_within_the_radius_are_coupled(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 1, 0, 0), ("c", 5, 0, 0))
        pairs = derive_pairs(
            rule(SpatialRuleKind.NEIGHBORHOOD, ("a", "b", "c"), radius=1.5, axis=Axis.X), members
        )
        assert [(members[i].key, members[j].key) for i, j in pairs] == [("a", "b")]

    def test_the_radius_is_euclidean_not_per_axis(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 1, 1, 1))
        pairs = derive_pairs(rule(SpatialRuleKind.NEIGHBORHOOD, ("a", "b"), radius=1.5), members)
        assert pairs == ()

    def test_bidirectional_emits_both_directions(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 0, 0, 1))
        pairs = derive_pairs(
            rule(SpatialRuleKind.NEIGHBORHOOD, ("a", "b"), radius=2.0, bidirectional=True),
            members,
        )
        assert [(members[i].key, members[j].key) for i, j in pairs] == [("a", "b"), ("b", "a")]


class TestLattice:
    def test_only_adjacent_cells_are_wired(self) -> None:
        # A 2x2 grid in the XZ plane: the diagonal is not an edge.
        members = placed(("a", 0, 0, 0), ("b", 1, 0, 0), ("c", 0, 0, 1), ("d", 1, 0, 1))
        pairs = derive_pairs(rule(SpatialRuleKind.LATTICE, ("a", "b", "c", "d")), members)
        wired = {(members[i].key, members[j].key) for i, j in pairs}
        assert wired == {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}

    def test_a_gap_of_two_cells_is_not_adjacency(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 1, 0, 0), ("c", 2, 0, 0))
        pairs = derive_pairs(rule(SpatialRuleKind.LATTICE, ("a", "b", "c")), members)
        assert [(members[i].key, members[j].key) for i, j in pairs] == [("a", "b"), ("b", "c")]

    def test_members_piled_on_one_point_are_not_a_lattice(self) -> None:
        members = placed(("a", 0, 0, 0), ("b", 0, 0, 0))
        with pytest.raises(RuleError, match="same point"):
            derive_pairs(rule(SpatialRuleKind.LATTICE, ("a", "b")), members)


def tower(count: int = 3, **generator: Any) -> Document:
    """A document whose root is nothing but one generator."""
    block = Module(
        id="block",
        params={"width": 8},
        inputs=(Port(name="in", direction=PortDirection.IN),),
        outputs=(Port(name="out", direction=PortDirection.OUT),),
        nodes=(Node(id="fc", op="ntb.linear", attrs={"in_features": 8, "out_features": "$width"}),),
    )
    root = Module(
        id="root",
        inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=("batch", 8))),),
        outputs=(Port(name="y", direction=PortDirection.OUT),),
        generators=(Generator(id="stack", module="block", count=count, **generator),),
    )
    return Document(name="tower", root="root", modules=(block, root))


class TestGenerators:
    def test_a_generator_becomes_one_instance_per_count(self) -> None:
        graph = resolve(tower(4))
        assert [node.id for node in graph.nodes] == [f"stack-{i}/fc" for i in range(4)]

    def test_chaining_wires_each_instance_into_the_next(self) -> None:
        graph = resolve(tower(3))
        assert [(str(e.src), str(e.dst)) for e in graph.edges] == [
            ("stack-0/fc.out", "stack-1/fc.in"),
            ("stack-1/fc.out", "stack-2/fc.in"),
        ]

    def test_an_unchained_generator_lays_the_instances_out_in_parallel(self) -> None:
        graph = resolve(tower(3, chain=False))
        assert graph.edges == ()

    def test_the_boundary_binds_to_the_ends_of_the_chain(self) -> None:
        graph = resolve(tower(3))
        assert str(graph.inputs[0].endpoint) == "stack-0/fc.in"
        assert str(graph.outputs[0].endpoint) == "stack-2/fc.out"

    def test_an_instance_knows_where_it_came_from(self) -> None:
        graph = resolve(tower(2))
        chain = next(e for e in graph.edges)
        assert chain.origin.generator == "stack"
        assert str(graph.nodes[1].origin) == "block/fc"

    def test_attribute_bindings_parameterise_each_instance(self) -> None:
        document = tower(3, attr_bindings={"width": "8 * 2 ** i"})
        widths = [node.attrs["out_features"] for node in resolve(document).nodes]
        assert widths == [8, 16, 32]

    def test_the_index_itself_is_available_as_a_parameter(self) -> None:
        block = Module(
            id="block",
            inputs=(Port(name="in", direction=PortDirection.IN),),
            outputs=(Port(name="out", direction=PortDirection.OUT),),
            nodes=(
                Node(id="fc", op="ntb.linear", attrs={"in_features": 8, "out_features": "$8 + i"}),
            ),
        )
        document = tower(3).model_copy(update={"modules": (block, tower(3).module("root"))})
        assert [n.attrs["out_features"] for n in resolve(document).nodes] == [8, 9, 10]

    def test_a_bad_binding_says_which_generator(self) -> None:
        document = tower(2, attr_bindings={"width": "nonesuch * 2"})
        with pytest.raises(ResolveError, match=r"generator 'stack'.*unknown parameter"):
            resolve(document)

    def test_a_generator_of_an_unknown_module_is_refused(self) -> None:
        root = Module(id="root", generators=(Generator(id="g", module="ghost", count=2),))
        with pytest.raises(ResolveError, match="instantiates unknown module 'ghost'"):
            resolve(Document(root="root", modules=(root,)))

    def test_chaining_a_module_with_no_ports_is_refused(self) -> None:
        block = Module(id="block", nodes=(Node(id="fc", op="ntb.relu"),))
        root = Module(id="root", generators=(Generator(id="g", module="block", count=2),))
        with pytest.raises(ResolveError, match="does not declare both an input and an output"):
            resolve(Document(root="root", modules=(block, root)))


def coupled(kind: SpatialRuleKind, **kwargs: Any) -> Document:
    """Three relus in a row, wired only by a spatial rule."""
    nodes = tuple(
        Node(id=f"p{i}", op="ntb.relu", placement=Placement(pos=(float(i), 0.0, 0.0)))
        for i in range(3)
    )
    module = Module(
        id="m",
        inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=("batch", 4))),),
        outputs=(Port(name="y", direction=PortDirection.OUT),),
        nodes=nodes,
        spatial_rules=(
            SpatialRule(id="r", kind=kind, members=("p0", "p1", "p2"), axis=Axis.X, **kwargs),
        ),
    )
    return Document(root="m", modules=(module,))


class TestRulesInADocument:
    def test_a_rule_becomes_ordinary_edges(self) -> None:
        graph = resolve(coupled(SpatialRuleKind.VERTICAL_STACK))
        assert [(str(e.src), str(e.dst)) for e in graph.edges] == [
            ("p0.out", "p1.in"),
            ("p1.out", "p2.in"),
        ]

    def test_a_generated_edge_records_the_rule_that_made_it(self) -> None:
        graph = resolve(coupled(SpatialRuleKind.VERTICAL_STACK))
        assert graph.edges[0].origin.rule == "r"
        assert str(graph.edges[0].origin) == "m/via r"

    def test_the_lowered_graph_is_a_dag_the_backends_can_walk(self) -> None:
        graph = resolve(coupled(SpatialRuleKind.VERTICAL_STACK))
        assert [n.id for n in graph.topological_order()] == ["p0", "p1", "p2"]

    def test_a_rule_over_a_generator_wires_its_instances(self) -> None:
        document = tower(3, chain=False, axis=Axis.Z, step=1.0)
        root = document.module("root")
        assert root is not None
        rules = (
            SpatialRule(
                id="stackup",
                kind=SpatialRuleKind.VERTICAL_STACK,
                members=("stack",),
                axis=Axis.Z,
                input_port="in",
                output_port="out",
            ),
        )
        updated = root.model_copy(update={"spatial_rules": rules})
        graph = resolve(
            document.model_copy(update={"modules": (document.module("block"), updated)})
        )
        assert [(str(e.src), str(e.dst)) for e in graph.edges] == [
            ("stack-0/fc.out", "stack-1/fc.in"),
            ("stack-1/fc.out", "stack-2/fc.in"),
        ]

    def test_a_rule_naming_something_that_is_not_there_is_refused(self) -> None:
        module = Module(
            id="m",
            nodes=(Node(id="p0", op="ntb.relu"), Node(id="p1", op="ntb.relu")),
            spatial_rules=(
                SpatialRule(id="r", kind=SpatialRuleKind.VERTICAL_STACK, members=("p0", "ghost")),
            ),
        )
        with pytest.raises(ResolveError, match="'ghost', which is neither a node nor a generator"):
            resolve(Document(root="m", modules=(module,)))

    def test_a_rule_naming_a_port_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ResolveError, match="has no input port 'nope'"):
            resolve(coupled(SpatialRuleKind.VERTICAL_STACK, input_port="nope"))

    def test_a_rule_may_share_a_module_with_explicit_edges(self) -> None:
        document = coupled(SpatialRuleKind.VERTICAL_STACK)
        module = document.root_module
        extra = Node(id="tail", op="ntb.relu", placement=Placement(pos=(9.0, 0.0, 0.0)))
        updated = module.model_copy(
            update={
                "nodes": (*module.nodes, extra),
                "edges": (
                    Edge(id="e", src=Endpoint(node="p2"), dst=Endpoint(node="tail", port="in")),
                ),
            }
        )
        graph = resolve(document.model_copy(update={"modules": (updated,)}))
        assert len(graph.edges) == 3
        assert str(graph.outputs[0].endpoint) == "tail.out"


class TestShippedExample:
    def test_the_tower_resolves_to_twenty_four_nodes(self) -> None:
        graph = resolve(io.load(EXAMPLES / "vertical_tower.ntb"))
        assert len(graph.nodes) == 24
        assert len(graph.edges) == 23
        graph.topological_order()

    def test_the_tower_types_end_to_end(self) -> None:
        from ntb.shapes import infer_shapes

        report = infer_shapes(resolve(io.load(EXAMPLES / "vertical_tower.ntb")))
        assert report.ok
        assert str(report.type_of("stack-11/act", "out")) == "float32[batch, 256]"


class TestPreview:
    """What the studio draws: authored blocks, generated ones, derived links."""

    def test_a_generator_shows_as_one_block_per_repetition(self) -> None:
        result = preview(io.load(EXAMPLES / "vertical_tower.ntb"))
        assert len(result.blocks) == 12
        assert {b.kind for b in result.blocks} == {BlockKind.GENERATED}
        assert result.blocks[3].pos == (0.0, 0.0, 3.0)
        assert result.blocks[3].index == 3

    def test_chaining_shows_as_links_between_repetitions(self) -> None:
        result = preview(io.load(EXAMPLES / "vertical_tower.ntb"))
        assert [link.kind for link in result.links] == [LinkKind.CHAIN] * 11
        assert result.links[0].source == "stack"

    def test_a_rule_shows_which_edges_it_derived(self) -> None:
        result = preview(io.load(EXAMPLES / "lattice_3d.ntb"))
        assert len(result.blocks) == 16
        assert all(link.kind is LinkKind.RULE for link in result.links)
        assert {link.source for link in result.links} == {"grid"}

    def test_authored_nodes_and_edges_come_through_as_drawn(self) -> None:
        result = preview(io.load(EXAMPLES / "mlp.ntb"))
        assert {b.kind for b in result.blocks} == {BlockKind.NODE}
        assert all(link.kind is LinkKind.EDGE for link in result.links)

    def test_a_rule_that_cannot_apply_is_reported_not_raised(self) -> None:
        # The studio has to keep drawing a document that is mid-edit.
        module = Module(
            id="m",
            nodes=(
                Node(id="a", op="ntb.relu", placement=Placement(pos=(0.0, 0.0, 0.0))),
                Node(id="b", op="ntb.relu", placement=Placement(pos=(0.0, 0.0, 0.0))),
            ),
            spatial_rules=(SpatialRule(id="r", kind=SpatialRuleKind.LATTICE, members=("a", "b")),),
        )
        result = preview(Document(root="m", modules=(module,)))
        assert result.links == ()
        assert "same point" in result.problems[0]

    def test_a_rule_with_too_few_blocks_is_reported(self) -> None:
        module = Module(
            id="m",
            nodes=(Node(id="a", op="ntb.relu"),),
            spatial_rules=(
                SpatialRule(id="r", kind=SpatialRuleKind.VERTICAL_STACK, members=("a",)),
            ),
        )
        assert "fewer than two" in preview(Document(root="m", modules=(module,))).problems[0]

    def test_an_unknown_module_previews_as_nothing(self) -> None:
        assert preview(io.load(EXAMPLES / "mlp.ntb"), "ghost").blocks == ()
