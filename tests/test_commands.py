"""The command bus: what each command does, and that every one of them undoes.

The round-trip property is the important test here. If applying a command and
then its inverse does not reproduce the original document byte for byte, undo
is silently lossy and no amount of UI polish fixes it.
"""

from __future__ import annotations

import pytest

from ntb.commands import (
    AddGenerator,
    AddModule,
    AddNode,
    AddRule,
    AnyCommand,
    Batch,
    CommandError,
    Connect,
    Disconnect,
    History,
    MoveNode,
    RemoveGenerator,
    RemoveModule,
    RemoveNode,
    RemoveRule,
    RenameNode,
    SetAttrs,
    SetMetadata,
    SetModulePorts,
    SetRoot,
    UpdateGenerator,
    UpdateRule,
    apply_all,
    apply_command,
    dump_command,
    parse_command,
)
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
from tests.conftest import EXAMPLES


def base() -> Document:
    """A two-node module: the smallest document that can lose an edge."""
    module = Module(
        id="m",
        inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=("batch", 8))),),
        outputs=(Port(name="y", direction=PortDirection.OUT),),
        nodes=(
            Node(id="fc1", op="ntb.linear", attrs={"in_features": 8, "out_features": 4}),
            Node(id="act", op="ntb.relu"),
            Node(id="fc2", op="ntb.linear", attrs={"in_features": 4, "out_features": 2}),
        ),
        edges=(
            Edge(id="e1", src=Endpoint(node="fc1"), dst=Endpoint(node="act", port="in")),
            Edge(id="e2", src=Endpoint(node="act"), dst=Endpoint(node="fc2", port="in")),
        ),
    )
    return Document(name="base", root="m", modules=(module,))


def commands() -> list[AnyCommand]:
    """One of every command, each valid against ``base()``."""
    return [
        AddNode(module="m", node=Node(id="fc3", op="ntb.relu")),
        AddNode(module="m", node=Node(id="fc3", op="ntb.relu"), index=0),
        RemoveNode(module="m", node="act"),
        RemoveNode(module="m", node="fc1"),
        MoveNode(module="m", node="act", placement=Placement(pos=(1.0, 2.0, 3.0))),
        SetAttrs(module="m", node="fc1", attrs={"in_features": 8, "out_features": 16}),
        RenameNode(module="m", node="fc1", name="first layer"),
        Connect(
            module="m",
            edge=Edge(id="e3", src=Endpoint(node="fc1"), dst=Endpoint(node="fc2", port="in")),
        ),
        Disconnect(module="m", edge="e1"),
        SetModulePorts(
            module="m",
            inputs=(Port(name="a", direction=PortDirection.IN),),
            outputs=(Port(name="b", direction=PortDirection.OUT),),
        ),
        AddModule(module=Module(id="block", nodes=(Node(id="n", op="ntb.relu"),))),
        SetRoot(root="m"),
        SetMetadata(name="renamed", doc="hello", metadata={"author": "someone"}),
        AddGenerator(module="m", generator=Generator(id="g", module="m", count=3)),
        AddRule(
            module="m",
            rule=SpatialRule(id="r", kind=SpatialRuleKind.VERTICAL_STACK, members=("fc1", "fc2")),
        ),
        Batch(
            commands=(
                AddNode(module="m", node=Node(id="tail", op="ntb.relu")),
                Connect(
                    module="m",
                    edge=Edge(
                        id="e9", src=Endpoint(node="fc2"), dst=Endpoint(node="tail", port="in")
                    ),
                ),
            ),
            label="append a tail",
        ),
    ]


class TestUndo:
    @pytest.mark.parametrize("command", commands(), ids=lambda c: c.kind)
    def test_a_command_and_its_inverse_restore_the_document(self, command: AnyCommand) -> None:
        document = base()
        result = apply_command(document, command)
        restored = apply_command(result.document, result.inverse).document
        assert io.dumps(restored) == io.dumps(document)

    @pytest.mark.parametrize("name", ["mlp", "transformer_block", "cnn3d"])
    def test_removing_any_node_of_an_example_undoes_exactly(self, name: str) -> None:
        # Real documents have edges in orders a hand-written fixture would not.
        document = io.load(EXAMPLES / f"{name}.ntb")
        for node in document.root_module.nodes:
            result = apply_command(document, RemoveNode(module=document.root, node=node.id))
            restored = apply_command(result.document, result.inverse).document
            assert io.dumps(restored) == io.dumps(document), node.id


class TestEditing:
    def test_adding_a_node_appends_it(self) -> None:
        result = apply_command(base(), AddNode(module="m", node=Node(id="new", op="ntb.relu")))
        assert [n.id for n in result.document.root_module.nodes] == ["fc1", "act", "fc2", "new"]

    def test_an_index_places_the_node(self) -> None:
        command = AddNode(module="m", node=Node(id="new", op="ntb.relu"), index=1)
        result = apply_command(base(), command)
        assert [n.id for n in result.document.root_module.nodes] == ["fc1", "new", "act", "fc2"]

    def test_removing_a_node_takes_its_edges_with_it(self) -> None:
        result = apply_command(base(), RemoveNode(module="m", node="act"))
        assert result.document.root_module.edges == ()

    def test_moving_a_node_changes_the_model(self) -> None:
        # Placement is semantic (ADR 0002), so this is an edit, not a view change.
        placement = Placement(pos=(0.0, 0.0, 4.0))
        result = apply_command(base(), MoveNode(module="m", node="act", placement=placement))
        assert result.document.root_module.node("act").placement.pos == (0.0, 0.0, 4.0)  # type: ignore[union-attr]

    def test_setting_attributes_replaces_them(self) -> None:
        result = apply_command(base(), SetAttrs(module="m", node="fc1", attrs={"out_features": 3}))
        assert result.document.root_module.node("fc1").attrs == {"out_features": 3}  # type: ignore[union-attr]

    def test_the_original_document_is_untouched(self) -> None:
        document = base()
        apply_command(document, RemoveNode(module="m", node="act"))
        assert document.root_module.node("act") is not None


class TestSpatialCommands:
    def spatial(self) -> Document:
        document = apply_command(
            base(), AddGenerator(module="m", generator=Generator(id="g", module="m", count=3))
        ).document
        return apply_command(
            document,
            AddRule(
                module="m",
                rule=SpatialRule(
                    id="r", kind=SpatialRuleKind.VERTICAL_STACK, members=("fc1", "fc2")
                ),
            ),
        ).document

    def test_a_generator_can_be_retuned_in_place(self) -> None:
        document = self.spatial()
        wider = Generator(id="g", module="m", count=12, axis=Axis.Z, step=2.0)
        result = apply_command(document, UpdateGenerator(module="m", generator=wider))
        assert result.document.root_module.generators[0].count == 12
        restored = apply_command(result.document, result.inverse).document
        assert io.dumps(restored) == io.dumps(document)

    def test_a_rule_can_be_retuned_in_place(self) -> None:
        document = self.spatial()
        wider = SpatialRule(
            id="r", kind=SpatialRuleKind.NEIGHBORHOOD, members=("fc1", "fc2"), radius=2.0
        )
        result = apply_command(document, UpdateRule(module="m", rule=wider))
        assert result.document.root_module.spatial_rules[0].radius == 2.0
        restored = apply_command(result.document, result.inverse).document
        assert io.dumps(restored) == io.dumps(document)

    def test_removing_a_generator_undoes_exactly(self) -> None:
        document = self.spatial()
        result = apply_command(document, RemoveGenerator(module="m", generator="g"))
        assert result.document.root_module.generators == ()
        restored = apply_command(result.document, result.inverse).document
        assert io.dumps(restored) == io.dumps(document)

    def test_removing_a_rule_undoes_exactly(self) -> None:
        document = self.spatial()
        result = apply_command(document, RemoveRule(module="m", rule="r"))
        assert result.document.root_module.spatial_rules == ()
        restored = apply_command(result.document, result.inverse).document
        assert io.dumps(restored) == io.dumps(document)

    def test_editing_something_that_is_not_there_is_refused(self) -> None:
        ghost = Generator(id="ghost", module="m", count=2)
        with pytest.raises(CommandError, match="has no generator 'ghost'"):
            apply_command(base(), UpdateGenerator(module="m", generator=ghost))

    def test_a_duplicate_generator_is_refused(self) -> None:
        with pytest.raises(CommandError, match="already has a generator 'g'"):
            apply_command(
                self.spatial(),
                AddGenerator(module="m", generator=Generator(id="g", module="m", count=2)),
            )

    def test_a_duplicate_rule_is_refused(self) -> None:
        rule = SpatialRule(id="r", kind=SpatialRuleKind.LATTICE, members=("fc1", "fc2"))
        with pytest.raises(CommandError, match="already has a rule 'r'"):
            apply_command(self.spatial(), AddRule(module="m", rule=rule))


class TestRefusals:
    def test_an_unknown_module_is_refused(self) -> None:
        with pytest.raises(CommandError, match="no module 'nope'"):
            apply_command(base(), RemoveNode(module="nope", node="act"))

    def test_an_unknown_node_is_refused(self) -> None:
        with pytest.raises(CommandError, match="has no node 'nope'"):
            apply_command(base(), RemoveNode(module="m", node="nope"))

    def test_a_duplicate_node_id_is_refused(self) -> None:
        with pytest.raises(CommandError, match="already has a node 'fc1'"):
            apply_command(base(), AddNode(module="m", node=Node(id="fc1", op="ntb.relu")))

    def test_a_duplicate_edge_id_is_refused(self) -> None:
        edge = Edge(id="e1", src=Endpoint(node="fc1"), dst=Endpoint(node="fc2", port="in"))
        with pytest.raises(CommandError, match="already has an edge 'e1'"):
            apply_command(base(), Connect(module="m", edge=edge))

    def test_connecting_a_node_that_is_not_there_is_refused(self) -> None:
        edge = Edge(id="e9", src=Endpoint(node="ghost"), dst=Endpoint(node="fc2", port="in"))
        with pytest.raises(CommandError, match="has no node 'ghost'"):
            apply_command(base(), Connect(module="m", edge=edge))

    def test_removing_an_absent_edge_is_refused(self) -> None:
        with pytest.raises(CommandError, match="has no edge 'e9'"):
            apply_command(base(), Disconnect(module="m", edge="e9"))

    def test_the_root_module_cannot_be_removed(self) -> None:
        with pytest.raises(CommandError, match="is the root"):
            apply_command(base(), RemoveModule(module="m"))

    def test_a_module_a_generator_uses_cannot_be_removed(self) -> None:
        document = io.load(EXAMPLES / "vertical_tower.ntb")
        generator = document.root_module.generators[0]
        with pytest.raises(CommandError, match="instantiated by generator"):
            apply_command(document, RemoveModule(module=generator.module))

    def test_an_unknown_root_is_refused(self) -> None:
        with pytest.raises(CommandError, match="no module 'ghost'"):
            apply_command(base(), SetRoot(root="ghost"))

    def test_a_port_facing_the_wrong_way_is_refused(self) -> None:
        command = SetModulePorts(module="m", inputs=(Port(name="a", direction=PortDirection.OUT),))
        with pytest.raises(CommandError, match="is not an in port"):
            apply_command(base(), command)

    def test_a_failed_batch_leaves_the_document_alone(self) -> None:
        document = base()
        batch = Batch(
            commands=(
                AddNode(module="m", node=Node(id="ok", op="ntb.relu")),
                RemoveNode(module="m", node="ghost"),
            )
        )
        with pytest.raises(CommandError):
            apply_command(document, batch)
        assert document.root_module.node("ok") is None


class TestWire:
    @pytest.mark.parametrize("command", commands(), ids=lambda c: c.kind)
    def test_a_command_survives_a_round_trip_through_json(self, command: AnyCommand) -> None:
        assert parse_command(dump_command(command)) == command

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(CommandError, match="not a valid command"):
            parse_command({"kind": "drop_database"})

    def test_a_malformed_command_is_refused(self) -> None:
        with pytest.raises(CommandError, match="not a valid command"):
            parse_command({"kind": "remove_node", "module": "m"})


class TestHistory:
    def test_undo_walks_back_through_a_whole_session(self) -> None:
        session: list[AnyCommand] = [
            AddNode(module="m", node=Node(id="fc3", op="ntb.relu")),
            Connect(
                module="m",
                edge=Edge(id="e3", src=Endpoint(node="fc2"), dst=Endpoint(node="fc3", port="in")),
            ),
            MoveNode(module="m", node="fc3", placement=Placement(pos=(0.0, 0.0, 2.0))),
            SetAttrs(module="m", node="fc1", attrs={"in_features": 8, "out_features": 32}),
            RemoveNode(module="m", node="act"),
            SetMetadata(name="session"),
        ]
        history = History(base())
        before = io.dumps(history.document)
        for command in session:
            history.do(command)
        for _ in session:
            history.undo()
        assert io.dumps(history.document) == before

    def test_redo_replays_what_undo_took_back(self) -> None:
        history = History(base())
        history.do(RemoveNode(module="m", node="act"))
        after = io.dumps(history.document)
        history.undo()
        history.redo()
        assert io.dumps(history.document) == after

    def test_a_new_edit_drops_the_redo_stack(self) -> None:
        history = History(base())
        history.do(RemoveNode(module="m", node="act"))
        history.undo()
        history.do(RenameNode(module="m", node="fc1", name="x"))
        assert not history.can_redo

    def test_nothing_to_undo_says_so(self) -> None:
        with pytest.raises(CommandError, match="nothing to undo"):
            History(base()).undo()

    def test_nothing_to_redo_says_so(self) -> None:
        with pytest.raises(CommandError, match="nothing to redo"):
            History(base()).redo()

    def test_a_failed_command_stays_out_of_the_history(self) -> None:
        history = History(base())
        with pytest.raises(CommandError):
            history.do(RemoveNode(module="m", node="ghost"))
        assert not history.can_undo

    def test_the_stack_is_bounded(self) -> None:
        history = History(base(), limit=2)
        for index in range(5):
            history.do(RenameNode(module="m", node="fc1", name=f"n{index}"))
        history.undo()
        history.undo()
        with pytest.raises(CommandError):
            history.undo()

    def test_opening_a_document_clears_the_history(self) -> None:
        history = History(base())
        history.do(RemoveNode(module="m", node="act"))
        history.reset(io.load(EXAMPLES / "mlp.ntb"))
        assert not history.can_undo and not history.can_redo

    def test_a_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            History(base(), limit=0)


class TestApplyAll:
    def test_commands_apply_as_one_step(self) -> None:
        document = base()
        result = apply_all(
            document,
            (
                RemoveNode(module="m", node="act"),
                AddNode(module="m", node=Node(id="act2", op="ntb.gelu")),
            ),
        )
        assert result.document.root_module.node("act") is None
        restored = apply_command(result.document, result.inverse).document
        assert io.dumps(restored) == io.dumps(document)
