"""The studio server: sessions, the HTTP surface and the WebSocket."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ntb.commands import AddNode, CommandError, dump_command
from ntb.ir import Node, io
from ntb.server import Session, blank_document, describe_op, op_catalog
from tests.conftest import EXAMPLES

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ntb.server.app import _origin_allowed, create_app  # noqa: E402


@pytest.fixture
def session() -> Session:
    return Session(io.load(EXAMPLES / "mlp.ntb"))


@pytest.fixture
def client(session: Session) -> TestClient:
    return TestClient(create_app(session))


def add_relu(node_id: str = "extra") -> dict[str, Any]:
    return dump_command(AddNode(module="mlp", node=Node(id=node_id, op="ntb.relu")))


class TestSession:
    def test_a_new_session_starts_from_an_empty_model(self) -> None:
        assert Session().document.root_module.nodes == ()

    def test_editing_bumps_the_revision(self, session: Session) -> None:
        before = session.revision
        session.apply(AddNode(module="mlp", node=Node(id="extra", op="ntb.relu")))
        assert session.revision == before + 1

    def test_derived_state_carries_diagnostics_types_and_code(self, session: Session) -> None:
        derived = session.derived()
        assert derived.diagnostics == ()
        assert derived.types["fc1.out"]
        assert "class" in derived.code

    def test_derived_state_is_cached_per_revision(self, session: Session) -> None:
        assert session.derived() is session.derived()
        session.apply(AddNode(module="mlp", node=Node(id="extra", op="ntb.relu")))
        assert session.derived().diagnostics  # the new node is unconnected

    def test_a_broken_document_reports_instead_of_raising(self, session: Session) -> None:
        session.apply(AddNode(module="mlp", node=Node(id="oops", op="ntb.nonesuch")))
        derived = session.derived()
        assert any(d["code"] == "unknown-op" for d in derived.diagnostics)
        assert derived.code == ""
        assert derived.code_error

    def test_a_document_with_generators_reports_every_instance(self) -> None:
        session = Session(io.load(EXAMPLES / "vertical_tower.ntb"))
        types = session.derived().types
        assert types["stack-0/fc.out"] == types["stack-11/fc.out"] == "float32[batch, 256]"

    def test_saving_and_reopening_round_trips(self, session: Session, tmp_path: Path) -> None:
        target = tmp_path / "saved.ntb"
        session.save(target)
        assert not session.dirty
        reopened = Session()
        reopened.open(target)
        assert io.dumps(reopened.document) == io.dumps(session.document)

    def test_saving_without_a_path_is_refused(self, session: Session) -> None:
        with pytest.raises(ValueError, match="no file yet"):
            session.save()

    def test_editing_makes_the_session_dirty(self, session: Session, tmp_path: Path) -> None:
        session.save(tmp_path / "saved.ntb")
        session.apply(AddNode(module="mlp", node=Node(id="extra", op="ntb.relu")))
        assert session.dirty

    def test_undo_and_redo_walk_the_session(self, session: Session) -> None:
        before = io.dumps(session.document)
        session.apply(AddNode(module="mlp", node=Node(id="extra", op="ntb.relu")))
        session.undo()
        assert io.dumps(session.document) == before
        session.redo()
        assert session.document.root_module.node("extra") is not None

    def test_a_blank_document_names_its_root(self) -> None:
        assert blank_document("draft").root_module.id == "model"


class TestSpatialState:
    """The studio cannot draw a generator or a rule without this."""

    def test_a_generated_block_reaches_the_client(self) -> None:
        session = Session(io.load(EXAMPLES / "lattice_3d.ntb"))
        derived = session.derived().as_json()
        assert len(derived["blocks"]) == 16
        assert derived["blocks"][0]["kind"] == "generated"
        assert {link["kind"] for link in derived["links"]} == {"rule"}

    def test_editing_a_generator_rebuilds_the_topology(self) -> None:
        from ntb.commands import UpdateGenerator

        session = Session(io.load(EXAMPLES / "lattice_3d.ntb"))
        before = len(session.derived().layout.links)
        generator = session.document.root_module.generators[0]
        taller = generator.model_copy(update={"count": 6})
        session.apply(UpdateGenerator(module="lattice", generator=taller))
        assert len(session.derived().layout.links) > before

    def test_a_broken_cell_says_which_repetition(self) -> None:
        from ntb.commands import UpdateGenerator

        session = Session(io.load(EXAMPLES / "lattice_3d.ntb"))
        stray = session.document.root_module.generators[3].model_copy(
            update={"origin": (12.0, 12.0, 0.0)}
        )
        session.apply(UpdateGenerator(module="lattice", generator=stray))
        diagnostics = session.derived().diagnostics
        assert diagnostics
        assert diagnostics[0]["block"] == "col3-0"
        assert "col3-0/mix" in diagnostics[0]["text"]


class TestCatalog:
    def test_every_registered_op_is_offered(self) -> None:
        from ntb.ops import REGISTRY

        assert len(op_catalog()) == len(REGISTRY)

    def test_an_op_describes_its_ports_and_attributes(self) -> None:
        from ntb.ops import REGISTRY

        described = describe_op(REGISTRY.require("ntb.linear"))
        assert [p["name"] for p in described["inputs"]] == ["in"]
        assert {a["name"] for a in described["attrs"]} >= {"in_features", "out_features"}
        assert "torch" in described["backends"]


class TestHttp:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/api/health").json()["status"] == "ok"

    def test_the_palette_comes_from_the_registry(self, client: TestClient) -> None:
        names = {op["name"] for op in client.get("/api/ops").json()}
        assert "ntb.conv3d" in names

    def test_state_carries_the_document(self, client: TestClient) -> None:
        state = client.get("/api/state").json()
        assert state["document"]["root"] == "mlp"
        assert state["derived"]["code"]

    def test_a_command_edits_the_document(self, client: TestClient) -> None:
        state = client.post("/api/command", json=add_relu()).json()
        assert any(n["id"] == "extra" for n in state["document"]["modules"][0]["nodes"])

    def test_a_bad_command_is_a_client_error(self, client: TestClient) -> None:
        response = client.post("/api/command", json=add_relu("fc1"))
        assert response.status_code == 400
        assert "already has a node" in response.json()["detail"]

    def test_an_unknown_command_is_refused(self, client: TestClient) -> None:
        assert client.post("/api/command", json={"kind": "rm_rf"}).status_code == 400

    def test_a_foreign_origin_is_refused(self, client: TestClient) -> None:
        response = client.get("/api/state", headers={"origin": "https://evil.example"})
        assert response.status_code == 403

    def test_the_placeholder_page_explains_the_missing_build(
        self, session: Session, tmp_path: Path
    ) -> None:
        # A source checkout has no bundle; the server has to say so rather than 404.
        with TestClient(create_app(session, static_dir=tmp_path)) as bare:
            assert "not built" in bare.get("/").text

    def test_a_built_frontend_is_served(self, session: Session, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("<h1>studio</h1>", encoding="utf-8")
        with TestClient(create_app(session, static_dir=tmp_path)) as built:
            assert "studio" in built.get("/").text


class TestOrigins:
    @pytest.mark.parametrize(
        ("origin", "allowed"),
        [
            (None, True),
            ("http://127.0.0.1:8756", True),
            ("http://localhost:5173", True),
            ("https://evil.example", False),
            ("http://127.0.0.1.evil.example", False),
        ],
    )
    def test_only_local_pages_may_drive_the_server(self, origin: str | None, allowed: bool) -> None:
        assert _origin_allowed(origin, "127.0.0.1:8756", ()) is allowed


class TestWebSocket:
    def test_the_state_arrives_on_connect(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as socket:
            message = socket.receive_json()
        assert message["type"] == "state"
        assert message["document"]["root"] == "mlp"

    def test_a_command_comes_back_as_state(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "command", "command": add_relu()})
            message = socket.receive_json()
        assert message["revision"] == 1
        assert any(n["id"] == "extra" for n in message["document"]["modules"][0]["nodes"])

    def test_undo_and_redo_travel_over_the_socket(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "command", "command": add_relu()})
            socket.receive_json()
            socket.send_json({"type": "undo"})
            assert not socket.receive_json()["derived"]["diagnostics"]
            socket.send_json({"type": "redo"})
            assert socket.receive_json()["derived"]["diagnostics"]

    def test_a_refused_command_is_an_error_not_a_disconnect(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "command", "command": add_relu("fc1")})
            error = socket.receive_json()
            assert error["type"] == "error"
            socket.send_json({"type": "refresh"})
            assert socket.receive_json()["type"] == "state"

    def test_an_unknown_message_is_an_error(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "launch_missiles"})
            assert socket.receive_json()["type"] == "error"

    def test_save_and_open_travel_over_the_socket(self, client: TestClient, tmp_path: Path) -> None:
        target = tmp_path / "from-ws.ntb"
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "save", "path": str(target)})
            assert socket.receive_json()["dirty"] is False
            socket.send_json({"type": "new"})
            assert socket.receive_json()["document"]["modules"][0]["nodes"] == []
            socket.send_json({"type": "open", "path": str(target)})
            assert socket.receive_json()["document"]["root"] == "mlp"

    def test_opening_a_missing_file_is_an_error(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "open", "path": "no/such/file.ntb"})
            assert socket.receive_json()["type"] == "error"

    def test_every_client_sees_the_edit(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
            first.receive_json()
            second.receive_json()
            first.send_json({"type": "command", "command": add_relu()})
            assert first.receive_json()["revision"] == 1
            assert second.receive_json()["revision"] == 1


class TestProtocol:
    def test_a_malformed_message_is_reported(self) -> None:
        from ntb.server.protocol import parse_message

        with pytest.raises(CommandError, match="not a valid message"):
            parse_message({"type": "open"})
