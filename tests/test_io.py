"""`.ntb` files must survive git, interrupted saves and version skew."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ntb.ir import Document, Edge, Endpoint, Module, Node, Placement, io
from ntb.ir.document import SCHEMA_VERSION
from ntb.ir.migrate import migrate

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def document() -> Document:
    module = Module(
        id="m",
        nodes=(
            Node(id="a", op="ntb.linear", attrs={"in_features": 4, "out_features": 8}),
            Node(id="b", op="ntb.relu", placement=Placement(pos=(1.0, 0.0, 0.0))),
        ),
        edges=(Edge(id="e", src=Endpoint(node="a"), dst=Endpoint(node="b", port="in")),),
    )
    return Document(name="doc", root="m", modules=(module,))


class TestCanonicalForm:
    def test_round_trip_is_byte_identical(self, document: Document) -> None:
        assert io.dumps(io.loads(io.dumps(document))) == io.dumps(document)

    def test_keys_are_sorted_so_diffs_stay_reviewable(self, document: Document) -> None:
        payload = json.loads(io.dumps(document))
        assert list(payload) == sorted(payload)

    def test_ends_with_exactly_one_newline(self, document: Document) -> None:
        text = io.dumps(document)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")

    def test_defaults_are_written_out(self, document: Document) -> None:
        # Defaults are serialised rather than omitted: a `.ntb` should read as a
        # complete description of the model, and a changed default in a later
        # NTB must not silently change what an existing file means.
        payload = json.loads(io.dumps(document))
        assert payload["modules"][0]["nodes"][0]["placement"]["pos"] == [0.0, 0.0, 0.0]


class TestLoad:
    def test_rejects_non_json(self) -> None:
        with pytest.raises(io.DocumentError, match="not valid JSON"):
            io.loads("{oops")

    def test_rejects_a_json_array(self) -> None:
        with pytest.raises(io.DocumentError, match="expected a JSON object"):
            io.loads("[]")

    def test_rejects_a_non_integer_schema_version(self) -> None:
        with pytest.raises(io.DocumentError, match="must be an integer"):
            io.loads('{"schema_version": "1", "root": "m", "modules": []}')

    def test_newer_schema_version_says_how_to_fix_it(self, document: Document) -> None:
        payload = json.loads(io.dumps(document))
        payload["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(io.DocumentError, match="pip install --upgrade ntb"):
            io.loads(json.dumps(payload))

    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(io.DocumentError, match="cannot read"):
            io.load(tmp_path / "absent.ntb")


class TestSave:
    def test_save_then_load(self, document: Document, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "model.ntb"
        io.save(document, path)
        assert io.load(path) == document

    def test_leaves_no_temporary_files_behind(self, document: Document, tmp_path: Path) -> None:
        io.save(document, tmp_path / "model.ntb")
        assert [p.name for p in tmp_path.iterdir()] == ["model.ntb"]

    def test_overwrite_is_atomic_in_place(self, document: Document, tmp_path: Path) -> None:
        path = tmp_path / "model.ntb"
        io.save(document, path)
        io.save(document.model_copy(update={"name": "renamed"}), path)
        assert io.load(path).name == "renamed"
        assert [p.name for p in tmp_path.iterdir()] == ["model.ntb"]


class TestMigrations:
    def test_current_version_is_a_no_op(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "root": "m"}
        assert migrate(dict(payload), from_version=SCHEMA_VERSION) == payload

    def test_a_missing_migration_is_an_explicit_failure(self) -> None:
        # Never silently accept a document we cannot actually understand.
        with pytest.raises(ValueError, match="no migration from schema v0"):
            migrate({"schema_version": 0}, from_version=0)


class TestShippedExamples:
    @pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.ntb")), ids=lambda p: p.name)
    def test_example_loads_and_is_canonically_formatted(self, path: Path) -> None:
        # Guards the docs as much as the code: a hand-edited example that no
        # longer round-trips would ship broken instructions to every new user.
        document = io.load(path)
        assert io.dumps(document) == path.read_text(encoding="utf-8")

    def test_the_tower_example_actually_exercises_the_spatial_layer(self) -> None:
        document = io.load(EXAMPLES / "vertical_tower.ntb")
        assert document.root_module.generators, "the tower example must use a Generator"
        lateral = document.module("lateral")
        assert lateral is not None and lateral.spatial_rules
