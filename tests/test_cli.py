"""The CLI surface: what a user hits first."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import EXAMPLES, SCHEMA_DIR
from typer.testing import CliRunner

from ntb import __version__
from ntb.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_ops_lists_every_registered_op() -> None:
    result = runner.invoke(app, ["ops"])
    assert result.exit_code == 0
    for name in ("ntb.linear", "ntb.relu", "ntb.conv2d"):
        assert name in result.stdout


def test_ops_filters_by_category() -> None:
    result = runner.invoke(app, ["ops", "--category", "activation"])
    assert result.exit_code == 0
    assert "ntb.relu" in result.stdout
    assert "ntb.conv2d" not in result.stdout


def test_info_summarises_an_example() -> None:
    result = runner.invoke(app, ["info", str(EXAMPLES / "vertical_tower.ntb")])
    assert result.exit_code == 0
    assert "1 generators" in result.stdout


def test_info_on_a_bad_path_exits_nonzero() -> None:
    result = runner.invoke(app, ["info", "does-not-exist.ntb"])
    assert result.exit_code == 1


def test_validate_accepts_the_examples() -> None:
    result = runner.invoke(app, ["validate", str(EXAMPLES / "mlp.ntb")])
    assert result.exit_code == 0


def test_checked_in_schema_matches_the_models() -> None:
    # The frontend's TypeScript types are generated from this file. If it drifts
    # from the pydantic models, the studio starts lying about the IR.
    result = runner.invoke(app, ["schema", "--check", "--directory", str(SCHEMA_DIR)])
    assert result.exit_code == 0, "run `ntb schema --write`"


def test_unimplemented_commands_say_which_phase_they_land_in() -> None:
    for command, phase in (("emit", "2"), ("studio", "3"), ("run", "6")):
        result = runner.invoke(app, [command])
        assert result.exit_code == 2
        assert f"phase {phase}" in result.output


def test_validate_reports_a_shape_error_and_exits_nonzero(tmp_path: Path) -> None:
    from ntb.ir import Document, Module, Node, Port, PortDirection, TensorType, io

    module = Module(
        id="m",
        inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=(1, 512))),),
        nodes=(Node(id="fc", op="ntb.linear", attrs={"in_features": 256, "out_features": 4}),),
    )
    path = tmp_path / "bad.ntb"
    io.save(Document(root="m", modules=(module,)), path)

    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "in_features is 256" in result.output


def test_shapes_prints_every_port() -> None:
    result = runner.invoke(app, ["shapes", str(EXAMPLES / "cnn3d.ntb")])
    assert result.exit_code == 0
    assert "head.out  float32[batch, 10]" in result.stdout


def test_shapes_refuses_a_document_it_cannot_lower_yet() -> None:
    result = runner.invoke(app, ["shapes", str(EXAMPLES / "vertical_tower.ntb")])
    assert result.exit_code == 1
    assert "phase 4" in result.output


def test_validate_strict_turns_warnings_into_failure() -> None:
    lenient = runner.invoke(app, ["validate", str(EXAMPLES / "vertical_tower.ntb")])
    strict = runner.invoke(app, ["validate", "--strict", str(EXAMPLES / "vertical_tower.ntb")])
    assert lenient.exit_code == 0
    assert strict.exit_code == 1
