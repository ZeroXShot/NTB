"""The CLI surface: what a user hits first."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ntb import __version__
from ntb.cli import app

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"


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
