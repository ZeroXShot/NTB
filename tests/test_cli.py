"""The CLI surface: what a user hits first."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ntb import __version__
from ntb.cli import app
from tests.conftest import EXAMPLES, SCHEMA_DIR

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


def test_studio_refuses_a_file_that_is_not_there() -> None:
    # Nothing here starts a server: the path check happens first.
    result = runner.invoke(app, ["studio", "no/such/model.ntb"])
    assert result.exit_code == 1
    assert "no such file" in result.output


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


def test_shapes_follows_a_generator_through_its_instances() -> None:
    result = runner.invoke(app, ["shapes", str(EXAMPLES / "vertical_tower.ntb")])
    assert result.exit_code == 0
    assert "stack-11/act.out  float32[batch, 256]" in result.stdout


def test_validate_strict_turns_warnings_into_failure(tmp_path: Path) -> None:
    from ntb.ir import Document, Module, Node, Port, PortDirection, TensorType, io

    # A model with no output port: worth a warning, not worth refusing to open.
    module = Module(
        id="m",
        inputs=(Port(name="x", direction=PortDirection.IN, type=TensorType(shape=("batch", 8))),),
        nodes=(Node(id="a", op="ntb.relu"),),
    )
    path = tmp_path / "no-output.ntb"
    io.save(Document(root="m", modules=(module,)), path)
    assert runner.invoke(app, ["validate", str(path)]).exit_code == 0
    assert runner.invoke(app, ["validate", "--strict", str(path)]).exit_code == 1


def test_emit_writes_torch_to_a_file(tmp_path: Path) -> None:
    target = tmp_path / "generated" / "model.py"
    result = runner.invoke(app, ["emit", str(EXAMPLES / "mlp.ntb"), "--out", str(target)])
    assert result.exit_code == 0
    assert "class Mlp(torch.nn.Module):" in target.read_text(encoding="utf-8")


def test_emit_prints_to_stdout_by_default() -> None:
    result = runner.invoke(app, ["emit", str(EXAMPLES / "mlp.ntb")])
    assert result.exit_code == 0
    assert result.stdout.startswith("import torch")


def test_emit_writes_keras() -> None:
    result = runner.invoke(app, ["emit", str(EXAMPLES / "mlp.ntb"), "--backend", "keras"])
    assert result.exit_code == 0
    assert "keras.Model(" in result.stdout


def test_emit_refuses_a_backend_that_does_not_exist() -> None:
    result = runner.invoke(app, ["emit", str(EXAMPLES / "mlp.ntb"), "--backend", "jax"])
    assert result.exit_code == 2
    assert "torch, keras or onnx" in result.output


def test_emit_expands_a_generator_into_real_layers() -> None:
    result = runner.invoke(app, ["emit", str(EXAMPLES / "vertical_tower.ntb")])
    assert result.exit_code == 0
    assert result.stdout.count("torch.nn.Linear(") == 12


def test_emit_onnx_writes_a_model(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    target = tmp_path / "model.onnx"
    result = runner.invoke(
        app, ["emit", str(EXAMPLES / "mlp.ntb"), "--backend", "onnx", "--out", str(target)]
    )
    assert result.exit_code == 0
    assert target.stat().st_size > 0


def test_emit_onnx_without_out_explains_why() -> None:
    result = runner.invoke(app, ["emit", str(EXAMPLES / "mlp.ntb"), "--backend", "onnx"])
    assert result.exit_code == 2
    assert "--out" in result.output


def test_import_seeds_a_document_from_onnx(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from ntb.emit import export_onnx_document
    from ntb.ir import io

    model = tmp_path / "mlp.onnx"
    target = tmp_path / "imported.ntb"
    export_onnx_document(io.load(EXAMPLES / "mlp.ntb")).save(model)

    result = runner.invoke(app, ["import", str(model), "--out", str(target)])
    assert result.exit_code == 0
    assert "3 nodes" in result.output
    assert "weights are not imported" in result.output
    assert runner.invoke(app, ["validate", str(target)]).exit_code == 0


def test_import_refuses_a_file_that_is_not_there(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    result = runner.invoke(
        app, ["import", str(tmp_path / "ghost.onnx"), "--out", str(tmp_path / "x.ntb")]
    )
    assert result.exit_code == 1


def test_run_trains_and_reports(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import shutil

    document = tmp_path / "mlp.ntb"
    shutil.copy(EXAMPLES / "mlp.ntb", document)
    result = runner.invoke(
        app,
        [
            "run",
            str(document),
            "--epochs",
            "1",
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--loss",
            "cross_entropy",
            "--root",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "done after 2 steps" in result.output
    assert "loss" in result.output

    listed = runner.invoke(app, ["runs", "--root", str(tmp_path / "runs")])
    assert listed.exit_code == 0
    assert "done" in listed.output


def test_run_refuses_an_option_that_is_not_a_choice(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", str(EXAMPLES / "mlp.ntb"), "--loss", "hinge", "--root", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "hinge" in result.output


def test_runs_reports_an_id_that_is_not_there(tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "--show", "nope", "--root", str(tmp_path / "runs")])
    assert result.exit_code == 1
