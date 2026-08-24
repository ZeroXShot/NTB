"""Training runs: the record, the manager, and the worker itself.

The worker is a real subprocess (ADR 11), so the tests that exercise it are
marked `torch` and are slower than the rest. Everything up to launching one is
plain Python and runs everywhere.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ntb.runs import (
    DataSource,
    Loss,
    Optimiser,
    RunConfig,
    RunError,
    RunManager,
    RunStore,
    Status,
)
from tests.conftest import EXAMPLES


@pytest.fixture
def document(tmp_path: Path) -> Path:
    """A copy of the MLP example, so checkpoints land in the temp directory."""
    target = tmp_path / "mlp.ntb"
    shutil.copy(EXAMPLES / "mlp.ntb", target)
    return target


def quick(document: Path, **overrides: Any) -> RunConfig:
    """A run short enough to wait for."""
    settings: dict[str, Any] = {
        "document": document,
        "epochs": 1,
        "steps_per_epoch": 3,
        "batch_size": 4,
        "loss": Loss.CROSS_ENTROPY,
    }
    return RunConfig(**{**settings, **overrides})


class TestConfig:
    def test_the_total_is_epochs_times_steps(self, document: Path) -> None:
        assert quick(document, epochs=4, steps_per_epoch=25).total_steps() == 100

    def test_it_refuses_a_learning_rate_of_zero(self, document: Path) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            quick(document, learning_rate=0.0)

    def test_the_options_are_a_closed_set(self) -> None:
        assert set(Optimiser) == {Optimiser.SGD, Optimiser.ADAM, Optimiser.ADAMW}
        assert set(DataSource) == {DataSource.SYNTHETIC, DataSource.SCRIPT}


class TestStore:
    def test_a_run_is_recorded_and_read_back(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.db")
        store.create("abc", "model.ntb", {"epochs": 1})
        record = store.get("abc")
        assert record is not None
        assert record.status is Status.RUNNING
        assert record.config == {"epochs": 1}

    def test_metrics_come_back_in_step_order(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.db")
        store.create("abc", "model.ntb", {})
        for step in (3, 1, 2):
            store.record("abc", step, 0, "loss", float(step), 0.0)
        assert [row["step"] for row in store.metrics("abc")] == [1, 2, 3]
        assert store.get("abc").last_step == 2  # type: ignore[union-attr]

    def test_finishing_stamps_the_status_and_the_time(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.db")
        store.create("abc", "model.ntb", {})
        store.finish("abc", Status.FAILED, "it broke")
        record = store.get("abc")
        assert record is not None
        assert record.status is Status.FAILED
        assert record.error == "it broke" and record.ended_at is not None

    def test_a_run_survives_the_process_that_wrote_it(self, tmp_path: Path) -> None:
        first = RunStore(tmp_path / "runs.db")
        first.create("abc", "model.ntb", {})
        first.record("abc", 1, 0, "loss", 0.5, 0.1)
        first.close()

        second = RunStore(tmp_path / "runs.db")
        assert second.metrics("abc")[0]["value"] == 0.5


class TestManagerWithoutTorch:
    def test_a_document_that_is_not_there_is_refused(self, tmp_path: Path) -> None:
        manager = RunManager(tmp_path / "runs")
        with pytest.raises(RunError, match="no such document"):
            manager.start(quick(tmp_path / "ghost.ntb"))
        manager.close()

    def test_resuming_something_that_never_ran_is_refused(self, tmp_path: Path) -> None:
        manager = RunManager(tmp_path / "runs")
        with pytest.raises(RunError, match="no run 'nope'"):
            manager.resume("nope")
        manager.close()

    def test_a_run_left_running_by_a_dead_session_is_marked_stopped(self, tmp_path: Path) -> None:
        first = RunManager(tmp_path / "runs")
        first.store.create("orphan", "model.ntb", {})
        first.store.close()

        second = RunManager(tmp_path / "runs")
        record = second.get("orphan")
        assert record is not None
        assert record.status is Status.STOPPED
        assert record.error is not None and "session" in record.error
        second.close()


@pytest.mark.torch
class TestTraining:
    """The whole thing: a subprocess trains a model NTB generated."""

    def test_a_run_trains_and_records_its_loss(self, document: Path, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        events: list[str] = []
        manager = RunManager(tmp_path / "runs", listener=lambda _, e: events.append(e["event"]))

        started = manager.start(quick(document))
        final = manager.wait(started.id, timeout=300)

        assert final.status is Status.DONE, final.error
        assert final.last_step == 3
        assert final.parameters == 101770
        assert [row["step"] for row in manager.metrics(started.id)] == [1, 2, 3]
        assert events[0] == "started" and "finished" in events
        manager.close()

    def test_a_checkpoint_is_written_and_can_be_resumed_from(
        self, document: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("torch")
        manager = RunManager(tmp_path / "runs")

        first = manager.wait(manager.start(quick(document)).id, timeout=300)
        assert first.checkpoint is not None and Path(first.checkpoint).is_file()

        second = manager.wait(manager.resume(first.id).id, timeout=300)
        assert second.status is Status.DONE
        # Resuming continues the step count rather than starting over.
        assert second.last_step == 6
        manager.close()

    def test_a_model_that_cannot_be_built_fails_the_run_not_the_manager(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("torch")
        from ntb.ir import Document, Module, Node, io

        broken = tmp_path / "broken.ntb"
        io.save(
            Document(root="m", modules=(Module(id="m", nodes=(Node(id="a", op="ntb.relu"),)),)),
            broken,
        )
        manager = RunManager(tmp_path / "runs")
        final = manager.wait(manager.start(quick(broken)).id, timeout=300)

        assert final.status is Status.FAILED
        assert final.error and "type-check" in final.error
        manager.close()

    def test_a_data_script_is_used_when_given(self, document: Path, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        script = tmp_path / "data.py"
        script.write_text(
            "import torch\n"
            "def dataloaders(batch_size):\n"
            "    batches = [\n"
            "        (torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,)))\n"
            "        for _ in range(2)\n"
            "    ]\n"
            "    return batches, batches\n",
            encoding="utf-8",
        )
        manager = RunManager(tmp_path / "runs")
        config = quick(document, data=DataSource.SCRIPT, data_script=script)
        final = manager.wait(manager.start(config).id, timeout=300)

        assert final.status is Status.DONE, final.error
        assert final.last_step == 2  # the script's two batches, not steps_per_epoch
        manager.close()

    def test_the_config_the_worker_ran_is_kept_next_to_its_output(
        self, document: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("torch")
        manager = RunManager(tmp_path / "runs")
        run = manager.wait(manager.start(quick(document)).id, timeout=300)

        saved = json.loads((tmp_path / "runs" / run.id / "config.json").read_text(encoding="utf-8"))
        assert saved["loss"] == "cross_entropy"
        manager.close()
