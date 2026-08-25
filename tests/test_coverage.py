"""Measuring the registry against the backends' own vocabularies.

The exact numbers are pinned by `docs/coverage.md` and `ntb coverage --check`,
not here: they move when torch does, and a test that fails on someone else's
release is a test nobody trusts. What is tested here is the matcher, which is
the part that can be wrong in a way that flatters the numbers.
"""

from __future__ import annotations

import pytest

from ntb.ops import REGISTRY
from ntb.ops.coverage import (
    EXCLUDED,
    as_markdown,
    audit,
    builtins,
    excluded_reason,
    normalise,
    reached,
)


class TestMatching:
    def test_a_target_matches_on_its_last_segment(self) -> None:
        assert normalise("torch.nn.functional.relu") == "relu"
        assert normalise("ReLU") == "relu"
        # Which is the whole point: NTB reaches ReLU through the functional form.
        assert normalise("torch.nn.functional.relu") == normalise("ReLU")

    def test_underscores_do_not_stop_a_match(self) -> None:
        assert normalise("keras.layers.ZeroPadding2D") == normalise("zero_padding2d")

    def test_an_exclusion_carries_a_reason(self) -> None:
        assert excluded_reason("Sequential")
        assert excluded_reason("RandomFlip") == "data augmentation, not architecture"
        assert excluded_reason("Linear") is None
        assert all(reason for reason in EXCLUDED.values()), "an exclusion without a reason"


class TestWhatIsMeasured:
    def test_only_the_ops_this_repo_ships_are_measured(self) -> None:
        assert all(spec.name.startswith("ntb.") for spec in builtins())

    def test_the_registry_targets_are_what_counts_as_reached(self) -> None:
        covered = reached(builtins(REGISTRY))
        assert "linear" in covered["torch"]
        assert "dense" in covered["keras"]
        assert "gemm" in covered["onnx"]
        # rank_targets and pad_target are targets too.
        assert "batchnorm1d" in covered["torch"]
        assert "zeropadding2d" in covered["keras"]

    def test_a_custom_onnx_op_is_not_coverage_of_onnx(self) -> None:
        # ntb.silu emits into the ntb.ops domain; ONNX has no Silu.
        assert "silu" not in reached(builtins(REGISTRY))["onnx"]


@pytest.mark.torch
class TestAgainstTorch:
    def test_the_layers_we_map_come_back_covered(self) -> None:
        pytest.importorskip("torch")
        surface = audit().surface("torch.nn")
        assert surface is not None and surface.available
        for name in ("Linear", "Conv2d", "ReLU", "LayerNorm", "MaxPool2d", "Embedding"):
            assert name in surface.covered, name

    def test_what_we_do_not_have_is_reported_missing(self) -> None:
        pytest.importorskip("torch")
        surface = audit().surface("torch.nn")
        assert surface is not None
        for name in ("LSTM", "GRU", "ConvTranspose2d", "GroupNorm"):
            assert name in surface.missing, name
        assert "Sequential" in surface.excluded

    def test_optimisers_are_honestly_reported_as_uncovered(self) -> None:
        pytest.importorskip("torch")
        surface = audit().surface("torch.optim")
        assert surface is not None
        # They are a closed enum in ntb.runs.config, not registry ops.
        assert surface.covered == () and "Adam" in surface.missing

    def test_a_loss_is_not_counted_as_a_layer(self) -> None:
        pytest.importorskip("torch")
        report = audit()
        layers, losses = report.surface("torch.nn"), report.surface("torch losses")
        assert layers is not None and losses is not None
        assert "NLLLoss2d" not in layers.missing
        assert "NLLLoss2d" in losses.missing


class TestTheDocument:
    def test_a_backend_that_is_not_installed_says_so_rather_than_failing(self) -> None:
        report = audit()
        for surface in report.surfaces:
            if not surface.available:
                assert surface.total == 0

    def test_the_generated_document_reports_what_was_measured(self) -> None:
        report = audit()
        text = as_markdown(report)
        assert "Do not edit" in text
        assert f"**{report.ops} ops**" in text
        # An exclusion has to be arguable, so the reason is published.
        assert "a container; NTB composes with Modules and edges" in text
