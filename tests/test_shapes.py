"""Dimension algebra: the part every shape rule leans on."""

from __future__ import annotations

import pytest

from ntb.shapes import ShapeError, conv_out_dim, dim, dims_equal, render


class TestDim:
    def test_integers_pass_through(self) -> None:
        assert dim(7) == 7

    def test_negative_integers_are_rejected(self) -> None:
        with pytest.raises(ShapeError, match="negative dimension"):
            dim(-1)

    def test_symbols_are_non_negative_integers(self) -> None:
        # Without these assumptions sympy will not simplify floor divisions, and
        # every convolution over a symbolic input would produce noise.
        symbol = next(iter(dim("batch").free_symbols))
        assert symbol.is_integer and symbol.is_nonnegative

    def test_unparseable_expression_is_reported(self) -> None:
        with pytest.raises(ShapeError, match="cannot parse"):
            dim("2 +")


class TestRender:
    def test_concrete_results_come_back_as_int(self) -> None:
        # Generated source must read `nn.Linear(512, 10)`, not `nn.Linear("512", 10)`.
        result = render(dim("seq") - dim("seq") + 128)
        assert result == 128
        assert isinstance(result, int)

    def test_symbolic_results_come_back_as_text(self) -> None:
        assert render(dim("batch") * 2) == "2*batch"

    def test_non_integer_constants_are_refused(self) -> None:
        with pytest.raises(ShapeError, match="non-integer constant"):
            render(dim(7) / 2)


class TestDimsEqual:
    def test_provably_equal(self) -> None:
        assert dims_equal(128, 128) is True
        assert dims_equal("2*batch", "batch + batch") is True

    def test_provably_different(self) -> None:
        assert dims_equal(128, 256) is False

    def test_undecidable_returns_none(self) -> None:
        # Two unrelated symbols may agree at runtime; callers must warn, not fail.
        assert dims_equal("batch", "n") is None


class TestConvOutDim:
    @pytest.mark.parametrize(
        ("size", "kernel", "stride", "padding", "dilation", "expected"),
        [
            (224, 7, 2, 3, 1, 112),
            (32, 3, 1, 1, 1, 32),
            (32, 3, 1, 0, 1, 30),
            (8, 3, 2, 0, 1, 3),
            (32, 3, 1, 0, 2, 28),
            (1, 1, 1, 0, 1, 1),
        ],
    )
    def test_matches_the_reference_formula(
        self, size: int, kernel: int, stride: int, padding: int, dilation: int, expected: int
    ) -> None:
        assert (
            conv_out_dim(size, kernel=kernel, stride=stride, padding=padding, dilation=dilation)
            == expected
        )

    def test_symbolic_input_stays_symbolic(self) -> None:
        result = conv_out_dim("h", kernel=3, stride=1, padding=1)
        assert result == "h"

    def test_symbolic_input_under_striding(self) -> None:
        result = conv_out_dim("h", kernel=1, stride=2, padding=0)
        assert isinstance(result, str) and "h" in result

    def test_window_larger_than_a_concrete_input_is_an_error(self) -> None:
        with pytest.raises(ShapeError, match="does not fit"):
            conv_out_dim(2, kernel=5, stride=1, padding=0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"kernel": 0, "stride": 1, "padding": 0}, "kernel must be >= 1"),
            ({"kernel": 3, "stride": 0, "padding": 0}, "stride must be >= 1"),
            ({"kernel": 3, "stride": 1, "padding": -1}, "padding must be >= 0"),
            ({"kernel": 3, "stride": 1, "padding": 0, "dilation": 0}, "dilation must be >= 1"),
        ],
    )
    def test_invalid_parameters_are_reported(self, kwargs: dict[str, int], message: str) -> None:
        with pytest.raises(ShapeError, match=message):
            conv_out_dim(32, **kwargs)
