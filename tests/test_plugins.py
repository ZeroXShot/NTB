"""Ops contributed from outside the repo (ADR 13).

The loader is what the tests here are about: what it imports, what it refuses,
and what it does with a plugin that breaks. The fixtures are real modules that
really register, so nothing below stubs the mechanism it is testing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from ntb.ops import REGISTRY
from ntb.ops.plugins import DISABLE, GROUP, load

FIXTURES = Path(__file__).parent / "fixtures" / "plugins"
PLUGIN_EXAMPLE = Path(__file__).parents[1] / "examples" / "plugin" / "src"


def point(module: str, name: str = "test") -> EntryPoint:
    return EntryPoint(name=name, value=module, group=GROUP)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Leave the registry as it was found: these tests really register ops."""
    # These tests are about loading, so the switch that turns it off is off
    # here even when the environment running them has it on.
    monkeypatch.delenv(DISABLE, raising=False)
    before = set(REGISTRY.names())
    modules = set(sys.modules)
    sys.path.insert(0, str(FIXTURES))
    sys.path.insert(0, str(PLUGIN_EXAMPLE))
    try:
        yield
    finally:
        sys.path.remove(str(PLUGIN_EXAMPLE))
        sys.path.remove(str(FIXTURES))
        for name in set(REGISTRY.names()) - before:
            REGISTRY.discard(name)
        # An import is cached, so a second load would register nothing.
        for name in set(sys.modules) - modules:
            del sys.modules[name]


class TestLoading:
    def test_an_op_from_outside_the_repo_becomes_an_op(self) -> None:
        report = load((point("good_plugin"),))

        assert report.problems == ()
        assert report.plugins[0].ops == ("test.good",)
        assert REGISTRY.get("test.good") is not None

    def test_an_entry_point_may_name_a_callable_instead(self) -> None:
        report = load((point("callable_plugin:setup"),))

        assert report.plugins[0].ops == ("test.called",)

    def test_a_plugin_that_breaks_is_reported_not_raised(self) -> None:
        report = load((point("angry_plugin"),))

        assert report.plugins == ()
        assert "this plugin is broken" in report.problems[0].reason
        # The point of reporting rather than raising.
        assert REGISTRY.get("ntb.relu") is not None

    def test_a_plugin_may_not_claim_the_ntb_namespace(self) -> None:
        report = load((point("greedy_plugin"),))

        assert report.plugins == ()
        assert "reserved" in report.problems[0].reason
        # And what it did register is taken back out again.
        assert REGISTRY.get("ntb.greedy") is None

    def test_one_broken_plugin_does_not_stop_the_next(self) -> None:
        report = load((point("angry_plugin", "angry"), point("good_plugin", "good")))

        assert len(report.problems) == 1
        assert report.plugins[0].ops == ("test.good",)

    def test_loading_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISABLE, "1")

        assert load((point("good_plugin"),)) == load(())
        assert REGISTRY.get("test.good") is None


class TestTheExamplePlugin:
    """`examples/plugin` is a real distribution, and the exit criterion."""

    def test_it_registers_and_reaches_every_backend(self) -> None:
        report = load((point("ntb_example_op", "example"),))
        assert report.problems == ()

        spec = REGISTRY.require("example.softsign")
        assert set(spec.backends()) == {"torch", "keras", "onnx"}
        # It declares a parity case, so the generated harness verifies it too.
        assert spec.parity is not None

    def test_it_emits_torch_like_any_other_op(self) -> None:
        load((point("ntb_example_op", "example"),))
        from ntb.emit import emit_torch_document
        from ntb.ir import Document, Module, Node, Port, PortDirection, TensorType

        document = Document(
            name="plugged",
            root="m",
            modules=(
                Module(
                    id="m",
                    nodes=(Node(id="act", op="example.softsign"),),
                    inputs=(
                        Port(
                            name="x",
                            direction=PortDirection.IN,
                            type=TensorType(shape=("batch", 8)),
                        ),
                    ),
                    outputs=(Port(name="y", direction=PortDirection.OUT),),
                ),
            ),
        )
        source = emit_torch_document(document).source
        assert "torch.nn.functional.softsign" in source
