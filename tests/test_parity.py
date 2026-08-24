"""Cross-backend numeric parity, generated from the op registry.

For every op that declares a ``ParityCase``, build a one-node graph, emit it to
both torch and ONNX, give both backends *the same* weights, feed both the same
input, and compare. Nothing here is written per op: adding an op to the registry
adds its parity test.

Weight transfer is what makes the comparison mean anything. Without it the two
backends would be running different models and any agreement would be luck.
"""

from __future__ import annotations

from typing import Any

import pytest

from ntb.emit.keras import emit as emit_keras
from ntb.emit.onnx import export, resolve_param_shape
from ntb.emit.torch import emit
from ntb.ir import (
    CoreEdge,
    CoreGraph,
    CoreNode,
    Endpoint,
    GraphInput,
    GraphOutput,
    TensorType,
)
from ntb.ir.core import Origin
from ntb.ir.types import DType
from ntb.ops import REGISTRY, OpSpec

np = pytest.importorskip("numpy")

VERIFIABLE = [spec for spec in REGISTRY if spec.parity is not None]
#: Ops whose Keras mapping cannot be given the same weights as torch, so a
#: comparison would be measuring two different models rather than two backends.
KERAS_UNTRANSFERABLE: tuple[str, ...] = ()
UNVERIFIED = [spec.name for spec in REGISTRY if spec.parity is None]


def build_graph(spec: OpSpec) -> CoreGraph:
    """A graph exercising ``spec`` on its declared parity case.

    Usually one node. A variadic port needs several sources, and a graph input
    can only land on a port once, so each of those arrives through its own relu
    -- an op the harness has already verified.
    """
    case = spec.parity
    assert case is not None
    origin = Origin(module="parity", node="n")
    nodes = [CoreNode(id="n", op=spec.name, attrs=dict(case.attrs), origin=origin)]
    inputs = [
        GraphInput(
            name=port,
            endpoint=Endpoint(node="n", port=port),
            type=TensorType(
                dtype=DType.INT64 if port in case.integer_inputs else DType.FLOAT32,
                shape=tuple(shape),
            ),
        )
        for port, shape in case.inputs.items()
    ]
    edges: list[CoreEdge] = []
    for port, shapes in case.fan_in.items():
        for index, shape in enumerate(shapes):
            feeder = f"src{index}"
            nodes.append(CoreNode(id=feeder, op="ntb.relu", origin=origin))
            inputs.append(
                GraphInput(
                    name=f"{port}{index}",
                    endpoint=Endpoint(node=feeder, port="in"),
                    type=TensorType(shape=tuple(shape)),
                )
            )
            edges.append(
                CoreEdge(
                    id=f"e{index}",
                    src=Endpoint(node=feeder, port="out"),
                    dst=Endpoint(node="n", port=port),
                    origin=origin,
                )
            )
    return CoreGraph(
        name=spec.name.replace(".", "_"),
        nodes=tuple(nodes),
        edges=tuple(edges),
        inputs=tuple(inputs),
        outputs=(GraphOutput(name="out", endpoint=Endpoint(node="n", port="out")),),
    )


def sample_inputs(spec: OpSpec, seed: int = 0) -> dict[str, Any]:
    case = spec.parity
    assert case is not None
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {}
    for port, shape in case.inputs.items():
        if port in case.integer_inputs:
            values[port] = rng.integers(0, case.index_limit, size=shape).astype(np.int64)
        else:
            values[port] = rng.standard_normal(shape).astype(np.float32)
    for port, shapes in case.fan_in.items():
        for index, shape in enumerate(shapes):
            values[f"{port}{index}"] = rng.standard_normal(shape).astype(np.float32)
    return values


class TestCoverage:
    def test_most_ops_declare_a_parity_case(self) -> None:
        assert len(VERIFIABLE) >= len(REGISTRY) - len(UNVERIFIED)

    def test_unverified_ops_are_the_known_list(self) -> None:
        # Fails when a new op ships without a ParityCase, which is the point:
        # an op nobody checks across backends is an op nobody can trust.
        assert sorted(UNVERIFIED) == [
            "ntb.attention",
            "ntb.dropout",
            "ntb.flatten",
            "ntb.reshape",
            "ntb.rmsnorm",
            "ntb.silu",
        ]


class TestGraphsBuild:
    @pytest.mark.parametrize("spec", VERIFIABLE, ids=lambda s: s.name)
    def test_the_parity_case_type_checks(self, spec: OpSpec) -> None:
        from ntb.shapes import infer_shapes

        report = infer_shapes(build_graph(spec))
        assert report.ok, [str(i) for i in report.issues]

    @pytest.mark.parametrize("spec", VERIFIABLE, ids=lambda s: s.name)
    def test_parameter_shapes_resolve(self, spec: OpSpec) -> None:
        assert spec.onnx is not None
        attrs = spec.resolved_attrs(spec.parity.attrs)  # type: ignore[union-attr]
        for param in spec.onnx.params:
            if param.when is not None and not attrs.get(param.when):
                continue
            shape = resolve_param_shape(param, attrs)
            assert all(d > 0 for d in shape), (spec.name, param.name)


@pytest.mark.torch
@pytest.mark.onnx
class TestNumericParity:
    @pytest.mark.parametrize("spec", VERIFIABLE, ids=lambda s: s.name)
    def test_torch_and_onnxruntime_agree(self, spec: OpSpec) -> None:
        torch = pytest.importorskip("torch")
        ort = pytest.importorskip("onnxruntime")

        graph = build_graph(spec)
        inputs = sample_inputs(spec)

        # torch
        emitted = emit(graph)
        namespace: dict[str, Any] = {}
        exec(compile(emitted.source, f"{spec.name}.py", "exec"), namespace)
        model = namespace[emitted.class_name]().eval()
        with torch.no_grad():
            torch_out = model(*(torch.from_numpy(v) for v in inputs.values()))
        if isinstance(torch_out, tuple):
            torch_out = torch_out[0]
        torch_out = torch_out.numpy()

        # onnx, given the very same weights
        exported = export(graph)
        _transfer_weights(spec, model, exported.model)
        session = ort.InferenceSession(
            exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        onnx_out = session.run(None, dict(inputs))[0]

        assert torch_out.shape == onnx_out.shape, spec.name
        np.testing.assert_allclose(
            torch_out, onnx_out, atol=spec.parity_atol, rtol=1e-4, err_msg=spec.name
        )


def _transfer_weights(spec: OpSpec, model: Any, onnx_model: Any) -> None:
    """Copy the torch module's tensors into the ONNX initialisers.

    The registry says which torch tensor backs which ONNX parameter
    (``ParamSpec.torch_name``), so this stays op-agnostic.
    """
    import onnx
    from onnx import numpy_helper

    assert spec.onnx is not None
    state = dict(model.named_parameters()) | dict(model.named_buffers())
    torch_by_suffix = {key.rsplit(".", 1)[-1]: value for key, value in state.items()}

    replacements: dict[str, Any] = {}
    for param in spec.onnx.params:
        if not param.torch_name:
            continue
        tensor = torch_by_suffix.get(param.torch_name)
        if tensor is None:
            continue
        replacements[f"n_{param.name}"] = tensor.detach().numpy().astype(np.float32)

    graph = onnx_model.graph
    for index, initialiser in enumerate(graph.initializer):
        replacement = replacements.get(initialiser.name)
        if replacement is None:
            continue
        expected = tuple(initialiser.dims)
        assert replacement.shape == expected, (
            f"{spec.name}: torch tensor {initialiser.name} is {replacement.shape}, "
            f"ONNX declared {expected}"
        )
        graph.initializer[index].CopyFrom(
            numpy_helper.from_array(replacement, name=initialiser.name)
        )
    onnx.checker.check_model(onnx_model)


def build_torch(spec: OpSpec, graph: CoreGraph) -> tuple[Any, Any]:
    """The emitted torch module, in eval mode, and the numpy result."""

    emitted = emit(graph)
    namespace: dict[str, Any] = {}
    exec(compile(emitted.source, f"{spec.name}.py", "exec"), namespace)
    return namespace[emitted.class_name]().eval(), emitted


def keras_weights(spec: OpSpec, model: Any) -> list[Any]:
    """The torch module's tensors, in the order and layout Keras keeps them."""
    assert spec.keras is not None
    state = dict(model.named_parameters()) | dict(model.named_buffers())
    by_suffix = {key.rsplit(".", 1)[-1]: value for key, value in state.items()}

    values: list[Any] = []
    for weight in spec.keras.weights:
        tensor = by_suffix.get(weight.torch_name)
        if tensor is None:
            continue
        array = tensor.detach().numpy().astype(np.float32)
        if weight.transform == "transpose":
            array = array.T
        elif weight.transform == "conv_kernel":
            # torch (out, in, *spatial) -> keras (*spatial, in, out)
            array = np.transpose(array, (*range(2, array.ndim), 1, 0))
        values.append(array)
    return values


@pytest.mark.torch
@pytest.mark.keras
class TestKerasParity:
    """torch against Keras 3, on the same weights and the same input.

    Keras 3 runs on TensorFlow, JAX or torch, so agreeing with torch here is
    what makes one `.ntb` mean the same model on all of them.
    """

    @pytest.mark.parametrize("spec", VERIFIABLE, ids=lambda s: s.name)
    def test_torch_and_keras_agree(self, spec: OpSpec) -> None:
        torch = pytest.importorskip("torch")
        pytest.importorskip("keras")
        if spec.name in KERAS_UNTRANSFERABLE:
            pytest.skip("no weight transfer for this op yet")

        graph = build_graph(spec)
        inputs = sample_inputs(spec)

        model, _ = build_torch(spec, graph)
        with torch.no_grad():
            torch_out = model(*(torch.from_numpy(v) for v in inputs.values()))
        if isinstance(torch_out, tuple):
            torch_out = torch_out[0]
        torch_out = torch_out.numpy()

        emitted = emit_keras(graph)
        namespace: dict[str, Any] = {}
        exec(compile(emitted.source, f"{spec.name}_keras.py", "exec"), namespace)
        built = namespace[emitted.class_name]()

        # Give Keras the torch weights, laid out the way Keras keeps them.
        transferred = keras_weights(spec, model)
        if transferred:
            layer = next(layer for layer in built.layers if layer.weights and layer.count_params())
            layer.set_weights(transferred)

        import keras

        values = list(inputs.values())
        produced = built(values[0] if len(values) == 1 else values)
        if isinstance(produced, (list, tuple)):
            produced = produced[0]
        keras_out = keras.ops.convert_to_numpy(produced)

        assert torch_out.shape == keras_out.shape, spec.name
        np.testing.assert_allclose(
            torch_out, keras_out, atol=spec.parity_atol, rtol=1e-4, err_msg=spec.name
        )
