"""The training loop, which runs in its own process.

Never in the server's. A training step allocates GPU memory, blocks, and can
take the interpreter down with it; the studio has to stay responsive and has to
survive the model that does not. See docs/adr/0011.

Everything this process has to say it says on stdout, one JSON object per line.
That is the whole protocol: the manager reads it, stores it, and forwards it.

    python -m ntb.runs.worker config.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from ntb.emit import emit_torch_document
from ntb.ir import io
from ntb.runs.config import DataSource, Loss, Optimiser, RunConfig
from ntb.shapes import infer_shapes
from ntb.spatial import resolve

#: Batch dimension used when a model input leaves it open.
DEFAULT_BATCH = 1


def emit(event: str, **fields: Any) -> None:
    """Say something to whoever is reading. Flushed: a stuck pipe is a dead UI."""
    print(json.dumps({"event": event, **fields}), flush=True)


def main(argv: list[str]) -> int:
    if len(argv) != 2:  # pragma: no cover - the manager always passes one
        print("usage: python -m ntb.runs.worker <config.json>", file=sys.stderr)
        return 2

    config = RunConfig.model_validate_json(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        _train(config)
    except Exception as exc:  # noqa: BLE001 - the manager needs the reason, not a traceback
        emit("failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


def _train(config: RunConfig) -> None:
    import torch

    torch.manual_seed(config.seed)
    document = io.load(config.document)
    emitted = emit_torch_document(document)

    namespace: dict[str, Any] = {}
    exec(compile(emitted.source, f"{config.document.stem}.py", "exec"), namespace)
    model = namespace[emitted.class_name]()
    device = torch.device(config.device)
    model.to(device)

    resumed_step = 0
    if config.resume_from is not None:
        state = torch.load(config.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        resumed_step = int(state.get("step", 0))
        emit("resumed", path=str(config.resume_from), step=resumed_step)

    optimiser = _optimiser(config, model.parameters())
    criterion = _criterion(config)
    loader = _data(config, document)

    parameters = sum(p.numel() for p in model.parameters())
    emit(
        "started",
        parameters=parameters,
        total_steps=config.total_steps() if config.data is DataSource.SYNTHETIC else None,
        device=str(device),
        model=emitted.class_name,
    )

    step = resumed_step
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        for inputs, target in loader(epoch):
            batch = [tensor.to(device) for tensor in inputs]
            target = target.to(device)

            optimiser.zero_grad(set_to_none=True)
            output = model(*batch)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)
            loss.backward()
            optimiser.step()

            step += 1
            emit(
                "metric",
                step=step,
                epoch=epoch,
                name="loss",
                value=float(loss.detach().cpu()),
                seconds=round(time.perf_counter() - started, 4),
            )
            if config.checkpoint_every and step % config.checkpoint_every == 0:
                _checkpoint(config, model, optimiser, step)

    _checkpoint(config, model, optimiser, step)
    emit("finished", steps=step, seconds=round(time.perf_counter() - started, 4))


def _optimiser(config: RunConfig, parameters: Any) -> Any:
    import torch

    kinds = {
        Optimiser.SGD: torch.optim.SGD,
        Optimiser.ADAM: torch.optim.Adam,
        Optimiser.ADAMW: torch.optim.AdamW,
    }
    return kinds[config.optimiser](parameters, lr=config.learning_rate)


def _criterion(config: RunConfig) -> Any:
    import torch

    kinds = {
        Loss.MSE: torch.nn.MSELoss,
        Loss.CROSS_ENTROPY: torch.nn.CrossEntropyLoss,
        Loss.BCE: torch.nn.BCEWithLogitsLoss,
    }
    return kinds[config.loss]()


def _data(config: RunConfig, document: Any) -> Any:
    if config.data is DataSource.SCRIPT:
        return _script_data(config)
    return _synthetic_data(config, document)


def _script_data(config: RunConfig) -> Any:
    """Load the user's own data. Their file, their rules, their process."""
    import importlib.util

    if config.data_script is None:
        raise ValueError("data is 'script' but no data_script was given")
    spec = importlib.util.spec_from_file_location("ntb_run_data", config.data_script)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {config.data_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "dataloaders"):
        raise ValueError(f"{config.data_script} defines no dataloaders(batch_size)")

    train, _ = module.dataloaders(config.batch_size)

    def loader(_: int) -> Any:
        for batch in train:
            inputs, target = batch
            yield (list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]), target

    return loader


def _synthetic_data(config: RunConfig, document: Any) -> Any:
    """Random tensors shaped like the model's own inputs.

    This answers "does this architecture train and how fast", which is the
    question you have while you are still drawing it. It is not a dataset and
    the loss it produces means nothing.
    """
    import torch

    graph = resolve(document)
    report = infer_shapes(graph)
    shapes = [_concrete(item.type.shape, config.batch_size) for item in graph.inputs]
    if not shapes:
        raise ValueError("the model declares no inputs, so there is nothing to feed it")

    endpoint = graph.outputs[0].endpoint if graph.outputs else None
    predicted = report.type_of(endpoint.node, endpoint.port) if endpoint else None
    if predicted is None:
        raise ValueError("the model's output type is unknown, so no target can be built")
    out_shape = _concrete(predicted.shape, config.batch_size)

    generator = torch.Generator().manual_seed(config.seed)

    def loader(epoch: int) -> Any:
        for _ in range(config.steps_per_epoch):
            inputs = [torch.randn(*shape, generator=generator) for shape in shapes]
            yield inputs, _target(config, out_shape, generator)

    return loader


def _target(config: RunConfig, shape: tuple[int, ...], generator: Any) -> Any:
    import torch

    if config.loss is Loss.CROSS_ENTROPY:
        return torch.randint(0, shape[-1], (shape[0],), generator=generator)
    if config.loss is Loss.BCE:
        return torch.randint(0, 2, shape, generator=generator).float()
    return torch.randn(*shape, generator=generator)


def _concrete(shape: tuple[int | str, ...], batch: int) -> tuple[int, ...]:
    """Symbolic dimensions become the batch size, or 1 if they are not the batch."""
    return tuple(
        dimension if isinstance(dimension, int) else (batch if index == 0 else DEFAULT_BATCH)
        for index, dimension in enumerate(shape)
    )


def _checkpoint(config: RunConfig, model: Any, optimiser: Any, step: int) -> None:
    import torch

    directory = config.workdir or Path("runs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.document.stem}-step{step}.pt"
    torch.save(
        {"model": model.state_dict(), "optimiser": optimiser.state_dict(), "step": step}, path
    )
    emit("checkpoint", step=step, path=str(path))


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main(sys.argv))
