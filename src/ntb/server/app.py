"""The studio server: one document, many views.

Every mutation arrives as a command and goes through the bus, so HTTP clients,
the WebSocket and the future MCP server share one implementation of editing.

Requests from another origin are refused. The server is a local process that can
read and write `.ntb` files, and a page on the internet must not be able to
drive it just because it is listening on loopback.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ntb import __version__
from ntb.commands import CommandError, parse_command
from ntb.ir import io
from ntb.runs import RunConfig, RunError, RunManager
from ntb.server.catalog import op_catalog
from ntb.server.protocol import error_message, parse_message, state_message
from ntb.server.session import Session, blank_document

STATIC_DIR = Path(__file__).resolve().parent.parent / "_static"

_MISSING_UI = """<!doctype html>
<title>NTB Studio</title>
<h1>NTB Studio is not built</h1>
<p>This install has no frontend bundle. From a source checkout run:</p>
<pre>cd apps/studio &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>The API is up: <a href="/api/ops">/api/ops</a></p>
"""


class Hub:
    """The connected clients. Broadcasts are best-effort: a dead socket is dropped."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    def add(self, client: WebSocket) -> None:
        self._clients.add(client)

    def remove(self, client: WebSocket) -> None:
        self._clients.discard(client)

    async def broadcast(self, message: dict[str, Any], *, skip: WebSocket | None = None) -> None:
        for client in list(self._clients):
            if client is skip:
                continue
            try:
                await client.send_json(message)
            except (RuntimeError, WebSocketDisconnect):
                self._clients.discard(client)


def create_app(
    session: Session | None = None,
    *,
    static_dir: Path | None = None,
    allow_origins: tuple[str, ...] = (),
    runs_root: Path | None = None,
) -> FastAPI:
    """Build the studio app around one session."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The reader threads need a loop to hand their events to.
        app.state.loop = asyncio.get_running_loop()
        yield
        app.state.runs.close()

    app = FastAPI(title="NTB Studio", version=__version__, lifespan=lifespan)
    app.state.session = session if session is not None else Session()
    app.state.hub = Hub()
    app.state.allow_origins = allow_origins
    app.state.loop = None

    def forward(run_id: str, event: dict[str, Any]) -> None:
        """A worker's reader thread has something to say; hand it to the loop."""
        loop: asyncio.AbstractEventLoop | None = app.state.loop
        if loop is None or loop.is_closed():
            return
        message = {"type": "run", "runId": run_id, **event}
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_broadcast(app, message)))

    app.state.runs = RunManager(runs_root or Path("runs"), listener=forward)

    @app.middleware("http")
    async def _same_origin_only(request: Request, call_next: Any) -> Any:
        origin = request.headers.get("origin")
        if not _origin_allowed(origin, request.headers.get("host"), allow_origins):
            return JSONResponse({"detail": f"origin {origin} is not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/ops")
    def ops() -> list[dict[str, Any]]:
        return op_catalog()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return _session(app).snapshot()

    @app.post("/api/command")
    async def command(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            _session(app).apply(parse_command(payload))
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        snapshot = _session(app).snapshot()
        await app.state.hub.broadcast(state_message(snapshot))
        return snapshot

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [record.as_json() for record in _runs(app).recent()]

    @app.post("/api/runs")
    def start_run(payload: dict[str, Any]) -> dict[str, Any]:
        studio = _session(app)
        # A worker loads the document from disk, so an unsaved one has nothing
        # to train. Saying so beats training a stale file.
        if studio.path is None:
            raise HTTPException(status_code=400, detail="save the document before training it")
        if studio.dirty:
            studio.save()
        try:
            config = RunConfig.model_validate({**payload, "document": str(studio.path)})
            return _runs(app).start(config).as_json()
        except (RunError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    def show_run(run_id: str) -> dict[str, Any]:
        record = _runs(app).get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id}")
        return {**record.as_json(), "metrics": _runs(app).metrics(run_id)}

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, Any]:
        try:
            return _runs(app).stop(run_id).as_json()
        except RunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str) -> dict[str, Any]:
        try:
            return _runs(app).resume(run_id).as_json()
        except RunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.websocket("/ws")
    async def websocket(client: WebSocket) -> None:
        origin = client.headers.get("origin")
        if not _origin_allowed(origin, client.headers.get("host"), allow_origins):
            await client.close(code=1008)
            return
        await client.accept()
        hub: Hub = app.state.hub
        hub.add(client)
        await client.send_json(state_message(_session(app).snapshot()))
        try:
            while True:
                await _handle(app, client, await client.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            hub.remove(client)

    _mount_ui(app, static_dir or STATIC_DIR)
    return app


async def _broadcast(app: FastAPI, message: dict[str, Any]) -> None:
    await app.state.hub.broadcast(message)


async def _handle(app: FastAPI, client: WebSocket, payload: Any) -> None:
    session = _session(app)
    try:
        message = parse_message(payload)
        if message.type == "command":
            session.apply(message.command)
        elif message.type == "undo":
            session.undo()
        elif message.type == "redo":
            session.redo()
        elif message.type == "save":
            session.save(Path(message.path) if message.path else None)
        elif message.type == "open":
            session.open(Path(message.path))
        elif message.type == "new":
            session.open_document(blank_document(message.name))
        # "refresh" falls through to the broadcast below.
    except (CommandError, io.DocumentError, ValueError, OSError) as exc:
        await client.send_json(error_message(str(exc)))
        return
    await app.state.hub.broadcast(state_message(session.snapshot()))


def _mount_ui(app: FastAPI, static_dir: Path) -> None:
    index = static_dir / "index.html"
    if index.is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="studio")
        return

    @app.get("/", response_class=HTMLResponse)
    def placeholder() -> str:
        return _MISSING_UI


def _origin_allowed(origin: str | None, host: str | None, extra: tuple[str, ...]) -> bool:
    """Same-origin, an explicit allow-list, or no Origin header at all (curl, tests)."""
    if origin is None:
        return True
    if origin in extra:
        return True
    parsed = urlsplit(origin)
    if host is not None and parsed.netloc == host:
        return True
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _session(app: FastAPI) -> Session:
    session: Session = app.state.session
    return session


def _runs(app: FastAPI) -> RunManager:
    manager: RunManager = app.state.runs
    return manager


def serve(
    document: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8756,
    open_browser: bool = True,
    runs_root: Path | None = None,
) -> None:  # pragma: no cover - exercised by `ntb studio`
    """Run the studio until interrupted."""
    import threading
    import webbrowser

    import uvicorn

    session = Session()
    if document is not None:
        session.open(document)
    app = create_app(session, runs_root=runs_root)

    if open_browser:
        url = f"http://{host}:{port}/"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
