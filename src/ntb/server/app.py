"""The studio server: one document, many views.

Every mutation arrives as a command and goes through the bus, so HTTP clients,
the WebSocket and the future MCP server share one implementation of editing.

Requests from another origin are refused. The server is a local process that can
read and write `.ntb` files, and a page on the internet must not be able to
drive it just because it is listening on loopback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ntb import __version__
from ntb.commands import CommandError, parse_command
from ntb.ir import io
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
) -> FastAPI:
    """Build the studio app around one session."""
    app = FastAPI(title="NTB Studio", version=__version__)
    app.state.session = session if session is not None else Session()
    app.state.hub = Hub()
    app.state.allow_origins = allow_origins

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


def serve(
    document: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8756,
    open_browser: bool = True,
) -> None:  # pragma: no cover - exercised by `ntb studio`
    """Run the studio until interrupted."""
    import threading
    import webbrowser

    import uvicorn

    session = Session()
    if document is not None:
        session.open(document)
    app = create_app(session)

    if open_browser:
        url = f"http://{host}:{port}/"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
