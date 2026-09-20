from __future__ import annotations

import asyncio
import base64
import hmac
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from pydantic import Field

from ..config import Config
from ..backend import BackendError
from ..budget import BudgetExceeded
from ..scheduler import QueueRejected
from ..runtime import Runtime
from ..types import ContractError, StrictModel, Watch


class FrameBody(StrictModel):
    source: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    timestamp: float = Field(ge=0)
    jpeg_base64: str = Field(max_length=16_000_000)


class SignalBody(StrictModel):
    source: str
    timestamp: float = Field(ge=0)
    kind: str
    text: str = Field(default="", max_length=100000)
    data: dict = Field(default_factory=dict)
    start: float | None = Field(default=None, ge=0)


class QueryBody(StrictModel):
    source: str
    question: str = Field(min_length=1, max_length=12000)
    question_id: str = Field(default="api-query", max_length=100)
    as_of: float | None = Field(default=None, ge=0)


def create_app(config: Config, workdir: Path, *, allow_network: bool = False,
               token: str | None = None, insecure_local: bool = False, live_clock: bool = True):
    try:
        from fastapi import FastAPI, Depends, HTTPException
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from fastapi.responses import JSONResponse, Response
    except ImportError as exc:
        raise RuntimeError("Server requires: pip install -e '.[server]'") from exc
    if not token and not insecure_local:
        raise ValueError("Set STREAMBUDGET_TOKEN or explicitly enable loopback-only --insecure-local")
    if (workdir / "memory.sqlite").exists():
        raise ValueError("Use a new workdir for a new live session; resume semantics are not implemented")
    runtime = None  # Created on the ASGI event-loop thread during lifespan startup.
    origin = 0.0

    async def heartbeat():
        while True:
            await runtime.tick(time.monotonic() - origin)
            await asyncio.sleep(.5)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal runtime, origin
        origin = time.monotonic()
        runtime = Runtime(config, workdir, allow_network=allow_network)
        if live_clock:
            runtime.timeline_clock = lambda: time.monotonic() - origin
        app.state.runtime = runtime
        runtime.start()
        task = asyncio.create_task(heartbeat()) if live_clock else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            try:
                await runtime.drain()
            except TimeoutError:
                pass
            await runtime.close()

    app = FastAPI(title="StreamBudget research runtime", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    bearer = HTTPBearer(auto_error=False)

    async def auth(credentials=Depends(bearer)):
        if token:
            if credentials is None or not hmac.compare_digest(credentials.credentials, token):
                raise HTTPException(401, "Invalid bearer token")

    @app.exception_handler(ContractError)
    async def contract_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(BudgetExceeded)
    async def budget_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=429)

    @app.exception_handler(QueueRejected)
    async def busy_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=503)

    @app.exception_handler(BackendError)
    async def model_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=502)

    @app.exception_handler(TimeoutError)
    async def timeout_error(request, exc):
        return JSONResponse({"error": "Inference deadline exceeded"}, status_code=504)

    @app.middleware("http")
    async def body_limit(request, call_next):
        try:
            size = int(request.headers.get("content-length", 0))
        except ValueError:
            return JSONResponse({"error": "Invalid content length"}, status_code=400)
        if size > 17_000_000:
            return JSONResponse({"error": "Body too large"}, status_code=413)
        return await call_next(request)

    def availability(timestamp):
        if not live_clock:
            return max(runtime.now, timestamp)
        now = time.monotonic() - origin
        if timestamp > now + .5:
            raise ContractError("Frame/signal is ahead of session clock; obtain session_time from /v1/status")
        return max(now, timestamp)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/status", dependencies=[Depends(auth)])
    async def status():
        return {"session_time": time.monotonic() - origin if live_clock else runtime.now, **runtime.report()}

    @app.post("/v1/frames", dependencies=[Depends(auth)])
    async def frame(body: FrameBody):
        try:
            data = base64.b64decode(body.jpeg_base64, validate=True)
        except ValueError as exc:
            raise ContractError("Invalid base64") from exc
        e = await runtime.ingest_frame(body.source, body.timestamp, data,
                                       available_at=availability(body.timestamp))
        return {"evidence_id": e.id, "accepted": True, "queue_depth": runtime.scheduler.queue.qsize()}

    @app.post("/v1/signals", dependencies=[Depends(auth)])
    async def signal(body: SignalBody):
        e = await runtime.ingest_signal(body.source, body.timestamp, body.kind, body.text, body.data,
                                        start=body.start, available_at=availability(body.timestamp))
        return {"evidence_id": e.id, "accepted": True}

    @app.post("/v1/watches", dependencies=[Depends(auth)])
    async def watch(body: Watch):
        runtime.register_watch(body)
        return {"watch_id": body.id}

    @app.delete("/v1/watches/{id}", dependencies=[Depends(auth)])
    async def cancel(id: str):
        runtime.cancel_watch(id)
        return {"cancelled": id}

    @app.post("/v1/query", dependencies=[Depends(auth)])
    async def query(body: QueryBody):
        result = await runtime.ask(body.question, body.source, as_of=body.as_of, question_id=body.question_id)
        return asdict(result)

    @app.get("/v1/events", dependencies=[Depends(auth)])
    async def events():
        return {"events": runtime.alerts[-200:]}

    @app.get("/v1/evidence/{id}", dependencies=[Depends(auth)])
    async def evidence(id: str):
        return runtime.store.get(id, runtime.snapshot()).view()

    @app.get("/v1/frame/{id}", dependencies=[Depends(auth)])
    async def image(id: str):
        e = runtime.store.get(id, runtime.snapshot())
        if e.kind != "frame":
            raise ContractError("Not frame evidence")
        return Response(runtime.media.read(e.payload["media_key"]), media_type="image/jpeg")

    return app


def serve(config: Config, workdir: Path, *, host="127.0.0.1", port=8765,
          allow_network=False, insecure_local=False):
    if insecure_local and host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--insecure-local is restricted to a loopback host")
    import uvicorn
    app = create_app(config, workdir, allow_network=allow_network,
                     token=os.getenv("STREAMBUDGET_TOKEN"), insecure_local=insecure_local)
    uvicorn.run(app, host=host, port=port)
