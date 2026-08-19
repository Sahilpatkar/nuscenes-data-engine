"""FastAPI application serving the current production detection model.

Endpoints:
    GET  /health                    liveness + model version + search readiness.
    POST /predict                   accept an image, return detections as JSON.
    POST /predict/annotated         accept an image, return it annotated as PNG.
    GET  /search?q=&k=              semantic text search over embedded frames.
    POST /search/image              semantic search by example image.
    GET  /search/similar/{token}    frames similar to a stored frame.
    POST /chat                      dataset-chat agent (text-to-SQL + vector search).
    POST /chat/stream               same agent, streamed as Server-Sent Events.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from nuscenes_data_engine import __version__
from nuscenes_data_engine.config import get_settings
from nuscenes_data_engine.monitoring.features import image_brightness
from nuscenes_data_engine.serving.model import load_production_model
from nuscenes_data_engine.serving.schemas import (
    ChatChart,
    ChatRequest,
    ChatResponse,
    ChatStep,
    Detection,
    HealthResponse,
    PredictResponse,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger("nuscenes_data_engine")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; serve degraded (503s) if the load fails."""
    settings = get_settings()
    app.state.settings = settings
    try:
        app.state.model, app.state.model_version = load_production_model(settings)
        logger.info(
            "Serving model version %s (imgsz=%d, conf=%.2f, device=%s)",
            app.state.model_version,
            settings.serving_imgsz,
            settings.serving_conf,
            settings.serving_device,
        )
    except Exception:
        logger.exception("Model load failed; /health reports degraded, /predict returns 503")
        app.state.model, app.state.model_version = None, None
    app.state.search_engine = None  # built lazily on the first /search call
    app.state.chat_catalog = None  # built lazily on the first /chat call
    app.state.graph = None  # knowledge-graph driver; built lazily, False once unreachable
    yield


app = FastAPI(title="nuScenes Data Engine — Detection API", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness probe."""
    state = request.app.state
    settings = state.settings
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        search_ready=state.search_engine is not None
        or Path(settings.search_lancedb_path).is_dir(),
        chat_provider=settings.chat_provider,
        chat_model=settings.chat_model
        if settings.chat_provider == "local"
        else settings.chat_anthropic_model,
    )


def _capture(
    request: Request, img: np.ndarray[Any, Any], n_detections: int, latency_ms: float
) -> None:
    """Append one drift-monitoring feature row per request; never fail the request."""
    path = request.app.state.settings.serving_capture_path
    if not path:
        return
    try:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            "n_detections": n_detections,
            "brightness": image_brightness(img),
            "latency_ms": round(latency_ms, 1),
            "model_version": request.app.state.model_version,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        logger.warning("Monitoring capture to %s failed", path, exc_info=True)


def _infer(request: Request, data: bytes) -> tuple[Any, np.ndarray[Any, Any]]:
    """Decode the uploaded bytes and run the model; return (result, image)."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Upload is not a decodable image.")
    settings = request.app.state.settings
    results = model.predict(
        img,
        imgsz=settings.serving_imgsz,
        conf=settings.serving_conf,
        device=settings.serving_device,
        verbose=False,
    )
    return results[0], img


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    """Run detection on an uploaded image and return the boxes as JSON."""
    start = time.perf_counter()
    result, img = _infer(request, await file.read())
    names = request.app.state.model.names
    detections = [
        Detection(
            label=names[int(box.cls)],
            confidence=float(box.conf),
            bbox=tuple(box.xyxy[0].tolist()),
        )
        for box in result.boxes
    ]
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("predict %s: %d detections in %.0f ms", file.filename, len(detections), latency_ms)
    _capture(request, img, len(detections), latency_ms)
    return PredictResponse(
        detections=detections,
        model_version=request.app.state.model_version,
        image_width=img.shape[1],
        image_height=img.shape[0],
        n_detections=len(detections),
    )


def _get_search_engine(request: Request) -> Any:
    """Build (once) and return the semantic-search engine; 503 when unavailable."""
    if request.app.state.search_engine is None:
        settings = request.app.state.settings
        try:
            from nuscenes_data_engine.data_engine.search import SearchEngine

            request.app.state.search_engine = SearchEngine(
                Path(settings.search_lancedb_path),
                settings.search_table,
                settings.search_model_name,
                device=settings.search_device,
            )
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Search unavailable: {exc}. Build/sync data/lancedb and install "
                "the 'engine' extra.",
            ) from exc
    return request.app.state.search_engine


def _search_response(results: list[dict[str, Any]], query: str, model_name: str) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(
                thumbnail_b64=base64.b64encode(row["thumbnail"]).decode(),
                **{k: v for k, v in row.items() if k not in ("thumbnail", "filename")},
            )
            for row in results
        ],
        query=query,
        embedding_model=model_name,
    )


@app.get("/search", response_model=SearchResponse)
def search(request: Request, q: str, k: int | None = None) -> SearchResponse:
    """Semantic text search over the embedded frames."""
    engine = _get_search_engine(request)
    k = k or request.app.state.settings.search_top_k
    start = time.perf_counter()
    results = engine.search_text(q, k)
    logger.info(
        "search %r: %d results in %.0f ms", q, len(results), (time.perf_counter() - start) * 1000
    )
    return _search_response(results, q, engine.model_name)


@app.post("/search/image", response_model=SearchResponse)
async def search_image(
    request: Request, file: UploadFile = File(...), k: int | None = None
) -> SearchResponse:
    """Semantic search by example image."""
    engine = _get_search_engine(request)
    k = k or request.app.state.settings.search_top_k
    try:
        results = engine.search_image(await file.read(), k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _search_response(results, f"image:{file.filename}", engine.model_name)


@app.get("/search/similar/{token}", response_model=SearchResponse)
def search_similar(request: Request, token: str, k: int | None = None) -> SearchResponse:
    """Frames most similar to a stored frame."""
    engine = _get_search_engine(request)
    k = k or request.app.state.settings.search_top_k
    try:
        results = engine.search_similar(token, k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _search_response(results, f"similar:{token}", engine.model_name)


def _get_chat_catalog(request: Request) -> Any:
    """Open (once) the DuckDB catalog; 503 when the processed tables are absent."""
    if request.app.state.chat_catalog is None:
        from nuscenes_data_engine.data_engine.chat.catalog import catalog_tables, open_catalog

        settings = request.app.state.settings
        con = open_catalog(
            Path(settings.processed_dir),
            labels_path=Path(settings.data_dir) / "autolabel" / "labels.parquet",
        )
        if not catalog_tables(con):
            raise HTTPException(
                status_code=503,
                detail=f"Chat unavailable: no Parquet tables under {settings.processed_dir}. "
                "Mount/sync data/processed.",
            )
        request.app.state.chat_catalog = con
    return request.app.state.chat_catalog


def _get_graph_driver(request: Request) -> Any:
    """Open (once) the Neo4j driver; cache ``False`` if unreachable so chat degrades.

    Chat stays fully functional on SQL + vector search when the graph is down — the
    ``run_cypher`` tool is simply not offered.
    """
    state = request.app.state
    if state.graph is None:  # not yet attempted this process
        try:
            from nuscenes_data_engine.data_engine.graph import connection

            state.graph = connection.get_driver(state.settings)
            logger.info("Knowledge graph connected — chat gains the run_cypher tool.")
        except Exception as exc:  # not installed / unreachable -> SQL+vector only
            logger.info("Knowledge graph unavailable (%s) — SQL+vector chat only.", exc)
            state.graph = False
    return state.graph or None


def _assemble_chat_response(result: Any) -> ChatResponse:
    """Build the ``ChatResponse``-shaped body from an agent ``ChatResult``.

    Shared by ``/chat`` and the ``final`` SSE event on ``/chat/stream`` so the two
    response shapes cannot drift apart.
    """
    return ChatResponse(
        answer=result.answer,
        model=result.model,
        steps=[ChatStep(**step) for step in result.steps],
        frames=[
            SearchResult(
                thumbnail_b64=base64.b64encode(frame["thumbnail"]).decode(),
                **{k: v for k, v in frame.items() if k not in ("thumbnail", "filename")},
            )
            for frame in result.frames
        ],
        charts=[ChatChart(**chart) for chart in result.charts],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Answer a dataset question via the tool-calling agent (SQL + vector + graph)."""
    from nuscenes_data_engine.data_engine.chat import agent
    from nuscenes_data_engine.data_engine.chat.transports import TransportError, make_transport

    settings = request.app.state.settings
    con = _get_chat_catalog(request)
    try:
        engine = _get_search_engine(request)
    except HTTPException:
        engine = None  # SQL-only chat still works without the LanceDB store
    try:
        transport = make_transport(settings)
    except (TransportError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Chat unavailable: {exc}") from exc

    start = time.perf_counter()
    try:
        result = agent.answer(
            body.message,
            transport=transport,
            con=con,
            search_engine=engine,
            history=[turn.model_dump() for turn in body.history],
            log_path=Path(settings.chat_log_path),
            graph_driver=_get_graph_driver(request),
            graph_database=settings.neo4j_database,
        )
    except TransportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info(
        "chat %r: %d steps, %d frames in %.0f ms",
        body.message[:80],
        len(result.steps),
        len(result.frames),
        (time.perf_counter() - start) * 1000,
    )
    return _assemble_chat_response(result)


@app.post("/chat/stream")
def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """Same agent as ``/chat``, streamed as Server-Sent Events.

    Resources are resolved through the same getters as ``/chat`` BEFORE the
    generator is defined, so a missing catalog/transport still raises an
    ``HTTPException`` that maps to a real status code (503) instead of a 200
    response that then fails mid-stream.

    Why a background thread: ``agent.answer`` is a synchronous, blocking call that
    invokes ``on_turn``/``on_token``/``on_step`` as plain Python callbacks — there is
    no way for a synchronous generator to ``yield`` from *inside* someone else's
    callback. The standard fix is to run ``agent.answer`` in a ``threading.Thread``
    and have the callbacks push ``(kind, payload)`` tuples onto a thread-safe
    ``queue.SimpleQueue``; the generator (run by Starlette in a threadpool, since
    it is a sync generator) drains that queue and yields one SSE frame per item.
    A ``None`` sentinel marks end-of-stream.

    If ``agent.answer`` itself raises, the thread has no other way to report the
    failure: the HTTP response — and its 200 status — was already sent with the
    first byte of the stream, so a 500 status is no longer possible. The failure
    is instead surfaced as a terminal ``error`` SSE event. The full traceback is
    always logged server-side first (``logger.exception``); what reaches the
    client mirrors ``/chat``'s own split — a ``TransportError``'s message is
    safe to show (it is already a curated, no-secrets string like "is `ollama
    serve` running?"), anything else becomes a generic message so an unexpected
    internal exception can never leak its text to the client.

    Payload encoding: every ``data:`` payload is a JSON value, so clients can
    ``json.loads`` each one uniformly — ``step``/``final`` are JSON objects,
    and ``token`` is a JSON-encoded string (``json.dumps(delta)``). The token
    encoding matters beyond consistency: a raw token delta routinely contains
    embedded newlines (markdown answers are full of them), and SSE framing is
    one event per blank-line-terminated block of `field: value` lines — an
    un-escaped ``\n`` inside a bare ``data: <delta>`` line would split into a
    second, un-prefixed line that a line-based SSE parser (browsers'
    ``EventSource`` included) silently drops instead of treating as part of the
    payload. JSON-encoding escapes it onto one line. ``turn``'s payload is the
    literal empty string, not JSON — there is nothing to decode.
    """
    from nuscenes_data_engine.data_engine.chat import agent
    from nuscenes_data_engine.data_engine.chat.transports import TransportError, make_transport

    settings = request.app.state.settings
    con = _get_chat_catalog(request)
    try:
        engine = _get_search_engine(request)
    except HTTPException:
        engine = None  # SQL-only chat still works without the LanceDB store
    try:
        transport = make_transport(settings)
    except (TransportError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Chat unavailable: {exc}") from exc
    graph_driver = _get_graph_driver(request)
    history = [turn.model_dump() for turn in body.history]

    def event_stream() -> Iterator[str]:
        events: queue.SimpleQueue[tuple[str, str] | None] = queue.SimpleQueue()

        def worker() -> None:
            try:
                result = agent.answer(
                    body.message,
                    transport=transport,
                    con=con,
                    search_engine=engine,
                    history=history,
                    log_path=Path(settings.chat_log_path),
                    graph_driver=graph_driver,
                    graph_database=settings.neo4j_database,
                    on_turn=lambda: events.put(("turn", "")),
                    on_token=lambda delta: events.put(("token", json.dumps(delta))),
                    on_step=lambda step: events.put(("step", json.dumps(step, default=str))),
                )
                final = _assemble_chat_response(result)
                events.put(("final", final.model_dump_json()))
            except Exception as exc:  # see docstring: headers are already sent
                logger.exception("chat_stream worker failed")
                message = (
                    str(exc)
                    if isinstance(exc, TransportError)
                    else "internal error — see server logs"
                )
                events.put(("error", message))
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while (item := events.get()) is not None:
            kind, payload = item
            yield f"event: {kind}\ndata: {payload}\n\n"
        thread.join()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/predict/annotated")
async def predict_annotated(request: Request, file: UploadFile = File(...)) -> Response:
    """Run detection and return the input image with boxes drawn, as PNG."""
    start = time.perf_counter()
    result, img = _infer(request, await file.read())
    ok, buf = cv2.imencode(".png", result.plot())
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    _capture(request, img, len(result.boxes), (time.perf_counter() - start) * 1000)
    return Response(content=buf.tobytes(), media_type="image/png")
