"""FastAPI application serving the current production detection model.

Endpoints:
    GET  /health             liveness + which model version is loaded.
    POST /predict            accept an image, return detections as JSON.
    POST /predict/annotated  accept an image, return it annotated as PNG.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile

from nuscenes_data_engine import __version__
from nuscenes_data_engine.config import get_settings
from nuscenes_data_engine.monitoring.features import image_brightness
from nuscenes_data_engine.serving.model import load_production_model
from nuscenes_data_engine.serving.schemas import Detection, HealthResponse, PredictResponse

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
    yield


app = FastAPI(title="nuScenes Data Engine — Detection API", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model is not None,
        model_version=request.app.state.model_version,
    )


def _capture(request: Request, img: np.ndarray[Any, Any], n_detections: int, latency_ms: float) -> None:
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
