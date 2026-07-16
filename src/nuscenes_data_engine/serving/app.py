"""FastAPI application serving the current production detection model.

Endpoints:
    GET  /health   liveness + whether a model is loaded.
    POST /predict  accept an image, return detections (JSON, optional annotated image).
"""

from __future__ import annotations

from fastapi import FastAPI

from nuscenes_data_engine import __version__
from nuscenes_data_engine.serving.schemas import HealthResponse

app = FastAPI(title="nuScenes Data Engine — Detection API", version=__version__)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    # TODO(Phase 4): report model_loaded based on registry state.
    return HealthResponse(status="ok", model_loaded=False)


# TODO(Phase 4): add POST /predict that loads the `production` model from the MLflow
# registry, runs inference on the uploaded image, and returns a PredictResponse.
