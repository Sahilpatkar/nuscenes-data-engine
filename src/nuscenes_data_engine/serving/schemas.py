"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """A single 2D detection."""

    label: str = Field(description="Predicted category name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence.")
    bbox: tuple[float, float, float, float] = Field(
        description="Axis-aligned box as (x_min, y_min, x_max, y_max) in pixels."
    )


class PredictResponse(BaseModel):
    """Response body for the ``/predict`` endpoint."""

    # `model_version` collides with pydantic's protected `model_` namespace.
    model_config = ConfigDict(protected_namespaces=())

    detections: list[Detection]
    model_version: str = Field(description="Registry version of the serving model.")
    image_width: int = Field(description="Decoded input image width in pixels.")
    image_height: int = Field(description="Decoded input image height in pixels.")
    n_detections: int = Field(description="Number of detections returned.")


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool = False
    model_version: str | None = None
    search_ready: bool = False


class SearchResult(BaseModel):
    """One frame returned by the semantic-search endpoints."""

    sample_data_token: str
    scene_name: str
    scene_description: str
    channel: str
    timestamp: int
    location: str
    is_night: bool
    is_rain: bool
    score: float = Field(description="Cosine similarity to the query (higher = closer).")
    thumbnail_b64: str = Field(description="Base64 JPEG thumbnail of the frame.")


class SearchResponse(BaseModel):
    """Response body for the ``/search*`` endpoints."""

    model_config = ConfigDict(protected_namespaces=())

    results: list[SearchResult]
    query: str = Field(description="The text query, or a marker for image/similar queries.")
    embedding_model: str
