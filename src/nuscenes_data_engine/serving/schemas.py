"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single 2D detection."""

    label: str = Field(description="Predicted category name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence.")
    bbox: tuple[float, float, float, float] = Field(
        description="Axis-aligned box as (x_min, y_min, x_max, y_max) in pixels."
    )


class PredictResponse(BaseModel):
    """Response body for the ``/predict`` endpoint."""

    detections: list[Detection]
    model_version: str = Field(description="Registry version of the serving model.")


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint."""

    status: str = "ok"
    model_loaded: bool = False
