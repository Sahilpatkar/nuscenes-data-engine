"""Fine-tune a pretrained YOLO detector and log the run to MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_model(config: dict[str, Any], yolo_data_yaml: Path) -> dict[str, Any]:
    """Fine-tune YOLO and return a summary of metrics + artifact locations.

    Args:
        config: Parsed ``configs/train.yaml`` contents.
        yolo_data_yaml: Path to the YOLO dataset descriptor.

    Returns:
        Run summary (metrics, weights path, data version hash) for MLflow logging.
    """
    # TODO(Phase 2): run ultralytics YOLO(...).train(...); log params, metrics,
    # weights, and sample predictions to MLflow.
    raise NotImplementedError
