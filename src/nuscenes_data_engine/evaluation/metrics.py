"""Detection metrics: overall and per-class mAP / precision / recall via Ultralytics val."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nuscenes_data_engine.training.runtime import configure_ultralytics

logger = logging.getLogger("nuscenes_data_engine")


def evaluate_model(
    weights: Path,
    data_yaml: Path,
    *,
    device: str = "0",
    imgsz: int = 640,
) -> dict[str, Any]:
    """Run validation and return aggregate + per-class detection metrics.

    Args:
        weights: Path to the model weights (``best.pt``).
        data_yaml: Ultralytics dataset descriptor whose ``val`` split is evaluated.
        device: Ultralytics device string.
        imgsz: Inference image size.

    Returns:
        ``{mAP50, mAP50-95, precision, recall, per_class: {name: mAP50-95}}``.
    """
    configure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml),
        split="val",
        device=device,
        imgsz=imgsz,
        verbose=False,
        plots=False,
    )
    box = m.box
    names = m.names
    per_class = {names[i]: round(float(v), 4) for i, v in enumerate(box.maps)}
    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }
