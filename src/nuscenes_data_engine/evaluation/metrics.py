"""Detection metrics: per-class mAP and precision/recall on a held-out split."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_metrics(
    predictions: Path,
    ground_truth: Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute overall and per-class mAP / precision / recall.

    Args:
        predictions: Model predictions for the eval split.
        ground_truth: Ground-truth labels for the eval split.
        iou_threshold: IoU threshold for a true positive.

    Returns:
        Nested metrics dict (overall + per-class).
    """
    # TODO(Phase 3): compute mAP / P / R (ultralytics val or a custom matcher).
    raise NotImplementedError
