"""Condition-sliced evaluation — the project's key differentiator.

Breaks metrics down by scene condition (night vs. day, rain vs. clear) using
nuScenes scene descriptions, mirroring how AV companies evaluate perception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def assign_slices(scene_description: str, slice_config: dict[str, Any]) -> dict[str, str]:
    """Map a scene description to its slice labels (e.g. ``{"time_of_day": "night"}``).

    Args:
        scene_description: The nuScenes ``scene.description`` string.
        slice_config: The ``slices`` block from ``configs/eval.yaml``.
    """
    # TODO(Phase 3): substring-match include/exclude rules against the description.
    raise NotImplementedError


def sliced_metrics(
    predictions: Path,
    ground_truth: Path,
    slice_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute metrics for each condition slice."""
    # TODO(Phase 3): group eval samples by slice and call metrics.compute_metrics.
    raise NotImplementedError
