"""Flatten the nuScenes relational JSON tables into denormalized records.

Walks `scene -> sample -> sample_data -> sample_annotation` joined with
`calibrated_sensor`, `ego_pose`, and `category`, producing one flat record per
(keyframe camera image, annotation) pair plus per-sample scene/weather context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_tables(dataroot: Path, version: str) -> dict[str, list[dict[str, Any]]]:
    """Load the raw nuScenes metadata tables via the devkit.

    Args:
        dataroot: Path to the nuScenes dataset root.
        version: Dataset version, e.g. ``"v1.0-trainval"``.

    Returns:
        Mapping of table name -> list of row dicts.
    """
    # TODO(Phase 1): use nuscenes.NuScenes(version, dataroot) to load tables.
    raise NotImplementedError


def flatten_annotations(dataroot: Path, version: str) -> list[dict[str, Any]]:
    """Produce one flat record per (camera keyframe, annotation) pair."""
    # TODO(Phase 1): join tables and emit denormalized annotation records.
    raise NotImplementedError
