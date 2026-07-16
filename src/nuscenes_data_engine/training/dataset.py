"""Convert the projected 2D annotations into an Ultralytics YOLO dataset."""

from __future__ import annotations

from pathlib import Path


def to_yolo_format(processed_dir: Path, yolo_dir: Path) -> Path:
    """Materialize a YOLO-format dataset (images + label txts + data.yaml).

    Args:
        processed_dir: Directory holding the Parquet annotation tables.
        yolo_dir: Output directory for the YOLO dataset.

    Returns:
        Path to the generated ``data.yaml`` describing the dataset.
    """
    # TODO(Phase 2): emit per-image label files and the YOLO data.yaml, using the
    # nuScenes official train/val scene split.
    raise NotImplementedError
