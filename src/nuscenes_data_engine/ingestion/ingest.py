"""Phase 1 ingestion orchestration: nuScenes -> validated Parquet metadata tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.ingestion.parquet import write_parquet
from nuscenes_data_engine.ingestion.parse import DEFAULT_CAMERAS, flatten, load_nusc

logger = logging.getLogger("nuscenes_data_engine")

DEFAULT_CONFIG = Path("configs/data.yaml")


def run_ingestion(
    config_path: Path = DEFAULT_CONFIG,
    *,
    limit_scenes: int | None = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline and write Parquet tables.

    Data root and version come from environment/`.env` (``Settings``), falling back to
    the values in ``config_path``. Camera list, projection thresholds, and output file
    names come from ``config_path``.

    Args:
        config_path: Path to ``data.yaml``.
        limit_scenes: If set, only the first N scenes are processed (fast dev runs).

    Returns:
        A summary dict (row counts and output paths).
    """
    settings = get_settings()
    cfg = load_yaml(config_path)

    source = cfg.get("source", {})
    dataroot = Path(settings.nuscenes_dataroot or source.get("dataroot"))
    version = settings.nuscenes_version or source.get("version")

    cameras = tuple(cfg.get("cameras", DEFAULT_CAMERAS))
    proj = cfg.get("projection", {})
    out = cfg.get("output", {})
    processed_dir = Path(out.get("processed_dir", settings.processed_dir))
    parquet_names = out.get("parquet", {})
    annotations_path = processed_dir / parquet_names.get("annotations", "annotations.parquet")
    samples_path = processed_dir / parquet_names.get("samples", "samples.parquet")

    nusc = load_nusc(dataroot, version)
    images, annotations = flatten(
        nusc,
        cameras=cameras,
        visibility_min=int(proj.get("visibility_min", 1)),
        min_box_area_px=float(proj.get("min_box_area_px", 0.0)),
        clip_to_image=bool(proj.get("clip_to_image", True)),
        limit_scenes=limit_scenes,
    )

    n_samples = write_parquet(images, samples_path)
    n_annotations = write_parquet(annotations, annotations_path)

    summary = {
        "version": version,
        "dataroot": str(dataroot),
        "scenes_processed": limit_scenes if limit_scenes else len(nusc.scene),
        "images": n_samples,
        "annotations": n_annotations,
        "samples_parquet": str(samples_path),
        "annotations_parquet": str(annotations_path),
    }
    logger.info(
        "Ingestion wrote %d image rows -> %s and %d annotation rows -> %s",
        n_samples,
        samples_path,
        n_annotations,
        annotations_path,
    )
    return summary
