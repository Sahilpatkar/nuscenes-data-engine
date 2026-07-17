"""Plain orchestration of the training pipeline (used by the CLI).

pull data version -> prepare YOLO dataset -> train + log to MLflow. The Dagster job in
:mod:`.pipeline` wraps the same primitives for orchestrated runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.training.dataset import build_yolo_dataset, compute_data_version
from nuscenes_data_engine.training.train import train_model

logger = logging.getLogger("nuscenes_data_engine")

DEFAULT_CONFIG = Path("configs/train.yaml")


def run_training(
    config_path: Path = DEFAULT_CONFIG,
    *,
    limit_scenes: int | None = None,
    cameras: list[str] | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    wandb_enabled: bool | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Run prepare -> train end to end and return the run summary."""
    settings = get_settings()
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})

    processed_dir = Path(data_cfg.get("processed_dir", settings.processed_dir))
    yolo_dir = Path(data_cfg.get("yolo_dir", processed_dir / "yolo"))
    dataroot = Path(settings.nuscenes_dataroot)

    data_version = compute_data_version(processed_dir)
    logger.info("Data version: %s", data_version)

    data_yaml, stats = build_yolo_dataset(
        processed_dir,
        dataroot,
        yolo_dir,
        cameras=cameras,
        limit_scenes=limit_scenes,
        force=force_rebuild,
    )
    summary = train_model(
        cfg,
        data_yaml,
        data_version=data_version,
        epochs=epochs,
        batch=batch,
        device=device,
        wandb_enabled=wandb_enabled,
    )
    summary["dataset_stats"] = stats
    return summary
