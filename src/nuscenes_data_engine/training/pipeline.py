"""Dagster job wiring the training stages into a reproducible pipeline.

Job graph: ``pull_data_version -> prepare_dataset -> train -> evaluate -> log_artifacts``.
The ops wrap the same primitives the CLI uses (:mod:`.dataset`, :mod:`.train`), so a
Dagster run and a `nuscenes-data-engine train` run do the same work.

Run with:  uv run dagster dev -m nuscenes_data_engine.training.pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dagster import Config, Definitions, OpExecutionContext, job, op

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.training.dataset import build_yolo_dataset, compute_data_version
from nuscenes_data_engine.training.train import train_model


class PipelineConfig(Config):
    """Run configuration for the training job."""

    config_path: str = "configs/train.yaml"
    limit_scenes: int | None = None
    cameras: list[str] | None = None
    epochs: int | None = None
    batch: int | None = None
    device: str | None = None
    wandb_enabled: bool | None = None


def _processed_dir(config_path: str) -> Path:
    settings = get_settings()
    cfg = load_yaml(Path(config_path))
    return Path(cfg.get("data", {}).get("processed_dir", settings.processed_dir))


@op
def pull_data_version(context: OpExecutionContext, config: PipelineConfig) -> str:
    """Fingerprint the processed dataset (provenance for the run)."""
    version = compute_data_version(_processed_dir(config.config_path))
    context.log.info("data version %s", version)
    return version


@op
def prepare_dataset(context: OpExecutionContext, config: PipelineConfig) -> str:
    """Materialize the YOLO dataset; return the data.yaml path."""
    settings = get_settings()
    cfg = load_yaml(Path(config.config_path))
    data_cfg = cfg.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", settings.processed_dir))
    yolo_dir = Path(data_cfg.get("yolo_dir", processed_dir / "yolo"))
    data_yaml, stats = build_yolo_dataset(
        processed_dir,
        Path(settings.nuscenes_dataroot),
        yolo_dir,
        cameras=config.cameras,
        limit_scenes=config.limit_scenes,
    )
    context.log.info("dataset stats %s", stats)
    return str(data_yaml)


@op
def train(
    context: OpExecutionContext,
    config: PipelineConfig,
    data_yaml: str,
    data_version: str,
) -> dict[str, Any]:
    """Fine-tune YOLO and log to MLflow."""
    cfg = load_yaml(Path(config.config_path))
    return train_model(
        cfg,
        Path(data_yaml),
        data_version=data_version,
        epochs=config.epochs,
        batch=config.batch,
        device=config.device,
        wandb_enabled=config.wandb_enabled,
    )


@op
def evaluate(context: OpExecutionContext, summary: dict[str, Any]) -> dict[str, Any]:
    """Surface the run's validation metrics (Phase 3 adds condition-sliced eval)."""
    context.log.info("metrics %s", summary.get("metrics"))
    return summary


@op
def log_artifacts(context: OpExecutionContext, summary: dict[str, Any]) -> None:
    """Weights + plots are logged to MLflow during training; record the location."""
    context.log.info("run '%s' logged; weights at %s", summary["run_name"], summary["best_weights"])


@job
def training_job() -> None:
    """pull_data_version -> prepare_dataset -> train -> evaluate -> log_artifacts."""
    version = pull_data_version()
    data_yaml = prepare_dataset()
    summary = train(data_yaml=data_yaml, data_version=version)
    log_artifacts(evaluate(summary))


defs = Definitions(jobs=[training_job])
