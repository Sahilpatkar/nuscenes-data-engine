"""Fine-tune a pretrained YOLO detector and log the run to MLflow (and optionally W&B).

Logs hyperparameters, the data-version hash, validation metrics, the best weights, and
the Ultralytics run directory (plots + sample prediction images). MLflow defaults to a
local SQLite store (`sqlite:///mlruns/mlflow.db`) so the GPU server needs no MLflow
server; sync `./mlruns` to the infra machine to view/register. Weights & Biases (cloud)
is enabled via config/`--wandb` and logs to the W&B project in `.env`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from nuscenes_data_engine.config import get_settings
from nuscenes_data_engine.training.runtime import REPO_ROOT, RUNS_DIR, configure_ultralytics

logger = logging.getLogger("nuscenes_data_engine")


def _set_up_experiment(mlflow: Any, tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the tracking store and ensure the experiment exists.

    For the local SQLite default, create the mlruns/ dir and give the experiment a local
    (rsyncable) artifact directory. Remote/HTTP tracking URIs are used as-is.
    """
    if tracking_uri.startswith("sqlite:"):
        (REPO_ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_location = None
        if tracking_uri.startswith("sqlite:"):
            artifact_dir = REPO_ROOT / "mlruns" / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_location = artifact_dir.as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def _run_name(model_cfg: dict[str, Any], imgsz: int, epochs: int) -> str:
    stem = Path(str(model_cfg.get("weights", "yolo"))).stem
    return f"{stem}_imgsz{imgsz}_e{epochs}"


def _init_wandb(run_name: str, params: dict[str, Any], experiment: str) -> Any | None:
    """Start a W&B run so Ultralytics logs into it; return the module or None.

    Requires ``WANDB_API_KEY`` (from ``.env`` or ``wandb login``) unless
    ``WANDB_MODE=offline``. Project/entity/mode come from settings.
    """
    settings = get_settings()
    if settings.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", settings.wandb_api_key)
    os.environ.setdefault("WANDB_MODE", settings.wandb_mode)

    try:
        import wandb
    except ImportError:
        logger.warning("W&B requested but `wandb` is not installed (uv sync --extra train).")
        return None
    if settings.wandb_mode != "offline" and not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "W&B enabled but no WANDB_API_KEY set; skipping (or use WANDB_MODE=offline)."
        )
        return None

    wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity or None,
        name=run_name,
        group=experiment,
        job_type="train",
        config=params,
    )
    if wandb.run is not None:
        logger.info("W&B run: %s", wandb.run.get_url())
    return wandb


def train_model(
    config: dict[str, Any],
    data_yaml: Path,
    *,
    data_version: str,
    epochs: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    wandb_enabled: bool | None = None,
) -> dict[str, Any]:
    """Fine-tune YOLO and log the run to MLflow (and optionally W&B); return a summary.

    Args:
        config: Parsed ``configs/train.yaml`` contents.
        data_yaml: Path to the YOLO dataset descriptor (from :func:`..dataset.build_yolo_dataset`).
        data_version: Content hash of the processed dataset (provenance).
        epochs: Override ``config['train']['epochs']`` (e.g. quick runs).
        device: Ultralytics device string (e.g. ``"0"``, ``"0,1"``, ``"cpu"``).
        wandb_enabled: Override ``config['tracking']['wandb']['enabled']``.
    """
    model_cfg = config["model"]
    train_cfg = config["train"]
    tracking = config.get("tracking", {})
    aug = train_cfg.get("augment", {}) or {}
    experiment = tracking.get("experiment_name", "nuscenes-yolo")

    use_wandb = (
        wandb_enabled
        if wandb_enabled is not None
        else bool(tracking.get("wandb", {}).get("enabled", False))
    )

    configure_ultralytics(enable_wandb=use_wandb)
    import mlflow
    from ultralytics import YOLO  # imported after config redirect

    n_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 1))
    n_batch = int(batch if batch is not None else train_cfg.get("batch", 16))
    imgsz = int(model_cfg.get("imgsz", 640))
    run_name = _run_name(model_cfg, imgsz, n_epochs)

    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri or tracking.get("mlflow_tracking_uri")
    _set_up_experiment(mlflow, tracking_uri, experiment)

    params = {
        "weights": model_cfg.get("weights"),
        "imgsz": imgsz,
        "epochs": n_epochs,
        "batch": n_batch,
        "lr0": train_cfg.get("lr0"),
        "optimizer": train_cfg.get("optimizer", "auto"),
        "seed": train_cfg.get("seed", 0),
        "classes": ",".join(model_cfg.get("classes", [])),
        "data_version": data_version,
        "data_yaml": str(data_yaml),
        **{f"aug_{k}": v for k, v in aug.items()},
    }

    wandb = _init_wandb(run_name, params, experiment) if use_wandb else None

    logger.info("MLflow run '%s' -> %s", run_name, tracking_uri)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = YOLO(str(model_cfg.get("weights", "yolov8n.pt")))
        results = model.train(
            data=str(data_yaml),
            epochs=n_epochs,
            imgsz=imgsz,
            batch=n_batch,
            lr0=train_cfg.get("lr0", 0.01),
            optimizer=train_cfg.get("optimizer", "auto"),
            seed=train_cfg.get("seed", 0),
            device=device if device is not None else "0",
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            plots=True,
            **{k: aug[k] for k in ("hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic") if k in aug},
        )

        save_dir = Path(results.save_dir)
        metrics = {
            k.replace("metrics/", "").replace("(B)", ""): float(v)
            for k, v in results.results_dict.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(metrics)

        best = save_dir / "weights" / "best.pt"
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="weights")
        # Plots + sample val-prediction images produced by Ultralytics.
        mlflow.log_artifacts(str(save_dir), artifact_path="ultralytics_run")

    wandb_url = None
    if wandb is not None and wandb.run is not None:
        wandb.summary.update(metrics)
        wandb_url = wandb.run.get_url()
        wandb.finish()

    summary = {
        "run_name": run_name,
        "save_dir": str(save_dir),
        "best_weights": str(best),
        "data_version": data_version,
        "metrics": metrics,
        "wandb_url": wandb_url,
    }
    logger.info("Training done: %s | metrics=%s", run_name, metrics)
    return summary
