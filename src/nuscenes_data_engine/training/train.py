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


def _run_name(weights: str, imgsz: int, epochs: int, suffix: str | None = None) -> str:
    name = f"{Path(weights).stem}_imgsz{imgsz}_e{epochs}"
    return f"{name}_{suffix}" if suffix else name


def _setup_wandb_env(settings: Any) -> bool:
    """Export W&B env vars so Ultralytics' callback (incl. DDP workers) targets our project.

    W&B is logged by Ultralytics' own callback rather than a manual ``wandb.init`` in this
    (main) process: under DDP the training loop runs in worker subprocesses, so a
    main-process run would stay empty while the worker created a second, auto-named run.
    The workers inherit these env vars. Returns False if W&B can't run (no key, online).
    """
    if settings.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", settings.wandb_api_key)
    os.environ.setdefault("WANDB_MODE", settings.wandb_mode)
    if settings.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", settings.wandb_entity)

    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("W&B requested but `wandb` is not installed (uv sync --extra train).")
        return False
    if settings.wandb_mode != "offline" and not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "W&B enabled but no WANDB_API_KEY set; skipping (or use WANDB_MODE=offline)."
        )
        return False
    return True


def _extract_metrics(results: Any, save_dir: Path) -> dict[str, float]:
    """Validation metrics from the results, falling back to results.csv (DDP-safe).

    Ultralytics returns a results object in single-GPU mode but a plain ``dict`` from the
    main process under DDP, so handle both.
    """
    raw = results if isinstance(results, dict) else getattr(results, "results_dict", {})
    metrics = {
        str(k).replace("metrics/", "").replace("(B)", ""): float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float))
    }
    if metrics:
        return metrics
    csv = save_dir / "results.csv"
    if csv.exists():
        import pandas as pd

        last = pd.read_csv(csv).iloc[-1]
        metrics = {
            c.strip().replace("metrics/", "").replace("(B)", ""): float(last[c])
            for c in last.index
            if "metrics/" in c
        }
    return metrics


def train_model(
    config: dict[str, Any],
    data_yaml: Path,
    *,
    data_version: str,
    epochs: int | None = None,
    batch: int | None = None,
    model: str | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    wandb_enabled: bool | None = None,
    run_suffix: str | None = None,
) -> dict[str, Any]:
    """Fine-tune YOLO and log the run to MLflow (and optionally W&B); return a summary.

    Args:
        config: Parsed ``configs/train.yaml`` contents.
        data_yaml: Path to the YOLO dataset descriptor (from :func:`..dataset.build_yolo_dataset`).
        data_version: Content hash of the processed dataset (provenance).
        epochs: Override ``config['train']['epochs']``.
        batch: Override ``config['train']['batch']``.
        model: Override ``config['model']['weights']`` (e.g. ``"yolov8s.pt"``) — sweeps.
        imgsz: Override ``config['model']['imgsz']`` (e.g. ``960``) — sweeps.
        device: Ultralytics device string (e.g. ``"0"``, ``"0,1"``, ``"cpu"``).
        wandb_enabled: Override ``config['tracking']['wandb']['enabled']``.
        run_suffix: Distinguish runs sharing (weights, imgsz, epochs) — e.g. active
            learning arms — which would otherwise clobber each other's output dirs.
    """
    model_cfg = config["model"]
    train_cfg = config["train"]
    tracking = config.get("tracking", {})
    aug = train_cfg.get("augment", {}) or {}
    experiment = tracking.get("experiment_name", "nuscenes-yolo")

    settings = get_settings()
    use_wandb = (
        wandb_enabled
        if wandb_enabled is not None
        else bool(tracking.get("wandb", {}).get("enabled", False))
    )
    if use_wandb:
        use_wandb = _setup_wandb_env(settings)

    configure_ultralytics(enable_wandb=use_wandb)
    import mlflow
    from ultralytics import YOLO  # imported after config redirect

    n_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 1))
    n_batch = int(batch if batch is not None else train_cfg.get("batch", 16))
    weights = model or str(model_cfg.get("weights", "yolov8n.pt"))
    n_imgsz = int(imgsz if imgsz is not None else model_cfg.get("imgsz", 640))
    run_name = _run_name(weights, n_imgsz, n_epochs, run_suffix)

    # Ultralytics uses `project` as BOTH the on-disk output dir and the W&B project name
    # (slashes sanitized), so with W&B on it must stay a clean relative name — the run
    # lands in the user's `nuscenes-data-engine` W&B project. Without W&B, use the
    # absolute repo runs dir: Ultralytics >=8.4 anchors relative project paths under its
    # own runs_dir/task, which would scatter outputs (runs/detect/runs/<name>).
    project = settings.wandb_project if use_wandb else str(RUNS_DIR)

    tracking_uri = settings.mlflow_tracking_uri or tracking.get("mlflow_tracking_uri")
    _set_up_experiment(mlflow, tracking_uri, experiment)

    cos_lr = bool(train_cfg.get("cos_lr", False))
    params = {
        "weights": weights,
        "imgsz": n_imgsz,
        "epochs": n_epochs,
        "batch": n_batch,
        "lr0": train_cfg.get("lr0"),
        "optimizer": train_cfg.get("optimizer", "auto"),
        "cos_lr": cos_lr,
        "seed": train_cfg.get("seed", 0),
        "classes": ",".join(model_cfg.get("classes", [])),
        "data_version": data_version,
        "data_yaml": str(data_yaml),
        **{f"aug_{k}": v for k, v in aug.items()},
    }

    logger.info("MLflow run '%s' -> %s | W&B: %s", run_name, tracking_uri, use_wandb)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        yolo = YOLO(weights)
        # W&B is logged by Ultralytics' callback (enabled via configure_ultralytics),
        # so it works in both single-GPU and DDP; provenance (data_version) is in MLflow.
        aug_keys = ("hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic", "mixup", "copy_paste")
        results = yolo.train(
            data=str(data_yaml),
            epochs=n_epochs,
            imgsz=n_imgsz,
            batch=n_batch,
            lr0=train_cfg.get("lr0", 0.01),
            optimizer=train_cfg.get("optimizer", "auto"),
            cos_lr=cos_lr,
            seed=train_cfg.get("seed", 0),
            device=device if device is not None else "0",
            project=project,
            name=run_name,
            exist_ok=True,
            plots=True,
            **{k: aug[k] for k in aug_keys if k in aug},
        )

        # yolo.train() returns a results object (single-GPU) or a dict (DDP); the trainer
        # always carries the resolved output dir.
        save_dir = Path(yolo.trainer.save_dir)
        metrics = _extract_metrics(results, save_dir)
        mlflow.log_metrics(metrics)

        best = save_dir / "weights" / "best.pt"
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="weights")
        # Plots + sample val-prediction images produced by Ultralytics.
        mlflow.log_artifacts(str(save_dir), artifact_path="ultralytics_run")

    summary = {
        "run_name": run_name,
        "save_dir": str(save_dir),
        "best_weights": str(best),
        "data_version": data_version,
        "metrics": metrics,
        "wandb": use_wandb,
    }
    logger.info("Training done: %s | metrics=%s", run_name, metrics)
    return summary
