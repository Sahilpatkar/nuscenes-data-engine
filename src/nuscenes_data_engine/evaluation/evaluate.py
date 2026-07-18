"""Phase 3 evaluation orchestration: overall + condition-sliced metrics, MLflow logging,
and registry promotion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.evaluation.metrics import evaluate_model
from nuscenes_data_engine.evaluation.registry import register_and_promote
from nuscenes_data_engine.evaluation.slices import build_slice_val_datasets

logger = logging.getLogger("nuscenes_data_engine")

DEFAULT_EVAL_CONFIG = Path("configs/eval.yaml")
DEFAULT_TRAIN_CONFIG = Path("configs/train.yaml")
MODEL_NAME = "nuscenes-yolo-detector"


def find_latest_weights(search_root: Path = Path("runs")) -> Path | None:
    """Most recently modified ``best.pt`` under ``search_root`` (default training output)."""
    candidates = sorted(
        search_root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def run_evaluation(
    eval_config_path: Path = DEFAULT_EVAL_CONFIG,
    train_config_path: Path = DEFAULT_TRAIN_CONFIG,
    *,
    weights: Path | None = None,
    device: str = "0",
    imgsz: int | None = None,
    register: bool = False,
) -> dict[str, Any]:
    """Evaluate a model overall and per condition slice; log to MLflow; optionally register.

    Returns a report dict: ``{weights, overall, slices, gate, passed, registry?}``.
    """
    settings = get_settings()
    eval_cfg = load_yaml(eval_config_path)
    train_cfg = load_yaml(train_config_path)
    data_cfg = train_cfg.get("data", {})
    # Evaluate at the model's training resolution; override for a specific checkpoint.
    imgsz = int(imgsz if imgsz is not None else train_cfg.get("model", {}).get("imgsz", 640))

    processed_dir = Path(data_cfg.get("processed_dir", settings.processed_dir))
    yolo_dir = Path(data_cfg.get("yolo_dir", processed_dir / "yolo"))

    weights = weights or find_latest_weights()
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError("No weights found; train a model first or pass --weights.")
    weights = Path(weights)
    logger.info("Evaluating %s", weights)

    # Overall metrics on the full val split.
    overall = evaluate_model(weights, yolo_dir / "data.yaml", device=device, imgsz=imgsz)
    logger.info("Overall: mAP50=%.4f mAP50-95=%.4f", overall["mAP50"], overall["mAP50-95"])

    # Condition-sliced metrics.
    slice_datasets = build_slice_val_datasets(processed_dir, yolo_dir, eval_cfg.get("slices", {}))
    slices: dict[str, Any] = {}
    for name, ds in slice_datasets.items():
        m = evaluate_model(weights, ds["yaml"], device=device, imgsz=imgsz)
        m["n_images"] = ds["n_images"]
        slices[name] = m
        logger.info(
            "Slice %-18s mAP50=%.4f mAP50-95=%.4f (%d imgs)",
            name,
            m["mAP50"],
            m["mAP50-95"],
            m["n_images"],
        )

    # Promotion gate.
    gate = eval_cfg.get("promotion", {})
    night = slices.get("time_of_day/night", {})
    passed = overall["mAP50-95"] >= gate.get("min_overall_map", 0.0) and night.get(
        "mAP50-95", 0.0
    ) >= gate.get("min_night_map", 0.0)

    report: dict[str, Any] = {
        "weights": str(weights),
        "overall": overall,
        "slices": slices,
        "gate": gate,
        "passed": passed,
    }

    _log_and_register(report, weights, imgsz, register)
    logger.info("Promotion gate: %s", "PASSED" if passed else "NOT MET")
    return report


def _log_and_register(report: dict[str, Any], weights: Path, imgsz: int, register: bool) -> None:
    """Log evaluation metrics to MLflow and, if requested, register + promote the model."""
    import mlflow

    from nuscenes_data_engine.training.train import _set_up_experiment

    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri
    _set_up_experiment(mlflow, tracking_uri, "nuscenes-yolo")

    run_name = f"eval_{weights.parent.parent.name}"
    with mlflow.start_run(run_name=run_name) as run:
        overall = report["overall"]
        metrics = {f"overall_{k}": v for k, v in overall.items() if isinstance(v, (int, float))}
        for name, m in report["slices"].items():
            tag = name.split("/")[-1]  # night/day/rain/clear
            for k in ("mAP50", "mAP50-95", "precision", "recall"):
                metrics[f"{tag}_{k}"] = m[k]
        mlflow.log_metrics(metrics)
        mlflow.log_param("weights", str(weights))
        mlflow.log_param("gate_passed", report["passed"])
        mlflow.log_artifact(str(weights), artifact_path="weights")

        if register:
            report["registry"] = register_and_promote(
                mlflow, run.info.run_id, MODEL_NAME, "weights", passed=report["passed"]
            )
