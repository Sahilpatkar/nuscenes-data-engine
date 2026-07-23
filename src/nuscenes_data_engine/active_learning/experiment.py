"""Experiment arms: build the arm's dataset, train, evaluate, record results.

Arms: ``baseline`` (25% train scenes), ``mined`` (baseline + mined frames),
``random`` (baseline + equal-sized random control). Each arm gets its own YOLO
dataset dir and run-name suffix; the val split is identical across arms by
construction and asserted at result-merge time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nuscenes_data_engine.config import get_settings, load_yaml

logger = logging.getLogger("nuscenes_data_engine")

ARMS = ("baseline", "mined", "random")


def resolve_arm_frames(state_dir: Path, processed_dir: Path, cfg: dict[str, Any], arm: str) -> set[str]:
    """The train-frame token set for one arm."""
    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    split = pd.read_parquet(state_dir / "split.parquet")
    baseline = frames_for_scenes(
        processed_dir, set(split[split["role"] == "baseline"]["scene_name"]), channel
    )
    if arm == "baseline":
        return baseline
    extra_file = {"mined": "mined.parquet", "random": "random.parquet"}[arm]
    extra = set(pd.read_parquet(state_dir / extra_file)["sample_data_token"])
    return baseline | extra


def overlay_train_config(base_config_path: Path, al_train: dict[str, Any]) -> dict[str, Any]:
    """train.yaml with the experiment's model/imgsz/epochs/batch overlaid."""
    cfg = load_yaml(base_config_path)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg["model"]["weights"] = al_train.get("model", cfg["model"].get("weights", "yolov8n.pt"))
    cfg["model"]["imgsz"] = int(al_train.get("imgsz", cfg["model"].get("imgsz", 640)))
    cfg["train"]["epochs"] = int(al_train.get("epochs", cfg["train"].get("epochs", 20)))
    cfg["train"]["batch"] = int(al_train.get("batch", cfg["train"].get("batch", 16)))
    return cfg


def merge_results(state_dir: Path, arm: str, record: dict[str, Any]) -> dict[str, Any]:
    """Add one arm's record to results.json, asserting val-set identity across arms."""
    results_path = state_dir / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text()) if results_path.is_file() else {}
    )
    for other, other_record in results.items():
        if other != arm and other_record.get("val_images") != record.get("val_images"):
            raise AssertionError(
                f"val split differs between arms {other} ({other_record.get('val_images')}) "
                f"and {arm} ({record.get('val_images')}) — comparison invalid"
            )
    results[arm] = record
    results_path.write_text(json.dumps(results, indent=2))
    return results


def run_arm(
    config_path: Path,
    *,
    arm: str,
    device: str | None = None,
    epochs: int | None = None,
    wandb_enabled: bool | None = None,
    force_rebuild: bool = False,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the arm's dataset, train, evaluate, and merge its results."""
    from nuscenes_data_engine.evaluation.evaluate import run_evaluation
    from nuscenes_data_engine.training.dataset import build_yolo_dataset, compute_data_version
    from nuscenes_data_engine.training.train import train_model

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r} (expected one of {ARMS})")
    cfg = load_yaml(config_path)
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    settings = get_settings()
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")

    tokens = resolve_arm_frames(state_dir, processed, cfg, arm)
    logger.info("Arm %s: %d train frames", arm, len(tokens))

    arm_dir = state_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    data_yaml, stats = build_yolo_dataset(
        processed,
        Path(settings.nuscenes_dataroot),
        arm_dir / "yolo",
        cameras=[channel],
        train_frames=tokens,
        force=force_rebuild,
    )

    train_cfg = overlay_train_config(
        Path(cfg.get("train_base_config", "configs/train.yaml")), cfg.get("train", {})
    )
    if epochs is not None:
        train_cfg["train"]["epochs"] = epochs
    train_cfg["data"] = {"processed_dir": str(processed), "yolo_dir": str(arm_dir / "yolo")}
    arm_train_config = arm_dir / "train_config.yaml"
    arm_train_config.write_text(yaml.safe_dump(train_cfg))

    summary = train_model(
        train_cfg,
        data_yaml,
        data_version=compute_data_version(processed),
        device=device,
        wandb_enabled=wandb_enabled,
        run_suffix=f"al-{arm}",
    )

    report = run_evaluation(
        Path(cfg.get("eval", {}).get("config", "configs/eval.yaml")),
        arm_train_config,
        weights=Path(summary["best_weights"]),
        device=device or "0",
        register=False,
    )
    record = {
        "n_train_images": stats.get("train_images"),
        "val_images": stats.get("val_images"),
        "run_name": summary["run_name"],
        "best_weights": str(summary["best_weights"]),
        "overall": report["overall"],
        "night": report["slices"].get("time_of_day/night", {}),
        "slices": {k: {m: v[m] for m in ("mAP50", "mAP50-95")} for k, v in report["slices"].items()},
    }
    results = merge_results(state_dir, arm, record)
    logger.info(
        "Arm %s: overall mAP50-95 %.3f, night mAP50-95 %.3f",
        arm,
        record["overall"]["mAP50-95"],
        record["night"].get("mAP50-95", float("nan")),
    )
    return results
