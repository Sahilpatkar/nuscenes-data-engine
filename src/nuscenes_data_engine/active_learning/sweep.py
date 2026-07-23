"""Failure sweep: run the baseline detector over the official val split.

Diagnoses where the baseline fails (false negatives + low-confidence hits per frame).
Deployment-proxy framing — val stands in for "frames seen in deployment"; the frames
mined from these failures come strictly from the train-scene pool, so no val frame
ever enters training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nuscenes_data_engine.active_learning.matching import match_frame
from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.ingestion.categories import CLASS_TO_INDEX

logger = logging.getLogger("nuscenes_data_engine")


def summarize_failures(failures: pd.DataFrame) -> dict[str, float]:
    """Top-line sweep stats (logged and W&B-reported)."""
    day = failures[~failures["is_night"]]
    night = failures[failures["is_night"]]
    return {
        "n_frames": float(len(failures)),
        "mean_failure_score": float(failures["failure_score"].mean()),
        "mean_failure_score_day": float(day["failure_score"].mean()) if len(day) else 0.0,
        "mean_failure_score_night": float(night["failure_score"].mean()) if len(night) else 0.0,
        "total_fn": float(failures["n_fn"].sum()),
        "total_low_conf": float(failures["n_low_conf"].sum()),
    }


def run_sweep(
    config_path: Path,
    *,
    weights: Path,
    device: str = "0",
    limit: int | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Sweep the detector over val frames; write and return failures.parquet."""
    from nuscenes.utils.splits import create_splits_scenes

    from nuscenes_data_engine.training.runtime import configure_ultralytics

    cfg = load_yaml(config_path)
    sweep_cfg = cfg.get("sweep", {})
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    settings = get_settings()

    samples = pd.read_parquet(
        processed / "samples.parquet",
        columns=["sample_data_token", "scene_name", "channel", "filename", "is_night"],
    )
    val_scenes = set(create_splits_scenes()["val"])
    frames = samples[
        samples["scene_name"].isin(val_scenes) & (samples["channel"] == channel)
    ].sort_values("sample_data_token", ignore_index=True)
    if limit is not None:
        frames = frames.head(limit)

    annotations = pd.read_parquet(
        processed / "annotations.parquet",
        columns=[
            "sample_data_token", "category_group", "visibility_token",
            "x_min", "y_min", "x_max", "y_max",
        ],
    )
    annotations = annotations.dropna(subset=["category_group"])
    visibility_min = sweep_cfg.get("visibility_min")
    if visibility_min is not None:
        visibility = pd.to_numeric(annotations["visibility_token"], errors="coerce")
        annotations = annotations[visibility >= int(visibility_min)]
    annotations = annotations[annotations["sample_data_token"].isin(frames["sample_data_token"])]
    gt_by_token = {token: group for token, group in annotations.groupby("sample_data_token")}

    configure_ultralytics()  # before importing ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    dataroot = Path(settings.nuscenes_dataroot)
    iou = float(sweep_cfg.get("iou", 0.5))
    conf_hit = float(sweep_cfg.get("conf_hit", 0.4))
    batch_size = int(sweep_cfg.get("batch", 64))

    records = []
    rows = frames.to_dict("records")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        paths = [str(dataroot / row["filename"]) for row in batch]
        results = model.predict(
            paths, imgsz=int(cfg.get("train", {}).get("imgsz", 640)),
            conf=float(sweep_cfg.get("conf_low", 0.05)), device=device, verbose=False,
        )
        for row, result in zip(batch, results, strict=True):
            boxes = result.boxes
            pred_boxes = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
            pred_classes = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)
            pred_conf = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
            gt = gt_by_token.get(row["sample_data_token"])
            if gt is None:
                gt_boxes, gt_classes = np.zeros((0, 4)), np.zeros(0, int)
            else:
                gt_boxes = gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
                gt_classes = gt["category_group"].map(CLASS_TO_INDEX).to_numpy(dtype=int)
            failure = match_frame(
                pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes,
                iou=iou, conf_hit=conf_hit,
            )
            records.append(
                {
                    "sample_data_token": row["sample_data_token"],
                    "scene_name": row["scene_name"],
                    "is_night": row["is_night"],
                    "n_gt": failure.n_gt,
                    "n_matched": failure.n_matched,
                    "n_fn": failure.n_fn,
                    "n_low_conf": failure.n_low_conf,
                    "failure_score": failure.failure_score,
                }
            )
        if (start // batch_size) % 10 == 0:
            logger.info("Sweep: %d/%d frames", min(start + batch_size, len(rows)), len(rows))

    failures = pd.DataFrame(records)
    state_dir.mkdir(parents=True, exist_ok=True)
    failures.to_parquet(state_dir / "failures.parquet", index=False)
    stats = summarize_failures(failures)
    logger.info("Sweep stats: %s", stats)
    return failures
