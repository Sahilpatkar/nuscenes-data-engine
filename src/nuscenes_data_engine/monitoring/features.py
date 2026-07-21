"""Per-image drift features: brightness, resolution, detection count.

Deliberately Evidently-free (cv2/numpy/pandas only — all base deps) so the reference
builder can run on the GPU server, which has the images, without extra installs. The
same :func:`image_brightness` is applied to decoded frames at serve time, so offline
reference and online capture share one definition — no train/serve skew.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

FEATURE_COLUMNS: tuple[str, ...] = ("brightness", "width", "height", "n_boxes")

# Serving-capture JSONL field -> feature-table column.
_JSONL_RENAMES = {"image_width": "width", "image_height": "height", "n_detections": "n_boxes"}


def image_brightness(img_bgr: np.ndarray[Any, Any]) -> float:
    """Mean luma-weighted gray value of a BGR image (0 dark .. 255 bright).

    Gray (Rec.601) approximates perceived brightness better than HSV's V channel,
    which is max(R,G,B) and overweights saturated pixels.
    """
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())


def compute_features(
    samples: pd.DataFrame,
    dataroot: Path | None,
    *,
    sample_images: int | None = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute the drift feature table for (a sample of) ``samples.parquet`` rows.

    Rows are sampled *first* so every emitted row has all four features. Brightness
    needs the image pixels under ``dataroot``; with ``dataroot=None`` or missing files
    it degrades to NaN (metadata-only mode for machines without the dataset).
    """
    if sample_images is not None and len(samples) > sample_images:
        samples = samples.sample(n=sample_images, random_state=seed)
    out = samples[["sample_data_token", "channel", "is_night", "width", "height", "n_boxes"]].copy()

    brightness: list[float] = []
    misses = 0
    for filename in samples["filename"]:
        img = cv2.imread(str(dataroot / filename)) if dataroot is not None else None
        if img is None:
            misses += 1
            brightness.append(float("nan"))
        else:
            brightness.append(image_brightness(img))
    if misses and dataroot is not None:
        logger.warning(
            "Brightness unavailable for %d/%d images under %s", misses, len(out), dataroot
        )
    out["brightness"] = brightness
    return out


def build_reference(
    processed_dir: Path,
    dataroot: Path | None,
    out_path: Path,
    *,
    condition: str = "all",
    sample_images: int | None = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a feature table from ``samples.parquet`` and write it to ``out_path``.

    ``condition`` filters on the materialized ``is_night`` column ("all"/"day"/"night"),
    which is by construction equivalent to the eval.yaml time-of-day slice rules —
    both come from ``"night" in scene_description.lower()`` at ingest time.
    """
    if condition not in ("all", "day", "night"):
        raise ValueError(f"condition must be all|day|night, got {condition!r}")
    samples = pd.read_parquet(processed_dir / "samples.parquet")
    if condition != "all":
        samples = samples[samples["is_night"] == (condition == "night")]
    features = compute_features(samples, dataroot, sample_images=sample_images, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Wrote %d %s-condition feature rows -> %s", len(features), condition, out_path)
    return features


def load_feature_table(path: Path) -> pd.DataFrame:
    """Load a feature table: a built parquet, or a serving-capture JSONL."""
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True).rename(columns=_JSONL_RENAMES)
    return pd.read_parquet(path)
