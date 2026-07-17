"""Convert the projected 2D annotations into an Ultralytics YOLO dataset.

Reads the processed Parquet tables and materializes a YOLO dataset directory:

    <out_dir>/
      images/{train,val}/*.jpg   -> symlinks to the read-only nuScenes JPEGs
      labels/{train,val}/*.txt   -> normalized `cls cx cy w h` boxes
      data.yaml                  -> Ultralytics dataset descriptor

The train/val split follows the official nuScenes scene split. Only annotations mapped
to one of the detector classes (see :mod:`..ingestion.categories`) become labels; images
with no target boxes are kept as backgrounds (empty label file).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from nuscenes.utils.splits import create_splits_scenes

from nuscenes_data_engine.ingestion.categories import CLASS_TO_INDEX, DETECTION_CLASSES

logger = logging.getLogger("nuscenes_data_engine")

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900


def compute_data_version(processed_dir: Path) -> str:
    """Content hash of the processed Parquet tables — the data-version fingerprint."""
    h = hashlib.sha256()
    for name in ("samples.parquet", "annotations.parquet"):
        h.update(name.encode())
        h.update((processed_dir / name).read_bytes())
    return h.hexdigest()[:16]


def _symlink(target: Path, link: Path) -> None:
    """Create/refresh a symlink ``link`` -> ``target``."""
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def build_yolo_dataset(
    processed_dir: Path,
    dataroot: Path,
    out_dir: Path,
    *,
    cameras: list[str] | None = None,
    limit_scenes: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build a YOLO dataset from the processed Parquet; return (data.yaml path, stats).

    Args:
        processed_dir: Directory holding samples/annotations Parquet.
        dataroot: nuScenes dataset root (symlink targets; read-only).
        out_dir: Output dataset directory.
        cameras: Restrict to these camera channels (default: all present).
        limit_scenes: Use only the first N scenes (by name) — fast dev datasets.
    """
    samples = pd.read_parquet(processed_dir / "samples.parquet")
    annotations = pd.read_parquet(processed_dir / "annotations.parquet")

    # Assign the official train/val split by scene name.
    split_scenes = create_splits_scenes()
    scene_to_split: dict[str, str] = {}
    for name in split_scenes["train"]:
        scene_to_split[name] = "train"
    for name in split_scenes["val"]:
        scene_to_split[name] = "val"
    samples = samples.assign(split=samples["scene_name"].map(scene_to_split))
    samples = samples[samples["split"].notna()]

    if cameras:
        samples = samples[samples["channel"].isin(cameras)]
    if limit_scenes:
        keep = sorted(samples["scene_name"].unique())[:limit_scenes]
        samples = samples[samples["scene_name"].isin(keep)]

    kept_tokens = set(samples["sample_data_token"])

    # Build normalized label lines for detector-class annotations only.
    ann = annotations[annotations["category_group"].notna()].copy()
    ann = ann[ann["sample_data_token"].isin(kept_tokens)]
    cls = ann["category_group"].map(CLASS_TO_INDEX)
    cx = ((ann["x_min"] + ann["x_max"]) / 2 / IMAGE_WIDTH).clip(0, 1)
    cy = ((ann["y_min"] + ann["y_max"]) / 2 / IMAGE_HEIGHT).clip(0, 1)
    bw = ((ann["x_max"] - ann["x_min"]) / IMAGE_WIDTH).clip(0, 1)
    bh = ((ann["y_max"] - ann["y_min"]) / IMAGE_HEIGHT).clip(0, 1)
    ann = ann.assign(
        _line=cls.astype(str)
        + " "
        + cx.round(6).astype(str)
        + " "
        + cy.round(6).astype(str)
        + " "
        + bw.round(6).astype(str)
        + " "
        + bh.round(6).astype(str)
    )
    labels_by_image: dict[str, list[str]] = (
        ann.groupby("sample_data_token")["_line"].apply(list).to_dict()
    )

    # Materialize image symlinks + label files.
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    counts = {"train": 0, "val": 0}
    for row in samples.itertuples(index=False):
        split = row.split
        stem = Path(row.filename).stem
        _symlink(dataroot / row.filename, out_dir / "images" / split / f"{stem}.jpg")
        lines = labels_by_image.get(row.sample_data_token, [])
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        n_boxes += len(lines)
        counts[split] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(DETECTION_CLASSES)),
            },
            sort_keys=False,
        )
    )

    stats = {
        "data_yaml": str(data_yaml),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "boxes": n_boxes,
        "classes": list(DETECTION_CLASSES),
    }
    logger.info(
        "YOLO dataset: %d train + %d val images, %d boxes -> %s",
        counts["train"],
        counts["val"],
        n_boxes,
        out_dir,
    )
    return data_yaml, stats
