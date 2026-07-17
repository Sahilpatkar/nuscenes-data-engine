"""Condition-sliced evaluation — the project's key differentiator.

Breaks metrics down by scene condition (night vs. day, rain vs. clear) using nuScenes
scene descriptions, mirroring how AV companies evaluate perception. Builds a per-slice
Ultralytics ``val`` dataset (an image-list txt + data.yaml over the official val split),
so each slice can be evaluated with the same code path as the overall set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from nuscenes.utils.splits import create_splits_scenes

from nuscenes_data_engine.ingestion.categories import DETECTION_CLASSES

logger = logging.getLogger("nuscenes_data_engine")


def _matches(description: str, rule: dict[str, Any]) -> bool:
    """Whether a scene description satisfies a slice's include/exclude substring rule."""
    d = description.lower()
    inc = [t.lower() for t in rule.get("include", [])]
    exc = [t.lower() for t in rule.get("exclude", [])]
    if any(t not in d for t in inc):
        return False
    return not any(t in d for t in exc)


def assign_slices(scene_description: str, slice_config: dict[str, Any]) -> dict[str, str]:
    """Map a scene description to its slice label per dimension.

    e.g. ``{"time_of_day": "night", "weather": "clear"}``.
    """
    labels: dict[str, str] = {}
    for dim, slices in slice_config.items():
        for name, rule in slices.items():
            if _matches(scene_description, rule):
                labels[dim] = name
                break
    return labels


def _slice_mask(descriptions: pd.Series, rule: dict[str, Any]) -> pd.Series:
    d = descriptions.str.lower()
    mask = pd.Series(True, index=descriptions.index)
    for t in rule.get("include", []):
        mask &= d.str.contains(t.lower(), regex=False)
    for t in rule.get("exclude", []):
        mask &= ~d.str.contains(t.lower(), regex=False)
    return mask


def build_slice_val_datasets(
    processed_dir: Path,
    yolo_dir: Path,
    slice_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Write a per-slice val image-list + data.yaml over the official val split.

    Returns ``{"<dim>/<slice>": {"yaml": Path, "n_images": int}}`` for slices with images.
    """
    samples = pd.read_parquet(processed_dir / "samples.parquet")
    val_scenes = set(create_splits_scenes()["val"])
    val = samples[samples["scene_name"].isin(val_scenes)]
    val_img_dir = (yolo_dir / "images" / "val").resolve()

    out: dict[str, dict[str, Any]] = {}
    for dim, slices in slice_config.items():
        for name, rule in slices.items():
            sub = val[_slice_mask(val["scene_description"], rule)]
            paths = [
                str(p)
                for f in sub["filename"]
                if (p := val_img_dir / f"{Path(f).stem}.jpg").exists()
            ]
            if not paths:
                logger.warning("slice %s/%s has no images; skipping", dim, name)
                continue

            txt = yolo_dir / f"val_{dim}_{name}.txt"
            txt.write_text("\n".join(paths))
            slice_yaml = yolo_dir / f"data_{dim}_{name}.yaml"
            slice_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(yolo_dir.resolve()),
                        "train": "images/train",
                        "val": str(txt.resolve()),
                        "names": dict(enumerate(DETECTION_CLASSES)),
                    },
                    sort_keys=False,
                )
            )
            out[f"{dim}/{name}"] = {"yaml": slice_yaml, "n_images": len(paths)}
            logger.info("slice %s/%s: %d val images", dim, name, len(paths))
    return out
