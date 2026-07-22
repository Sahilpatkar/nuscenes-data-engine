"""Data-availability manifest: which referenced sensor files actually exist on disk.

The nuScenes metadata references every sensor blob, but the shared server's copy is
partial (all radar absent, LiDAR sweeps absent). The devkit loads metadata fine and
only crashes when something touches a missing file — so every downstream stage filters
on this manifest instead of trusting the metadata's file references.

Parses the metadata JSONs directly (no devkit; base deps only) and checks presence via
one directory listing per referenced directory rather than millions of per-file stats —
the difference between minutes and hours on NFS.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")


def _load_table(dataroot: Path, version: str, name: str) -> list[dict[str, object]]:
    with open(dataroot / version / f"{name}.json", encoding="utf-8") as fh:
        data: list[dict[str, object]] = json.load(fh)
    return data


def _modality(channel: str) -> str:
    return channel.split("_")[0].lower()  # CAM_FRONT -> cam, LIDAR_TOP -> lidar, RADAR_* -> radar


def build_manifest(dataroot: Path, version: str, out_path: Path) -> pd.DataFrame:
    """Cross-check every sample_data record against the filesystem; write a parquet.

    One row per record: sample_data_token, filename, channel, modality, is_key_frame,
    scene_token, scene_name, present.
    """
    sample_to_scene = {
        s["token"]: s["scene_token"] for s in _load_table(dataroot, version, "sample")
    }
    scene_names = {s["token"]: s["name"] for s in _load_table(dataroot, version, "scene")}
    records = _load_table(dataroot, version, "sample_data")

    listed: dict[str, set[str]] = {}  # parent dir -> filenames on disk
    rows = []
    for rec in records:
        filename = str(rec["filename"])
        parent, _, name = filename.rpartition("/")
        if parent not in listed:
            try:
                with os.scandir(dataroot / parent) as it:
                    listed[parent] = {e.name for e in it}
            except FileNotFoundError:
                listed[parent] = set()
        channel = parent.rsplit("/", 1)[-1]  # samples/CAM_FRONT -> CAM_FRONT
        rows.append(
            {
                "sample_data_token": rec["token"],
                "filename": filename,
                "channel": channel,
                "modality": _modality(channel),
                "is_key_frame": bool(rec["is_key_frame"]),
                "scene_token": sample_to_scene.get(rec["sample_token"], ""),
                "scene_name": scene_names.get(sample_to_scene.get(rec["sample_token"], ""), ""),
                "present": name in listed[parent],
            }
        )

    manifest = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    logger.info(
        "Manifest: %d records, %d present -> %s", len(manifest), manifest["present"].sum(), out_path
    )
    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per (channel, is_key_frame): referenced vs present counts."""
    grouped = manifest.groupby(["channel", "is_key_frame"], as_index=False).agg(
        n_referenced=("present", "size"), n_present=("present", "sum")
    )
    return grouped.sort_values(["channel", "is_key_frame"], ignore_index=True)


def camera_keyframes_complete(manifest: pd.DataFrame) -> bool:
    """True when every referenced camera keyframe exists — the training working set."""
    cams = manifest[(manifest["modality"] == "cam") & manifest["is_key_frame"]]
    return bool(cams["present"].all())
