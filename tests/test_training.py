"""Tests for the YOLO dataset builder and data-version hashing (no training)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("nuscenes")  # data extra: the dataset builder needs the devkit splits

from nuscenes_data_engine.training.dataset import (
    build_yolo_dataset,
    compute_data_version,
)

# scene-0001 is an official train scene, scene-0003 an official val scene.
TRAIN_SCENE = "scene-0001"
VAL_SCENE = "scene-0003"


def _write_processed(processed_dir: Path, dataroot: Path) -> None:
    """Write a tiny samples/annotations pair plus dummy image files to symlink to."""
    rows_img, rows_ann = [], []
    for scene in (TRAIN_SCENE, VAL_SCENE):
        fname = f"samples/CAM_FRONT/{scene}.jpg"
        (dataroot / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
        (dataroot / fname).write_bytes(b"\xff\xd8\xff")  # tiny fake jpeg
        sd_token = f"sd-{scene}"
        rows_img.append(
            {
                "sample_data_token": sd_token,
                "sample_token": f"s-{scene}",
                "channel": "CAM_FRONT",
                "filename": fname,
                "width": 1600,
                "height": 900,
                "timestamp": 0,
                "n_boxes": 1,
                "scene_token": scene,
                "scene_name": scene,
                "scene_description": "x",
                "log_token": "l",
                "location": "singapore-onenorth",
                "is_night": False,
                "is_rain": False,
            }
        )
        rows_ann.append(
            {
                "annotation_token": f"ann-{scene}",
                "sample_data_token": sd_token,
                "sample_token": f"s-{scene}",
                "channel": "CAM_FRONT",
                "category_name": "vehicle.car",
                "category_group": "car",
                "visibility_token": "4",
                "num_lidar_pts": 5,
                "num_radar_pts": 1,
                "x_min": 100.0,
                "y_min": 200.0,
                "x_max": 300.0,
                "y_max": 400.0,
                "bbox_area": 40000.0,
                "scene_token": scene,
                "scene_name": scene,
                "scene_description": "x",
                "log_token": "l",
                "location": "singapore-onenorth",
                "is_night": False,
                "is_rain": False,
            }
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_img).to_parquet(processed_dir / "samples.parquet")
    pd.DataFrame(rows_ann).to_parquet(processed_dir / "annotations.parquet")


def test_build_yolo_dataset_splits_and_labels(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    dataroot = tmp_path / "nuscenes"
    out = tmp_path / "yolo"
    _write_processed(processed, dataroot)

    data_yaml, stats = build_yolo_dataset(processed, dataroot, out, cameras=["CAM_FRONT"])

    assert stats["train_images"] == 1 and stats["val_images"] == 1
    assert stats["boxes"] == 2

    # Official split honored: scene-0001 -> train, scene-0003 -> val.
    assert (out / "images" / "train" / f"{TRAIN_SCENE}.jpg").is_symlink()
    assert (out / "images" / "val" / f"{VAL_SCENE}.jpg").is_symlink()

    # Label is normalized `cls cx cy w h` with class 0 (car).
    label = (out / "labels" / "train" / f"{TRAIN_SCENE}.txt").read_text().split()
    assert label[0] == "0"
    cx, cy, w, h = map(float, label[1:])
    assert (cx, cy) == pytest.approx((0.125, 0.3333), abs=1e-3)
    assert (w, h) == pytest.approx((0.125, 0.2222), abs=1e-3)

    meta = yaml.safe_load(data_yaml.read_text())
    assert meta["names"] == {0: "car", 1: "truck", 2: "bus", 3: "pedestrian", 4: "bicycle"}


def test_rebuild_skipped_when_unchanged(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    dataroot = tmp_path / "nuscenes"
    out = tmp_path / "yolo"
    _write_processed(processed, dataroot)

    build_yolo_dataset(processed, dataroot, out, cameras=["CAM_FRONT"])
    label = out / "labels" / "train" / f"{TRAIN_SCENE}.txt"
    first_mtime = label.stat().st_mtime_ns
    assert (out / ".build_manifest.json").exists()

    # Second build with the same inputs must skip (label file left untouched).
    _, stats = build_yolo_dataset(processed, dataroot, out, cameras=["CAM_FRONT"])
    assert label.stat().st_mtime_ns == first_mtime
    assert stats["train_images"] == 1 and stats["val_images"] == 1


def test_rebuild_triggered_when_data_changes(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    dataroot = tmp_path / "nuscenes"
    out = tmp_path / "yolo"
    _write_processed(processed, dataroot)
    build_yolo_dataset(processed, dataroot, out, cameras=["CAM_FRONT"])
    label = out / "labels" / "train" / f"{TRAIN_SCENE}.txt"
    first_mtime = label.stat().st_mtime_ns

    # Change the underlying data -> new data_version -> rebuild (label rewritten).
    anns = pd.read_parquet(processed / "annotations.parquet")
    anns.loc[0, "x_max"] = 320.0
    anns.to_parquet(processed / "annotations.parquet")
    build_yolo_dataset(processed, dataroot, out, cameras=["CAM_FRONT"])
    assert label.stat().st_mtime_ns != first_mtime


def test_stale_train_files_pruned_when_train_frames_shrinks(tmp_path: Path) -> None:
    """A rebuild must not leave a prior build's images/labels lying around.

    The accepted/train-frame set can shrink between builds (e.g. re-tuning `al
    pseudo-label`'s conf/tolerance) — a create-only materialize loop would leave the
    dropped frame's stale .jpg/.txt in place for Ultralytics to glob up, while
    stats["train_images"] under-reports the true (stale-inflated) image count.
    """
    processed = tmp_path / "processed"
    dataroot = tmp_path / "nuscenes"
    out = tmp_path / "yolo"
    (dataroot / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)

    tokens = ["t1", "t2", "t3"]
    rows_img = []
    for token in tokens:
        fname = f"samples/CAM_FRONT/{token}.jpg"
        (dataroot / fname).write_bytes(b"\xff\xd8\xff")
        rows_img.append(
            {
                "sample_data_token": token,
                "sample_token": f"s-{token}",
                "channel": "CAM_FRONT",
                "filename": fname,
                "width": 1600,
                "height": 900,
                "timestamp": 0,
                "n_boxes": 0,
                "scene_token": TRAIN_SCENE,
                "scene_name": TRAIN_SCENE,
                "scene_description": "x",
                "log_token": "l",
                "location": "singapore-onenorth",
                "is_night": False,
                "is_rain": False,
            }
        )
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_img).to_parquet(processed / "samples.parquet")
    # Empty but real-schema annotations (no boxes for any of these frames); numeric
    # columns need explicit dtypes so an empty frame doesn't come out as `object`.
    pd.DataFrame(
        {
            "annotation_token": pd.Series([], dtype=str),
            "sample_data_token": pd.Series([], dtype=str),
            "sample_token": pd.Series([], dtype=str),
            "channel": pd.Series([], dtype=str),
            "category_name": pd.Series([], dtype=str),
            "category_group": pd.Series([], dtype=str),
            "visibility_token": pd.Series([], dtype=str),
            "num_lidar_pts": pd.Series([], dtype=int),
            "num_radar_pts": pd.Series([], dtype=int),
            "x_min": pd.Series([], dtype=float),
            "y_min": pd.Series([], dtype=float),
            "x_max": pd.Series([], dtype=float),
            "y_max": pd.Series([], dtype=float),
            "bbox_area": pd.Series([], dtype=float),
            "scene_token": pd.Series([], dtype=str),
            "scene_name": pd.Series([], dtype=str),
            "scene_description": pd.Series([], dtype=str),
            "log_token": pd.Series([], dtype=str),
            "location": pd.Series([], dtype=str),
            "is_night": pd.Series([], dtype=bool),
            "is_rain": pd.Series([], dtype=bool),
        }
    ).to_parquet(processed / "annotations.parquet")

    build_yolo_dataset(processed, dataroot, out, cameras=["CAM_FRONT"], train_frames=set(tokens))
    assert (out / "images" / "train" / "t3.jpg").is_symlink()
    assert (out / "labels" / "train" / "t3.txt").is_file()

    # Rebuild with 't3' dropped from the accepted set -> the build key changes (the
    # train_frames hash differs), so this is a real rebuild, not the up-to-date skip.
    _, stats = build_yolo_dataset(
        processed, dataroot, out, cameras=["CAM_FRONT"], train_frames={"t1", "t2"}
    )
    assert stats["train_images"] == 2
    assert not (out / "images" / "train" / "t3.jpg").exists()
    assert not (out / "labels" / "train" / "t3.txt").exists()


def test_compute_data_version_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    dataroot = tmp_path / "nuscenes"
    _write_processed(processed, dataroot)

    v1 = compute_data_version(processed)
    assert v1 == compute_data_version(processed)  # stable

    # Mutate the annotations -> different fingerprint.
    anns = pd.read_parquet(processed / "annotations.parquet")
    anns.loc[0, "x_max"] = 350.0
    anns.to_parquet(processed / "annotations.parquet")
    assert compute_data_version(processed) != v1
