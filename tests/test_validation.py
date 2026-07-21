"""Tests for the Great Expectations validation suites (tiny synthetic dataset)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("great_expectations")  # data extra

from nuscenes_data_engine.validation.expectations import validate_dataset


def _write_valid_dataset(processed_dir: Path) -> None:
    """Write a minimal but schema-valid samples + annotations pair."""
    # One scene with the minimum plausible image count (15 keyframes x 6 cameras).
    images = []
    for k in range(15):
        for cam in (
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ):
            images.append(
                {
                    "sample_data_token": f"sd-{k}-{cam}",
                    "sample_token": f"s-{k}",
                    "channel": cam,
                    "filename": f"samples/{cam}/img-{k}.jpg",
                    "width": 1600,
                    "height": 900,
                    "n_boxes": 1 if cam == "CAM_FRONT" else 0,
                    "scene_token": "scene-1",
                }
            )
    annotations = [
        {
            "annotation_token": f"ann-{k}",
            "sample_data_token": f"sd-{k}-CAM_FRONT",
            "channel": "CAM_FRONT",
            "category_name": "vehicle.car",
            "category_group": "car",
            "visibility_token": "4",
            "x_min": 100.0,
            "y_min": 200.0,
            "x_max": 300.0,
            "y_max": 400.0,
            "bbox_area": 40000.0,
        }
        for k in range(15)
    ]
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(images).to_parquet(processed_dir / "samples.parquet")
    pd.DataFrame(annotations).to_parquet(processed_dir / "annotations.parquet")


def test_validate_clean_dataset_passes(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    assert validate_dataset(tmp_path) is True


def test_out_of_bounds_box_fails(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    anns = pd.read_parquet(tmp_path / "annotations.parquet")
    anns.loc[0, "x_max"] = 9999.0  # outside image width
    anns.to_parquet(tmp_path / "annotations.parquet")
    assert validate_dataset(tmp_path) is False


def test_invalid_category_fails(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    anns = pd.read_parquet(tmp_path / "annotations.parquet")
    anns.loc[0, "category_name"] = "not.a.real.category"
    anns.to_parquet(tmp_path / "annotations.parquet")
    assert validate_dataset(tmp_path) is False
