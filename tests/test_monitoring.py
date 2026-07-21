"""Tests for Phase 5 drift monitoring (synthetic images + parquet; offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("evidently")  # data extra

import cv2
import numpy as np
import pandas as pd

from nuscenes_data_engine.monitoring.drift import (
    build_drift_report,
    save_drift_report,
    summarize_drift,
)
from nuscenes_data_engine.monitoring.features import (
    build_reference,
    compute_features,
    image_brightness,
    load_feature_table,
)


def _write_samples(root: Path, *, with_images: bool = True) -> Path:
    """Synthetic samples.parquet (+ tiny real JPEGs): 3 day rows (bright), 2 night (dark)."""
    rows = []
    for i in range(5):
        night = i >= 3
        filename = f"samples/CAM_FRONT/frame_{i}.jpg"
        if with_images:
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            value = 20 if night else 200
            cv2.imwrite(str(path), np.full((90, 160, 3), value, np.uint8))
        rows.append(
            {
                "sample_data_token": f"sd{i}",
                "sample_token": f"s{i}",
                "channel": "CAM_FRONT",
                "filename": filename,
                "width": 1600,
                "height": 900,
                "timestamp": 1_526_915_243_012_465 + i,
                "n_boxes": 2 if night else 12,
                "scene_token": "sc0",
                "scene_name": "scene-0001",
                "scene_description": "Night drive" if night else "Sunny day",
                "log_token": "l0",
                "location": "boston-seaport",
                "is_night": night,
                "is_rain": False,
            }
        )
    processed = root / "processed"
    processed.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(processed / "samples.parquet", index=False)
    return processed


class TestFeatures:
    def test_image_brightness_gray_mean(self) -> None:
        assert image_brightness(np.full((4, 4, 3), 200, np.uint8)) == pytest.approx(200.0)
        assert image_brightness(np.zeros((4, 4, 3), np.uint8)) == 0.0

    def test_compute_features_reads_images(self, tmp_path: Path) -> None:
        processed = _write_samples(tmp_path)
        samples = pd.read_parquet(processed / "samples.parquet")
        features = compute_features(samples, tmp_path, sample_images=None)
        assert len(features) == 5
        day = features[~features["is_night"]]["brightness"]
        night = features[features["is_night"]]["brightness"]
        assert day.mean() == pytest.approx(200.0, abs=2)
        assert night.mean() == pytest.approx(20.0, abs=2)
        assert set(features["width"]) == {1600}

    def test_missing_images_yield_nan(self, tmp_path: Path) -> None:
        processed = _write_samples(tmp_path, with_images=False)
        samples = pd.read_parquet(processed / "samples.parquet")
        for dataroot in (tmp_path / "nowhere", None):
            features = compute_features(samples, dataroot, sample_images=None)
            assert features["brightness"].isna().all()

    def test_build_reference_condition_filter(self, tmp_path: Path) -> None:
        processed = _write_samples(tmp_path)
        out = tmp_path / "night.parquet"
        features = build_reference(processed, tmp_path, out, condition="night", sample_images=None)
        assert len(features) == 2
        assert features["is_night"].all()
        assert out.is_file()
        with pytest.raises(ValueError, match="condition"):
            build_reference(processed, tmp_path, out, condition="dusk")

    def test_load_serving_jsonl_renames_columns(self, tmp_path: Path) -> None:
        log = tmp_path / "requests.jsonl"
        rows = [
            {"ts": "t", "image_width": 1600, "image_height": 900, "n_detections": 4,
             "brightness": 101.5, "latency_ms": 90.0, "model_version": "2"}
            for _ in range(3)
        ]
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        table = load_feature_table(log)
        assert len(table) == 3
        assert {"width", "height", "n_boxes", "brightness"} <= set(table.columns)


def _feature_frame(rng: np.random.Generator, b_mean: float, b_std: float, lam: float) -> pd.DataFrame:
    n = 300
    return pd.DataFrame(
        {
            "brightness": rng.normal(b_mean, b_std, n),
            "width": 1600,
            "height": 900,
            "n_boxes": rng.poisson(lam, n),
        }
    )


class TestDrift:
    def test_drift_detected_on_shifted_current(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        _feature_frame(rng, 110, 20, 12).to_parquet(tmp_path / "ref.parquet", index=False)
        _feature_frame(rng, 40, 15, 5).to_parquet(tmp_path / "cur.parquet", index=False)
        summary = summarize_drift(
            build_drift_report(tmp_path / "ref.parquet", tmp_path / "cur.parquet")
        )
        assert summary["columns"]["brightness"]["drift_detected"] is True
        assert summary["dataset_drift"] is True
        assert summary["n_drifted"] >= 1

    def test_no_drift_on_identical_distribution(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        _feature_frame(rng, 110, 20, 12).to_parquet(tmp_path / "ref.parquet", index=False)
        _feature_frame(rng, 110, 20, 12).to_parquet(tmp_path / "cur.parquet", index=False)
        summary = summarize_drift(
            build_drift_report(tmp_path / "ref.parquet", tmp_path / "cur.parquet")
        )
        assert summary["dataset_drift"] is False
        assert all(not v["drift_detected"] for v in summary["columns"].values())

    def test_save_writes_html_and_json(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        _feature_frame(rng, 110, 20, 12).to_parquet(tmp_path / "ref.parquet", index=False)
        _feature_frame(rng, 40, 15, 5).to_parquet(tmp_path / "cur.parquet", index=False)
        snapshot = build_drift_report(tmp_path / "ref.parquet", tmp_path / "cur.parquet")
        html_path, json_path = save_drift_report(snapshot, tmp_path / "out")
        # Pins the Evidently gotcha: save_html silently no-ops on a Path argument.
        assert html_path.is_file() and html_path.stat().st_size > 0
        assert json.loads(json_path.read_text())["dataset_drift"] is True

    def test_all_nan_column_dropped(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        ref = _feature_frame(rng, 110, 20, 12)
        ref["brightness"] = float("nan")  # metadata-only reference
        ref.to_parquet(tmp_path / "ref.parquet", index=False)
        _feature_frame(rng, 40, 15, 5).to_parquet(tmp_path / "cur.parquet", index=False)
        summary = summarize_drift(
            build_drift_report(tmp_path / "ref.parquet", tmp_path / "cur.parquet")
        )
        assert "brightness" not in summary["columns"]
