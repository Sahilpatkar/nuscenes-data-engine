"""Tests for VLM weak supervision: pure verification core + dataset seam."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nuscenes_data_engine.active_learning.pseudo_label import (
    VLM_TO_DETECTOR,
    detection_counts,
    verify_frames,
)


def test_vlm_to_detector_covers_every_detector_class() -> None:
    from nuscenes_data_engine.ingestion.categories import DETECTION_CLASSES

    assert set(VLM_TO_DETECTOR.values()) == set(DETECTION_CLASSES)
    # The five VLM classes with no detector counterpart are deliberately absent.
    assert "traffic_cones" not in VLM_TO_DETECTOR
    assert "construction_vehicles" not in VLM_TO_DETECTOR


def test_detection_counts_per_frame_and_class() -> None:
    boxes = pd.DataFrame(
        {
            "sample_data_token": ["f1", "f1", "f1", "f2"],
            "category_group": ["car", "car", "pedestrian", "bus"],
        }
    )
    counts = detection_counts(boxes)
    assert counts["f1"]["car"] == 2
    assert counts["f1"]["pedestrian"] == 1
    assert counts["f1"].get("bus", 0) == 0
    assert counts["f2"]["bus"] == 1


def _labels(**counts: int) -> dict[str, Any]:
    """A parsed VLM label row with all ten count fields (unspecified ones zero)."""
    row = {
        "cars": 0, "trucks": 0, "buses": 0, "trailers": 0, "construction_vehicles": 0,
        "motorcycles": 0, "bicycles": 0, "pedestrians": 0, "traffic_cones": 0,
        "barriers": 0, "parse_status": "ok",
    }
    row.update(counts)
    return row


def test_verify_frames_accepts_within_tolerance() -> None:
    det = {"f1": {"car": 3, "pedestrian": 1}}
    vlm = {"f1": _labels(cars=4, pedestrians=1)}  # car off by 1 -> still accepted
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == ["f1"]
    assert diagnostics["n_candidates"] == 1 and diagnostics["n_accepted"] == 1


def test_verify_frames_rejects_when_any_class_disagrees() -> None:
    det = {"f1": {"car": 3, "pedestrian": 1}}
    vlm = {"f1": _labels(cars=3, pedestrians=5)}  # pedestrians off by 4
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == []
    assert diagnostics["rejected_by_class"]["pedestrian"] == 1


def test_verify_frames_tolerance_boundary_is_inclusive() -> None:
    det = {"f1": {"car": 2}}
    assert verify_frames(det, {"f1": _labels(cars=3)}, tolerance=1)[0] == ["f1"]
    assert verify_frames(det, {"f1": _labels(cars=4)}, tolerance=1)[0] == []


def test_verify_frames_zero_detections_matches_zero_counts() -> None:
    # A genuine empty frame: detector found nothing, VLM saw nothing -> accept
    # (an empty label file is a valid background training example).
    accepted, _ = verify_frames({"f1": {}}, {"f1": _labels()}, tolerance=1)
    assert accepted == ["f1"]


def test_verify_frames_rejects_missing_or_unparsed_labels() -> None:
    det = {"f1": {"car": 1}, "f2": {"car": 1}}
    vlm = {"f2": _labels(cars=1, parse_status="error")}  # f1 absent, f2 unparsed
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == []
    assert diagnostics["n_no_label"] == 1
    assert diagnostics["n_unparsed"] == 1


def test_verify_frames_is_deterministic_and_sorted() -> None:
    det = {t: {"car": 1} for t in ("f3", "f1", "f2")}
    vlm = {t: _labels(cars=1) for t in ("f3", "f1", "f2")}
    accepted, _ = verify_frames(det, vlm, tolerance=1)
    assert accepted == ["f1", "f2", "f3"]
