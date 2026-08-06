"""Tests for VLM weak supervision: pure verification core + dataset seam."""

from __future__ import annotations

from pathlib import Path
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


def test_detection_counts_empty_inputs_return_empty() -> None:
    from nuscenes_data_engine.active_learning.pseudo_label import detection_counts

    assert detection_counts(pd.DataFrame([])) == {}  # column-less: propose_boxes' empty shape
    assert detection_counts(pd.DataFrame({"sample_data_token": [], "category_group": []})) == {}


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
    det = {"f1": {"car": 1}, "f2": {"car": 1}, "f3": {"car": 1}}
    vlm = {"f3": _labels(cars=1, parse_status="error")}  # f1, f2 absent; f3 unparsed
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == []
    assert diagnostics["n_no_label"] == 2
    assert diagnostics["n_unparsed"] == 1


def test_verify_frames_mapping_is_not_permutable() -> None:
    """Distinct nonzero counts per class, so a swapped mapping cannot pass."""
    det = {"f1": {"car": 1, "truck": 2, "bus": 3, "pedestrian": 4, "bicycle": 5}}
    vlm = {"f1": _labels(cars=1, trucks=2, buses=3, pedestrians=4, bicycles=5)}
    assert verify_frames(det, vlm, tolerance=0)[0] == ["f1"]
    swapped = {"f1": _labels(cars=1, trucks=3, buses=2, pedestrians=4, bicycles=5)}
    assert verify_frames(det, swapped, tolerance=0)[0] == []


def test_verify_frames_reports_mutual_zero_agreement() -> None:
    det = {"f1": {"car": 2}, "f2": {"car": 1, "pedestrian": 1}}
    vlm = {"f1": _labels(cars=2), "f2": _labels(cars=1, pedestrians=1)}
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == ["f1", "f2"]
    # f1 has neither detector nor VLM pedestrians -> the blind-spot bucket; f2 has both.
    assert diagnostics["accepted_mutual_zero_by_class"]["pedestrian"] == 1
    assert diagnostics["accepted_mutual_zero_by_class"]["bus"] == 2  # neither frame has buses
    assert "car" not in diagnostics["accepted_mutual_zero_by_class"]


def test_verify_frames_is_deterministic_and_sorted() -> None:
    det = {t: {"car": 1} for t in ("f3", "f1", "f2")}
    vlm = {t: _labels(cars=1) for t in ("f3", "f1", "f2")}
    accepted, _ = verify_frames(det, vlm, tolerance=1)
    assert accepted == ["f1", "f2", "f3"]


def test_boxes_to_rows_projects_to_annotations_schema() -> None:
    import numpy as np

    from nuscenes_data_engine.active_learning.pseudo_label import boxes_to_rows

    rows = boxes_to_rows(
        "f1",
        np.array([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 50.0, 50.0]]),
        np.array([0, 3]),  # car, pedestrian (CLASS_TO_INDEX order)
        np.array([0.9, 0.7]),
    )
    assert [r["category_group"] for r in rows] == ["car", "pedestrian"]
    assert rows[0]["sample_data_token"] == "f1"
    assert (rows[0]["x_min"], rows[0]["y_min"]) == (10.0, 20.0)
    assert (rows[0]["x_max"], rows[0]["y_max"]) == (110.0, 220.0)
    assert rows[0]["score"] == pytest.approx(0.9)
    assert set(rows[0]) == {
        "sample_data_token", "category_group", "x_min", "y_min", "x_max", "y_max", "score",
    }


def test_boxes_to_rows_rejects_out_of_taxonomy_class_index() -> None:
    import numpy as np

    from nuscenes_data_engine.active_learning.pseudo_label import boxes_to_rows

    with pytest.raises(ValueError, match="wrong weights file"):
        boxes_to_rows("f1", np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([79]), np.array([0.9]))


def test_boxes_to_rows_empty_frame_yields_no_rows() -> None:
    import numpy as np

    from nuscenes_data_engine.active_learning.pseudo_label import boxes_to_rows

    assert boxes_to_rows("f1", np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)) == []


def test_build_weak_sample_selects_only_unlabelled_arm_frames(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.pseudo_label import build_weak_sample

    samples = pd.DataFrame(
        {
            "sample_data_token": ["a", "b", "c"],
            "sample_token": ["sa", "sb", "sc"],
            "channel": ["CAM_FRONT"] * 3,
            "filename": ["a.jpg", "b.jpg", "c.jpg"],
            "width": [1600] * 3,
            "height": [900] * 3,
            "timestamp": [1, 2, 3],
            "n_boxes": [1, 2, 3],
            "scene_token": ["s1"] * 3,
            "scene_name": ["scene-0001"] * 3,
            "scene_description": ["x"] * 3,
            "log_token": ["l1"] * 3,
            "location": ["boston-seaport"] * 3,
            "is_night": [False] * 3,
            "is_rain": [False] * 3,
        }
    )
    arm_tokens = ["a", "b", "c"]
    already = pd.DataFrame({"sample_data_token": ["b"], "parse_status": ["ok"]})

    weak = build_weak_sample(samples, arm_tokens, already)
    assert list(weak["sample_data_token"]) == ["a", "c"]  # 'b' already labelled
    # Columns the 6b submit path reads must all be present.
    for column in ("filename", "sample_token", "scene_name", "present", "in_opus_subset"):
        assert column in weak.columns
    assert weak["present"].all() and not weak["in_opus_subset"].any()


def test_build_weak_sample_relabels_unparsed_frames(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.pseudo_label import build_weak_sample

    samples = pd.DataFrame(
        {
            "sample_data_token": ["a"], "sample_token": ["sa"], "channel": ["CAM_FRONT"],
            "filename": ["a.jpg"], "width": [1600], "height": [900], "timestamp": [1],
            "n_boxes": [1], "scene_token": ["s1"], "scene_name": ["scene-0001"],
            "scene_description": ["x"], "log_token": ["l1"], "location": ["boston-seaport"],
            "is_night": [False], "is_rain": [False],
        }
    )
    already = pd.DataFrame({"sample_data_token": ["a"], "parse_status": ["error"]})
    weak = build_weak_sample(samples, ["a"], already)
    assert list(weak["sample_data_token"]) == ["a"]  # unparsed -> label again
