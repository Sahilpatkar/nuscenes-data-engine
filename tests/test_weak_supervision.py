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


def test_build_weak_sample_excludes_frames_missing_from_availability() -> None:
    from nuscenes_data_engine.active_learning.pseudo_label import build_weak_sample

    samples = pd.DataFrame(
        {
            "sample_data_token": ["a", "b"], "sample_token": ["sa", "sb"],
            "channel": ["CAM_FRONT"] * 2, "filename": ["a.jpg", "b.jpg"],
            "width": [1600] * 2, "height": [900] * 2, "timestamp": [1, 2],
            "n_boxes": [1, 2], "scene_token": ["s1"] * 2, "scene_name": ["scene-0001"] * 2,
            "scene_description": ["x"] * 2, "log_token": ["l1"] * 2,
            "location": ["boston-seaport"] * 2, "is_night": [False] * 2,
            "is_rain": [False] * 2,
        }
    )
    # 'b' is recorded but its image is not on disk -> must not be sent to the VLM.
    availability = pd.DataFrame(
        {"sample_data_token": ["a", "b"], "present": [True, False]}
    )
    weak = build_weak_sample(samples, ["a", "b"], None, availability=availability)
    assert list(weak["sample_data_token"]) == ["a"]
    assert weak["present"].all()


def test_build_weak_sample_without_availability_keeps_all_frames() -> None:
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
    weak = build_weak_sample(samples, ["a"], None)
    assert list(weak["sample_data_token"]) == ["a"] and weak["present"].all()


def test_run_pseudo_sample_writes_sample_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from nuscenes_data_engine.active_learning.pseudo_label import run_pseudo_sample

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nodata"))  # no 6b labels

    pd.DataFrame({"sample_data_token": ["a", "b"]}).to_parquet(
        state / "random.parquet", index=False
    )
    pd.DataFrame(
        {
            "sample_data_token": ["a", "b"], "sample_token": ["sa", "sb"],
            "channel": ["CAM_FRONT"] * 2, "filename": ["a.jpg", "b.jpg"],
            "width": [1600] * 2, "height": [900] * 2, "timestamp": [1, 2],
            "n_boxes": [1, 2], "scene_token": ["s1"] * 2, "scene_name": ["scene-0001"] * 2,
            "scene_description": ["x"] * 2, "log_token": ["l1"] * 2,
            "location": ["boston-seaport"] * 2, "is_night": [False] * 2,
            "is_rain": [False] * 2,
        }
    ).to_parquet(processed / "samples.parquet", index=False)

    config = tmp_path / "al.yaml"
    config.write_text(yaml.safe_dump({"state": {"dir": str(state)}}))
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    first = run_pseudo_sample(config, weak_config, arm="random", processed_dir=processed)
    assert first["n_needing_labels"] == 2
    assert (weak_state / "sample.parquet").is_file()

    # Simulate a successful collect: 'a' now labelled ok -> only 'b' remains.
    pd.DataFrame(
        {"sample_data_token": ["a"], "parse_status": ["ok"]}
    ).to_parquet(weak_state / "labels.parquet", index=False)
    second = run_pseudo_sample(config, weak_config, arm="random", processed_dir=processed)
    assert second["n_needing_labels"] == 1


def test_run_pseudo_sample_rejects_tokens_missing_from_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from nuscenes_data_engine.active_learning.pseudo_label import run_pseudo_sample

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nodata"))
    pd.DataFrame({"sample_data_token": ["a", "ghost"]}).to_parquet(
        state / "random.parquet", index=False
    )
    pd.DataFrame(
        {
            "sample_data_token": ["a"], "sample_token": ["sa"], "channel": ["CAM_FRONT"],
            "filename": ["a.jpg"], "width": [1600], "height": [900], "timestamp": [1],
            "n_boxes": [1], "scene_token": ["s1"], "scene_name": ["scene-0001"],
            "scene_description": ["x"], "log_token": ["l1"], "location": ["boston-seaport"],
            "is_night": [False], "is_rain": [False],
        }
    ).to_parquet(processed / "samples.parquet", index=False)
    config = tmp_path / "al.yaml"
    config.write_text(yaml.safe_dump({"state": {"dir": str(state)}}))
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(tmp_path / "weak")}}))

    with pytest.raises(ValueError, match="missing from samples"):
        run_pseudo_sample(config, weak_config, arm="random", processed_dir=processed)


def test_run_pseudo_label_writes_accepted_and_pseudo_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from nuscenes_data_engine.active_learning import pseudo_label as pl

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    weak_state.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nodata"))  # no 6b labels

    # Arm has three frames; f1 agrees, f2 disagrees on pedestrians, f3 disagrees on cars.
    # (Every arm frame must carry a label — run_pseudo_label now requires full
    # coverage before it will spend the GPU — so f3 gets a label too, just one that
    # disagrees, to keep it rejected for a count reason rather than a coverage gap.)
    pd.DataFrame({"sample_data_token": ["f1", "f2", "f3"]}).to_parquet(
        state / "random.parquet", index=False
    )
    pd.DataFrame(
        {
            "sample_data_token": ["f1", "f2", "f3"],
            "filename": ["f1.jpg", "f2.jpg", "f3.jpg"],
            "channel": ["CAM_FRONT"] * 3,
            "scene_name": ["scene-0001"] * 3,
        }
    ).to_parquet(processed / "samples.parquet", index=False)
    base = {
        "cars": 0, "trucks": 0, "buses": 0, "trailers": 0, "construction_vehicles": 0,
        "motorcycles": 0, "bicycles": 0, "pedestrians": 0, "traffic_cones": 0,
        "barriers": 0, "parse_status": "ok",
    }
    pd.DataFrame(
        [
            {**base, "sample_data_token": "f1", "cars": 2, "pedestrians": 1},
            {**base, "sample_data_token": "f2", "cars": 2, "pedestrians": 9},
            {**base, "sample_data_token": "f3", "cars": 9},
        ]
    ).to_parquet(weak_state / "labels.parquet", index=False)

    def _fake_propose(weights, frames, dataroot, **kwargs):
        return pd.DataFrame(
            [
                {"sample_data_token": "f1", "category_group": "car",
                 "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0, "score": 0.9},
                {"sample_data_token": "f1", "category_group": "car",
                 "x_min": 20.0, "y_min": 0.0, "x_max": 30.0, "y_max": 10.0, "score": 0.8},
                {"sample_data_token": "f1", "category_group": "pedestrian",
                 "x_min": 5.0, "y_min": 5.0, "x_max": 9.0, "y_max": 25.0, "score": 0.7},
                {"sample_data_token": "f2", "category_group": "car",
                 "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0, "score": 0.9},
                {"sample_data_token": "f2", "category_group": "car",
                 "x_min": 20.0, "y_min": 0.0, "x_max": 30.0, "y_max": 10.0, "score": 0.8},
                {"sample_data_token": "f3", "category_group": "car",
                 "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0, "score": 0.9},
            ]
        )

    monkeypatch.setattr(pl, "propose_boxes", _fake_propose)
    monkeypatch.setattr(pl, "_official_val_scenes", lambda: {"scene-9999"})  # devkit-free

    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "state": {"dir": str(state)},
                "pseudo": {"conf": 0.5, "tolerance": 1, "imgsz": 640, "batch": 32},
                "train": {"imgsz": 640},
            }
        )
    )
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    summary = pl.run_pseudo_label(
        config, weak_config, arm="random", weights=tmp_path / "best.pt",
        processed_dir=processed, device="cpu",
    )

    assert summary["n_accepted"] == 1  # only f1
    assert summary["retention"] == pytest.approx(1 / 3)
    accepted = pd.read_parquet(state / "random_accepted.parquet")
    assert list(accepted["sample_data_token"]) == ["f1"]
    pseudo = pd.read_parquet(state / "random_pseudo_labels.parquet")
    assert set(pseudo["sample_data_token"]) == {"f1"}  # rejected frames' boxes dropped
    assert len(pseudo) == 3
    assert "score" not in pseudo.columns or pseudo["score"].notna().all()


def test_run_pseudo_label_requires_labels_for_every_arm_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running before `autolabel collect` finishes must fail loudly, not silently shrink."""
    import yaml

    from nuscenes_data_engine.active_learning import pseudo_label as pl

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    weak_state.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nodata"))
    pd.DataFrame({"sample_data_token": ["f1", "f2"]}).to_parquet(
        state / "random.parquet", index=False
    )
    pd.DataFrame(
        {"sample_data_token": ["f1", "f2"], "filename": ["f1.jpg", "f2.jpg"],
         "channel": ["CAM_FRONT"] * 2, "scene_name": ["scene-0001"] * 2}
    ).to_parquet(processed / "samples.parquet", index=False)
    base = {
        "cars": 0, "trucks": 0, "buses": 0, "trailers": 0, "construction_vehicles": 0,
        "motorcycles": 0, "bicycles": 0, "pedestrians": 0, "traffic_cones": 0,
        "barriers": 0, "parse_status": "ok",
    }
    # Only f1 was labelled; f2 is still pending.
    pd.DataFrame([{**base, "sample_data_token": "f1"}]).to_parquet(
        weak_state / "labels.parquet", index=False
    )

    called = []
    monkeypatch.setattr(pl, "propose_boxes", lambda *a, **k: called.append(1) or pd.DataFrame())
    monkeypatch.setattr(pl, "_official_val_scenes", lambda: {"scene-9999"})
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump({"state": {"dir": str(state)}, "pseudo": {"conf": 0.5, "tolerance": 1}})
    )
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    with pytest.raises(ValueError, match="have no VLM label"):
        pl.run_pseudo_label(
            config, weak_config, arm="random", weights=tmp_path / "best.pt",
            processed_dir=processed, device="cpu",
        )
    assert not called, "must fail before spending the GPU proposal"


def test_run_pseudo_label_empty_acceptance_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from nuscenes_data_engine.active_learning import pseudo_label as pl

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    weak_state.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nodata"))
    pd.DataFrame({"sample_data_token": ["f1"]}).to_parquet(state / "random.parquet", index=False)
    pd.DataFrame(
        {"sample_data_token": ["f1"], "filename": ["f1.jpg"], "channel": ["CAM_FRONT"],
         "scene_name": ["scene-0001"]}
    ).to_parquet(processed / "samples.parquet", index=False)
    pd.DataFrame(
        [{"sample_data_token": "f1", "parse_status": "error"}]
    ).to_parquet(weak_state / "labels.parquet", index=False)

    monkeypatch.setattr(
        pl, "propose_boxes",
        lambda *a, **k: pd.DataFrame(
            [{"sample_data_token": "f1", "category_group": "car",
              "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "score": 0.9}]
        ),
    )
    monkeypatch.setattr(pl, "_official_val_scenes", lambda: {"scene-9999"})  # devkit-free
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump({"state": {"dir": str(state)}, "pseudo": {"conf": 0.5, "tolerance": 1}})
    )
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    with pytest.raises(ValueError, match="no frames survived"):
        pl.run_pseudo_label(
            config, weak_config, arm="random", weights=tmp_path / "best.pt",
            processed_dir=processed, device="cpu",
        )


def test_run_pseudo_label_rejects_val_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A val frame in the accepted set would corrupt every arm's comparison."""
    import yaml

    from nuscenes_data_engine.active_learning import pseudo_label as pl

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    weak_state.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nodata"))
    pd.DataFrame({"sample_data_token": ["v1"]}).to_parquet(state / "random.parquet", index=False)
    pd.DataFrame(
        {"sample_data_token": ["v1"], "filename": ["v1.jpg"], "channel": ["CAM_FRONT"],
         "scene_name": ["scene-0003"]}
    ).to_parquet(processed / "samples.parquet", index=False)
    base = {
        "cars": 0, "trucks": 0, "buses": 0, "trailers": 0, "construction_vehicles": 0,
        "motorcycles": 0, "bicycles": 0, "pedestrians": 0, "traffic_cones": 0,
        "barriers": 0, "parse_status": "ok",
    }
    pd.DataFrame([{**base, "sample_data_token": "v1", "cars": 1}]).to_parquet(
        weak_state / "labels.parquet", index=False
    )
    monkeypatch.setattr(
        pl, "propose_boxes",
        lambda *a, **k: pd.DataFrame(
            [{"sample_data_token": "v1", "category_group": "car",
              "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "score": 0.9}]
        ),
    )
    monkeypatch.setattr(pl, "_official_val_scenes", lambda: {"scene-0003"})  # v1 IS val
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump({"state": {"dir": str(state)}, "pseudo": {"conf": 0.5, "tolerance": 1}})
    )
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    with pytest.raises(ValueError, match="val split"):
        pl.run_pseudo_label(
            config, weak_config, arm="random", weights=tmp_path / "best.pt",
            processed_dir=processed, device="cpu",
        )
