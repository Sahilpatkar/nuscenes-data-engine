"""Tests for the demo curation pipeline (pure; no torch, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nuscenes_data_engine.active_learning.matching import match_frame, match_frame_boxes


def test_match_frame_boxes_assigns_statuses() -> None:
    gt_boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=float)
    gt_classes = np.array([0, 1])
    pred_boxes = np.array(
        [[5, 5, 105, 105], [400, 400, 500, 500], [210, 210, 290, 290]], dtype=float
    )
    pred_classes = np.array([0, 0, 1])
    pred_conf = np.array([0.9, 0.8, 0.2])
    result = match_frame_boxes(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    assert list(result.pred_status) == ["tp", "fp", "low_conf"]
    assert result.pred_matched_gt[0] == 0          # first pred matched GT row 0
    assert result.pred_matched_gt[1] == -1         # unmatched
    # IoU(pred2, gt1) = 0.64 >= iou threshold, so the low-conf pred DOES claim GT
    # row 1 — match_frame counts low-conf hits toward n_matched/matched_gt too
    # (verified directly against match_frame on this exact fixture), so gt_matched
    # must be [True, True] here for the equivalence property to hold in general.
    assert list(result.gt_matched) == [True, True]  # low-conf pred still matches GT


def test_match_frame_boxes_one_gt_consumes_one_pred() -> None:
    gt_boxes = np.array([[0, 0, 100, 100]], dtype=float)
    gt_classes = np.array([0])
    pred_boxes = np.array([[0, 0, 100, 100], [1, 1, 99, 99]], dtype=float)
    pred_classes = np.array([0, 0])
    pred_conf = np.array([0.9, 0.85])
    result = match_frame_boxes(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    assert list(result.pred_status).count("tp") == 1
    assert list(result.pred_status).count("fp") == 1


def test_match_frame_boxes_iou_boundary() -> None:
    # IoU exactly at the threshold counts as a match iff match_frame counts it.
    gt_boxes = np.array([[0, 0, 100, 100]], dtype=float)
    gt_classes = np.array([0])
    pred_boxes = np.array([[0, 0, 100, 50]], dtype=float)  # IoU = 0.5 exactly
    pred_classes = np.array([0])
    pred_conf = np.array([0.9])
    boxes = match_frame_boxes(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    frame = match_frame(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    assert (boxes.pred_status[0] == "tp") == (frame.n_matched == 1)


def test_match_frame_boxes_counts_agree_with_match_frame() -> None:
    """Property: box-level results aggregate to exactly match_frame's counts."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        n_gt, n_pred = rng.integers(0, 8), rng.integers(0, 8)
        gt_boxes = rng.uniform(0, 800, (n_gt, 2))
        gt_boxes = np.hstack([gt_boxes, gt_boxes + rng.uniform(20, 200, (n_gt, 2))])
        gt_classes = rng.integers(0, 3, n_gt)
        pred_boxes = rng.uniform(0, 800, (n_pred, 2))
        pred_boxes = np.hstack([pred_boxes, pred_boxes + rng.uniform(20, 200, (n_pred, 2))])
        pred_classes = rng.integers(0, 3, n_pred)
        pred_conf = rng.uniform(0.05, 1.0, n_pred)
        frame = match_frame(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
        )
        boxes = match_frame_boxes(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
        )
        assert boxes.gt_matched.sum() == frame.n_matched
        assert (~boxes.gt_matched).sum() == frame.n_fn
        assert list(boxes.pred_status).count("low_conf") == frame.n_low_conf


def _curation_fixture(tmp_path: Path) -> dict[str, Path]:
    processed = tmp_path / "processed"
    processed.mkdir()
    al = tmp_path / "al"
    al.mkdir()
    pd.DataFrame({
        "sample_data_token": [f"v{i}" for i in range(8)],
        "scene_name": ["s"] * 8,
        "is_night": [True, True, True, False, False, False, False, False],
        "n_gt": [5, 4, 3, 6, 2, 3, 4, 3],
        "n_matched": [1, 2, 3, 2, 2, 3, 4, 3],
        "n_fn": [4, 2, 0, 4, 0, 0, 0, 0],
        "n_low_conf": [2, 1, 0, 3, 0, 0, 0, 0],
        "failure_score": [0.9, 0.7, 0.2, 0.8, 0.1, 0.05, 0.02, 0.01],
    }).to_parquet(al / "failures.parquet")
    pd.DataFrame({
        "sample_data_token": [f"v{i}" for i in range(8)] + ["t1", "t2", "t3"],
        "sample_token": [f"st{i}" for i in range(8)] + ["st8", "st9", "st10"],
        "channel": ["CAM_FRONT"] * 11,
        "filename": [f"samples/CAM_FRONT/{i}.jpg" for i in range(11)],
    }).to_parquet(processed / "samples.parquet")
    pd.DataFrame({
        "sample_token": ["st0", "st9"], "is_hard_braking": [True, True],
    }).to_parquet(processed / "canbus.parquet")
    pd.DataFrame({"sample_data_token": ["t1", "t2", "t3"]}).to_parquet(
        al / "graph_rate_night.parquet"
    )
    pd.DataFrame({"sample_data_token": ["t1"]}).to_parquet(
        al / "graph_rate_night_accepted.parquet"
    )
    return {"processed": processed, "al": al}


def test_curate_buckets_split_and_dedup(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    quotas = {"night_failure": 2, "day_failure": 1, "hard_braking": 2, "al_selected": 3,
              "weak_accepted": 1, "weak_rejected": 2, "clean_success": 1, "semantic": 2}
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging", quotas=quotas,
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: {"v5": ["semantic"], "t2": ["semantic"]},
        semantic_queries=["q"],
    )
    by_token = manifest.set_index("sample_data_token")
    assert by_token.loc["v0", "split"] == "val"          # night_failure top score
    assert by_token.loc["t1", "split"] == "train_pool"    # weak_accepted
    assert "weak_rejected" in by_token.loc["t2", "curation_buckets"]
    assert "semantic" in by_token.loc["t2", "curation_buckets"]  # dedup: multi-bucket
    assert len(manifest) == len(set(manifest["sample_data_token"]))
    # hard_braking maps sample_token -> CAM_FRONT sample_data_token
    assert "hard_braking" in by_token.loc["v0", "curation_buckets"]
    # filelist written, relative, unique
    filelist = (tmp_path / "staging" / "rsync_filelist.txt").read_text().splitlines()
    assert len(filelist) == len(manifest)
    assert all(not line.startswith("/") for line in filelist)


def test_curate_is_deterministic(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    kwargs = dict(
        processed_dir=paths["processed"], al_dir=paths["al"],
        quotas={"night_failure": 2, "day_failure": 1, "hard_braking": 1, "al_selected": 2,
                "weak_accepted": 1, "weak_rejected": 1, "clean_success": 1, "semantic": 0},
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: {}, semantic_queries=[],
    )
    first = run_curate(staging_dir=tmp_path / "s1", **kwargs)
    second = run_curate(staging_dir=tmp_path / "s2", **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_curate_clean_success_requires_no_failures(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging",
        quotas={"night_failure": 0, "day_failure": 0, "hard_braking": 0, "al_selected": 0,
                "weak_accepted": 0, "weak_rejected": 0, "clean_success": 2, "semantic": 0},
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: {}, semantic_queries=[],
    )
    stats = manifest.set_index("sample_data_token")
    assert all(stats["n_fn"] == 0) and all(stats["n_gt"] >= 3)
