"""Tests for the demo curation pipeline (pure; no torch, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
