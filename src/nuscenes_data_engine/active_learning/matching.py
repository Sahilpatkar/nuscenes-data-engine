"""Prediction↔ground-truth box matching — pure numpy so it tests in torch-free CI.

Greedy per-class matching: predictions in descending-confidence order each claim their
best-IoU unmatched GT box (IoU >= threshold). Unclaimed GT boxes are false negatives;
matches below ``conf_hit`` are "low-confidence hits" — both feed the failure score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FrameFailure:
    """Failure accounting for one frame."""

    n_gt: int
    n_matched: int
    n_fn: int
    n_low_conf: int

    @property
    def failure_score(self) -> float:
        return self.n_fn + 0.5 * self.n_low_conf


def iou_matrix(boxes_a: np.ndarray[Any, Any], boxes_b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Pairwise IoU for two (N,4)/(M,4) xyxy arrays -> (N,M)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_frame(
    pred_boxes: np.ndarray[Any, Any],
    pred_classes: np.ndarray[Any, Any],
    pred_conf: np.ndarray[Any, Any],
    gt_boxes: np.ndarray[Any, Any],
    gt_classes: np.ndarray[Any, Any],
    *,
    iou: float = 0.5,
    conf_hit: float = 0.4,
) -> FrameFailure:
    """Greedy per-class matching of one frame's predictions against its GT."""
    n_gt = len(gt_boxes)
    matched_gt = np.zeros(n_gt, dtype=bool)
    n_matched = 0
    n_low_conf = 0

    for cls in np.unique(gt_classes) if n_gt else []:
        gt_idx = np.flatnonzero(gt_classes == cls)
        pred_idx = np.flatnonzero(pred_classes == cls)
        if len(pred_idx) == 0:
            continue
        order = pred_idx[np.argsort(-pred_conf[pred_idx])]
        ious = iou_matrix(pred_boxes[order], gt_boxes[gt_idx])
        claimed = np.zeros(len(gt_idx), dtype=bool)
        for row, pred_i in enumerate(order):
            candidates = np.where(claimed, -1.0, ious[row])
            best = int(np.argmax(candidates)) if len(candidates) else -1
            if best >= 0 and candidates[best] >= iou:
                claimed[best] = True
                matched_gt[gt_idx[best]] = True
                n_matched += 1
                if pred_conf[pred_i] < conf_hit:
                    n_low_conf += 1
        del claimed

    return FrameFailure(
        n_gt=n_gt,
        n_matched=n_matched,
        n_fn=int(n_gt - matched_gt.sum()),
        n_low_conf=n_low_conf,
    )
