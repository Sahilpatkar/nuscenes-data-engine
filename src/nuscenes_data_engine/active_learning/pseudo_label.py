"""VLM-verified self-training: pseudo-label mined frames without ground truth.

Phase 6b's VLM emits scene-level *counts*, not boxes, so it cannot label frames for a
detector directly. Instead the baseline detector proposes boxes and the VLM verifies
them: a frame is kept only when every detector class the VLM can see agrees within a
tolerance. This leans on 6b's measured strength (presence tagging) rather than its
weakness (exact counts when crowded). Design:
docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

# VLM ObjectCounts field -> detector class. The five VLM classes with no detector
# counterpart (trailers, construction_vehicles, motorcycles, traffic_cones, barriers)
# are intentionally unmapped: the detector cannot be wrong about what it never predicts.
VLM_TO_DETECTOR: dict[str, str] = {
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "pedestrians": "pedestrian",
    "bicycles": "bicycle",
}


def detection_counts(boxes: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-frame, per-detector-class box counts from an annotations-schema frame."""
    counts: dict[str, dict[str, int]] = {}
    for token, group in boxes.groupby("sample_data_token")["category_group"]:
        counts[str(token)] = {str(k): int(v) for k, v in Counter(group).items()}
    return counts


def verify_frames(
    det_counts: dict[str, dict[str, int]],
    vlm_labels: dict[str, dict[str, Any]],
    tolerance: int,
) -> tuple[list[str], dict[str, Any]]:
    """Frames whose pseudo-labels the VLM corroborates, plus rejection diagnostics.

    A frame is accepted iff, for every mapped class, ``|detector - vlm| <= tolerance``.
    Frames with no VLM label, or whose label failed to parse, are rejected — the
    verifier can only vouch for what it actually saw.
    """
    accepted: list[str] = []
    rejected_by_class: Counter[str] = Counter()
    n_no_label = 0
    n_unparsed = 0

    for token in sorted(det_counts):
        label = vlm_labels.get(token)
        if label is None:
            n_no_label += 1
            continue
        if label.get("parse_status") != "ok":
            n_unparsed += 1
            continue
        disagreeing = [
            det_class
            for vlm_field, det_class in VLM_TO_DETECTOR.items()
            if abs(int(det_counts[token].get(det_class, 0)) - int(label[vlm_field]))
            > tolerance
        ]
        if disagreeing:
            rejected_by_class.update(disagreeing)
            continue
        accepted.append(token)

    diagnostics = {
        "n_candidates": len(det_counts),
        "n_accepted": len(accepted),
        "n_no_label": n_no_label,
        "n_unparsed": n_unparsed,
        "retention": len(accepted) / len(det_counts) if det_counts else 0.0,
        "rejected_by_class": dict(rejected_by_class),
    }
    return accepted, diagnostics
