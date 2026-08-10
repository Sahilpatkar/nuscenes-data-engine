"""VLM-verified self-training: pseudo-label mined frames without ground truth.

Phase 6b's VLM emits scene-level *counts*, not boxes, so it cannot label frames for a
detector directly. Instead the baseline detector proposes boxes and the VLM verifies
them: a frame is kept only when every detector class the VLM can see agrees within a
tolerance. This leans on 6b's measured strength (presence tagging) rather than its
weakness (exact counts when crowded). Design:
docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md
"""

from __future__ import annotations

import json
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
    if boxes.empty:
        return {}
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

    ``accepted_mutual_zero_by_class`` counts accepted frames where detector and VLM both
    reported zero of a class — agreement that may reflect a shared blind spot rather than
    a true negative (6b measured pedestrian presence recall at 0.58), so it is reported
    alongside retention rather than gating acceptance.
    """
    accepted: list[str] = []
    rejected_by_class: Counter[str] = Counter()
    mutual_zero: Counter[str] = Counter()
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
        mutual_zero.update(
            det_class
            for vlm_field, det_class in VLM_TO_DETECTOR.items()
            if det_counts[token].get(det_class, 0) == 0 and int(label[vlm_field]) == 0
        )

    diagnostics = {
        "n_candidates": len(det_counts),
        "n_accepted": len(accepted),
        "n_no_label": n_no_label,
        "n_unparsed": n_unparsed,
        "retention": len(accepted) / len(det_counts) if det_counts else 0.0,
        "rejected_by_class": dict(rejected_by_class),
        "accepted_mutual_zero_by_class": dict(mutual_zero),
    }
    return accepted, diagnostics


def boxes_to_rows(
    token: str,
    xyxy: Any,
    classes: Any,
    scores: Any,
) -> list[dict[str, Any]]:
    """One detector frame's boxes as annotations-schema rows (pixel xyxy + score).

    NOTE: a checkpoint whose taxonomy merely *overlaps* ours (e.g. COCO, where class 0
    is `person` and ours is `car`) cannot be detected here — the index is in range and
    the mislabel is silent. Always pass a checkpoint fine-tuned on this taxonomy.
    """
    from nuscenes_data_engine.ingestion.categories import DETECTION_CLASSES

    rows: list[dict[str, Any]] = []
    for box, cls, score in zip(xyxy, classes, scores, strict=True):
        index = int(cls)
        if not 0 <= index < len(DETECTION_CLASSES):
            raise ValueError(
                f"class index {index} out of range for {DETECTION_CLASSES} — "
                "wrong weights file (not fine-tuned on this taxonomy)?"
            )
        rows.append(
            {
                "sample_data_token": token,
                "category_group": DETECTION_CLASSES[index],
                "x_min": float(box[0]),
                "y_min": float(box[1]),
                "x_max": float(box[2]),
                "y_max": float(box[3]),
                "score": float(score),
            }
        )
    return rows


def propose_boxes(
    weights: Path,
    frames: pd.DataFrame,
    dataroot: Path,
    *,
    conf: float,
    imgsz: int,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Run the baseline detector over ``frames`` (needs GPU + ultralytics + images).

    ``frames`` needs ``sample_data_token`` and ``filename``. Returns annotations-schema
    rows for every detection at or above ``conf`` — frames with no detection simply
    contribute no rows (a valid empty/background label file downstream).
    """
    from nuscenes_data_engine.training.runtime import configure_ultralytics

    configure_ultralytics()  # before importing ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    records = frames.to_dict("records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [str(dataroot / r["filename"]) for r in batch]
        results = model.predict(paths, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for record, result in zip(batch, results, strict=True):
            boxes = result.boxes
            if not len(boxes):
                continue
            rows.extend(
                boxes_to_rows(
                    str(record["sample_data_token"]),
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy().astype(int),
                    boxes.conf.cpu().numpy(),
                )
            )
        if (start // batch_size) % 10 == 0:
            logger.info("Proposing: %d/%d frames", min(start + batch_size, len(records)), len(records))
    columns = [
        "sample_data_token", "category_group", "x_min", "y_min", "x_max", "y_max", "score",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_weak_sample(
    samples: pd.DataFrame,
    arm_tokens: list[str],
    existing_labels: pd.DataFrame | None,
    *,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The arm frames still needing a VLM label, in Phase 6b's ``sample.parquet`` shape.

    Frames already labelled with ``parse_status == 'ok'`` are skipped; unparsed ones are
    re-labelled. The extra columns (``present``/``in_opus_subset``/``stratum``) exist so
    the unchanged 6b submit/collect path can consume this table.

    ``availability`` mirrors the filter ``sampling.build_sample`` applies against
    ``data/processed/availability.parquet`` (``sample_data_token``, ``present``): frames
    absent from the manifest or marked not-present are excluded — the image isn't on
    disk, so the VLM can't be sent it — rather than defaulted to ``present=True``. When
    ``availability`` is omitted, ``present`` is set to ``True`` for every row (today's
    behaviour, unchanged for existing callers).
    """
    labelled: set[str] = set()
    if existing_labels is not None and len(existing_labels):
        ok = existing_labels[existing_labels["parse_status"] == "ok"]
        labelled = set(ok["sample_data_token"])
    wanted = [t for t in arm_tokens if t not in labelled]
    weak = samples[samples["sample_data_token"].isin(wanted)].copy()
    if availability is not None:
        manifest = availability[["sample_data_token", "present"]]
        weak = weak.merge(manifest[manifest["present"]], on="sample_data_token")
    else:
        weak["present"] = True
    weak["in_opus_subset"] = False
    weak["stratum"] = "weak-supervision"
    return weak.sort_values("sample_data_token", ignore_index=True)


def run_pseudo_sample(
    config_path: Path, weak_config_path: Path, *, arm: str, processed_dir: Path | None = None
) -> dict[str, Any]:
    """Write ``sample.parquet`` (6b shape) for the arm frames still needing VLM labels.

    Raises ``ValueError`` if any arm token is missing from ``samples.parquet`` — a
    mismatch there means the arm was mined against different processed data. When
    ``availability.parquet`` exists it is passed to :func:`build_weak_sample` so frames
    without an on-disk image are dropped rather than silently sent to the VLM.
    """
    from nuscenes_data_engine.config import get_settings, load_yaml

    cfg = load_yaml(config_path)
    weak_cfg = load_yaml(weak_config_path)
    settings = get_settings()
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    weak_state = Path(weak_cfg.get("state", {}).get("dir", "data/active_learning/autolabel_weak"))

    arm_path = state_dir / f"{arm}.parquet"
    if not arm_path.is_file():
        raise ValueError(f"No mined frames for arm {arm!r} at {arm_path}")
    arm_tokens = list(pd.read_parquet(arm_path)["sample_data_token"])

    samples = pd.read_parquet(processed / "samples.parquet")
    missing = set(arm_tokens) - set(samples["sample_data_token"])
    if missing:
        raise ValueError(
            f"{len(missing)} arm frames missing from samples.parquet "
            f"(first: {sorted(missing)[0]})"
        )

    labels_path = Path(settings.data_dir) / "autolabel" / "labels.parquet"
    existing = pd.read_parquet(labels_path) if labels_path.is_file() else None
    weak_labels_path = weak_state / "labels.parquet"
    if weak_labels_path.is_file():  # a previous weak run already labelled some
        weak_existing = pd.read_parquet(weak_labels_path)
        existing = weak_existing if existing is None else pd.concat([existing, weak_existing])

    availability_path = processed / "availability.parquet"
    availability = pd.read_parquet(availability_path) if availability_path.is_file() else None

    weak = build_weak_sample(samples, arm_tokens, existing, availability=availability)
    if availability is not None:
        n_before_availability = len(build_weak_sample(samples, arm_tokens, existing))
        n_not_present = n_before_availability - len(weak)
        logger.info(
            "Arm %s: %d frames dropped as not present on disk (availability manifest)",
            arm, n_not_present,
        )
    weak_state.mkdir(parents=True, exist_ok=True)
    weak.to_parquet(weak_state / "sample.parquet", index=False)
    summary = {
        "arm": arm,
        "n_arm_frames": len(arm_tokens),
        "n_needing_labels": len(weak),
        "sample_parquet": str(weak_state / "sample.parquet"),
    }
    logger.info(
        "Arm %s: %d/%d frames need VLM labels -> %s",
        arm, len(weak), len(arm_tokens), weak_state / "sample.parquet",
    )
    return summary


def _official_val_scenes() -> set[str]:
    """The nuScenes val scene names (devkit); isolated so tests can monkeypatch it."""
    from nuscenes.utils.splits import create_splits_scenes

    return set(create_splits_scenes()["val"])


def run_pseudo_label(
    config_path: Path,
    weak_config_path: Path,
    *,
    arm: str,
    weights: Path,
    processed_dir: Path | None = None,
    device: str = "0",
) -> dict[str, Any]:
    """Propose boxes for an arm's frames, verify them against VLM counts, write both.

    Writes ``<arm>_pseudo_labels.parquet`` (accepted frames' boxes),
    ``<arm>_accepted.parquet`` (the token list both weak arms train on), and
    ``<arm>_pseudo_summary.json`` (retention/rejection diagnostics).

    Every cheap check — arm/sample consistency, VLM label coverage, val-split
    membership — runs before :func:`propose_boxes` so a run started too early (before
    ``autolabel collect`` finished labelling every arm frame) fails loudly instead of
    silently shrinking to whatever subset happened to have a label.
    """
    from nuscenes_data_engine.config import get_settings, load_yaml

    cfg = load_yaml(config_path)
    weak_cfg = load_yaml(weak_config_path)
    settings = get_settings()
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    pseudo_cfg = cfg.get("pseudo") or {}
    conf = float(pseudo_cfg.get("conf", 0.5))
    tolerance = int(pseudo_cfg.get("tolerance", 1))
    imgsz = int(pseudo_cfg.get("imgsz", cfg.get("train", {}).get("imgsz", 640)))
    batch_size = int(pseudo_cfg.get("batch", 32))
    weak_state = Path(weak_cfg.get("state", {}).get("dir", "data/active_learning/autolabel_weak"))

    arm_path = state_dir / f"{arm}.parquet"
    if not arm_path.is_file():
        raise ValueError(f"No mined frames for arm {arm!r} at {arm_path}")
    arm_tokens = set(pd.read_parquet(arm_path)["sample_data_token"])

    samples = pd.read_parquet(processed / "samples.parquet")
    missing = arm_tokens - set(samples["sample_data_token"])
    if missing:
        raise ValueError(
            f"{len(missing)} arm frames missing from samples (first: {sorted(missing)[0]})"
        )
    frames = samples[samples["sample_data_token"].isin(arm_tokens)]

    labels: list[pd.DataFrame] = []
    for path in (
        Path(settings.data_dir) / "autolabel" / "labels.parquet",
        weak_state / "labels.parquet",
    ):
        if path.is_file():
            labels.append(pd.read_parquet(path))
    if not labels:
        raise ValueError("No VLM labels found — run `al pseudo-sample` then autolabel first.")
    all_labels = pd.concat(labels, ignore_index=True)
    # Prefer a parsed label over an unparsed one regardless of source order; the
    # weak table only ever holds tokens build_weak_sample deemed unlabelled, so this
    # matters mainly for a re-run or a future arm.
    all_labels = (
        all_labels.assign(_ok=(all_labels["parse_status"] == "ok").astype(int))
        .sort_values("_ok", kind="stable")
        .drop_duplicates(subset="sample_data_token", keep="last")
        .drop(columns="_ok")
    )
    vlm_labels = {
        str(row["sample_data_token"]): row
        for row in all_labels.to_dict("records")
        if str(row["sample_data_token"]) in arm_tokens
    }

    # By construction every arm frame carries a VLM label by this point (pseudo-sample
    # + autolabel collect ran first); if any don't, this ran before collect finished —
    # fail now, before the GPU proposal, rather than quietly verifying a shrunken subset.
    unlabelled = sorted(t for t in arm_tokens if t not in vlm_labels)
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} of {len(arm_tokens)} arm frames have no VLM label "
            f"(first: {unlabelled[0]}) — run `al pseudo-sample --arm {arm}` and finish "
            "`autolabel collect` before pseudo-labelling"
        )

    # Defence in depth: the arm parquet is pool-only by construction (round-1 mining
    # guards), but a pseudo-labelled val frame would silently corrupt every arm's
    # comparison, so check it here too rather than trusting an upstream invariant.
    # Computed before propose_boxes so a missing devkit or missing scene_name column
    # also surfaces before the GPU run.
    val_scenes = _official_val_scenes()
    val_frames = set(samples[samples["scene_name"].isin(val_scenes)]["sample_data_token"])

    boxes = propose_boxes(
        weights, frames, Path(settings.nuscenes_dataroot),
        conf=conf, imgsz=imgsz, device=device, batch_size=batch_size,
    )
    # Frames with no detection still count as candidates (empty label = background).
    det_counts: dict[str, dict[str, int]] = {token: {} for token in sorted(arm_tokens)}
    det_counts.update(detection_counts(boxes))

    accepted, diagnostics = verify_frames(det_counts, vlm_labels, tolerance)
    if not accepted:
        raise ValueError(
            f"Verification kept nothing for arm {arm!r}: no frames survived "
            f"(candidates {diagnostics['n_candidates']}, no-label {diagnostics['n_no_label']}, "
            f"unparsed {diagnostics['n_unparsed']})"
        )

    accepted_set = set(accepted)
    if not accepted_set <= arm_tokens:
        raise ValueError("accepted frames must come from the arm's mined frames")
    if accepted_set & val_frames:
        raise ValueError(
            f"{len(accepted_set & val_frames)} accepted frames are in the val split"
        )

    pseudo = boxes[boxes["sample_data_token"].isin(accepted_set)].reset_index(drop=True)
    pd.DataFrame({"sample_data_token": accepted}).to_parquet(
        state_dir / f"{arm}_accepted.parquet", index=False
    )
    pseudo.to_parquet(state_dir / f"{arm}_pseudo_labels.parquet", index=False)

    # How the detector's kept-frame boxes compare to ground truth, when available —
    # tells the write-up whether the detector systematically under-labels the frames
    # it keeps (a training-data property distinct from verification retention).
    gt_mean = None
    annotations_path = processed / "annotations.parquet"
    if annotations_path.is_file():
        gt = pd.read_parquet(annotations_path, columns=["sample_data_token", "category_group"])
        gt = gt[gt["category_group"].notna() & gt["sample_data_token"].isin(accepted_set)]
        gt_mean = len(gt) / len(accepted)

    summary = {
        "arm": arm,
        "conf": conf,
        "tolerance": tolerance,
        "n_boxes": len(pseudo),
        "mean_boxes_per_accepted_frame": len(pseudo) / len(accepted),
        "mean_gt_boxes_per_accepted_frame": gt_mean,
        **diagnostics,
    }
    (state_dir / f"{arm}_pseudo_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Arm %s: %d/%d frames accepted (retention %.2f), %d pseudo boxes; "
        "rejected by class: %s; accepted mutual-zero by class: %s",
        arm, diagnostics["n_accepted"], diagnostics["n_candidates"],
        diagnostics["retention"], len(pseudo), diagnostics["rejected_by_class"],
        diagnostics["accepted_mutual_zero_by_class"],
    )
    return summary
