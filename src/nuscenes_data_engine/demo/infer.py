"""``demo infer`` — local box-level inference over the curated frame manifest.

Reads the manifest written by ``demo curate``, crops every frame (val and
train_pool) to a fixed display size, and — for ``val`` frames only — runs each
injected model, box-matches its predictions against ground truth with
``match_frame_boxes`` (the same greedy semantics ``active_learning/sweep.py`` uses
for ``failures.parquet``), and writes ``predictions.parquet`` + ``gt_boxes.parquet``.
``frame_manifest.parquet`` is rewritten in place with ``fixes_fn_vs_<a>_<b>``
exemplar columns for every ordered model pair.

Visibility filter (2026-08-18 correction): ``active_learning/sweep.py`` filters GT to
``visibility_token >= visibility_min`` (``configs/active_learning.yaml``'s
``sweep.visibility_min``, default ``"2"``) before matching — see
``sweep.py:77-80``. This module mirrors that exactly so the demo's tp/fp/fn language
means what ``failures.parquet`` means: GT rows below the threshold are EXCLUDED from
matching (their ``matched_<model>`` columns stay NA, not False — NA is "not
evaluated", False would falsely claim "evaluated and missed") but are still written
to ``gt_boxes.parquet`` with ``below_visibility_min=True`` so the UI can render them
ghosted without counting them as false negatives.

The model call is an injected adapter (``predict_fn(image_path) -> {"boxes",
"conf", "classes"}``), so everything here is torch-free and unit-tested.
``make_ultralytics_predictor`` is the one real (lazy-imported, untested) adapter —
ultralytics/torch are not in the local venv; it is exercised only by the real
``demo infer`` CLI run once ``uv sync --extra train --extra engine`` is done.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from nuscenes_data_engine.active_learning.matching import match_frame_boxes
from nuscenes_data_engine.ingestion.categories import CLASS_TO_INDEX, DETECTION_CLASSES

logger = logging.getLogger("nuscenes_data_engine")

_GT_COLUMNS = (
    "annotation_token",
    "sample_data_token",
    "category_group",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
)

_PREDICTION_COLUMNS = (
    "sample_data_token",
    "model",
    "category_group",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "conf",
    "status",
    "matched_annotation_token",
)


def run_infer(
    *,
    staging_dir: Path,
    annotations: pd.DataFrame,
    models: dict[str, Callable[[Path], dict[str, Any]]],
    crop_size: tuple[int, int],
    iou: float,
    conf_hit: float,
    images_root: Path | None = None,
    visibility_min: str = "2",
) -> dict[str, Any]:
    """Run inference + box matching over the staged curation manifest.

    Every manifest row's image is resolved and cropped (val and train_pool alike);
    a missing image raises naming the token before any model is called. Only
    ``val`` frames are matched against GT — ``train_pool`` frames have no
    failure-ledger equivalent to agree with, so running models over them would
    invite exactly the kind of un-auditable number the visibility correction exists
    to avoid.
    """
    staging_dir = Path(staging_dir)
    images_root = Path(images_root) if images_root is not None else staging_dir
    manifest = pd.read_parquet(staging_dir / "frame_manifest.parquet")
    model_names = list(models)

    # Step 1: resolve + crop every frame (both splits) before any model runs, so a
    # missing image always fails loudly up front rather than mid-inference.
    crops_dir = staging_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, Path] = {}
    for row in manifest.itertuples():
        token = row.sample_data_token
        image_path = images_root / row.filename
        if not image_path.is_file():
            raise ValueError(f"demo infer: image missing for token {token!r}: {image_path}")
        image_paths[token] = image_path
        with Image.open(image_path) as img:
            img.convert("RGB").resize(crop_size, Image.Resampling.LANCZOS).save(
                crops_dir / f"{token}.jpg"
            )

    val_tokens = list(manifest.loc[manifest["split"] == "val", "sample_data_token"])

    # GT prep mirrors sweep.py:76-80 exactly (dropna category_group, then the
    # visibility filter) except the visibility filter FLAGS instead of dropping —
    # gt_boxes.parquet must carry every curated-token GT row.
    curated_tokens = set(manifest["sample_data_token"])
    gt_all = annotations[annotations["sample_data_token"].isin(curated_tokens)].copy()
    gt_all = gt_all.dropna(subset=["category_group"]).reset_index(drop=True)
    visibility = pd.to_numeric(gt_all["visibility_token"], errors="coerce")
    gt_all["below_visibility_min"] = ~(visibility >= int(visibility_min))
    for model_name in model_names:
        gt_all[f"matched_{model_name}"] = pd.array([pd.NA] * len(gt_all), dtype="boolean")

    prediction_rows: list[dict[str, Any]] = []
    # (token, model_a, model_b) -> bool: model_a had an unmatched GT that model_b matched.
    fixes: dict[str, dict[tuple[str, str], bool]] = {}

    for token in val_tokens:
        image_path = image_paths[token]
        frame_mask = (gt_all["sample_data_token"] == token) & (~gt_all["below_visibility_min"])
        frame_gt = gt_all.loc[frame_mask]
        gt_boxes = frame_gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
        gt_classes = frame_gt["category_group"].map(CLASS_TO_INDEX).to_numpy(dtype=int)
        gt_index = frame_gt.index.to_numpy()

        gt_matched_by_model: dict[str, np.ndarray[Any, Any]] = {}
        for model_name, predict_fn in models.items():
            result = predict_fn(image_path)
            pred_boxes = np.asarray(result["boxes"], dtype=float).reshape(-1, 4)
            pred_conf = np.asarray(result["conf"], dtype=float)
            classes_raw = list(result["classes"])
            pred_classes = np.array([CLASS_TO_INDEX[c] for c in classes_raw], dtype=int)

            matches = match_frame_boxes(
                pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes,
                iou=iou, conf_hit=conf_hit,
            )
            gt_matched_by_model[model_name] = matches.gt_matched
            gt_all.loc[gt_index, f"matched_{model_name}"] = matches.gt_matched

            for i in range(len(pred_boxes)):
                gt_row = int(matches.pred_matched_gt[i])
                matched_token = (
                    frame_gt.iloc[gt_row]["annotation_token"] if gt_row >= 0 else None
                )
                prediction_rows.append({
                    "sample_data_token": token,
                    "model": model_name,
                    "category_group": classes_raw[i],
                    "x_min": float(pred_boxes[i, 0]),
                    "y_min": float(pred_boxes[i, 1]),
                    "x_max": float(pred_boxes[i, 2]),
                    "y_max": float(pred_boxes[i, 3]),
                    "conf": float(pred_conf[i]),
                    "status": matches.pred_status[i],
                    "matched_annotation_token": matched_token,
                })

        for model_a in model_names:
            fn_a = ~gt_matched_by_model[model_a]
            for model_b in model_names:
                if model_a == model_b:
                    continue
                fixed = bool(np.any(fn_a & gt_matched_by_model[model_b])) if len(fn_a) else False
                fixes.setdefault(token, {})[(model_a, model_b)] = fixed

    predictions = pd.DataFrame(prediction_rows, columns=list(_PREDICTION_COLUMNS))
    predictions.to_parquet(staging_dir / "predictions.parquet")

    gt_out_columns = [*_GT_COLUMNS, "below_visibility_min", *(f"matched_{m}" for m in model_names)]
    gt_all[gt_out_columns].to_parquet(staging_dir / "gt_boxes.parquet")

    pair_columns = [f"fixes_fn_vs_{a}_{b}" for a in model_names for b in model_names if a != b]
    for col in pair_columns:
        manifest[col] = pd.array([pd.NA] * len(manifest), dtype="boolean")
    for idx, row in manifest.iterrows():
        token_fixes = fixes.get(row["sample_data_token"])
        if token_fixes is None:
            continue
        for (model_a, model_b), value in token_fixes.items():
            manifest.at[idx, f"fixes_fn_vs_{model_a}_{model_b}"] = value
    manifest.to_parquet(staging_dir / "frame_manifest.parquet")

    logger.info(
        "demo infer: %d val frames, %d models, %d predictions -> %s",
        len(val_tokens), len(model_names), len(predictions), staging_dir,
    )
    return {
        "n_predictions": len(predictions),
        "n_val_frames": len(val_tokens),
        "n_train_pool_frames": len(manifest) - len(val_tokens),
    }


def make_ultralytics_predictor(
    weights: Path, *, imgsz: int = 640, conf: float = 0.05, device: str = "cpu"
) -> Callable[[Path], dict[str, Any]]:
    """Wrap an ultralytics YOLO checkpoint as a ``predict_fn`` for ``run_infer``.

    NOT unit-tested — ultralytics/torch are not in the local venv (see the
    ``uv sync --extra train --extra engine`` prerequisite in ``docs/DEMO.md``); this
    thin adapter is exercised only by the real ``demo infer`` CLI run. Mirrors
    ``active_learning/sweep.py``'s ``configure_ultralytics()`` call (before importing
    ultralytics) and its ``conf=0.05`` proposal threshold (catch low-confidence hits
    for the demo's low_conf status, same as the sweep does for failures.parquet).
    """
    from nuscenes_data_engine.training.runtime import configure_ultralytics

    configure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(image_path: Path) -> dict[str, Any]:
        results = model.predict(
            [str(image_path)], imgsz=imgsz, conf=conf, device=device, verbose=False
        )
        boxes = results[0].boxes
        if len(boxes) == 0:
            return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "classes": []}
        pred_cls_idx = boxes.cls.cpu().numpy().astype(int)
        return {
            "boxes": boxes.xyxy.cpu().numpy(),
            "conf": boxes.conf.cpu().numpy(),
            "classes": [DETECTION_CLASSES[i] for i in pred_cls_idx],
        }

    return predict
