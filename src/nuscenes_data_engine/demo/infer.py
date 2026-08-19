"""``demo infer`` — local box-level inference over the curated frame manifest.

Reads the manifest written by ``demo curate``, crops every frame (val and
train_pool) to a fixed display size, and — for ``val`` frames only — runs each
injected model, box-matches its predictions against ground truth with
``match_frame_boxes`` (the same greedy semantics ``active_learning/sweep.py`` uses
for ``failures.parquet``), and writes ``predictions.parquet`` + ``gt_boxes.parquet``.
``frame_manifest.parquet`` is rewritten in place with ``fixes_fn_vs_<a>_<b>``
exemplar columns for every ordered model pair, plus one ``n_preds_<model>`` nullable-
Int64 column per configured model: the raw prediction count (incl. low_conf) for
``val`` rows, NA for ``train_pool`` rows never evaluated. Zero is a legitimate,
recorded finding ("ran and found nothing" — exactly a total-miss frame the demo
wants to show); NA means the model never ran on that token at all. This distinction
is what ``demo build``'s val-coverage validation checks — ``predictions.parquet``
alone can't tell "found nothing" from "never ran" apart, since both write zero rows.

Visibility filter (2026-08-18 correction): ``active_learning/sweep.py`` filters GT to
``visibility_token >= visibility_min`` (``configs/active_learning.yaml``'s
``sweep.visibility_min``, default ``"2"``) before matching — see
``sweep.py:77-80``. This module mirrors that exactly so the demo's tp/fp/fn language
means what ``failures.parquet`` means: GT rows below the threshold are EXCLUDED from
matching (their ``matched_<model>`` columns stay NA, not False — NA is "not
evaluated", False would falsely claim "evaluated and missed") but are still written
to ``gt_boxes.parquet`` with ``below_visibility_min=True`` so the UI can render them
ghosted without counting them as false negatives. ``visibility_min=None`` mirrors
sweep.py's own null handling (``if visibility_min is not None: ...``): no filter at
all, every GT row in scope, none ghosted — NOT the string ``"None"`` fed to ``int()``,
which crashes (a real config divergence the CLI must avoid by passing ``None``
through rather than ``str()``-wrapping whatever ``sweep_cfg.get(...)`` returns).

NOTE: on today's ingested data ``below_visibility_min`` is always False — ingestion
already applies ``projection.visibility_min: 2`` (``configs/data.yaml:19``) before
``annotations.parquet`` is ever written, so no row with ``visibility_token < 2``
exists to flag. The column is parity-defensive (data ingested without that floor, or
a future ``visibility_min`` lower than ingestion's) rather than something today's
demo package actually exercises — Phase 3's UI should not build ghost-box rendering
against it without first confirming ghost rows occur in the data it is fed.

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
from nuscenes_data_engine.training.dataset import IMAGE_HEIGHT, IMAGE_WIDTH

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
    "imgsz",
)

# Explicit dtypes so predictions.parquet's schema is stable regardless of row count —
# an all-empty ``prediction_rows`` (e.g. a manifest with no val frames) would otherwise
# infer every column as object/float64 from zero samples, which can silently differ
# from a non-empty run's schema and break downstream readers expecting a fixed schema.
_PREDICTION_DTYPES: dict[str, str] = {
    "sample_data_token": "object",
    "model": "object",
    "category_group": "object",
    "x_min": "float64",
    "y_min": "float64",
    "x_max": "float64",
    "y_max": "float64",
    "conf": "float64",
    "status": "object",
    "matched_annotation_token": "object",
    "imgsz": "Int64",  # nullable: only make_ultralytics_predictor's results report it
}


def _map_pred_classes(classes_raw: list[str], *, model_name: str) -> np.ndarray[Any, Any]:
    """Map predicted class NAMES to detector indices, mirroring sweep.py:112's GT map.

    A class outside ``CLASS_TO_INDEX`` means the injected model isn't speaking this
    project's taxonomy (e.g. a COCO-class name) — a bare ``KeyError`` would name
    neither the offending model nor the accepted vocabulary, so it's translated into
    a ``ValueError`` that names both.
    """
    try:
        return np.array([CLASS_TO_INDEX[c] for c in classes_raw], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"demo infer: model {model_name!r} predicted unmapped class {exc.args[0]!r} "
            f"— expected one of {sorted(CLASS_TO_INDEX)}"
        ) from exc


def run_infer(
    *,
    staging_dir: Path,
    annotations: pd.DataFrame,
    models: dict[str, Callable[[Path], dict[str, Any]]],
    crop_size: tuple[int, int],
    iou: float,
    conf_hit: float,
    images_root: Path | None = None,
    visibility_min: str | None = "2",
) -> dict[str, Any]:
    """Run inference + box matching over the staged curation manifest.

    Every manifest row's image is resolved, size-checked, and cropped (val and
    train_pool alike); a missing or wrong-sized image raises naming the token before
    any model is called. Only ``val`` frames are matched against GT —
    ``train_pool`` frames have no failure-ledger equivalent to agree with, so running
    models over them would invite exactly the kind of un-auditable number the
    visibility correction exists to avoid.
    """
    staging_dir = Path(staging_dir)
    images_root = Path(images_root) if images_root is not None else staging_dir
    manifest = pd.read_parquet(staging_dir / "frame_manifest.parquet")
    model_names = list(models)

    # Step 1: resolve + crop every frame (both splits) before any model runs, so a
    # missing/wrong-sized image always fails loudly up front rather than mid-inference.
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
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(
                    f"demo infer: token {token!r} image is {img.size}, expected "
                    f"{(IMAGE_WIDTH, IMAGE_HEIGHT)} — predictions/GT would be scored "
                    "in the wrong pixel space"
                )
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
    if visibility_min is None:
        # Mirrors sweep.py's `if visibility_min is not None: ...` — null means no
        # filter at all, not "filter at threshold 0". str(None) fed to int() would
        # crash; every GT row stays in scope, none are ghosted.
        gt_all["below_visibility_min"] = False
    else:
        visibility = pd.to_numeric(gt_all["visibility_token"], errors="coerce")
        gt_all["below_visibility_min"] = ~(visibility >= int(visibility_min))
    for model_name in model_names:
        gt_all[f"matched_{model_name}"] = pd.array([pd.NA] * len(gt_all), dtype="boolean")

    prediction_rows: list[dict[str, Any]] = []
    # (token, model_a, model_b) -> bool: model_a had an unmatched GT that model_b matched.
    fixes: dict[str, dict[tuple[str, str], bool]] = {}
    # token -> {model: raw prediction count, incl. low_conf}. Recorded even when 0 —
    # "ran and found nothing" is a legitimate finding (exactly the total-miss frame
    # the demo wants to show) and must read differently on the manifest than "never
    # ran" (NA), since predictions.parquet has zero rows for either case and can't
    # tell them apart on its own.
    n_preds: dict[str, dict[str, int]] = {}

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
            pred_classes = _map_pred_classes(classes_raw, model_name=model_name)
            imgsz = result.get("imgsz")

            matches = match_frame_boxes(
                pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes,
                iou=iou, conf_hit=conf_hit,
            )
            gt_matched_by_model[model_name] = matches.gt_matched
            gt_all.loc[gt_index, f"matched_{model_name}"] = matches.gt_matched
            n_preds.setdefault(token, {})[model_name] = len(pred_boxes)

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
                    "imgsz": imgsz,
                })

        for model_a in model_names:
            fn_a = ~gt_matched_by_model[model_a]
            for model_b in model_names:
                if model_a == model_b:
                    continue
                fixed = bool(np.any(fn_a & gt_matched_by_model[model_b]))
                fixes.setdefault(token, {})[(model_a, model_b)] = fixed

    predictions = pd.DataFrame(prediction_rows, columns=list(_PREDICTION_COLUMNS))
    predictions = predictions.astype(_PREDICTION_DTYPES)
    predictions.to_parquet(staging_dir / "predictions.parquet")

    gt_out_columns = [*_GT_COLUMNS, "below_visibility_min", *(f"matched_{m}" for m in model_names)]
    gt_all[gt_out_columns].to_parquet(staging_dir / "gt_boxes.parquet")

    # Drop any exemplar/coverage columns from a PRIOR run's model roster before
    # adding this run's — rerunning with a different model set must not ship stale
    # fixes_fn_vs_<a>_<b> or n_preds_<model> columns alongside the current ones.
    # Column names are built by joining against model_names (not parsed back via
    # string-splitting), since model names may themselves contain underscores (e.g.
    # "graph_rate_night").
    manifest = manifest.drop(
        columns=[
            c for c in manifest.columns
            if c.startswith("fixes_fn_vs_") or c.startswith("n_preds_")
        ]
    )
    pair_columns = [f"fixes_fn_vs_{a}_{b}" for a in model_names for b in model_names if a != b]
    for col in pair_columns:
        manifest[col] = pd.array([pd.NA] * len(manifest), dtype="boolean")
    n_preds_columns = [f"n_preds_{m}" for m in model_names]
    for col in n_preds_columns:
        manifest[col] = pd.array([pd.NA] * len(manifest), dtype="Int64")
    for idx, row in manifest.iterrows():
        token = row["sample_data_token"]
        token_fixes = fixes.get(token)
        if token_fixes is not None:
            for (model_a, model_b), value in token_fixes.items():
                manifest.at[idx, f"fixes_fn_vs_{model_a}_{model_b}"] = value
        token_counts = n_preds.get(token)
        if token_counts is not None:
            for model_name, count in token_counts.items():
                manifest.at[idx, f"n_preds_{model_name}"] = count
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
    ultralytics).

    ``imgsz``/``conf`` are the caller's responsibility, not this adapter's: each
    checkpoint must be predicted at (or near) the ``imgsz`` it was trained at — the
    demo's three checkpoints do NOT all share one (see ``configs/demo.yaml``
    ``models.<name>.imgsz``, sourced from each run's own ``args.yaml``) — and the
    demo CLI reads ``conf`` from ``configs/active_learning.yaml``'s
    ``sweep.conf_low`` rather than relying on this function's own default. The
    ``imgsz=640``/``conf=0.05`` defaults here exist only as a fallback for direct,
    non-CLI use; they are not claimed to "mirror" anything mechanically.

    Raises ``ValueError`` if the checkpoint's class taxonomy doesn't match
    ``DETECTION_CLASSES`` — a checkpoint that merely overlaps ours (e.g. a stock COCO
    checkpoint, where class 0 is "person") would otherwise silently mislabel every
    prediction (see ``active_learning/pseudo_label.py``'s ``boxes_to_rows`` for the
    same class of bug on the index side; this is the checkpoint-identity side).
    """
    from nuscenes_data_engine.training.runtime import configure_ultralytics

    configure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    if tuple(names[i] for i in range(len(names))) != DETECTION_CLASSES:
        raise ValueError(
            f"{weights}: checkpoint classes {names} != {DETECTION_CLASSES} — "
            "not fine-tuned on this taxonomy? (a COCO checkpoint would silently mislabel)"
        )

    def predict(image_path: Path) -> dict[str, Any]:
        results = model.predict(
            [str(image_path)], imgsz=imgsz, conf=conf, device=device, verbose=False
        )
        boxes = results[0].boxes
        if len(boxes) == 0:
            return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "classes": [], "imgsz": imgsz}
        pred_cls_idx = boxes.cls.cpu().numpy().astype(int)
        return {
            "boxes": boxes.xyxy.cpu().numpy(),
            "conf": boxes.conf.cpu().numpy(),
            "classes": [DETECTION_CLASSES[i] for i in pred_cls_idx],
            "imgsz": imgsz,
        }

    return predict
