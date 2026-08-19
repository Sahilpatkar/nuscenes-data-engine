"""Pure failure-filter logic for the Failure Explorer (no Streamlit import — the
AST guard in tests/test_demo_app.py applies here too, so this module stays
importable and testable without a Streamlit runtime; pandas is the only dependency.
"""

from __future__ import annotations

import pandas as pd

_LIGHTING = {"night": True, "day": False}
_FAILURE_TYPES = ("has_fn", "has_fp", "has_low_conf", "clean")


def failure_flags(
    manifest: pd.DataFrame, gt: pd.DataFrame, preds: pd.DataFrame, *, model: str
) -> pd.DataFrame:
    """Per-val-frame failure flags for ``model``: has_fn / has_fp / has_low_conf / clean.

    Restricted to ``split == "val"`` rows (the train pool has no per-model
    evaluation to flag). ``has_fn`` is True when any GT row for the frame has
    ``matched_<model>`` explicitly False — the nullable-boolean column's NA ("not
    evaluated") does not count as a miss, mirroring ``render.draw_overlay``'s
    NA-safe rendering. ``has_fp`` / ``has_low_conf`` come from that model's
    prediction rows for the frame carrying ``status == "fp"`` / ``"low_conf"``.
    ``clean`` is true when none of the three apply, including frames with zero GT
    rows and zero predictions.
    """
    matched_col = f"matched_{model}"
    val = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    val_tokens = set(val["sample_data_token"])

    gt_val = gt.loc[gt["sample_data_token"].isin(val_tokens)]
    is_fn = gt_val[matched_col].eq(False).fillna(False)  # NA -> "not evaluated", not FN
    fn_tokens = set(gt_val.loc[is_fn, "sample_data_token"])

    preds_val = preds.loc[
        preds["sample_data_token"].isin(val_tokens) & (preds["model"] == model)
    ]
    fp_tokens = set(preds_val.loc[preds_val["status"] == "fp", "sample_data_token"])
    low_conf_tokens = set(preds_val.loc[preds_val["status"] == "low_conf", "sample_data_token"])

    out = val.copy()
    out["has_fn"] = out["sample_data_token"].isin(fn_tokens)
    out["has_fp"] = out["sample_data_token"].isin(fp_tokens)
    out["has_low_conf"] = out["sample_data_token"].isin(low_conf_tokens)
    out["clean"] = ~(out["has_fn"] | out["has_fp"] | out["has_low_conf"])
    return out


def filter_frames(
    *,
    manifest: pd.DataFrame,
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    model: str,
    lighting: str | None = None,
    rain: bool | None = None,
    category: str | None = None,
    size_bucket: str | None = None,
    distance_range: tuple[float, float] | None = None,
    failure_type: str | None = None,
    bucket: str | None = None,
) -> pd.DataFrame:
    """Compose ``failure_flags`` with the sidebar's filter criteria.

    ``lighting``: "night" | "day", matched against ``is_night``. ``rain``: exact
    match against ``is_rain``. ``category`` / ``size_bucket`` / ``distance_range``:
    keep a frame if ANY of its GT rows (val-only, any model — GT boxes are not
    model-specific) satisfies the criterion. ``failure_type``: one of
    "has_fn"/"has_fp"/"has_low_conf"/"clean". ``bucket``: membership in the frame's
    ``curation_buckets`` list. Every criterion left as None is a no-op; with none
    set, the full val-split frame set (with flags attached) is returned.
    """
    flags = failure_flags(manifest, gt, preds, model=model)
    gt_val = gt.loc[gt["sample_data_token"].isin(set(flags["sample_data_token"]))]

    mask = pd.Series(True, index=flags.index)

    if lighting is not None:
        if lighting not in _LIGHTING:
            raise ValueError(f"filter_frames: unknown lighting {lighting!r} — expected 'night'/'day'")
        mask &= flags["is_night"] == _LIGHTING[lighting]

    if rain is not None:
        mask &= flags["is_rain"] == rain

    if category is not None:
        tokens = set(gt_val.loc[gt_val["category_group"] == category, "sample_data_token"])
        mask &= flags["sample_data_token"].isin(tokens)

    if size_bucket is not None:
        tokens = set(gt_val.loc[gt_val["size_bucket"] == size_bucket, "sample_data_token"])
        mask &= flags["sample_data_token"].isin(tokens)

    if distance_range is not None:
        lo, hi = distance_range
        in_range = gt_val.loc[gt_val["distance_to_ego_m"].between(lo, hi)]
        mask &= flags["sample_data_token"].isin(set(in_range["sample_data_token"]))

    if failure_type is not None:
        if failure_type not in _FAILURE_TYPES:
            raise ValueError(
                f"filter_frames: unknown failure_type {failure_type!r} — expected "
                f"one of {_FAILURE_TYPES}"
            )
        mask &= flags[failure_type].astype(bool)

    if bucket is not None:
        mask &= flags["curation_buckets"].apply(lambda buckets: bucket in buckets)

    return flags[mask].reset_index(drop=True)
