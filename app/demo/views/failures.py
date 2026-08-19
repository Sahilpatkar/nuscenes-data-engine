"""Failure Explorer — GT vs prediction overlays, filterable by condition (design:
docs/superpowers/specs/2026-08-18-demo-phase3-design.md).

Val frames only: the train pool carries no predictions by design (nothing to
diagnose a "failure" against), so every table here is pre-filtered to
``split == "val"`` via ``filters.failure_flags``/``filter_frames``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from filters import filter_frames
from PIL import Image
from render import draw_overlay

from data import crop_path, load_frame_manifest, load_gt_boxes, load_predictions, thumb_path

_MODE_LABELS = {"Overlay": "overlay", "GT only": "gt", "Predictions only": "pred"}
_FAILURE_LABELS = {
    "All": None,
    "Has FN": "has_fn",
    "Has FP": "has_fp",
    "Has low-conf": "has_low_conf",
    "Clean": "clean",
}


def _models_from_gt(gt: pd.DataFrame) -> list[str]:
    return sorted(col.removeprefix("matched_") for col in gt.columns if col.startswith("matched_"))


def _models_from_manifest(manifest: pd.DataFrame) -> list[str]:
    return sorted(
        col.removeprefix("n_preds_") for col in manifest.columns if col.startswith("n_preds_")
    )


def _bool_options(series: pd.Series, *, true_label: str, false_label: str) -> list[str]:
    """Option list built from the values actually present, True before False."""
    present = sorted(series.dropna().unique().tolist(), reverse=True)
    labels = {True: true_label, False: false_label}
    return ["All"] + [labels[v] for v in present]


def _box_counts(gt: pd.DataFrame, preds: pd.DataFrame, token: str, model: str) -> tuple[int, int, int]:
    """(n_fn, n_fp, n_low_conf) for one frame+model — counts, not the boolean
    flags failure_flags exposes, for the grid caption."""
    gt_token = gt.loc[gt["sample_data_token"] == token]
    n_fn = int(gt_token[f"matched_{model}"].eq(False).sum())
    preds_token = preds.loc[(preds["sample_data_token"] == token) & (preds["model"] == model)]
    n_fp = int((preds_token["status"] == "fp").sum())
    n_low_conf = int((preds_token["status"] == "low_conf").sum())
    return n_fn, n_fp, n_low_conf


def _frame_image_path(token: str) -> Path | None:
    thumb = thumb_path(token)
    if thumb.is_file():
        return thumb
    crop = crop_path(token)
    return crop if crop.is_file() else None


def render() -> None:
    st.title("Failure Explorer")
    st.caption(
        "GT vs prediction overlays for the held-out val split, filterable by "
        "condition, class, and failure type. The train pool carries no "
        "predictions by design and is not shown here — see Active Learning and "
        "Weak Supervision for what it's used for."
    )

    manifest = load_frame_manifest()
    gt = load_gt_boxes()
    preds = load_predictions()

    # below_visibility_min rows are dropped entirely -- everywhere this page
    # touches gt_boxes, including the per-box table -- rather than rendered as
    # "ghost boxes" the model was never scored against. infer.py's docstring note
    # (2026-08-18 correction) is explicit that this column is parity-defensive:
    # today's ingested data is all-False (the visibility floor is already applied
    # at ingestion) and Phase 3's UI "should not build ghost-box rendering ...
    # without first confirming ghost rows occur in the data it is fed" -- so this
    # page drops them instead of building that unconfirmed rendering mode.
    gt = gt.loc[~gt["below_visibility_min"]].reset_index(drop=True)

    val_manifest = manifest.loc[manifest["split"] == "val"]
    gt_models = _models_from_gt(gt)
    if not gt_models:
        st.warning("No per-model matched columns in gt_boxes — nothing to explore yet.")
        return

    st.sidebar.header("Filters")
    default_index = gt_models.index("baseline") if "baseline" in gt_models else 0
    model = st.sidebar.radio("Model", gt_models, index=default_index)

    lighting_label = st.sidebar.selectbox(
        "Lighting", _bool_options(val_manifest["is_night"], true_label="Night", false_label="Day")
    )
    lighting = {"Night": "night", "Day": "day"}.get(lighting_label)

    rain_label = st.sidebar.selectbox(
        "Rain", _bool_options(val_manifest["is_rain"], true_label="Rain", false_label="No rain")
    )
    rain = {"Rain": True, "No rain": False}.get(rain_label)

    val_tokens = set(val_manifest["sample_data_token"])
    gt_val = gt.loc[gt["sample_data_token"].isin(val_tokens)]

    category_choice = st.sidebar.selectbox(
        "Category", ["All", *sorted(gt_val["category_group"].dropna().unique())]
    )
    category = None if category_choice == "All" else category_choice

    size_choice = st.sidebar.selectbox(
        "Size", ["All", *sorted(gt_val["size_bucket"].dropna().unique())]
    )
    size_bucket = None if size_choice == "All" else size_choice

    distances = gt_val["distance_to_ego_m"].dropna()
    min_dist = float(distances.min()) if not distances.empty else 0.0
    max_dist = float(distances.max()) if not distances.empty else 100.0
    if max_dist <= min_dist:
        max_dist = min_dist + 1.0
    distance_range = st.sidebar.slider(
        "Distance to ego (m)", min_value=min_dist, max_value=max_dist, value=(min_dist, max_dist)
    )

    failure_label = st.sidebar.radio("Failure type", list(_FAILURE_LABELS))
    failure_type = _FAILURE_LABELS[failure_label]

    all_buckets = sorted({b for buckets in val_manifest["curation_buckets"] for b in buckets})
    bucket_choice = st.sidebar.multiselect("Curation bucket", all_buckets)

    frames = filter_frames(
        manifest=manifest,
        gt=gt,
        preds=preds,
        model=model,
        lighting=lighting,
        rain=rain,
        category=category,
        size_bucket=size_bucket,
        distance_range=distance_range,
        failure_type=failure_type,
    )
    if bucket_choice:
        frames = frames.loc[
            frames["curation_buckets"].apply(lambda bs: any(b in bs for b in bucket_choice))
        ].reset_index(drop=True)

    st.caption(f"{len(frames)} / {len(val_manifest)} val frames match the current filters")

    columns = st.columns(4)
    for position, row in enumerate(frames.itertuples()):
        token = row.sample_data_token
        image_path = _frame_image_path(token)
        n_fn, n_fp, n_low_conf = _box_counts(gt, preds, token, model)
        with columns[position % 4]:
            if image_path is not None:
                st.image(str(image_path))
            st.caption(f"{row.scene_name} · {n_fn}fn/{n_fp}fp/{n_low_conf}lowconf")
            if st.button("View", key=f"failure_select_{token}"):
                st.session_state["failure_token"] = token

    selected_token = st.session_state.get("failure_token")
    if not selected_token or selected_token not in set(frames["sample_data_token"]):
        return

    st.divider()
    st.subheader("Detail")
    frame_row = val_manifest.loc[val_manifest["sample_data_token"] == selected_token].iloc[0]
    gt_token = gt.loc[gt["sample_data_token"] == selected_token]
    preds_token = preds.loc[
        (preds["sample_data_token"] == selected_token) & (preds["model"] == model)
    ]

    crop = crop_path(selected_token)
    if crop.is_file():
        mode_label = st.radio("View", list(_MODE_LABELS), horizontal=True)
        gt_for_render = gt_token.rename(columns={f"matched_{model}": "matched"})
        image = draw_overlay(
            Image.open(crop), gt_for_render, preds_token, mode=_MODE_LABELS[mode_label], scale=0.6
        )
        st.image(image)

    meta_cols = st.columns(4)
    meta_cols[0].metric("Scene", str(frame_row.get("scene_name", "n/a")))
    meta_cols[1].metric("Night", "Yes" if frame_row.get("is_night") else "No")
    meta_cols[2].metric("Rain", "Yes" if frame_row.get("is_rain") else "No")
    # curation_buckets round-trips through parquet as a numpy array (pyarrow's
    # list dtype), not a plain Python list -- `array or []` would raise
    # ("truth value of an array with more than one element is ambiguous") for any
    # frame in more than one bucket, so check emptiness by length, not truthiness.
    buckets = frame_row.get("curation_buckets")
    buckets_list = list(buckets) if buckets is not None else []
    meta_cols[3].metric("Buckets", ", ".join(buckets_list) if buckets_list else "none")

    manifest_models = _models_from_manifest(manifest)
    if manifest_models:
        pred_cols = st.columns(len(manifest_models))
        for col, other_model in zip(pred_cols, manifest_models, strict=True):
            col.metric(f"n_preds ({other_model})", str(frame_row.get(f"n_preds_{other_model}")))

    # NA-safe status label, mirroring render.py's draw_overlay: NA in
    # matched_<model> means "not evaluated" and must read as neither a match nor
    # a miss.
    matched_series = gt_token[f"matched_{model}"]
    gt_status = matched_series.map({True: "matched", False: "fn"}).fillna("not_evaluated")
    gt_table = gt_token.assign(
        kind="gt", conf=pd.NA, status=gt_status
    )[["kind", "category_group", "conf", "status", "distance_to_ego_m", "size_bucket"]]
    pred_table = preds_token.assign(kind="pred", distance_to_ego_m=pd.NA, size_bucket=pd.NA)[
        ["kind", "category_group", "conf", "status", "distance_to_ego_m", "size_bucket"]
    ]
    st.dataframe(pd.concat([gt_table, pred_table], ignore_index=True))

    exemplar_cols = [c for c in manifest.columns if c.startswith(f"fixes_fn_vs_{model}_")]
    is_exemplar = any(bool(frame_row.get(c)) for c in exemplar_cols if pd.notna(frame_row.get(c)))
    if is_exemplar:
        st.success(
            f"Exemplar: `{model}` fixes a false negative another model missed on this frame."
        )

    st.caption(
        "FN = GT unmatched by the selected model; low-confidence claims count as "
        "matches, mirroring the AL pipeline."
    )
