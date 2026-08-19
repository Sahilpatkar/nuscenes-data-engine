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
from filters import failure_counts, filter_frames, sort_frames
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
_SORT_LABELS = {
    "Failure count": "failure_count",
    "Distance (nearest GT)": "distance",
    "Scene name": "scene_name",
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
    # Explicit key: tests drive the model choice directly via
    # at.radio(key="failure_model").set_value(...) rather than relying on
    # streamlit's identity-derived default key, which isn't guaranteed stable
    # across reruns for a plain positional call like this.
    model = st.sidebar.radio("Model", gt_models, index=default_index, key="failure_model")

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
        "Distance to ego (m)",
        min_value=min_dist,
        max_value=max_dist,
        value=(min_dist, max_dist),
        help=(
            "Narrowing this excludes frames with zero GT boxes entirely (there's "
            "nothing that can be 'in range') -- e.g. an all-hallucination frame "
            "during hard braking. Leave at the full extent to include those."
        ),
    )
    # filters.py's contract: distance_range=None is a no-op (every frame passes);
    # a concrete range only keeps a frame if >= 1 of its GT rows falls inside it,
    # so a frame with zero GT rows can never be "in range" under ANY concrete
    # range. The slider always returns a concrete tuple, so passing it straight
    # through would make the default (full-extent) state silently exclude every
    # zero-GT frame -- exactly the all-hallucination frames this page exists to
    # surface. Only pass a real range once the user has actually narrowed it;
    # narrowing is then an intentional opt-in to GT-based filtering, which
    # legitimately excludes zero-GT frames (nothing there to match).
    distance_arg = None if distance_range == (min_dist, max_dist) else distance_range

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
        distance_range=distance_arg,
        failure_type=failure_type,
        # Empty multiselect -> None: filter_frames's own no-op contract (buckets
        # not None but empty would still be a no-op via `if buckets:`, but None
        # says so explicitly at the call site).
        buckets=bucket_choice or None,
    )

    st.caption(f"{len(frames)} / {len(val_manifest)} val frames match the current filters")

    sort_label = st.sidebar.selectbox("Sort by", list(_SORT_LABELS))
    frames = sort_frames(frames, gt=gt, preds=preds, model=model, key=_SORT_LABELS[sort_label])

    # One failure_counts call over every filtered token, not per-row inside the
    # grid loop: the same counts also back the "failure_count" sort key above, so
    # computing them once keeps the displayed n_fn/n_fp/n_low_conf and the sort
    # order from ever silently disagreeing with each other.
    counts = failure_counts(gt, preds, frames["sample_data_token"], model=model).set_index(
        "sample_data_token"
    )

    # No pagination: the curated set tops out at 125 val frames today, which
    # renders comfortably in one scroll. If that grows substantially, page
    # `frames` here (e.g. st.session_state-backed offset + a fixed page size)
    # before this loop rather than rendering the whole filtered set.
    columns = st.columns(4)
    for position, row in enumerate(frames.itertuples()):
        token = row.sample_data_token
        image_path = _frame_image_path(token)
        n_fn, n_fp, n_low_conf = (
            int(counts.loc[token, "n_fn"]),
            int(counts.loc[token, "n_fp"]),
            int(counts.loc[token, "n_low_conf"]),
        )
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
    frame_buckets = frame_row.get("curation_buckets")
    frame_buckets_list = list(frame_buckets) if frame_buckets is not None else []
    meta_cols[3].metric("Buckets", ", ".join(frame_buckets_list) if frame_buckets_list else "none")

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
    # Explicit dtypes on the NA-filled columns (not a bare `pd.NA`/object-dtype
    # broadcast): pandas emits a FutureWarning ("concatenation with empty or
    # all-NA entries is deprecated") when concat has to guess the dtype of a
    # column that's entirely NA in one frame — giving it the same dtype the
    # OTHER frame's real values already have (float64 for conf/distance, object
    # for size_bucket) sidesteps the guess entirely.
    gt_table = gt_token.assign(
        kind="gt",
        conf=pd.Series([float("nan")] * len(gt_token), dtype="float64", index=gt_token.index),
        status=gt_status,
    )[["kind", "category_group", "conf", "status", "distance_to_ego_m", "size_bucket"]]
    pred_table = preds_token.assign(
        kind="pred",
        distance_to_ego_m=pd.Series(
            [float("nan")] * len(preds_token), dtype="float64", index=preds_token.index
        ),
        size_bucket=pd.Series([None] * len(preds_token), dtype="object", index=preds_token.index),
    )[["kind", "category_group", "conf", "status", "distance_to_ego_m", "size_bucket"]]
    st.dataframe(pd.concat([gt_table, pred_table], ignore_index=True))

    # fixes_fn_vs_<A>_<B> means "A missed this GT box, B caught it" (infer.py's
    # fixes-loop: fn_a = ~gt_matched_by_model[A]; fixed = any(fn_a &
    # gt_matched_by_model[B])). The selected model is credited as the FIXER (B),
    # so the column to look up per candidate "other" model is
    # fixes_fn_vs_{other}_{model} -- NOT fixes_fn_vs_{model}_{other}, which would
    # instead ask "did the selected model miss something `other` caught" and
    # credit the model that MISSED. The model list is taken from gt_models (the
    # matched_<model> columns already discovered above), never parsed back out
    # of a fixes_fn_vs_ column name: model names themselves contain underscores
    # (e.g. "graph_rate_night"), so a fixes_fn_vs_baseline_graph_rate_night
    # column name cannot be split unambiguously into its two model names.
    fixers_of = []
    for other in gt_models:
        if other == model:
            continue
        fixed = frame_row.get(f"fixes_fn_vs_{other}_{model}")
        if pd.notna(fixed) and bool(fixed):
            fixers_of.append(other)
    for other in fixers_of:
        st.success(f"Exemplar: `{model}` catches a box `{other}` misses on this frame")

    st.caption(
        "FN = GT unmatched by the selected model; low-confidence claims count as "
        "matches, mirroring the AL pipeline."
    )
