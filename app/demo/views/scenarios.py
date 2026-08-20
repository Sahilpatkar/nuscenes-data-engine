"""Scenario Search — preset driving-scenario queries over ``scenario_events.parquet``,
a synchronized event viewer (camera + ego dynamics + scene context + model context)
with a t-2..t+2 filmstrip, and the recorded semantic gallery (design:
docs/superpowers/specs/2026-08-20-demo-phase5-design.md).

Two preset families, honestly separated — mirrors ``demo/events.py``'s own split:
*dynamics* presets are dataset-wide, GT-only queries over every CAM_FRONT keyframe;
*model-result* presets only cover the 125 curated val frames that carry recorded
predictions. Every preset's scope is stated on screen so neither family reads as
more complete than it actually is.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd
import streamlit as st
from filters import rank_events
from PIL import Image
from render import draw_overlay, recorded_banner

from data import (
    crop_path,
    load_events,
    load_frame_manifest,
    load_gt_boxes,
    load_predictions,
    load_semsearch,
    thumb_path,
)

# configs/demo.yaml's presets.model_for_results -- which model's matched_<model> /
# predictions the two model-result presets (and this page's model panel) key off.
_MODEL_FOR_RESULTS = "baseline"

_STANDARD_GRAVITY_MPS2 = 9.80665  # braking-g captions divide accel_long_min_mps2 by this

_NOT_CURATED_CAPTION = "not in the curated prediction set"

_DYNAMICS_SCOPE = "Dataset-wide — all 34,149 CAM_FRONT keyframes."
_MODEL_SCOPE = "Curated val split only — 125 frames with recorded predictions."


class _Preset(TypedDict):
    label: str
    description: str       # one-liner shown under the preset's button
    family: str             # "dynamics" | "model"
    scope_caption: str
    severity_caption: str
    verdict: str | None     # model-family only: what a card in this preset means


PRESETS: dict[str, _Preset] = {
    "hard_braking_near_pedestrians": {
        "label": "Hard braking near pedestrians",
        "description": "Hard braking with a pedestrian within 10m — the flagship graph/CAN query.",
        "family": "dynamics",
        "scope_caption": _DYNAMICS_SCOPE,
        "severity_caption": "Ranked by strongest braking first.",
        "verdict": None,
    },
    "night_pedestrians": {
        "label": "Night scenes, nearby pedestrians",
        "description": "Night driving with a pedestrian within 10m.",
        "family": "dynamics",
        "scope_caption": _DYNAMICS_SCOPE,
        "severity_caption": "Ranked by nearest pedestrian first.",
        "verdict": None,
    },
    "fast_cyclists": {
        "label": "Nearby cyclists at speed",
        "description": "A cyclist within 10m while driving at highway speed.",
        "family": "dynamics",
        "scope_caption": _DYNAMICS_SCOPE,
        "severity_caption": "Ranked by fastest ego speed first.",
        "verdict": None,
    },
    "rain_vru": {
        "label": "Rain, nearby vulnerable road users",
        "description": "Rain with a pedestrian or cyclist within 10m.",
        "family": "dynamics",
        "scope_caption": _DYNAMICS_SCOPE,
        "severity_caption": "Ranked by nearest VRU (pedestrian or cyclist) first.",
        "verdict": None,
    },
    "fn_pedestrians_night": {
        "label": "False-negative pedestrians at night",
        "description": "Pedestrians the model missed, at night.",
        "family": "model",
        "scope_caption": _MODEL_SCOPE,
        "severity_caption": "Ranked by nearest missed pedestrian first.",
        "verdict": f"{_MODEL_FOR_RESULTS}: missed this pedestrian (false negative)",
    },
    "low_conf_braking": {
        "label": "Low-confidence detections while braking",
        "description": "Low-confidence detections during hard braking.",
        "family": "model",
        "scope_caption": _MODEL_SCOPE,
        "severity_caption": "Ranked by lowest confidence first.",
        "verdict": f"{_MODEL_FOR_RESULTS}: low-confidence detection during braking",
    },
}

_FLAGSHIP_PRESET = "hard_braking_near_pedestrians"

# Filmstrip step order, before/after the current frame -- plain ASCII hyphens
# (ruff RUF001 flags the design doc's typographic U+2212 minus sign as an
# ambiguous character), matching the t_minus/t_plus column-name convention.
_BEFORE_STEPS = (("t_minus2", "t-2"), ("t_minus1", "t-1"))
_AFTER_STEPS = (("t_plus1", "t+1"), ("t_plus2", "t+2"))
_CURRENT_STEP = "current"


def _card_image_path(token: str) -> Path | None:
    """Thumb first (small, cheap), crop as a fallback -- same idiom as the Failure
    Explorer's ``_frame_image_path``."""
    thumb = thumb_path(token)
    if thumb.is_file():
        return thumb
    crop = crop_path(token)
    return crop if crop.is_file() else None


def _gt_for_render(gt: pd.DataFrame, token: str) -> pd.DataFrame:
    """``gt_boxes`` rows for ``token``, visibility-floor rows dropped, ``matched_
    <model>`` renamed to ``matched`` for ``draw_overlay``.

    Safe to call for a NON-curated token too: ``gt_boxes.parquet`` only ever
    contains rows for curated frame_manifest tokens (``build.py::_include_
    curation``), so a non-curated token always yields zero rows here -- an empty
    frame ``draw_overlay`` renders as "no GT boxes", exactly the honest picture for
    a token with no curated ground truth to show.
    """
    subset = gt.loc[gt["sample_data_token"] == token]
    subset = subset.loc[~subset["below_visibility_min"]]
    return subset.rename(columns={f"matched_{_MODEL_FOR_RESULTS}": "matched"})


def _render_preset_buttons() -> None:
    columns = st.columns(3)
    for index, (name, preset) in enumerate(PRESETS.items()):
        with columns[index % 3]:
            if st.button(preset["label"], key=f"scenario_preset_{name}"):
                st.session_state["scenario_preset"] = name
            st.caption(preset["description"])


def _format_braking_g(accel_mps2: float) -> str:
    if pd.isna(accel_mps2):
        return "no braking data"
    g_force = abs(float(accel_mps2)) / _STANDARD_GRAVITY_MPS2
    return f"{g_force:.2f}g braking"


def _format_distance(meters: float, *, label: str) -> str:
    return f"{meters:.1f}m {label}" if pd.notna(meters) else f"no {label}"


def _render_card_grid(ranked: pd.DataFrame, preset: _Preset) -> None:
    if ranked.empty:
        st.info("No events match this preset in the current package.")
        return

    columns = st.columns(4)
    for position, row in enumerate(ranked.itertuples()):
        token = row.sample_data_token
        with columns[position % 4]:
            image_path = _card_image_path(token)
            if image_path is not None:
                st.image(str(image_path))
            caption_parts = [
                _format_braking_g(row.accel_long_min_mps2),
                _format_distance(row.min_dist_pedestrian_m, label="ped"),
                f"{row.speed_mps:.1f} m/s" if pd.notna(row.speed_mps) else "no speed",
            ]
            if row.is_night:
                caption_parts.append("night")
            if row.is_rain:
                caption_parts.append("rain")
            st.caption(" · ".join(caption_parts))
            if preset["family"] == "model" and preset["verdict"]:
                st.caption(preset["verdict"])
            if st.button("View", key=f"scenario_select_{token}"):
                st.session_state["scenario_token"] = token


def _render_viewer(row: pd.Series) -> None:
    token = row.sample_data_token
    gt = _gt_for_render(load_gt_boxes(), token)

    if row.in_curated_set:
        crop = crop_path(token)
        if crop.is_file():
            preds = load_predictions()
            preds_token = preds.loc[
                (preds["sample_data_token"] == token) & (preds["model"] == _MODEL_FOR_RESULTS)
            ]
            image = draw_overlay(Image.open(crop), gt, preds_token, mode="overlay", scale=0.6)
            st.image(image)
        return

    thumb = thumb_path(token)
    if thumb.is_file():
        # GT boxes only (no predictions exist for a non-curated token): `gt` is
        # always empty here (see _gt_for_render's docstring), so this renders as
        # the plain thumb -- still routed through draw_overlay, not a special
        # case, so a future token that DOES somehow have curated-scoped GT rows
        # (there is none today) draws correctly rather than being silently
        # skipped by a hand-rolled bypass.
        image = draw_overlay(Image.open(thumb), gt, pd.DataFrame(), mode="gt", scale=0.16)
        st.image(image)
    st.caption(_NOT_CURATED_CAPTION)


def _render_ego_panel(row: pd.Series) -> None:
    st.markdown("**Ego dynamics**")
    columns = st.columns(3)
    columns[0].metric("Speed", f"{row.speed_mps:.1f} m/s" if pd.notna(row.speed_mps) else "n/a")
    columns[1].metric(
        "Peak decel",
        f"{row.accel_long_min_mps2:.2f} m/s²" if pd.notna(row.accel_long_min_mps2) else "n/a",
    )
    columns[2].metric("Hard braking", "Yes" if row.is_hard_braking else "No")


def _render_context_panel(row: pd.Series) -> None:
    st.markdown("**Scene context**")
    dist_cols = st.columns(3)
    dist_cols[0].metric(
        "Min ped dist",
        f"{row.min_dist_pedestrian_m:.1f} m" if pd.notna(row.min_dist_pedestrian_m) else "n/a",
    )
    dist_cols[1].metric(
        "Min vehicle dist",
        f"{row.min_dist_vehicle_m:.1f} m" if pd.notna(row.min_dist_vehicle_m) else "n/a",
    )
    dist_cols[2].metric(
        "Min cyclist dist",
        f"{row.min_dist_cyclist_m:.1f} m" if pd.notna(row.min_dist_cyclist_m) else "n/a",
    )

    other_cols = st.columns(3)
    other_cols[0].metric(
        "Peds within 10m",
        str(int(row.n_peds_within_10m)) if pd.notna(row.n_peds_within_10m) else "0",
    )
    other_cols[1].metric("Night", "Yes" if row.is_night else "No")
    other_cols[2].metric("Rain", "Yes" if row.is_rain else "No")


def _render_model_panel(row: pd.Series, preset: _Preset) -> None:
    st.markdown("**Model context**")
    if not row.in_curated_set:
        st.caption(_NOT_CURATED_CAPTION)
        return

    manifest = load_frame_manifest()
    frame_rows = manifest.loc[manifest["sample_data_token"] == row.sample_data_token]
    model_cols = [c for c in manifest.columns if c.startswith("n_preds_")]
    if frame_rows.empty or not model_cols:
        st.caption(_NOT_CURATED_CAPTION)
        return

    frame_row = frame_rows.iloc[0]
    columns = st.columns(len(model_cols))
    for column, model_col in zip(columns, model_cols, strict=True):
        model_name = model_col.removeprefix("n_preds_")
        column.metric(f"n_preds ({model_name})", str(frame_row.get(model_col)))

    if preset["family"] == "model" and preset["verdict"]:
        st.caption(preset["verdict"])


def _render_filmstrip(row: pd.Series) -> None:
    st.markdown("**Filmstrip**")

    # (label, token, speed, accel) for every non-NA t-2..t+2 neighbor, plus the
    # current frame in the middle -- NA neighbors (scene edges) are omitted from
    # the slider entirely, per the design doc.
    steps: list[tuple[str, str, float, float]] = []
    for column, label in _BEFORE_STEPS:
        token = getattr(row, column)
        if pd.notna(token):
            steps.append((label, token, getattr(row, f"speed_{column}"), getattr(row, f"accel_{column}")))
    steps.append((_CURRENT_STEP, row.sample_data_token, row.speed_mps, row.accel_long_min_mps2))
    for column, label in _AFTER_STEPS:
        token = getattr(row, column)
        if pd.notna(token):
            steps.append((label, token, getattr(row, f"speed_{column}"), getattr(row, f"accel_{column}")))

    by_label = {label: (token, speed, accel) for label, token, speed, accel in steps}
    selected_label = st.select_slider(
        "Step", options=[label for label, *_ in steps], value=_CURRENT_STEP, key="scenario_filmstrip"
    )
    step_token, step_speed, step_accel = by_label[selected_label]

    thumb = thumb_path(step_token)
    if thumb.is_file():
        st.image(str(thumb))
    speed_text = f"{step_speed:.1f} m/s" if pd.notna(step_speed) else "n/a"
    accel_text = f"{step_accel:.2f} m/s²" if pd.notna(step_accel) else "n/a"
    st.caption(f"{selected_label}: speed {speed_text}, accel {accel_text}")


def _render_semantic_gallery() -> None:
    st.subheader("Recorded semantic search")
    recorded_banner("Recorded semantic search results — live search runs in the local stack")

    semsearch = load_semsearch()
    if semsearch.empty:
        st.info("No recorded semantic search results in this package.")
        return

    for query, group in semsearch.groupby("query", sort=False):
        st.markdown(f"**{query}**")
        ranked_group = group.sort_values("rank")
        columns = st.columns(len(ranked_group))
        for column, hit in zip(columns, ranked_group.itertuples(), strict=True):
            with column:
                thumb = thumb_path(hit.sample_data_token)
                if thumb.is_file():
                    st.image(str(thumb))
                st.caption(f"score {hit.score:.2f}")


def render() -> None:
    st.title("Scenario Search")
    st.caption(
        "Preset driving-scenario queries over the full dataset's CAN-bus + GT "
        "dynamics, plus two curated-val model-result presets. Pick a preset, then "
        "an event, to open the synchronized viewer below."
    )

    events = load_events()
    st.session_state.setdefault("scenario_preset", _FLAGSHIP_PRESET)

    _render_preset_buttons()

    preset_name = str(st.session_state["scenario_preset"])
    preset = PRESETS[preset_name]
    ranked = rank_events(events, preset_name)

    st.divider()
    st.subheader(preset["label"])
    st.caption(f"{preset['scope_caption']} {preset['severity_caption']}")

    if preset_name == _FLAGSHIP_PRESET:
        st.success(
            f"{len(ranked)} events — identical count in DuckDB SQL and Neo4j "
            "Cypher (Cypher sourced from GRAPH.md until Phase 6)"
        )

    _render_card_grid(ranked, preset)

    selected_token = st.session_state.get("scenario_token")
    if selected_token and selected_token in set(ranked["sample_data_token"]):
        row = ranked.loc[ranked["sample_data_token"] == selected_token].iloc[0]
        st.divider()
        st.subheader("Event viewer")
        _render_viewer(row)
        _render_ego_panel(row)
        _render_context_panel(row)
        _render_model_panel(row, preset)
        _render_filmstrip(row)

    st.divider()
    _render_semantic_gallery()
