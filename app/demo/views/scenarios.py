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
from typing import Any, TypedDict

import pandas as pd
import streamlit as st
from filters import rank_events, severity_caption, speed_caption, subgraph_narrative
from PIL import Image
from render import draw_overlay, recorded_banner

from data import (
    crop_path,
    events_available,
    load_events,
    load_frame_manifest,
    load_gt_boxes,
    load_predictions,
    load_semsearch,
    load_subgraphs,
    subgraphs_available,
    thumb_path,
)

# configs/demo.yaml's presets.model_for_results -- which model's matched_<model> /
# predictions the two model-result presets (and this page's model panel) key off.
_MODEL_FOR_RESULTS = "baseline"

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
        "description": "A cyclist within 10m while driving above 10 m/s (~36 km/h).",
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

# Phase 6 (Task 3): the interactive subgraph panel -- on_path nodes get the accent
# color + the larger size, off-path (context) nodes are faded + smaller. Two colors
# total, not one per node type (spec docs/superpowers/specs/2026-08-20-demo-phase6-
# design.md §2: "on_path nodes in the accent color ... off-path nodes faded") --
# the six node TYPES are distinguished by their label text (in the node's own label
# and the legend caption below), not by six separate colors.
_NODE_ON_PATH_COLOR = "#FF851B"
_NODE_OFF_PATH_COLOR = "#BBBBBB"
_NODE_ON_PATH_SIZE = 22
_NODE_OFF_PATH_SIZE = 12
_GRAPH_LEGEND = "Node types: Scene · Sample · EgoPose · ObjectObservation · Category · Location"
_GRAPH_ABSENT_NOTE = "graph export not included in this package"
_MODEL_GT_ONLY_NOTE = "graph holds GT only — model verdict comes from the prediction set"

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

    Safe to call for a NON-curated token too. ``gt_boxes.parquet`` covers every
    curated frame_manifest token -- val AND train_pool -- while ``in_curated_set``
    is val-only, so a non-curated event usually yields zero rows, but a curated
    train_pool frame (14 of the 119 non-curated events in the shipped package)
    yields its GT with ``matched`` all-NA, which ``draw_overlay`` renders as plain
    GT boxes (never as misses). Predictions are val-only, so the "not in the
    curated prediction set" caption stays honest in both cases.
    """
    subset = gt.loc[gt["sample_data_token"] == token]
    subset = subset.loc[~subset["below_visibility_min"]]
    return subset.rename(columns={f"matched_{_MODEL_FOR_RESULTS}": "matched"})


def _render_parity_line(preset_name: str, preset: _Preset) -> None:
    """Phase 6 (Task 3): the preset header's SQL/Cypher parity line.

    Model-result presets always get the fixed GT-only note (their verdict is
    never graph-derivable, regardless of whether `demo subgraphs` ran). Dynamics
    presets read their own `graph_subgraphs/<preset>.json`'s `cypher_count`/
    `sql_count`/`parity` -- written honestly for all four by `demo subgraphs`
    (the flagship's mismatch would already have failed the build; the other
    three's mismatch, if any, is recorded plainly here as "mismatch recorded",
    never hidden). Replaces the old flagship-only badge that read overview_
    metrics.json's sourced-until-Phase-6 value: every dynamics preset now gets
    this line, not just the flagship, and the package having no subgraphs staged
    at all is a silent no-op here (nothing to report) rather than a stale claim --
    the viewer's graph panel below carries the honest absent note for that case.
    """
    if preset["family"] == "model":
        st.caption(_MODEL_GT_ONLY_NOTE)
        return
    if not subgraphs_available():
        return
    subgraph_payload = load_subgraphs(preset_name)
    if subgraph_payload is None:
        return
    cypher_count = subgraph_payload.get("cypher_count")
    sql_count = subgraph_payload.get("sql_count")
    symbol = "✓" if subgraph_payload.get("parity") else "✗ mismatch recorded"
    st.caption(f"Cypher: {cypher_count} · SQL: {sql_count} {symbol}")


def _short_id(node_id: str) -> str:
    """The id portion after its "<type>:" prefix, truncated for a readable agraph
    node label -- real graph tokens are long hex strings; Category/Location ids
    are names, usually already short."""
    token = node_id.split(":", 1)[1] if ":" in node_id else node_id
    return token if len(token) <= 10 else f"{token[:10]}..."


def _render_graph_panel(row: pd.Series, preset_name: str) -> None:
    """Phase 6 (Task 3): the interactive subgraph panel -- the Scenario page's
    long-standing Phase-6 slot, now implemented.

    Degrades honestly at three levels, each without raising: no `demo subgraphs`
    staged at all (`subgraphs_available()` False), this preset's own JSON missing
    (defensive -- `demo build` ships all six together or none), or this specific
    event absent from that preset's `events` dict (its subgraph export capped at
    <=30 events per preset, or the event simply isn't tagged with this preset).
    All three show the same honest note rather than a crash or a silent gap.

    `streamlit_agraph` is imported here (not at module level) so a package/
    environment missing it still renders every other page element -- an
    ImportError becomes an `st.warning` naming the package, not a fatal import
    error for the whole module.
    """
    st.markdown("**Interactive graph**")

    if not subgraphs_available():
        st.info(_GRAPH_ABSENT_NOTE)
        return
    subgraph_payload = load_subgraphs(preset_name)
    if subgraph_payload is None:
        st.info(_GRAPH_ABSENT_NOTE)
        return
    events: dict[str, Any] = subgraph_payload.get("events", {})
    event_subgraph: dict[str, Any] | None = events.get(str(row.sample_data_token))
    if event_subgraph is None:
        st.info(_GRAPH_ABSENT_NOTE)
        return

    try:
        from streamlit_agraph import Config, Edge, Node, agraph
    except ImportError:
        st.warning(
            "streamlit-agraph is not installed — the interactive graph panel is unavailable."
        )
        return

    nodes_data: list[dict[str, Any]] = event_subgraph.get("nodes", [])
    edges_data: list[dict[str, Any]] = event_subgraph.get("edges", [])
    nodes_by_id = {n["id"]: n for n in nodes_data}

    agraph_nodes = [
        Node(
            id=n["id"],
            label=f"{n['label']}: {_short_id(n['id'])}",
            size=_NODE_ON_PATH_SIZE if n.get("on_path") else _NODE_OFF_PATH_SIZE,
            color=_NODE_ON_PATH_COLOR if n.get("on_path") else _NODE_OFF_PATH_COLOR,
        )
        for n in nodes_data
    ]
    agraph_edges = [
        Edge(source=e["source"], target=e["target"], label=e["label"]) for e in edges_data
    ]
    config = Config(width=700, height=420, directed=True, physics=False, hierarchical=False)

    graph_col, detail_col = st.columns([2, 1])
    with graph_col:
        clicked = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    with detail_col:
        clicked_node = nodes_by_id.get(clicked) if clicked else None
        if clicked_node is not None:
            meta = clicked_node.get("meta", {})
            st.table(
                pd.DataFrame({"key": list(meta.keys()), "value": [str(v) for v in meta.values()]})
            )
        else:
            st.caption(subgraph_narrative(event_subgraph))

    st.caption(_GRAPH_LEGEND)


def _render_preset_buttons() -> None:
    columns = st.columns(3)
    for index, (name, preset) in enumerate(PRESETS.items()):
        with columns[index % 3]:
            if st.button(preset["label"], key=f"scenario_preset_{name}"):
                st.session_state["scenario_preset"] = name
            st.caption(preset["description"])


def _render_card_grid(ranked: pd.DataFrame, preset_name: str, preset: _Preset) -> None:
    if ranked.empty:
        st.info("No events match this preset in the current package.")
        return

    columns = st.columns(4)
    for position, row in enumerate(ranked.itertuples()):
        token = row.sample_data_token
        row_dict = row._asdict()
        with columns[position % 4]:
            image_path = _card_image_path(token)
            if image_path is not None:
                st.image(str(image_path))
            # The headline figure is THIS preset's own ranking quantity (item 6,
            # consolidated review) -- braking g for hard_braking_near_pedestrians,
            # speed for fast_cyclists, the missed pedestrian's own distance for
            # fn_pedestrians_night, etc. -- not a fixed field shown regardless of
            # which preset produced the card.
            caption_parts = [severity_caption(preset_name, row_dict)]
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
        preds = load_predictions()
        preds_token = preds.loc[
            (preds["sample_data_token"] == token) & (preds["model"] == _MODEL_FOR_RESULTS)
        ]
        crop = crop_path(token)
        if crop.is_file():
            image = draw_overlay(Image.open(crop), gt, preds_token, mode="overlay", scale=0.6)
            st.image(image)
            return
        thumb = thumb_path(token)
        if thumb.is_file():
            # Crop missing (a partial/corrupted package) -- fall back to the
            # thumb at its own scale rather than rendering nothing (item 5,
            # consolidated review): the same resilience the non-curated branch
            # below already has.
            image = draw_overlay(Image.open(thumb), gt, preds_token, mode="overlay", scale=0.16)
            st.image(image)
        else:
            st.caption("no image available for this curated frame")
        return

    thumb = thumb_path(token)
    if thumb.is_file():
        # GT boxes only (no predictions exist for a non-curated token). `gt` is
        # empty unless the token is a curated train_pool frame, in which case its
        # GT draws with matched all-NA (plain green, never orange) -- see
        # _gt_for_render's docstring. Routed through draw_overlay either way so
        # those frames' GT is shown rather than being silently
        # skipped by a hand-rolled bypass.
        image = draw_overlay(Image.open(thumb), gt, pd.DataFrame(), mode="gt", scale=0.16)
        st.image(image)
    st.caption(_NOT_CURATED_CAPTION)


def _render_ego_panel(row: pd.Series) -> None:
    st.markdown("**Ego dynamics**")
    columns = st.columns(3)
    columns[0].metric("Speed", speed_caption(row.speed_mps))
    accel = row.accel_long_min_mps2
    # "Peak decel" only makes sense for a NEGATIVE (decelerating) reading -- 27/126
    # real flagship-adjacent events have a positive accel_long_min_mps2 (the frame
    # was accelerating, not braking, at its most extreme longitudinal sample), and
    # labeling that a "decel" figure misrepresents the frame (item 1, consolidated
    # review).
    decel_label = "Peak decel" if pd.notna(accel) and accel < 0 else "Peak long. accel"
    columns[1].metric(decel_label, f"{accel:.2f} m/s²" if pd.notna(accel) else "n/a")
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
    # the strip entirely, per the design doc.
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

    # All five ship already (item 12, consolidated review): a strip of every
    # available step's thumbnail, the selected one marked via its own caption --
    # not just the single selected image the slider alone would show.
    strip_columns = st.columns(len(steps))
    for strip_col, (label, step_tok, _speed, _accel) in zip(strip_columns, steps, strict=True):
        with strip_col:
            step_thumb = thumb_path(step_tok)
            if step_thumb.is_file():
                st.image(str(step_thumb))
            marker = " (selected)" if label == selected_label else ""
            st.caption(f"{label}{marker}")

    _step_token, step_speed, step_accel = by_label[selected_label]
    speed_text = speed_caption(step_speed)
    accel_text = f"{step_accel:.2f} m/s²" if pd.notna(step_accel) else "n/a"
    st.caption(f"{selected_label}: {speed_text}, accel {accel_text}")


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
        target_k = (
            int(ranked_group["k"].iloc[0])
            if "k" in ranked_group.columns and pd.notna(ranked_group["k"].iloc[0])
            else len(ranked_group)
        )
        st.caption(f"{len(ranked_group)} of {target_k} front-camera hits")
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

    if not events_available():
        # scenario_events.parquet shipped starting package_version 0.4 (Phase 5) --
        # a stale/older demo_data/ package must not crash the page with a bare
        # traceback (item 4, consolidated review).
        st.error("needs demo_data >= 0.4 — rerun demo build")
        return

    events = load_events()
    st.session_state.setdefault("scenario_preset", _FLAGSHIP_PRESET)

    _render_preset_buttons()

    preset_name = str(st.session_state["scenario_preset"])
    preset = PRESETS[preset_name]
    ranked = rank_events(events, preset_name)

    st.divider()
    st.subheader(preset["label"])
    st.caption(f"{preset['scope_caption']} {preset['severity_caption']}")
    # Phase 6 (Task 3): every dynamics preset gets its own Cypher/SQL parity
    # line (read from graph_subgraphs/<preset>.json, written by `demo
    # subgraphs`); model presets get the fixed GT-only note. Supersedes the old
    # flagship-only badge that read overview_metrics.json's sourced-until-
    # Phase-6 value -- see _render_parity_line's docstring.
    _render_parity_line(preset_name, preset)

    _render_card_grid(ranked, preset_name, preset)

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
        # Phase-6 slot (item 11, consolidated review): implemented -- the
        # interactive subgraph panel (Task 3).
        _render_graph_panel(row, preset_name)

    st.divider()
    _render_semantic_gallery()
