"""Scenario Search — preset driving-scenario queries over ``scenario_events.parquet``,
a one-screen event viewer (the frame beside its CAN speed/acceleration curves, one
row of dynamics/context/model cards, a t-2..t+2 filmstrip whose step slider is the
curves' cursor), and the recorded semantic gallery (designs:
docs/superpowers/specs/2026-08-20-demo-phase5-design.md, and 2026-08-22-demo-phase9b-
design.md §2 for the one-screen relayout).

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
from filters import (
    FILMSTRIP_STEPS,
    FilmstripCurve,
    FilmstripStep,
    filmstrip_steps,
    graph_node_label,
    parity_caption,
    parity_short,
    path_steps,
    rank_events,
    severity_caption,
    speed_caption,
    subgraph_narrative,
)
from PIL import Image
from render import (
    curve_caption,
    curve_charts,
    draw_overlay,
    loop_breadcrumb,
    metric_cards,
    provenance,
    recorded_banner,
)

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
# the six node TYPES are distinguished by their label text (filters.graph_node_label
# writes a semantic one per type) and by the legend caption below, not by six
# separate colors.
_NODE_ON_PATH_COLOR = "#FF851B"
_NODE_OFF_PATH_COLOR = "#BBBBBB"
_NODE_ON_PATH_SIZE = 22
_NODE_OFF_PATH_SIZE = 12
_EDGE_OFF_PATH_COLOR = "#DDDDDD"
_EDGE_LABEL_FONT = {"size": 10, "color": "#555555", "align": "middle"}

# Phase 9b (Task 5): the progressive reveal's two on-path states. "Amber" IS the
# Phase-6 path color -- a step at or before the slider's position keeps it, so the
# legend's "Orange = the matched path" stays literally true and the default (last
# step) draws exactly what this panel always drew. A path node whose step hasn't
# been reached yet goes HOLLOW instead of disappearing: white fill, grey rim, grey
# label. It has to stay in the node set or vis.js re-solves the force layout at
# every step and the graph jumps around under the viewer.
_AMBER = _NODE_ON_PATH_COLOR
_NODE_HOLLOW_FILL = "#FFFFFF"
_NODE_HOLLOW_BORDER = _NODE_OFF_PATH_COLOR
_NODE_HOLLOW_BORDER_WIDTH = 2
_NODE_HOLLOW_LABEL_COLOR = "#777777"

# The reveal's two widget keys -- read through their own return values (both
# widgets are instantiated ABOVE the graph), never written.
_PATH_STEP_KEY = "scenario_path_step"
_GRAPH_CONTEXT_KEY = "scenario_graph_context"

# Only these relationship types get their label drawn. A Sample hub carries up to 14
# HAS_OBJECT edges, each to an OF_CATEGORY edge of its own, so labeling those turned
# the star around the hub into an unreadable pile of repeated words (visual check,
# 2026-08-20) -- the object edges' meaning is already carried by their endpoints'
# own labels ("adult 9.9 m" -> "human.pedestrian.adult").
_STRUCTURAL_EDGES = frozenset({"AT_POSE", "IN_SCENE", "IN_LOCATION", "NEXT"})

_GRAPH_LEGEND = (
    "Orange = the matched path (Scene → Sample → EgoPose → matching objects → "
    "Category); grey = context (other observations, nearest 12 kept). Edge labels "
    "are shown for structural edges only. Click a node for its properties. "
    "Node types: Scene · Sample · EgoPose · ObjectObservation · Category · Location"
)
# Phase 9b (Task 5): one more line under the (unchanged) legend, for the two states
# only the reveal can produce.
#
# The second sentence states the CONSTANT NODE SET, which is this panel's own
# doing, and stops there (item I4, Phase 9b consolidated review): it used to promise
# "so the layout does not jump as you reveal it", a claim about how the browser's
# force layout behaves that nothing here verifies -- agraph re-solves the physics on
# every render.
_GRAPH_REVEAL_LEGEND = (
    "Amber = revealed up to the slider's step; hollow = on the matched path but not "
    "revealed yet. Every path node stays drawn at every step — only its colour "
    "changes."
)
# The detail column's own caption under the step facts: the export has no CAN node
# of its own, so this says where the two CAN readings actually live rather than
# letting the panel imply a node that isn't there.
_STEP_FACTS_CAPTION = (
    "CAN speed and minimum longitudinal acceleration are properties of the EgoPose "
    "node — the graph export has no separate CAN node."
)
_GRAPH_ABSENT_NOTE = "graph export not included in this package"
# Distinct from _GRAPH_ABSENT_NOTE (item M3, Phase 6 review): the package DOES carry
# a graph export, this one event just isn't in it. Unreachable for a package built
# since the stale-staging guard (build.py::_include_subgraphs), but the page should
# still say what it actually found.
_GRAPH_EVENT_ABSENT_NOTE = "no subgraph staged for this event — re-run demo subgraphs"
_MODEL_GT_ONLY_NOTE = "graph holds GT only — model verdict comes from the prediction set"

# Phase 9a (Task 6): the event viewer's own overlay provenance -- every branch of
# _render_viewer that actually draws an image (curated w/ crop, curated w/
# thumb-fallback, non-curated) says the same thing about where it came from.
_VIEWER_PROVENANCE_DETAIL = (
    "overlay from gt_boxes.parquet / predictions.parquet; dynamics from "
    "scenario_events.parquet"
)

# Filmstrip step order, before/after the current frame -- this page's two-sided
# shape of filters.FILMSTRIP_STEPS (Phase 9b: one definition, shared with the tour
# and with filters.filmstrip_steps; the copy-contract test compares all three).
# The `is not None` guard is what tells the type checker that the two neighbour
# halves are (column, label) pairs with a REAL column -- only the middle entry,
# the event's own frame, carries None, and it is _CURRENT_STEP below.
_BEFORE_STEPS: tuple[tuple[str, str], ...] = tuple(
    (column, label) for column, label in FILMSTRIP_STEPS[:2] if column is not None
)
_AFTER_STEPS: tuple[tuple[str, str], ...] = tuple(
    (column, label) for column, label in FILMSTRIP_STEPS[3:] if column is not None
)
_CURRENT_STEP = FILMSTRIP_STEPS[2][1]

# label -> the neighbour's own column prefix, for the readings the filmstrip readout
# still shows in the EVENT TABLE's units (m/s, from ego_pose) rather than the
# curve's km/h -- filters.FilmstripStep carries the km/h reading and the token, not
# which column the step came from.
_STEP_COLUMNS: dict[str, str] = {
    label: column for column, label in (*_BEFORE_STEPS, *_AFTER_STEPS)
}

# The filmstrip slider's key: read (never written) before the widget exists, so the
# curves' rule follows the step the viewer last picked -- see _selected_step.
_FILMSTRIP_KEY = "scenario_filmstrip"

# Phase 9b (Task 4): the event header's provenance -- the dynamics figures on this
# screen (the metric row, both curves, the filmstrip readout) all come from the one
# table `demo build` computed from the CAN bus and the GT annotations.
_EVENT_PROVENANCE_DETAIL = "CAN-bus and GT dynamics computed at build (scenario_events.parquet)"


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


def _render_parity_line(preset_name: str, preset: _Preset, n_shown: int) -> None:
    """Phase 6 (Task 3): the preset header's SQL/Cypher parity line.

    Model-result presets always get the fixed GT-only note (their verdict is
    never graph-derivable, regardless of whether `demo subgraphs` ran). Dynamics
    presets read their own `graph_subgraphs/<preset>.json`'s `cypher_count`/
    `sql_count`/`parity` -- written honestly for all four by `demo subgraphs`
    (the flagship's mismatch would already have failed the build; the other
    three's mismatch, if any, is recorded plainly here, never hidden). Replaces
    the old flagship-only badge that read overview_metrics.json's sourced-until-
    Phase-6 value: every dynamics preset now gets this line, not just the
    flagship, and the package having no subgraphs staged at all is a silent no-op
    here (nothing to report) rather than a stale claim -- the viewer's graph panel
    below carries the honest absent note for that case.

    The line says WHAT the counts count and how many of them the grid below
    actually shows (item I2, review): a bare "Cypher: 786 · SQL: 786 ✓" sat above
    a 30-card grid with nothing on screen reconciling 786 with 30 -- though the
    "showing the top N" clause only appears when the grid actually cut something
    (`n_shown < sql_count`; item 6, Phase 6 follow-up review), so a preset whose
    full population fits on screen doesn't claim a cap that never bit. And a
    recorded MISMATCH goes through st.warning rather than the same grey caption a
    clean parity gets (item M4) -- it is a finding, not small print. `parity is
    None` (model presets only, which never reach here) draws no symbol at all
    rather than being called a mismatch by default. The exact wording (singular/
    plural "keyframe", the mismatch text, the suppressed clause) lives in the
    pure `filters.parity_caption`, unit-tested there without a Streamlit runtime;
    this function only decides st.warning vs st.caption off `parity is False`.
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
    # sql_count is always written by `demo subgraphs` for a dynamics preset (unlike
    # cypher_count, which is legitimately null only for a model preset -- never
    # reached here). The 0 default is a typed fallback for mypy, not a real case.
    sql_count = subgraph_payload.get("sql_count", 0)
    parity = subgraph_payload.get("parity")
    caption = parity_caption(sql_count, cypher_count, parity, n_shown)
    if parity is False:
        st.warning(caption)
    else:
        st.caption(caption)
    # Phase 9a (Task 6): the provenance for the counts the line above just showed --
    # only reached once a real parity line (clean or a recorded mismatch) actually
    # drew, never for the model-family/absent-package/absent-payload no-ops above.
    provenance("recorded", "SQL and Cypher counts computed at build (demo subgraphs)")


def _node_font(on_path: bool, *, revealed: bool = True) -> dict[str, Any]:
    """vis.js font spec for one node's label -- readable at the size the panel draws.

    On-path labels are bigger, near-black, and sit on a translucent white plate so
    they stay legible where the force-directed layout overlaps them on an edge; the
    context nodes' labels are small and grey so the matched path reads first. A path
    node the reveal hasn't reached yet (Task 5) keeps the size and the plate -- the
    layout must not shift when it lights up -- and goes grey with its outline.
    """
    if on_path:
        color = "#222222" if revealed else _NODE_HOLLOW_LABEL_COLOR
        return {"size": 13, "color": color, "background": "#FFFFFFCC"}
    return {"size": 10, "color": "#777777"}


def _node_color(on_path: bool, revealed: bool) -> str | dict[str, str]:
    """One node's vis.js color: the path amber once its step is revealed, a hollow
    white-on-grey outline before it, today's flat grey for a context node.

    vis.js reads a STRING as "fill and border alike", which is why hollow has to be
    the dict form -- a white string would draw a white node with a white rim, i.e.
    an invisible one.
    """
    if not on_path:
        return _NODE_OFF_PATH_COLOR
    if revealed:
        return _AMBER
    return {"background": _NODE_HOLLOW_FILL, "border": _NODE_HOLLOW_BORDER}


def _render_graph_panel(row: pd.Series, preset_name: str) -> None:
    """Phase 6 (Task 3): the interactive subgraph panel -- the Scenario page's
    long-standing Phase-6 slot, now implemented.

    Degrades honestly at three levels, each without raising: no `demo subgraphs`
    staged at all (`subgraphs_available()` False), this preset's own JSON missing
    (defensive -- `demo build` ships all six together or none), or this specific
    event absent from that preset's `events` dict. The first two are the same
    "no graph export in this package" state; the third is a DIFFERENT one and now
    says so (item M3, review) -- the package does carry the export, this event
    just isn't in it (unreachable for a package built since the stale-staging
    guard in build.py::_include_subgraphs, but the page still reports what it
    actually found rather than a note that would be wrong).

    Phase 9b (Task 5) makes the panel a PROGRESSIVE reveal rather than 27 nodes at
    once: `filters.path_steps` turns the export's own `path` chains into the ordered
    story (Scene → keyframe → each matching object, nearest first → ego pose), a
    `select_slider` walks it, and a toggle folds the context nodes away. Two rules
    keep it honest and stable: every ON-PATH node is drawn at every step (only its
    colour changes, so vis.js keeps solving the same graph and the layout doesn't
    jump), and an edge is only drawn when both of its endpoints are. The default is
    the LAST step -- the whole path revealed, which is exactly what Phase 6 drew.

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
        st.info(_GRAPH_EVENT_ABSENT_NOTE)
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
    on_path_ids = {n["id"] for n in nodes_data if n.get("on_path")}

    # Phase 9b (Task 5): the reveal's two widgets sit ABOVE the graph, so their
    # return values are already in hand when the node list below is built -- no
    # session_state read-before-the-widget dance (and neither key is ever written).
    steps = path_steps(event_subgraph)
    labels = [step.label for step in steps]
    step_column, context_column = st.columns([3, 1])
    with step_column:
        selected_label = (
            st.select_slider(
                "Reveal the matched path",
                options=labels,
                value=labels[-1],   # the whole path, revealed: this panel's Phase-6 state
                key=_PATH_STEP_KEY,
            )
            if labels
            else None
        )
    with context_column:
        show_context = st.toggle("Show context nodes", value=False, key=_GRAPH_CONTEXT_KEY)

    # A stale step from the previously-selected event deserializes back to the
    # default (streamlit's SelectSliderSerde falls back when the stored option is
    # gone), so this only has to cover the no-steps case.
    selected_index = (
        labels.index(selected_label)
        if selected_label is not None and selected_label in labels
        else len(labels) - 1
    )
    step_of_node: dict[str, int] = {}
    for index, step in enumerate(steps):
        for node_id in step.node_ids:
            step_of_node.setdefault(node_id, index)

    def _revealed(node_id: str) -> bool:
        """Whether a node's step has been reached. With no path chains staged there
        is nothing to reveal progressively and every on-path node is drawn in the
        path color, exactly as this panel did before the reveal existed. An on-path
        node no chain names (defensive) belongs to the last step."""
        if not steps:
            return True
        return step_of_node.get(node_id, len(steps) - 1) <= selected_index

    # The on-path nodes are ALWAYS in the node set -- only their color changes with
    # the step, so vis.js keeps solving the same graph and the layout stays put.
    # Context nodes are the ones the toggle adds or removes.
    visible_ids = {n["id"] for n in nodes_data if n.get("on_path") or show_context}

    agraph_nodes = [
        Node(
            id=n["id"],
            # The SEMANTIC label (filters.graph_node_label): "adult 9.9 m",
            # "ego -4.5 m/s²", "scene-1084". The graph TYPE and the raw token stay
            # on hover, and the full meta is one click away in the panel beside.
            label=graph_node_label(n),
            title=f"{n['label']} {n['id'].split(':', 1)[-1]}",
            size=_NODE_ON_PATH_SIZE if n.get("on_path") else _NODE_OFF_PATH_SIZE,
            color=_node_color(bool(n.get("on_path")), _revealed(n["id"])),
            borderWidth=(
                _NODE_HOLLOW_BORDER_WIDTH
                if n.get("on_path") and not _revealed(n["id"])
                else 1
            ),
            font=_node_font(bool(n.get("on_path")), revealed=_revealed(n["id"])),
        )
        for n in nodes_data
        if n["id"] in visible_ids
    ]

    agraph_edges = []
    for edge in edges_data:
        # An edge whose other end isn't drawn would dangle -- the context object and
        # the matched one share a Category node, so this has to check BOTH ends.
        if edge["source"] not in visible_ids or edge["target"] not in visible_ids:
            continue
        # An edge is on the path only when BOTH its endpoints are -- a HAS_OBJECT
        # edge from the (on-path) Sample hub to a CONTEXT object is context, not
        # path -- and it lights up only once both of them have been revealed.
        on_path = edge["source"] in on_path_ids and edge["target"] in on_path_ids
        revealed = on_path and _revealed(edge["source"]) and _revealed(edge["target"])
        agraph_edges.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                label=edge["label"] if edge["label"] in _STRUCTURAL_EDGES else "",
                color=_AMBER if revealed else _EDGE_OFF_PATH_COLOR,
                width=2 if revealed else 1,
                font=dict(_EDGE_LABEL_FONT),
            )
        )
    # physics ON: vis.js is given no seeded coordinates, so with physics off it
    # drops ~30 nodes at (near-)random positions -- verified in a real browser
    # (2026-08-20), an unreadable pile. Force-direction spreads the star around the
    # Sample hub. The spec's "static highlight (no animation)" is about the PATH
    # COLORING, not the layout solver.
    config = Config(width=700, height=460, directed=True, physics=True, hierarchical=False)

    graph_col, detail_col = st.columns([2, 1])
    with graph_col:
        clicked = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    with detail_col:
        clicked_node = nodes_by_id.get(clicked) if clicked else None
        if clicked_node is not None:
            # A clicked node still wins the column: the viewer asked for THAT node's
            # properties, not for the step they happen to be standing on.
            meta = clicked_node.get("meta", {})
            st.table(
                pd.DataFrame({"key": list(meta.keys()), "value": [str(v) for v in meta.values()]})
            )
        elif steps:
            step = steps[selected_index]
            st.table(
                pd.DataFrame({
                    "fact": [label for label, _ in step.facts],
                    "value": [value for _, value in step.facts],
                })
            )
            st.caption(_STEP_FACTS_CAPTION)
        # The "why this event" one-liner stays visible in both states -- it is the
        # whole path in one sentence, which is what a step's facts deliberately are
        # not.
        st.caption(subgraph_narrative(event_subgraph))

    st.caption(_GRAPH_LEGEND)
    if steps:
        st.caption(_GRAPH_REVEAL_LEGEND)


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
            provenance("recomputed", _VIEWER_PROVENANCE_DETAIL)
            return
        thumb = thumb_path(token)
        if thumb.is_file():
            # Crop missing (a partial/corrupted package) -- fall back to the
            # thumb at its own scale rather than rendering nothing (item 5,
            # consolidated review): the same resilience the non-curated branch
            # below already has.
            image = draw_overlay(Image.open(thumb), gt, preds_token, mode="overlay", scale=0.16)
            st.image(image)
            provenance("recomputed", _VIEWER_PROVENANCE_DETAIL)
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
        provenance("recomputed", _VIEWER_PROVENANCE_DETAIL)
    st.caption(_NOT_CURATED_CAPTION)


def _render_event_header(row: pd.Series, preset: _Preset, preset_name: str) -> None:
    """The one-line answer to "what am I looking at": the scene, this preset's own
    severity figure, the compact SQL<->Graph parity line, and where the numbers on
    this screen came from.

    The parity line (Phase 9a) is a dynamics preset's own trust indicator, distinct
    from (and additional to) the preset header's pinned ``parity_caption`` line
    higher up the page; model presets get nothing here, their GT-only note already
    covers it. A package with no subgraphs staged simply omits the line rather than
    making a stale claim.
    """
    st.markdown(f"**{row.scene_name}** · {severity_caption(preset_name, row.to_dict())}")
    if preset["family"] == "dynamics":
        subgraph_payload = load_subgraphs(preset_name) if subgraphs_available() else None
        if subgraph_payload is not None:
            st.caption(
                parity_short(
                    int(subgraph_payload["sql_count"]),
                    subgraph_payload["cypher_count"],
                    subgraph_payload["parity"],
                )
            )
    provenance("recorded", _EVENT_PROVENANCE_DETAIL)


def _selected_step(curve: FilmstripCurve) -> str | None:
    """Which step the curves' rule sits on, read BEFORE the filmstrip slider is
    instantiated (so this run's charts follow the step the viewer just picked, not
    the previous one) and never written back -- the widget owns its own key.

    A stored label the CURRENT event has no step for (the viewer moved the slider,
    then opened an event whose scene edge drops that neighbour) falls back to the
    event's own frame: ``render.line_chart`` refuses to draw a rule at a step that
    is not on the axis, and silently pointing at nothing would be worse.
    """
    labels = [step.label for step in curve.steps]
    if not labels:
        return None
    current = next((step.label for step in curve.steps if step.is_current), labels[0])
    stored = str(st.session_state.get(_FILMSTRIP_KEY, current))
    return stored if stored in labels else current


def _render_curves(curve: FilmstripCurve, selected: str | None) -> None:
    """The two step curves (CAN speed, CAN longitudinal acceleration) beside the
    frame -- the motion cue a single still frame cannot give. Both are built by
    ``render.curve_charts`` so tour step 2 draws the identical pair."""
    if not curve.steps:
        return
    speed, accel = curve_charts(curve, selected=selected)
    st.altair_chart(speed, width="stretch")
    st.altair_chart(accel, width="stretch")
    st.caption(curve_caption(curve))


def _model_result(row: pd.Series) -> str:
    """The model figure for the event frame: how many boxes ``_MODEL_FOR_RESULTS``
    (the model whose predictions the overlay beside it draws) claimed on it.

    Every other case is the page's own pinned "not in the curated prediction set"
    wording -- the event outside the curated val split, and the curated event whose
    ``frame_manifest`` row or per-model column the package doesn't carry, which is
    exactly what the Phase-5 model panel said for those two cases before this row
    merged them into one card.
    """
    if not row.in_curated_set:
        return _NOT_CURATED_CAPTION
    manifest = load_frame_manifest()
    column = f"n_preds_{_MODEL_FOR_RESULTS}"
    if column not in manifest.columns:
        return _NOT_CURATED_CAPTION
    frame_rows = manifest.loc[manifest["sample_data_token"] == row.sample_data_token]
    if frame_rows.empty or pd.isna(frame_rows.iloc[0][column]):
        return _NOT_CURATED_CAPTION
    count = int(frame_rows.iloc[0][column])
    return f"{count} pred{'' if count == 1 else 's'} ({_MODEL_FOR_RESULTS})"


# Which VRU the distance card names, per preset (item M2, Phase 9b consolidated
# review: the row showed the PEDESTRIAN distance under every preset, including the
# one whose whole subject is a cyclist). The card SWITCHES -- the row stays six
# cards -- because a second distance card would be an empty figure on five of the
# six presets. rain_vru keeps the pedestrian card: its own ranking quantity is the
# nearer of the two VRUs, and that figure is already the card-grid caption.
_VRU_DISTANCE_CARDS: dict[str, tuple[str, str]] = {
    "fast_cyclists": ("Min cyclist dist", "min_dist_cyclist_m"),
}
_DEFAULT_VRU_DISTANCE_CARD = ("Min ped dist", "min_dist_pedestrian_m")

# The "Model result" card's VALUE when the frame has no model figure, short enough
# to survive as one of six st.metric values (item M3: the full _NOT_CURATED_CAPTION
# sentence is 33 characters and truncates in a sixth-width card -- and this is the
# flagship preset's own default state). The sentence itself is the caption under the
# row, so the page still says it in full.
_NOT_CURATED_VALUE = "not curated"


def _render_metric_row(
    row: pd.Series, preset: _Preset, preset_name: str, curve: FilmstripCurve
) -> None:
    """One row of six cards -- the ego-dynamics, scene-context and model panels'
    figures merged so the viewer fits on one screen (spec sec2).

    The speed card is the EVENT FRAME's own step of the curve above it, so the two
    always agree (and it says "Ego speed" on a pre-0.8 package, where that reading
    is the GT ego pose rather than the CAN bus). "Min long. accel" names the column
    it shows (``accel_long_min_mps2``, the frame's most extreme longitudinal
    sample) rather than the old panel's "Peak decel", which only made sense for a
    negative reading -- 27 of the 126 shipped events are accelerating at that
    sample, and calling that a deceleration figure misread the frame (item 1,
    Phase-5 consolidated review; the sign-neutral name now covers both).

    The VRU distance card follows ``preset_name`` (``_VRU_DISTANCE_CARDS``), and the
    model card's value is short with the full sentence as a caption under the row
    (``_NOT_CURATED_VALUE``) -- items M2 and M3 of the Phase 9b consolidated review.
    """
    current = next((step for step in curve.steps if step.is_current), None)
    speed = current.can_speed_kmh if current is not None else None
    accel = row.accel_long_min_mps2
    vru_label, vru_column = _VRU_DISTANCE_CARDS.get(
        preset_name, _DEFAULT_VRU_DISTANCE_CARD
    )
    vru_distance = row.get(vru_column)
    result = _model_result(row)
    not_curated = result == _NOT_CURATED_CAPTION
    lighting = " · ".join([
        "night" if row.is_night else "day", *(["rain"] if row.is_rain else [])
    ])
    metric_cards(
        [
            (
                "CAN speed" if curve.speed_is_can else "Ego speed",
                f"{speed:.1f} km/h" if speed is not None else "n/a",
            ),
            ("Min long. accel", f"{accel:.2f} m/s²" if pd.notna(accel) else "n/a"),
            (
                vru_label,
                f"{vru_distance:.1f} m" if pd.notna(vru_distance) else "n/a",
            ),
            (
                "Peds within 10m",
                str(int(row.n_peds_within_10m)) if pd.notna(row.n_peds_within_10m) else "0",
            ),
            ("Lighting / rain", lighting),
            ("Model result", _NOT_CURATED_VALUE if not_curated else result),
        ],
        per_row=6,
    )
    if not_curated:
        st.caption(_NOT_CURATED_CAPTION)
    if preset["family"] == "model" and preset["verdict"]:
        st.caption(preset["verdict"])


def _step_speed_mps(row: pd.Series, step: FilmstripStep) -> float:
    """One step's GT EGO-POSE speed in m/s -- a different reading, in a different
    unit, from the CAN speed the curve and the metric card above show.

    Both are kept: the readout answers "how fast at this step" in the unit the rest
    of the page's speed captions use (``filters.speed_caption``), while the card and
    the curve are the CAN bus. They can differ by ~25 % on the same keyframe, so the
    readout NAMES its source (item I3, Phase 9b consolidated review) -- unlabelled,
    the two figures read as one page contradicting itself.
    """
    column = _STEP_COLUMNS.get(step.label)
    value = row.speed_mps if column is None else getattr(row, f"speed_{column}")
    return float(value) if pd.notna(value) else float("nan")


def _render_filmstrip(row: pd.Series, curve: FilmstripCurve) -> None:
    """The t-2..t+2 strip, its step slider and the selected step's readout.

    The steps come from ``filters.filmstrip_steps`` (which drops an NA neighbour at
    a scene edge, as this strip always has) so the slider, the thumbnails and the
    curves above are driven by ONE list -- a step the strip offered but the curve
    did not draw would put the rule somewhere the viewer never selected.
    """
    st.markdown("**Filmstrip**")
    steps = curve.steps
    if not steps:
        return
    labels = [step.label for step in steps]
    selected_label = st.select_slider(
        "Step",
        options=labels,
        value=_CURRENT_STEP if _CURRENT_STEP in labels else labels[0],
        key=_FILMSTRIP_KEY,
    )

    # All five ship already (item 12, consolidated review): a strip of every
    # available step's thumbnail, the selected one marked via its own caption --
    # not just the single selected image the slider alone would show.
    strip_columns = st.columns(len(steps))
    for strip_col, step in zip(strip_columns, steps, strict=True):
        with strip_col:
            step_thumb = thumb_path(step.token)
            if step_thumb.is_file():
                st.image(str(step_thumb))
            marker = " (selected)" if step.label == selected_label else ""
            st.caption(f"{step.label}{marker}")

    selected_step = next(step for step in steps if step.label == selected_label)
    # "ego", so this m/s figure cannot be mistaken for the CAN km/h one on the card
    # above it (see _step_speed_mps). speed_caption itself is unchanged -- it is the
    # same "N.N m/s" every other ego-speed caption on the page uses.
    speed_text = speed_caption(_step_speed_mps(row, selected_step))
    accel = selected_step.accel_mps2
    accel_text = f"{accel:.2f} m/s²" if accel is not None else "n/a"
    st.caption(f"{selected_label}: ego {speed_text}, accel {accel_text}")


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
    loop_breadcrumb(["Mine"])
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
    _render_parity_line(preset_name, preset, len(ranked))

    _render_card_grid(ranked, preset_name, preset)

    selected_token = st.session_state.get("scenario_token")
    if selected_token and selected_token in set(ranked["sample_data_token"]):
        row = ranked.loc[ranked["sample_data_token"] == selected_token].iloc[0]
        st.divider()
        st.subheader("Event viewer")
        # Phase 9b (Task 4): the viewer on ONE screen -- header line, then the
        # frame beside its two step curves, then the single merged metric row,
        # then the filmstrip that drives the curves' rule. The three stacked
        # full-width panels this replaced (ego dynamics / scene context / model
        # context) pushed the filmstrip and the graph below two scrolls, so the
        # motion the event is ABOUT was never on screen with the frame.
        _render_event_header(row, preset, preset_name)
        curve = filmstrip_steps(row)
        # Read the slider's step BEFORE the widget that owns it is instantiated
        # further down: on the rerun after a move, session_state already holds the
        # new step, so the charts drawn here follow it in the same run.
        selected_step = _selected_step(curve)
        frame_column, curve_column = st.columns([3, 2])
        with frame_column:
            _render_viewer(row)
        with curve_column:
            _render_curves(curve, selected_step)
        _render_metric_row(row, preset, preset_name, curve)
        _render_filmstrip(row, curve)
        # Phase-6 slot (item 11, consolidated review): implemented -- the
        # interactive subgraph panel (Task 3).
        _render_graph_panel(row, preset_name)

    st.divider()
    _render_semantic_gallery()
