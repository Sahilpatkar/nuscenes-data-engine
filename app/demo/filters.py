"""Pure filter/formatting logic shared by the Failure Explorer and Scenario Search
pages (no Streamlit import — the AST guard in tests/test_demo_app.py applies here
too, so this module stays importable and testable without a Streamlit runtime;
pandas is the only dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

_LIGHTING = {"night": True, "day": False}
_FAILURE_TYPES = ("has_fn", "has_fp", "has_low_conf", "clean")
_SORT_KEYS = ("failure_count", "distance", "scene_name")

# Standard gravity (m/s^2) -- braking_caption divides accel_long_min_mps2 by this
# to express peak deceleration in g's.
_STANDARD_GRAVITY_MPS2 = 9.80665

_SEVERITY_PRESETS = (
    "hard_braking_near_pedestrians",
    "night_pedestrians",
    "fast_cyclists",
    "rain_vru",
    "fn_pedestrians_night",
    "low_conf_braking",
)


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
    buckets: list[str] | None = None,
) -> pd.DataFrame:
    """Compose ``failure_flags`` with the sidebar's filter criteria.

    ``lighting``: "night" | "day", matched against ``is_night``. ``rain``: exact
    match against ``is_rain``. ``category`` / ``size_bucket`` / ``distance_range``
    are CONJUNCTIVE over the SAME GT row (val-only, any model — GT boxes are not
    model-specific): keep a frame only if at least one of its GT rows satisfies
    EVERY one of these three that is set, not each criterion independently
    against possibly-different rows. E.g. category="pedestrian" +
    distance_range=(80, 123) requires an actual pedestrian box in that distance
    range — a frame with a near pedestrian and an unrelated far car does not
    qualify just because it has *some* row matching each criterion separately
    (a real bug an 81k-combination behavioral sweep caught: that independent-
    per-criterion version returned frames with zero boxes actually matching the
    combined query). ``failure_type``: one of "has_fn"/"has_fp"/"has_low_conf"/
    "clean". ``buckets``: any-overlap membership against the frame's
    ``curation_buckets`` list (keep a frame if ANY of its own buckets appears in
    ``buckets`` — the sidebar's multiselect is OR, not AND, across selections).
    ``None`` or an empty list is a no-op, same as every other criterion left as
    ``None``; with nothing set, the full val-split frame set (with flags
    attached) is returned.
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

    # category/size_bucket/distance_range: ONE conjunctive mask over GT rows,
    # built from whichever of the three are set, then a frame matches if ANY of
    # its rows satisfies all of them together -- not three independent token-set
    # intersections (which would let a near pedestrian and an unrelated far car
    # on the same frame satisfy "pedestrian" + "far" separately without either
    # box actually being both).
    box_criteria_set = category is not None or size_bucket is not None or distance_range is not None
    if box_criteria_set:
        box_mask = pd.Series(True, index=gt_val.index)
        if category is not None:
            box_mask &= gt_val["category_group"] == category
        if size_bucket is not None:
            box_mask &= gt_val["size_bucket"] == size_bucket
        if distance_range is not None:
            lo, hi = distance_range
            box_mask &= gt_val["distance_to_ego_m"].between(lo, hi)
        matching_tokens = set(gt_val.loc[box_mask, "sample_data_token"])
        mask &= flags["sample_data_token"].isin(matching_tokens)

    if failure_type is not None:
        if failure_type not in _FAILURE_TYPES:
            raise ValueError(
                f"filter_frames: unknown failure_type {failure_type!r} — expected "
                f"one of {_FAILURE_TYPES}"
            )
        mask &= flags[failure_type].astype(bool)

    if buckets:
        bucket_set = set(buckets)
        mask &= flags["curation_buckets"].apply(lambda bs: bool(bucket_set & set(bs)))

    return flags[mask].reset_index(drop=True)


def failure_counts(
    gt: pd.DataFrame, preds: pd.DataFrame, tokens: Iterable[str], *, model: str
) -> pd.DataFrame:
    """Per-token (n_fn, n_fp, n_low_conf, failure_count) for ``model``, restricted
    to ``tokens`` — the grid caption's "2fn/1fp/0lowconf" summary and
    ``sort_frames``'s "failure_count" key share this, so the two always agree.
    Unlike ``failure_flags``'s booleans, these are real counts (a frame can have
    more than one FN GT box or FP/low-conf prediction). A token with none of the
    three gets zeros, not a missing row.
    """
    token_list = list(dict.fromkeys(tokens))  # de-duplicate, preserve order
    matched_col = f"matched_{model}"

    gt_scope = gt.loc[gt["sample_data_token"].isin(token_list)]
    is_fn = gt_scope[matched_col].eq(False).fillna(False)
    n_fn = gt_scope.loc[is_fn].groupby("sample_data_token").size()

    preds_scope = preds.loc[
        preds["sample_data_token"].isin(token_list) & (preds["model"] == model)
    ]
    n_fp = preds_scope.loc[preds_scope["status"] == "fp"].groupby("sample_data_token").size()
    n_low_conf = preds_scope.loc[preds_scope["status"] == "low_conf"].groupby(
        "sample_data_token"
    ).size()

    # dtype="object" forced explicitly: an empty Python list defaults to float64
    # when pandas infers a column's dtype, which then breaks sort_frames's merge
    # against a real (always object-dtype, since tokens are strings) frame
    # column -- "merge on object and float64 columns" (caught by the new empty-
    # state page path, final-review round). A non-empty token_list would infer
    # object anyway; this just makes the empty case match it explicitly.
    out = pd.DataFrame(
        {"sample_data_token": pd.array(token_list, dtype="object")}
    ).set_index("sample_data_token")
    out["n_fn"] = n_fn
    out["n_fp"] = n_fp
    out["n_low_conf"] = n_low_conf
    out = out.fillna(0).astype(int)
    out["failure_count"] = out["n_fn"] + out["n_fp"] + out["n_low_conf"]
    return out.reset_index()


def sort_frames(
    frames: pd.DataFrame, *, gt: pd.DataFrame, preds: pd.DataFrame, model: str, key: str
) -> pd.DataFrame:
    """Sort a ``filter_frames``/``failure_flags`` result for the grid.

    ``key``: "failure_count" (``failure_counts``'s n_fn+n_fp+n_low_conf for
    ``model`` — descending, worst first), "distance" (each token's nearest/min GT
    ``distance_to_ego_m`` — ascending, NA last, so a frame with no GT distance
    data doesn't float to the top by accident), or "scene_name" (alphabetical).
    Returns a new, re-indexed DataFrame; ``frames`` itself is not mutated.
    """
    if key not in _SORT_KEYS:
        raise ValueError(f"sort_frames: unknown key {key!r} — expected one of {_SORT_KEYS}")

    if key == "scene_name":
        return frames.sort_values("scene_name", kind="stable").reset_index(drop=True)

    if key == "distance":
        min_dist = gt.groupby("sample_data_token")["distance_to_ego_m"].min()
        out = frames.assign(_sort_key=frames["sample_data_token"].map(min_dist))
        out = out.sort_values("_sort_key", kind="stable", na_position="last")
        return out.drop(columns="_sort_key").reset_index(drop=True)

    counts = failure_counts(gt, preds, frames["sample_data_token"], model=model)
    out = frames.merge(
        counts[["sample_data_token", "failure_count"]], on="sample_data_token", how="left"
    )
    out = out.sort_values("failure_count", ascending=False, kind="stable")
    return out.drop(columns="failure_count").reset_index(drop=True)


def rank_events(events: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Scenario Search's card grid: ``events`` rows tagged with ``preset``, sorted by
    that preset's own severity rank (ascending — rank 1 is most severe/first).

    ``demo/events.py::build_events`` writes one ``preset_rank_<name>`` column per
    known preset (Int64, NA for a row not tagged with that preset, even if it's
    tagged with a DIFFERENT preset) — filtering on that column being non-null is
    exactly "tagged with this preset", so no separate membership check against
    ``preset_tags`` is needed. An unknown preset name (no matching column at all)
    raises rather than silently returning nothing, mirroring ``sort_frames``'s
    ``unknown key`` guard.
    """
    rank_col = f"preset_rank_{preset}"
    if rank_col not in events.columns:
        raise ValueError(
            f"rank_events: unknown preset {preset!r} — no {rank_col!r} column in events"
        )
    tagged = events.loc[events[rank_col].notna()]
    return tagged.sort_values(rank_col, kind="stable").reset_index(drop=True)


# --- Scenario Search card/panel caption formatting (consolidated review, items 1 & 6) --
#
# Pure string formatters over already-extracted scalars -- no pandas DataFrame
# required, so each is unit-testable directly without building a fixture frame.


def braking_caption(accel_mps2: float) -> str:
    """"X.XXg braking" for a NEGATIVE (decelerating) accel_long_min_mps2; the same
    "no braking data" string covers both NA and a non-negative value.

    27/126 real flagship-adjacent events have a POSITIVE accel_long_min_mps2 (the
    frame was accelerating, not braking, at its most extreme longitudinal sample)
    -- labeling that a braking-g figure misrepresents the frame (item 1,
    consolidated review).
    """
    if pd.isna(accel_mps2) or accel_mps2 >= 0:
        return "no braking data"
    g_force = abs(float(accel_mps2)) / _STANDARD_GRAVITY_MPS2
    return f"{g_force:.2f}g braking"


def distance_caption(meters: float, *, label: str) -> str:
    return f"{meters:.1f}m {label}" if pd.notna(meters) else f"no {label}"


def speed_caption(mps: float) -> str:
    return f"{mps:.1f} m/s" if pd.notna(mps) else "no speed"


def confidence_caption(conf: float) -> str:
    return f"{conf:.2f} conf" if pd.notna(conf) else "no low-conf detection"


def _min_skip_na(*values: float) -> float:
    """``min()`` over the given values, skipping NA -- mirrors ``events.py``'s own
    ``vru_dist`` computation (``pd.concat(...).min(skipna=True)``) but for two
    already-extracted scalars rather than a whole column."""
    present = [v for v in values if pd.notna(v)]
    return min(present) if present else float("nan")


def severity_caption(preset: str, row: Mapping[str, Any]) -> str:
    """The headline severity figure for one Scenario Search event card -- the SAME
    quantity that ``preset`` ranks by (its own severity key in
    ``demo/events.py::build_events``), not a fixed field shown regardless of which
    preset produced the card (item 6, consolidated review: cards previously always
    showed min-pedestrian distance, even for e.g. fast_cyclists, whose own ranking
    key is speed).

    ``row`` needs whichever of ``accel_long_min_mps2`` / ``min_dist_pedestrian_m`` /
    ``speed_mps`` / ``min_dist_cyclist_m`` / ``fn_ped_min_dist_m`` /
    ``low_conf_min_conf`` the given ``preset`` reads (a dict, or a ``scenario_
    events`` row's ``._asdict()`` — anything supporting ``row["col"]``).
    """
    if preset == "hard_braking_near_pedestrians":
        return braking_caption(row["accel_long_min_mps2"])
    if preset == "night_pedestrians":
        return distance_caption(row["min_dist_pedestrian_m"], label="ped")
    if preset == "fast_cyclists":
        return speed_caption(row["speed_mps"])
    if preset == "rain_vru":
        vru = _min_skip_na(row["min_dist_pedestrian_m"], row["min_dist_cyclist_m"])
        return distance_caption(vru, label="VRU")
    if preset == "fn_pedestrians_night":
        return distance_caption(row["fn_ped_min_dist_m"], label="missed ped")
    if preset == "low_conf_braking":
        return confidence_caption(row["low_conf_min_conf"])
    raise ValueError(f"severity_caption: unknown preset {preset!r} — expected one of {_SEVERITY_PRESETS}")


def subgraph_narrative(subgraph: Mapping[str, Any]) -> str:
    """The "why this event" one-liner for one event's assembled subgraph (Phase 6's
    ``subgraph_export.assemble_subgraph`` output: ``{nodes, edges, path}``, nodes
    carrying ``{id, label, group, on_path, meta}``): "Scene scene-0123 → Sample →
    EgoPose (hard braking -4.50 m/s²) → pedestrian at 3.80 m".

    ``path``'s ordered id chains pick out WHICH node fills each slot -- the backbone
    ``[scene_id, sample_id, ego_id]`` and the first ``[sample_id, object_id,
    category_id]`` chain (the NEAREST matching on_path ObjectObservation, since
    ``assemble_subgraph`` sorts them by distance) -- while each node's own ``meta``
    supplies the values. Every piece degrades independently rather than raising or
    emitting a "None" literal: a missing Scene name, EgoPose accel, or on_path
    object each just drops its own segment. "Sample" always appears (even for a
    totally empty ``subgraph``) so the panel never shows a blank caption.

    Which node fills the Scene / EgoPose slot is decided by node LABEL, never by
    position in ``path`` (item I3): ``assemble_subgraph`` appends the backbone chain
    ONLY when BOTH the Scene and the EgoPose exist, so on a subgraph missing either
    of them ``path[0]`` is an object chain -- and reading its third element as the
    EgoPose slot reported the Category node as an EgoPose ("Sample -> EgoPose" for a
    sample that has no ego pose at all). Both nodes are unique in a subgraph, so the
    label lookup gives the same answer as the backbone chain whenever that chain
    exists, and keeps each segment dropping INDEPENDENTLY when it doesn't. The
    backbone chain is still located (by its EgoPose-labelled tail) so it is not
    mistaken for the object chain.
    """
    nodes = subgraph.get("nodes") or []
    path = subgraph.get("path") or []
    by_id = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}

    def _first_labelled(label: str) -> dict[str, Any] | None:
        return next((n for n in by_id.values() if n.get("label") == label), None)

    backbone_index = next(
        (
            index
            for index, chain in enumerate(path)
            if len(chain) == 3 and (by_id.get(chain[2]) or {}).get("label") == "EgoPose"
        ),
        None,
    )
    scene = _first_labelled("Scene")
    ego = _first_labelled("EgoPose")

    parts: list[str] = []

    scene_name = (scene.get("meta") or {}).get("name") if scene else None
    if scene_name:
        parts.append(f"Scene {scene_name}")

    parts.append("Sample")

    if ego is not None:
        ego_meta = ego.get("meta") or {}
        accel = ego_meta.get("accel_long_min_mps2")
        if accel is not None:
            prefix = "hard braking " if ego_meta.get("is_hard_braking") else ""
            parts.append(f"EgoPose ({prefix}{float(accel):.2f} m/s²)")
        else:
            parts.append("EgoPose")

    object_chain = next(
        (
            chain
            for index, chain in enumerate(path)
            if index != backbone_index and len(chain) == 3
        ),
        None,
    )
    obj = by_id.get(object_chain[1]) if object_chain else None
    if obj is not None:
        obj_meta = obj.get("meta") or {}
        category = obj_meta.get("group") or obj_meta.get("category")
        distance = obj_meta.get("distance_to_ego_m")
        if category and distance is not None:
            parts.append(f"{category} at {float(distance):.2f} m")
        elif category:
            parts.append(str(category))

    return " → ".join(parts)


# The agraph panel's node label for each graph node type (Phase 6 fix round). Pure
# and Streamlit-free like every other formatter here, so it is unit-testable
# directly; views/scenarios.py's _render_graph_panel is the only caller.
_DEFAULT_CATEGORY_TAIL = "object"


def _numeric(value: Any) -> float | None:
    """``value`` as a float, or None when it is missing/NA/non-numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def graph_node_label(node: Mapping[str, Any]) -> str:
    """The short, SEMANTIC label one subgraph node draws with in the agraph panel.

    Nodes used to be labelled "<graph type>: <first 10 characters of the id>", which
    tells a viewer nothing for a hex token ("ObjectObservation: 1bc3b8d...") and
    actively misleads for a Category: head-truncating the id made
    ``human.pedestrian.adult`` and ``human.pedestrian.construction_worker`` BOTH
    render as "Category: human.pede..." -- duplicated labels on 30/126 real events
    (item I4). Each type now shows the quantity a viewer reads the graph for:

    - ObjectObservation: category tail + distance ("adult 9.9 m", "car 5.3 m"), or
      the tail alone without a usable distance.
    - EgoPose: signed peak longitudinal accel ("ego -4.5 m/s²"), the flagship
      preset's own severity figure; "ego pose" when it is absent.
    - Scene: its name ("scene-1084").
    - Sample: "sample (t)" for the event's own frame (the on_path one), "sample" for
      its prev/next temporal-context neighbours.
    - Category / Location: the id tail in FULL -- it is already the name.

    The node's graph type stays available on hover (``_render_graph_panel`` puts it
    in the node's ``title``) and its full ``meta`` is one click away.
    """
    label = str(node.get("label") or "")
    meta = node.get("meta") or {}

    if label == "ObjectObservation":
        category = str(meta.get("category") or _DEFAULT_CATEGORY_TAIL)
        tail = category.rsplit(".", 1)[-1]
        distance = _numeric(meta.get("distance_to_ego_m"))
        return tail if distance is None else f"{tail} {distance:.1f} m"
    if label == "EgoPose":
        accel = _numeric(meta.get("accel_long_min_mps2"))
        return "ego pose" if accel is None else f"ego {accel:+.1f} m/s²"
    if label == "Scene":
        return str(meta.get("name") or "scene")
    if label == "Sample":
        return "sample (t)" if node.get("on_path") else "sample"
    if label in ("Category", "Location"):
        node_id = str(node.get("id") or "")
        return node_id.split(":", 1)[1] if ":" in node_id else node_id
    return label


def parity_caption(
    sql_count: int, cypher_count: int | None, parity: bool | None, n_shown: int
) -> str:
    """The exact wording of a dynamics preset's SQL/Cypher parity line (item 6,
    Phase 6 review) -- pure so it is unit-testable without a Streamlit runtime;
    ``views/scenarios.py::_render_parity_line`` is the only caller, and decides
    ``st.warning`` vs ``st.caption`` off ``parity is False`` itself (a rendering
    choice, not a wording one).

    "keyframe" is singular only when ``sql_count == 1`` -- plural otherwise, same
    as any other count-noun agreement. The " · showing the top N" clause is
    included only when the grid actually cut something (``n_shown < sql_count``);
    a tiny/test fixture where every matching keyframe got a card would otherwise
    claim a cap that never bit. A recorded mismatch (``parity is False``) folds
    "✗ mismatch recorded" into the same string, ahead of that clause, rather than
    a separate ✓/✗ symbol -- it is a finding, not small print (item M4). ``parity
    is None`` (model presets never reach here; a dynamics preset's JSON always
    carries a bool) draws no symbol at all rather than being called a mismatch by
    default.
    """
    noun = "keyframe" if sql_count == 1 else "keyframes"
    counts = (
        f"{sql_count} matching {noun} dataset-wide — "
        f"Cypher {cypher_count} · SQL {sql_count}"
    )
    shown = f" · showing the top {n_shown}" if n_shown < sql_count else ""
    if parity is False:
        return f"{counts} ✗ mismatch recorded{shown}"
    symbol = " ✓" if parity is True else ""
    return f"{counts}{symbol}{shown}"


# --- Phase 7 (Task 4): Active Learning page helpers ------------------------------
#
# All pure, like everything else in this module: the Active Learning page's three
# derived claims -- which boxes the arm actually fixed, what the night floor did to
# the all-night community's quota, and why one frame was selected -- are computed
# here and unit-tested without a Streamlit runtime.

# Only these two statuses can be attached to a GT box: `demo infer` writes
# matched_annotation_token on a prediction that matched a GT box (tp, or low_conf
# when it is below the confidence floor); an "fp" carries no annotation token.
_CLAIM_STATUSES = ("tp", "low_conf")

_FIXED_BOX_COLUMNS = [
    "annotation_token",
    "category_group",
    "distance_to_ego_m",
    "baseline_claim",
    "arm_claim",
]

# configs/demo.yaml's `models:` -- "champion" is the yolov8m @960 checkpoint, a
# MODEL-SIZE champion from the Phase-3 comparison, NOT an active-learning arm and
# not the best AL result (that is `graph`, +0.0344 overall). Labelling it plainly
# wherever it appears keeps the two kinds of "winner" apart on screen.
_MODEL_LABELS = {"champion": "champion (yolov8m @960)"}


def model_label(model: str) -> str:
    """The on-screen label for a model name (see ``_MODEL_LABELS``)."""
    return _MODEL_LABELS.get(model, model)


def _claims(preds: pd.DataFrame, model: str) -> dict[str, tuple[str, float]]:
    """``{annotation_token: (status, conf)}`` for one model's GT-matched predictions,
    keeping the highest-confidence claim when several rows match the same box."""
    rows = preds.loc[
        (preds["model"] == model)
        & preds["matched_annotation_token"].notna()
        & preds["status"].isin(_CLAIM_STATUSES)
    ]
    best: dict[str, tuple[str, float]] = {}
    for row in rows.itertuples(index=False):
        token = str(row.matched_annotation_token)
        conf = float(row.conf)
        if token not in best or conf > best[token][1]:
            best[token] = (str(row.status), conf)
    return best


def visible_gt_boxes(gt: pd.DataFrame) -> pd.DataFrame:
    """``gt_boxes`` (any number of frames) with the visibility-floor rows dropped.

    Rows under the visibility floor were never scored against, so no page counts
    them, draws them, or explains them. The ``below_visibility_min`` column is
    checked for rather than assumed (an older package predates it) and its NA is
    read as "not below the floor" -- the column is a nullable boolean, and a plain
    ``~column`` on it propagates NA into the mask instead of keeping the row.

    THE definition of the visibility rule (Phase 9a review M10): the Failure
    Explorer, the guided tour and ``visible_gt`` below all call this rather than
    each spelling the same filter out, so the rule cannot drift between pages.
    """
    if "below_visibility_min" not in gt.columns:
        return gt
    return gt.loc[~gt["below_visibility_min"].fillna(False)]


def visible_gt(gt: pd.DataFrame, token: str) -> pd.DataFrame:
    """One frame's GT rows, visibility-floor rows dropped (``visible_gt_boxes``
    narrowed to ``token``).

    Shared by the Active Learning and Weak Supervision pages (consolidated review
    M6 -- it was duplicated in both views).
    """
    return visible_gt_boxes(gt.loc[gt["sample_data_token"] == token])


def gt_for_render(gt_rows: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """``matched_<model>`` renamed to the ``matched`` column ``draw_overlay`` reads.

    ``model=None`` (or a model this package has no column for) yields an all-NA
    ``matched``: a train-pool frame was never evaluated by anyone, and draw_overlay
    renders NA as plain GT rather than as a miss -- "not evaluated" is not the same
    claim as "missed". The Weak Supervision page always passes None (its frames are
    train-pool ones and the comparison it draws is GT vs the pseudo labels).
    """
    column = f"matched_{model}"
    if model is not None and column in gt_rows.columns:
        return gt_rows.rename(columns={column: "matched"})
    return gt_rows.assign(matched=pd.Series(pd.NA, index=gt_rows.index, dtype="boolean"))


def fixed_boxes(
    gt: pd.DataFrame, preds: pd.DataFrame, *, baseline: str, arm: str
) -> pd.DataFrame:
    """The GT boxes ``arm`` upgraded over ``baseline`` on ONE frame, nearest first.

    ``gt``/``preds`` are that frame's rows (the caller slices by
    ``sample_data_token``); ``gt`` needs annotation_token, category_group,
    distance_to_ego_m and below_visibility_min, ``preds`` needs model, status, conf
    and matched_annotation_token.

    A row is an upgrade when the arm DETECTS the box (``status == "tp"``) and the
    baseline either never claimed it (``baseline_claim == "none"``) or only claimed
    it below the confidence floor (``"low-conf 0.22"``). Both models detecting it,
    the arm being unsure itself, and boxes below the visibility floor are excluded:
    those are not a before/after. This is the same rule ``demo build`` validates the
    hand-approved exemplar tokens with (exporters.py::_upgraded_gt_boxes) -- the app
    can't import that module, so the two are deliberate twins.
    """
    visible = (
        gt.loc[~gt["below_visibility_min"].fillna(False)]
        if "below_visibility_min" in gt.columns
        else gt
    )
    arm_claims = _claims(preds, arm)
    baseline_claims = _claims(preds, baseline)

    records = []
    for row in visible.itertuples(index=False):
        annotation = str(row.annotation_token)
        arm_claim = arm_claims.get(annotation)
        if arm_claim is None or arm_claim[0] != "tp":
            continue
        baseline_claim = baseline_claims.get(annotation)
        if baseline_claim is not None and baseline_claim[0] == "tp":
            continue
        records.append(
            {
                "annotation_token": annotation,
                "category_group": row.category_group,
                "distance_to_ego_m": row.distance_to_ego_m,
                "baseline_claim": (
                    "none" if baseline_claim is None else f"low-conf {baseline_claim[1]:.2f}"
                ),
                "arm_claim": f"{arm_claim[1]:.2f}",
            }
        )
    frame = pd.DataFrame.from_records(records, columns=_FIXED_BOX_COLUMNS)
    # na_position="last": a box whose annotation didn't join onto annotations_3d has
    # no distance, and belongs after the ones that can be placed in the scene.
    return frame.sort_values("distance_to_ego_m", na_position="last").reset_index(drop=True)


def community_jump(
    communities: pd.DataFrame, *, before: str, after: str
) -> dict[str, Any] | None:
    """What the night floor did to the largest ALL-NIGHT community's quota.

    ``communities`` is ``al_communities.parquet``; ``before``/``after`` name its two
    quota columns (``quota_graph_rate`` / ``quota_graph_rate_night`` in the shipped
    package). Returns ``{community, size, quota_before, quota_after, ratio}`` for the
    largest community whose members are ALL night frames -- the one the page's
    caption points at, because it is where the night floor's effect is unambiguous
    (in the real package: #10301, 916 frames, 83 -> 323, 3.9x) -- or ``None`` when no
    such community exists, in which case the page simply doesn't make the claim.

    The ``community == -1`` backfill sentinel is excluded: it is a bucket for frames
    the night backfill drew from outside every community, not a community.
    """
    if communities.empty:
        return None
    real = communities.loc[communities["community"] >= 0]
    all_night = real.loc[real["night_members"] == real["size"]]
    if all_night.empty:
        return None
    row = all_night.sort_values(["size", "community"], ascending=[False, True]).iloc[0]
    quota_before = int(row[before])
    quota_after = int(row[after])
    return {
        "community": int(row["community"]),
        "size": int(row["size"]),
        "quota_before": quota_before,
        "quota_after": quota_after,
        # A community that got nothing under the first arm has no ratio to report
        # (the page says the two quotas instead of an infinite multiple).
        "ratio": (quota_after / quota_before) if quota_before else None,
    }


_PICK_PASS_LABELS = {"main": "main pass", "backfill": "seeded backfill"}


# 11th/12th/13th (and the whole 111-119 family) break the last-digit rule, so the
# teens are handled before this lookup is consulted.
_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(number: int) -> str:
    """1 -> "1st", 2 -> "2nd", 11 -> "11th", 21 -> "21st"."""
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{_ORDINAL_SUFFIXES.get(number % 10, 'th')}"


def _quota_text(row: Mapping[str, Any]) -> str:
    """A community's quota, saying where it came from when mass didn't buy it.

    ``select_by_mass`` gives every community that got no night pick a FLOOR of one
    frame, so a community with zero routed failure mass still mines one. Printing a
    bare "1 frame" under "Community failure mass: 0.00 (rank 81 of 97)" reads as a
    quota the mass bought -- it didn't (consolidated review I3: 6 of the 78 gallery
    frames are in zero-mass communities).
    """
    quota = int(row["community_quota"])
    frames = f"{quota} frame{'' if quota == 1 else 's'}"
    if float(row["community_mass"]) == 0:
        return f"{frames} (per-community floor — this community drew no routed failure mass)"
    return frames


def _optional_int(value: Any) -> int | None:
    """An Int64/NA column's value as a plain int, or None when it is missing."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def selection_factors(
    row: Mapping[str, Any], *, n_communities: int, night_floor: int | None = None
) -> list[tuple[str, str, bool | None]]:
    """One selected frame's ``al_selection_explain.parquet`` row as an ordered
    ``(label, value, flag)`` panel, in the order the mechanism actually runs.

    ``flag`` is a bool only where a check/cross reads as a fact (night or not);
    None everywhere else, since a quota or a rank is not a pass/fail.

    The ORDER encodes the honest causal story (Task 3's finding): a frame is
    selected because its COMMUNITY carries failure mass, which buys the community a
    quota, and the frame ranks high enough within it by similarity degree.
    ``n_failures_routed``/``mass_routed`` come LAST and are context, not the reason
    -- 1249 of the 1500 real selected frames have no routed mass at all, so
    presenting it as "high failure rate ✓" would be a fabricated explanation for
    five out of six frames.
    """
    community = int(row["community"])
    mass_rank = _optional_int(row.get("community_mass_rank"))
    degree_rank = _optional_int(row.get("degree_rank_in_community"))
    n_routed = int(row.get("n_failures_routed") or 0)
    pick_pass = str(row["pick_pass"])
    if pick_pass == "night":
        pass_label = (
            "night pass" if night_floor is None else f"night pass (night floor {night_floor})"
        )
    else:
        pass_label = _PICK_PASS_LABELS.get(pick_pass, pick_pass)

    return [
        ("Night frame", "yes" if bool(row["is_night"]) else "no", bool(row["is_night"])),
        (
            "Community",
            f"#{community} · {int(row['community_size'])} frames · "
            f"{int(row['community_night_members'])} at night",
            None,
        ),
        (
            "Community failure mass",
            f"{float(row['community_mass']):.2f} "
            f"(rank {mass_rank if mass_rank is not None else 'n/a'} of {n_communities})",
            None,
        ),
        ("Community quota", _quota_text(row), None),
        ("Picked in", pass_label, None),
        (
            "Similarity-degree rank",
            "n/a"
            if degree_rank is None
            else f"{_ordinal(degree_rank)} of {int(row['community_size'])}",
            None,
        ),
        (
            "Routed failures",
            "none — not itself a routing target"
            if n_routed == 0
            else f"{n_routed} failure{'' if n_routed == 1 else 's'}, "
            f"mass {float(row['mass_routed']):.1f}",
            None,
        ),
    ]


# --- Phase 7 (Task 5): Weak Supervision page helpers ------------------------------
#
# Pure, like the rest of this module: the Weak Supervision page's three derived
# shapes -- one frame's verdict badge and VLM-vs-GT counts, a base arm's loss
# split, and the accepted/rejected crowding pairs -- are computed here and unit-
# tested without a Streamlit runtime.

# The five detector classes the VLM was asked to count, in exporters.py's own
# _VLM_CLASS_COLUMNS order: vlm_counts.parquet carries a vlm_<class>/gt_<class>
# pair for each of them.
_VLM_CLASSES = ("car", "truck", "bus", "pedestrian", "bicycle")

_COUNT_TABLE_COLUMNS = ["class", "vlm_count", "gt_count"]
_LOSS_COLUMNS = ["base_arm", "component", "value", "share"]
_CROWDING_COLUMNS = ["arm", "side", "gt_boxes_per_frame"]


def _count_text(value: Any) -> str:
    """A count as a plain integer string, or "n/a" when it is missing.

    The VLM's counts are float columns and a curated frame the VLM never labelled
    has NA in every one of them (export_vlm_counts fills GT counts with a real 0,
    but never invents a VLM count) -- rendering that NaN into the table would read
    as "the VLM counted zero", which is a different claim from "the VLM never
    looked at this frame".
    """
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value)}"


def weak_frame_summary(
    verdict: str | None, vlm_rows: pd.DataFrame, counts_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    """One curated weak frame's verdict badge and its VLM-vs-GT count table.

    ``verdict`` is ``frame_manifest.weak_verdict`` ("accepted"/"rejected"/NA),
    ``vlm_rows`` that token's ``weak_labels.parquet`` rows (its verified pseudo
    boxes), ``counts_row`` its ``vlm_counts.parquet`` row (or None when the VLM
    never labelled it). Returns ``{verdict_label, mutual_zero, table}``.

    "accepted" with ZERO pseudo boxes is the mutual-zero case and is labelled as
    such: the detector proposed nothing on that frame and the VLM's counts agreed,
    so it was accepted carrying no supervision at all -- materially different from
    "accepted, here are its boxes", and not rare (see weak_verifier_by_class's
    mutual_zero_share). "rejected" frames never have boxes at all, by construction,
    so their label says so instead of reading as an absence of data.

    (The boxes themselves are the baseline detector's; the VLM emits per-class
    counts and verifies them -- see active_learning/pseudo_label.py.)
    """
    label = None if verdict is None or pd.isna(verdict) else str(verdict)
    n_boxes = len(vlm_rows)
    mutual_zero = bool(label == "accepted" and n_boxes == 0)
    if mutual_zero:
        verdict_label = "accepted — 0 pseudo boxes (mutual zero)"
    elif label == "accepted":
        verdict_label = f"accepted — {n_boxes} pseudo box{'' if n_boxes == 1 else 'es'}"
    elif label == "rejected":
        verdict_label = "rejected — no pseudo boxes by construction"
    else:
        verdict_label = "not a candidate of the weak-supervision arm"

    table = pd.DataFrame(
        [
            {
                "class": category,
                "vlm_count": _count_text(
                    None if counts_row is None else counts_row.get(f"vlm_{category}")
                ),
                "gt_count": _count_text(
                    None if counts_row is None else counts_row.get(f"gt_{category}")
                ),
            }
            for category in _VLM_CLASSES
        ],
        columns=_COUNT_TABLE_COLUMNS,
    )
    return {"verdict_label": verdict_label, "mutual_zero": mutual_zero, "table": table}


def loss_long(row: Mapping[str, Any]) -> pd.DataFrame:
    """One ``weak_loss_decomposition`` row as the three stacked components, in the
    order they stack: what weak supervision RETAINED of the GT arm's gain, then
    each of the two ways it lost the rest (frames the verifier dropped, then label
    noise on the frames it kept).

    ``share`` for the retained component is the RAW ``weak_gain / gt_gain`` ratio,
    not the stored ``retention`` (which the exporter rounds to 4 dp so the build
    can assert it against overview_metrics.json) -- so the three shares sum to 1.0
    to float precision, which is what a stacked bar claims visually.
    """
    gt_gain = float(row["gt_gain"])
    weak_gain = float(row["weak_gain"])
    retained_share = weak_gain / gt_gain if gt_gain else float("nan")
    return pd.DataFrame(
        [
            {
                "base_arm": row["base_arm"], "component": "retained",
                "value": weak_gain, "share": retained_share,
            },
            {
                "base_arm": row["base_arm"], "component": "dropped-frame cost",
                "value": float(row["dropped_frame_cost"]),
                "share": float(row["dropped_frame_share"]),
            },
            {
                "base_arm": row["base_arm"], "component": "label cost",
                "value": float(row["label_cost"]), "share": float(row["label_share"]),
            },
        ],
        columns=_LOSS_COLUMNS,
    )


def crowding_long(results: pd.DataFrame) -> pd.DataFrame:
    """``weak_supervision_results`` as paired (arm, side, gt_boxes_per_frame) rows,
    accepted before rejected within each arm -- the shape the paired bar chart
    needs to show what the verifier's agreement rule actually selects for: sparse
    frames pass, crowded ones are rejected.
    """
    records = [
        {
            "arm": row.arm,
            "side": side,
            "gt_boxes_per_frame": float(
                row.gt_boxes_per_accepted_frame
                if side == "accepted"
                else row.gt_boxes_per_rejected_frame
            ),
        }
        for row in results.itertuples(index=False)
        for side in ("accepted", "rejected")
    ]
    return pd.DataFrame.from_records(records, columns=_CROWDING_COLUMNS)


# --- the recorded chat replays ----------------------------------------------------
#
# The page reads `demo chat-record`'s own records (chat_record.RECORD_KEYS /
# FRAME_COLUMNS) and the agent's step summaries verbatim; these helpers only turn
# them into display text, so every claim on screen traces back to a stored value.

# The agent's tools, in the order the "tools exercised" card names them (agent.py's
# TOOL_SPECS plus the graph tool, which is offered only when Neo4j was reachable).
_REPLAY_TOOLS = ("run_sql", "run_cypher", "search_frames", "show_frames", "make_chart")

# Every check `evaluate.grade_case` can emit, in the order the verdict line lists
# them. A check that did NOT apply to a case is absent from the record's `checks`
# dict (never False), so this order filters rather than fills.
_CHECK_ORDER = ("english", "tool_use", "grounded", "numeric", "frames")

# The two query tools, mapped to (the step-input key holding the query, the language
# its code block is highlighted in) -- equal today, but they are two different things.
# Every other tool carries no code block at all.
_STEP_CODE = {"run_sql": ("sql", "sql"), "run_cypher": ("cypher", "cypher")}


def replay_tool_counts(replays: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many times each agent tool ran across ``replays``, plus ``charts`` (the
    total number of charts recorded).

    An errored replay's steps are counted like any other: the agent really did run
    those tools before the session broke, and dropping them would understate what
    the recording exercised. Every ``_REPLAY_TOOLS`` key is always present (the card
    reads them straight), and an unknown tool name is counted under its own key
    rather than dropped.
    """
    counts = dict.fromkeys(_REPLAY_TOOLS, 0)
    counts["charts"] = 0
    for replay in replays:
        for step in replay.get("steps") or []:
            tool = str(step.get("tool", ""))
            counts[tool] = counts.get(tool, 0) + 1
        counts["charts"] += len(replay.get("charts") or [])
    return counts


def _reference_text(value: Any) -> str:
    """A stored reference value as display text: an integral one without decimals
    ("66", however JSON typed it), a fractional one exactly as recorded ("2.91").
    Never re-rounded — the number is the eval harness's own reference."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def verdict_line(replay: Mapping[str, Any]) -> str:
    """The graded verdict for one replay: which checks applied and passed, or which
    ones failed by name, plus the reference value the numeric check was graded
    against.

    The reference is stated; the model's own number is NOT extracted from its prose
    — the answer is on screen in full above this line, and picking "the" number out
    of a sentence would be the page inventing a fact the package does not carry.
    A showcase replay was never graded and says so rather than borrowing a verdict.
    """
    checks = replay.get("checks")
    if replay.get("kind") == "showcase" or not isinstance(checks, dict):
        return "showcase — not graded"
    error = replay.get("error")
    if error:
        return f"✗ errored — {error}"

    applicable = [name for name in _CHECK_ORDER if isinstance(checks.get(name), bool)]
    reference = (
        f" · reference {_reference_text(checks['expected'])}" if "expected" in checks else ""
    )
    if checks.get("passed"):
        return f"✓ passed ({', '.join(applicable)}){reference}"
    failed = [name for name in applicable if not checks[name]]
    return f"✗ failed: {', '.join(failed)}{reference}"


def frame_caption(frame: Mapping[str, Any]) -> str:
    """One retrieved frame's caption: scene, location, and the conditions the
    package recorded for it. Rain is named only when it rained — a "no rain" label
    on every dry frame is noise, not information."""
    conditions = "night" if frame.get("is_night") else "day"
    if frame.get("is_rain"):
        conditions += ", rain"
    parts = [str(frame[key]) for key in ("scene_name", "location") if frame.get(key)]
    parts.append(conditions)
    return " · ".join(parts)


def step_detail(step: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    """One agent step as (outcome line, code block, code language).

    The outcome passes the recorder's own summary through VERBATIM (the agent wrote
    it with ``agent._summarize``, including an ``error: …`` one), with the step's
    input adding what the summary alone does not say: the query a search ran, how
    many tokens a show_frames call named. SQL and Cypher come back as code so they
    can be read (and copied) as queries; the other tools carry no code.
    """
    tool = str(step.get("tool", ""))
    step_input = step.get("input") or {}
    outcome = f"`{tool}` → {step.get('output', '')}"

    if tool in _STEP_CODE:
        key, language = _STEP_CODE[tool]
        code = step_input.get(key)
        return outcome, (None if code is None else str(code)), (None if code is None else language)
    if tool == "search_frames" and step_input.get("query"):
        return f"{outcome} · query: {step_input['query']}", None, None
    if tool == "show_frames":
        tokens = step_input.get("sample_data_tokens") or []
        return f"{outcome} · {len(tokens)} tokens", None, None
    return outcome, None, None


# --- Phase 9a (Task 1): guided-tour helpers ---------------------------------------


def parity_short(
    sql_count: int, cypher_count: int | None, parity: bool | None, *, noun: str = "events"
) -> str:
    """The guided tour's one-line SQL/Cypher parity summary (a shorter cousin of
    ``parity_caption``, which stays as the Scenario preset header's own pinned
    wording -- this is a distinct string for a distinct place).

    ``cypher_count is None`` (a GT-only preset never ran the Cypher side) reads
    "Graph n/a (GT-only preset)" rather than a mismatch. Otherwise a ✓/✗ symbol
    follows the Cypher count exactly as ``parity_caption`` does: ✓ for
    ``parity is True``, "✗ mismatch recorded" for ``parity is False``, nothing for
    ``parity is None``. ``noun`` singularizes (trailing "s" stripped, or pass an
    already-singular noun e.g. "event") when ``sql_count == 1``.
    """
    label = noun[:-1] if sql_count == 1 and noun.endswith("s") else noun
    if cypher_count is None:
        graph = "n/a (GT-only preset)"
    elif parity is False:
        graph = f"{cypher_count} ✗ mismatch recorded"
    elif parity is True:
        graph = f"{cypher_count} ✓"
    else:
        graph = f"{cypher_count}"
    return f"{sql_count} {label} found · SQL {sql_count} / Graph {graph}"


def tour_frame_candidates(
    manifest: pd.DataFrame, explain: pd.DataFrame, gt: pd.DataFrame, *, arm: str
) -> list[str]:
    """Ranked candidate frames for the guided tour's arm-selected steps ("why this
    frame", before/after).

    Restricted to manifest rows curated for ``arm`` (``al_selected_by`` -- an older
    package without the column, or with no rows selected by this arm, yields no
    candidates rather than raising, mirroring the Active Learning page's own
    degradation) that also have a row in ``al_selection_explain.parquet``: the
    "why selected" step has nothing to explain for a frame the explain group never
    covered.

    Ranked night frames first -- the arm this tour walks is night-targeted, so the
    frame it leads with must show what the arm was built to find -- then by the
    frame's community's failure-mass rank ascending, then by how many visible
    pedestrian GT boxes the frame carries, then by token for a stable tie-break.

    The mass rank leads the two content keys (Phase 9a review I3) because the step
    it feeds explains WHY a frame was selected, and the answer is the mass → quota
    mechanism: the frame whose community carried the most routed failure mass is
    the one that best exemplifies it. A frame whose explain row carries no
    ``community_mass_rank`` (or a package whose explain group predates the column)
    sorts after every ranked one rather than ahead of them.
    """
    if manifest.empty or explain.empty or "al_selected_by" not in manifest.columns:
        return []
    explained = set(explain["sample_data_token"].astype(str))
    selected = manifest.loc[
        (manifest["al_selected_by"] == arm)
        & manifest["sample_data_token"].astype(str).isin(explained)
    ]
    if selected.empty:
        return []

    # A community's mass rank is 1-based over the communities, so it can never
    # exceed the number of explain rows -- len(explain) + 1 is "unranked, last".
    unranked = len(explain) + 1
    mass_ranks: dict[str, int] = {}
    if "community_mass_rank" in explain.columns:
        for explain_row in explain.itertuples(index=False):
            value = explain_row.community_mass_rank
            if bool(pd.notna(value)):
                mass_ranks[str(explain_row.sample_data_token)] = int(value)

    def _pedestrian_count(token: str) -> int:
        rows = visible_gt(gt, token)
        return int((rows["category_group"] == "pedestrian").sum())

    # NA-safe: an unenriched frame's is_night can be pd.NA/None, and bool(pd.NA)
    # raises rather than sorting -- it goes last (as if day), the same as every
    # other night-unknown frame, rather than crashing the tour's step.
    night = selected["is_night"].fillna(False).astype(bool)
    ranked = sorted(
        (
            not is_night,
            mass_ranks.get(str(row.sample_data_token), unranked),
            -_pedestrian_count(str(row.sample_data_token)),
            str(row.sample_data_token),
        )
        for is_night, row in zip(night, selected.itertuples(index=False), strict=True)
    )
    return [token for _night, _mass_rank, _peds, token in ranked]
