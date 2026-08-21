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
