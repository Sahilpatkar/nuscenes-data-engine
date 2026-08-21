"""``demo/events.py`` — scenario event features, six presets, filmstrip neighbors.

Pure function building ``scenario_events.parquet``'s content: per-CAM_FRONT-keyframe
ego dynamics + scene context (``canbus`` ⋈ ``ego_pose`` ⋈ ``annotations_3d`` ⋈
``samples``), tagged against six named preset predicates, ranked and capped per
preset, with t-2..t+2 filmstrip neighbor tokens/readouts for the event viewer. See
docs/superpowers/specs/2026-08-20-demo-phase5-design.md §1.

Two preset families, honestly separated:

- *dynamics presets* (dataset-wide, GT-only, always computed): ``hard_braking_near_
  pedestrians`` (the flagship), ``night_pedestrians``, ``fast_cyclists``, ``rain_vru``.
- *model-result presets* (curated-val-only — predictions exist only for the staged
  curation frames): ``fn_pedestrians_night``, ``low_conf_braking``. These read the
  STAGING parquets (``frame_manifest.parquet``, ``gt_boxes.parquet``,
  ``predictions.parquet``) written by ``demo curate`` + ``demo infer`` (the same
  files ``build.py::_include_curation`` copies into the package) — never the
  committed demo_data — and are silently skipped (with a ``logger.warning``) when
  that staging isn't ready, exactly like ``_include_curation`` itself.

The flagship predicate (``hard_braking_near_pedestrians``) must agree bit-for-bit
with ``demo/exporters.py::FLAGSHIP_SQL`` — ``build_events``'s ``flagship_expected``
argument lets a caller (``demo build``) assert the pre-cap count against
``configs/demo.yaml``'s ``flagship.expected_sql_count`` in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

# category_group (annotations_3d) -> the coarser "context group" the presets reason
# about. car/truck/bus collapse to "vehicle"; bicycle is nuScenes' only cyclist
# category (docs/DATA.md, Verified facts in the phase-5 plan).
_CONTEXT_GROUPS: dict[str, str] = {
    "pedestrian": "pedestrian",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "cyclist",
}

_DYNAMICS_PRESETS = (
    "hard_braking_near_pedestrians",
    "night_pedestrians",
    "fast_cyclists",
    "rain_vru",
)
_MODEL_PRESETS = ("fn_pedestrians_night", "low_conf_braking")
_ALL_PRESETS = (*_DYNAMICS_PRESETS, *_MODEL_PRESETS)

# label -> pandas .shift() amount within a scene, timestamp-sorted. shift(k) for
# k > 0 pulls the value from k rows EARLIER (a "t_minus" neighbor); k < 0 pulls from
# |k| rows LATER (a "t_plus" neighbor).
_NEIGHBORS = (("t_minus2", 2), ("t_minus1", 1), ("t_plus1", -1), ("t_plus2", -2))

_NULLABLE_FLOAT_COLUMNS = (
    "speed_mps",
    "accel_long_min_mps2",
    "min_dist_pedestrian_m",
    "min_dist_vehicle_m",
    "min_dist_cyclist_m",
    "fn_ped_min_dist_m",
    "low_conf_min_conf",
    *(f"speed_{label}" for label, _ in _NEIGHBORS),
    *(f"accel_{label}" for label, _ in _NEIGHBORS),
)

_COLUMN_ORDER = (
    "sample_data_token",
    "sample_token",
    "scene_name",
    "timestamp",
    "speed_mps",
    "accel_long_min_mps2",
    "is_hard_braking",
    "min_dist_pedestrian_m",
    "min_dist_vehicle_m",
    "min_dist_cyclist_m",
    "n_peds_within_10m",
    "is_night",
    "is_rain",
    "preset_tags",
    "in_curated_set",
    # Model-preset-specific severity figures (item 6, consolidated review): the
    # MISSED pedestrian's own distance and the lowest low-conf-braking confidence
    # -- distinct from the frame-wide min_dist_pedestrian_m above, which can be a
    # DIFFERENT, caught pedestrian on the same frame.
    "fn_ped_min_dist_m",
    "low_conf_min_conf",
    *(label for label, _ in _NEIGHBORS),
    *(f"speed_{label}" for label, _ in _NEIGHBORS),
    *(f"accel_{label}" for label, _ in _NEIGHBORS),
    *(f"preset_rank_{name}" for name in _ALL_PRESETS),
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required demo-events input missing: {path}")
    return path


def _null_safe_bool(series: pd.Series) -> pd.Series:
    """``bool | null`` -> plain ``bool``, null treated as False.

    curate.py precedent: ``.where(notna, False).astype(bool)`` rather than
    ``.fillna(False).astype(bool)`` — the latter is semantically identical but
    pandas 2.x emits a FutureWarning about its implicit object->bool downcast.
    """
    return series.where(series.notna(), False).astype(bool)


def _is_definitely_false(series: pd.Series) -> pd.Series:
    """Nullable-boolean column -> plain bool mask of rows that are explicitly False.

    Unlike ``_null_safe_bool`` (where null means "assume not"), an unscored
    ``matched_<model>`` row (null) must NOT be counted as a false negative — null
    means "not evaluated", not "definitely missed". ``fillna(True)`` maps True/null
    both to True, so inverting leaves True exactly where the source was False.
    """
    return ~series.fillna(True).astype(bool)


def _load_base(processed_dir: Path) -> pd.DataFrame:
    """CAM_FRONT keyframes ⋈ canbus ⋈ ego_pose, one row per ``sample_token``.

    All three are documented 1:1 (34,149 == 34,149, Verified facts) — an inner join
    with ``validate="one_to_one"`` catches a duplicate key on either side, and the
    row-count check below catches a silent coverage gap that ``validate`` alone
    would let through (a missing key on one side is still a valid one-to-one join).
    """
    samples = pd.read_parquet(
        _require(processed_dir / "samples.parquet"),
        columns=[
            "sample_data_token",
            "sample_token",
            "channel",
            "scene_token",
            "scene_name",
            "timestamp",
            "is_night",
            "is_rain",
        ],
    )
    cam_front = samples.loc[samples["channel"] == "CAM_FRONT"].drop(columns=["channel"])
    n_cam_front = len(cam_front)

    canbus = pd.read_parquet(
        _require(processed_dir / "canbus.parquet"),
        columns=["sample_token", "is_hard_braking", "accel_long_min_mps2"],
    )
    canbus = canbus.assign(is_hard_braking=_null_safe_bool(canbus["is_hard_braking"]))

    ego_pose = pd.read_parquet(
        _require(processed_dir / "ego_pose.parquet"),
        columns=["sample_token", "speed_mps"],
    )

    merged = cam_front.merge(canbus, on="sample_token", how="inner", validate="one_to_one")
    if len(merged) != n_cam_front:
        raise ValueError(
            f"demo events: canbus join dropped rows ({n_cam_front} CAM_FRONT "
            f"keyframes -> {len(merged)} after joining canbus) — expected canbus "
            "to be 1:1 with CAM_FRONT samples"
        )
    merged = merged.merge(ego_pose, on="sample_token", how="inner", validate="one_to_one")
    if len(merged) != n_cam_front:
        raise ValueError(
            f"demo events: ego_pose join dropped rows ({n_cam_front} CAM_FRONT "
            f"keyframes -> {len(merged)} after joining ego_pose) — expected ego_pose "
            "to be 1:1 with CAM_FRONT samples"
        )
    return merged


def _context_features(processed_dir: Path, near_dist_m: float) -> pd.DataFrame:
    """Per-``sample_token`` min distance per context group + peds within ``near_dist_m``.

    ``category_group`` is null for the ~1/3 of real annotations_3d rows outside the
    five tracked classes (barriers, traffic cones, trailers, motorcycles, animals,
    ...) — that's an expected "not one of car/truck/bus/pedestrian/bicycle" value,
    not a data error, so those rows are silently dropped from context (they were
    never going to move the pedestrian/vehicle/cyclist minimums). A non-null value
    that still isn't in ``_CONTEXT_GROUPS`` IS an error — the five tracked category
    names have changed underneath this module without a matching update here.
    """
    ann = pd.read_parquet(
        _require(processed_dir / "annotations_3d.parquet"),
        columns=["sample_token", "category_group", "distance_to_ego_m"],
    )
    context_group = ann["category_group"].map(_CONTEXT_GROUPS)
    unmapped = ann["category_group"].notna() & context_group.isna()
    unknown = sorted(set(ann.loc[unmapped, "category_group"].unique()))
    if unknown:
        raise ValueError(f"demo events: unmapped annotations_3d category_group(s): {unknown}")
    ann = ann.assign(context_group=context_group).dropna(subset=["context_group"])

    min_dist = (
        ann.groupby(["sample_token", "context_group"])["distance_to_ego_m"]
        .min()
        .unstack("context_group")
        .reindex(columns=["pedestrian", "vehicle", "cyclist"])
    )
    min_dist.columns = [f"min_dist_{group}_m" for group in min_dist.columns]

    peds = ann.loc[ann["context_group"] == "pedestrian"]
    n_peds_within = (
        peds.loc[peds["distance_to_ego_m"] < near_dist_m]
        .groupby("sample_token")
        .size()
        .rename("n_peds_within_10m")
    )

    return min_dist.join(n_peds_within, how="left").reset_index()


def _add_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scene, timestamp-ordered t-2..t+2 neighbor tokens + speed/accel readouts."""
    df = df.sort_values(
        ["scene_token", "timestamp", "sample_data_token"], kind="stable"
    ).reset_index(drop=True)
    grouped = df.groupby("scene_token", sort=False)
    for label, shift_amount in _NEIGHBORS:
        df[label] = grouped["sample_data_token"].shift(shift_amount)
        df[f"speed_{label}"] = grouped["speed_mps"].shift(shift_amount)
        df[f"accel_{label}"] = grouped["accel_long_min_mps2"].shift(shift_amount)
    return df


def _load_staging(
    staging_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load the three staged curation parquets, or ``None`` when not ready.

    "Ready" mirrors ``build.py::_include_curation``'s all-or-nothing absent check:
    all three files must be present. Anything less (staging_dir unset, or a partial
    ``demo curate``-but-not-``demo infer`` state) skips the model-result presets with
    a warning rather than raising — an operator mid-flow shouldn't block the
    dynamics-only presets from being computed.
    """
    if staging_dir is None:
        logger.warning(
            "demo events: no staging_dir configured — skipping model-result presets "
            "(fn_pedestrians_night, low_conf_braking)"
        )
        return None
    staging_dir = Path(staging_dir)
    manifest_path = staging_dir / "frame_manifest.parquet"
    gt_path = staging_dir / "gt_boxes.parquet"
    pred_path = staging_dir / "predictions.parquet"
    if not (manifest_path.is_file() and gt_path.is_file() and pred_path.is_file()):
        logger.warning(
            "demo events: no curation staging at %s — skipping model-result presets "
            "(fn_pedestrians_night, low_conf_braking); run `demo curate` + `demo "
            "infer` first",
            staging_dir,
        )
        return None
    manifest = pd.read_parquet(manifest_path)
    gt_boxes = pd.read_parquet(gt_path)
    predictions = pd.read_parquet(pred_path)
    return manifest, gt_boxes, predictions


def _load_annotation_distances(processed_dir: Path) -> pd.DataFrame:
    """``annotation_token -> distance_to_ego_m``, for joining a missed pedestrian's
    OWN per-box distance onto the ``fn_pedestrians_night`` preset's severity key.

    The staged ``gt_boxes.parquet`` itself carries no distance column — that
    enrichment only happens at build time (``build.py::_include_curation``, from
    this same ``annotations_3d`` table) — so this preset joins it in directly
    rather than waiting for the package-time copy.
    """
    return pd.read_parquet(
        _require(processed_dir / "annotations_3d.parquet"),
        columns=["annotation_token", "distance_to_ego_m"],
    )


def _model_preset_tags(
    df: pd.DataFrame,
    staging: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    model: str,
    annotation_distances: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """``(in_curated_set, fn_pedestrians_night, fn_ped_min_dist_m, low_conf_braking,
    low_conf_min_conf)``.

    All five are boolean/float Series aligned to ``df``'s index; when staging isn't
    ready the tags are all-False, the two float series are all-NaN, and
    ``in_curated_set`` is all-False (the honest "not curated" state the UI is
    expected to render, per the design doc).
    """
    in_curated_set = pd.Series(False, index=df.index)
    fn_pedestrians_night = pd.Series(False, index=df.index)
    fn_ped_min_dist = pd.Series(np.nan, index=df.index, dtype="float64")
    low_conf_braking = pd.Series(False, index=df.index)
    low_conf_min_conf = pd.Series(np.nan, index=df.index, dtype="float64")

    if staging is None:
        return in_curated_set, fn_pedestrians_night, fn_ped_min_dist, low_conf_braking, low_conf_min_conf

    manifest, gt_boxes, predictions = staging
    val_tokens = set(manifest.loc[manifest["split"] == "val", "sample_data_token"])
    in_curated_set = df["sample_data_token"].isin(val_tokens)

    matched_col = f"matched_{model}"
    if matched_col not in gt_boxes.columns:
        raise ValueError(
            f"demo events: gt_boxes.parquet has no column {matched_col!r} — check "
            "presets.model_for_results in configs/demo.yaml"
        )
    if "annotation_token" not in gt_boxes.columns:
        raise ValueError("demo events: gt_boxes.parquet missing annotation_token column")

    # below_visibility_min rows are dropped everywhere else the app touches GT
    # (views/failures.py, views/scenarios.py render only ~below_visibility_min
    # rows) rather than treated as "ghost boxes" -- a missed one must not count
    # as a false negative here either, or this preset would flag a pedestrian the
    # rest of the app never shows as a miss. The real staged gt_boxes.parquet
    # (from `demo infer`) always carries this column; it's optional here (default:
    # nothing flagged) only so older/synthetic fixtures that predate it don't need
    # updating for an invariant they aren't exercising.
    if "below_visibility_min" in gt_boxes.columns:
        visible_gt = gt_boxes.loc[~_null_safe_bool(gt_boxes["below_visibility_min"])]
    else:
        visible_gt = gt_boxes
    ped_gt = visible_gt.loc[
        visible_gt["sample_data_token"].isin(val_tokens)
        & (visible_gt["category_group"] == "pedestrian")
    ]
    fn_ped_gt = ped_gt.loc[_is_definitely_false(ped_gt[matched_col])]
    fn_tokens = set(fn_ped_gt["sample_data_token"])
    night_tokens = set(df.loc[df["is_night"], "sample_data_token"])
    fn_pedestrians_night = df["sample_data_token"].isin(fn_tokens & night_tokens)

    # Per-token min distance among the MISSED pedestrians specifically -- not the
    # frame's nearest pedestrian overall (min_dist_pedestrian_m), which can be a
    # DIFFERENT, CAUGHT box on the same frame. The card's verdict text says "this
    # [missed] pedestrian", so its distance must be that box's own.
    fn_with_dist = fn_ped_gt.merge(annotation_distances, on="annotation_token", how="left")
    per_token_min_fn_dist = fn_with_dist.groupby("sample_data_token")["distance_to_ego_m"].min()
    fn_ped_min_dist = df["sample_data_token"].map(per_token_min_fn_dist).astype("float64")

    if "model" not in predictions.columns or "status" not in predictions.columns:
        raise ValueError("demo events: predictions.parquet missing model/status columns")
    low_conf_preds = predictions.loc[
        (predictions["model"] == model) & (predictions["status"] == "low_conf")
    ]
    low_conf_tokens = set(low_conf_preds["sample_data_token"])
    per_token_min_conf = low_conf_preds.groupby("sample_data_token")["conf"].min()
    low_conf_min_conf = df["sample_data_token"].map(per_token_min_conf).astype("float64")
    low_conf_braking = df["sample_data_token"].isin(low_conf_tokens) & df["is_hard_braking"]

    return in_curated_set, fn_pedestrians_night, fn_ped_min_dist, low_conf_braking, low_conf_min_conf


def _cap_preset(
    df: pd.DataFrame,
    tagged: pd.Series,
    *,
    key: pd.Series,
    ascending: bool,
    cap: int,
    preset: str,
) -> tuple[pd.Series, pd.Series]:
    """Rank ``preset``'s tagged rows by severity, keep the top ``cap``.

    Returns ``(kept_mask, rank)`` aligned to ``df``'s full index — ``rank`` is
    ``Int64`` (NA for untagged/capped-out rows), ``kept_mask`` is the boolean rows
    that retain the tag. Ties break on ``sample_data_token`` ascending
    (deterministic, independent of input row order).
    """
    tagged_idx = df.index[tagged]
    order = pd.DataFrame(
        {"token": df.loc[tagged_idx, "sample_data_token"], "key": key.loc[tagged_idx]},
        index=tagged_idx,
    ).sort_values(["key", "token"], ascending=[ascending, True], kind="stable")
    kept_idx = order.index[:cap]

    if len(order) > cap:
        logger.info("demo events: preset %r capped %d -> %d events", preset, len(order), cap)

    rank = pd.Series(pd.NA, index=df.index, dtype="Int64")
    rank.loc[kept_idx] = np.arange(1, len(kept_idx) + 1)
    kept_mask = pd.Series(False, index=df.index)
    kept_mask.loc[kept_idx] = True
    return kept_mask, rank


def build_events(
    *,
    processed_dir: Path,
    staging_dir: Path | None,
    presets_cfg: dict[str, Any],
    flagship_expected: int | None,
) -> pd.DataFrame:
    """Compute the scenario-events table: features, six preset tags, filmstrip neighbors.

    Only events matching >=1 preset are returned (rows tagged by zero presets after
    capping are dropped). Deterministic: stable sorts throughout, token-ascending
    tie-breaks in the per-preset cap.

    Raises ``ValueError`` if ``flagship_expected`` is given and the PRE-CAP
    ``hard_braking_near_pedestrians`` count doesn't match it — the intended use is
    asserting parity with ``exporters.FLAGSHIP_SQL`` before the package ships.
    """
    processed_dir = Path(processed_dir)
    near_dist_m = float(presets_cfg["near_dist_m"])
    high_speed_mps = float(presets_cfg["high_speed_mps"])
    cap_per_preset = int(presets_cfg["cap_per_preset"])
    model = str(presets_cfg["model_for_results"])

    # The exported n_peds_within_10m column NAME promises a literal 10m threshold
    # -- item 9, consolidated review. presets.near_dist_m is configurable for
    # every OTHER preset predicate, but this one specific column would silently
    # lie about its own name if near_dist_m ever drifted from 10.0; rename the
    # column (and every reader of it, app/demo included) before changing this.
    if near_dist_m != 10.0:
        raise ValueError(
            f"demo events: presets.near_dist_m={near_dist_m} but the exported "
            "n_peds_within_10m column name promises exactly 10.0 — rename the "
            "column everywhere it's read before changing this threshold"
        )

    df = _load_base(processed_dir)
    context = _context_features(processed_dir, near_dist_m)
    df = df.merge(context, on="sample_token", how="left")
    df["n_peds_within_10m"] = df["n_peds_within_10m"].fillna(0).astype("int64")

    df = _add_neighbors(df)

    staging = _load_staging(staging_dir)
    # Only read annotations_3d a second time (for the fn distance join) when
    # staging is actually ready -- an absent/partial staging dir never touches
    # this frame (_model_preset_tags's early-return path), so there is no reason
    # to pay for the read.
    annotation_distances = (
        _load_annotation_distances(processed_dir)
        if staging is not None
        else pd.DataFrame(columns=["annotation_token", "distance_to_ego_m"])
    )
    (
        in_curated_set,
        fn_pedestrians_night,
        fn_ped_min_dist,
        low_conf_braking,
        low_conf_min_conf,
    ) = _model_preset_tags(df, staging, model, annotation_distances)
    df["in_curated_set"] = in_curated_set
    df["fn_ped_min_dist_m"] = fn_ped_min_dist
    df["low_conf_min_conf"] = low_conf_min_conf

    vru_dist = pd.concat([df["min_dist_pedestrian_m"], df["min_dist_cyclist_m"]], axis=1).min(
        axis=1, skipna=True
    )

    tags = {
        "hard_braking_near_pedestrians": (
            df["is_hard_braking"] & (df["min_dist_pedestrian_m"] < near_dist_m)
        ),
        "night_pedestrians": df["is_night"] & (df["min_dist_pedestrian_m"] < near_dist_m),
        "fast_cyclists": (
            (df["speed_mps"] >= high_speed_mps) & (df["min_dist_cyclist_m"] < near_dist_m)
        ),
        "rain_vru": df["is_rain"] & (vru_dist < near_dist_m),
        "fn_pedestrians_night": fn_pedestrians_night,
        "low_conf_braking": low_conf_braking,
    }
    tag_matrix = pd.DataFrame({name: tags[name] for name in _ALL_PRESETS}, index=df.index)

    flagship_precap = int(tag_matrix["hard_braking_near_pedestrians"].sum())
    if flagship_expected is not None and flagship_precap != flagship_expected:
        raise ValueError(
            f"demo events: flagship pre-cap count {flagship_precap} != expected "
            f"{flagship_expected} — hard_braking_near_pedestrians has diverged from "
            "exporters.FLAGSHIP_SQL"
        )

    severity = {
        "hard_braking_near_pedestrians": (df["accel_long_min_mps2"], True),
        "night_pedestrians": (df["min_dist_pedestrian_m"], True),
        "fast_cyclists": (df["speed_mps"], False),
        "rain_vru": (vru_dist, True),
        # Ranked by the MISSED pedestrian's own distance (item 3, consolidated
        # review) -- not min_dist_pedestrian_m, which can be a different, caught
        # box on the same frame.
        "fn_pedestrians_night": (df["fn_ped_min_dist_m"], True),
        "low_conf_braking": (df["low_conf_min_conf"], True),
    }

    kept_masks: dict[str, pd.Series] = {}
    ranks: dict[str, pd.Series] = {}
    for name in _ALL_PRESETS:
        key, ascending = severity[name]
        kept_masks[name], ranks[name] = _cap_preset(
            df, tag_matrix[name], key=key, ascending=ascending, cap=cap_per_preset, preset=name
        )

    for name in _ALL_PRESETS:
        df[f"preset_rank_{name}"] = ranks[name]

    kept_matrix = pd.DataFrame({name: kept_masks[name] for name in _ALL_PRESETS}, index=df.index)
    kept_values = kept_matrix.to_numpy(dtype=bool)
    df["preset_tags"] = [
        [name for name, flag in zip(_ALL_PRESETS, row, strict=True) if flag] for row in kept_values
    ]

    result = df.loc[kept_matrix.any(axis=1)].copy()
    result = result.sort_values("sample_data_token", kind="stable").reset_index(drop=True)

    for col in _NULLABLE_FLOAT_COLUMNS:
        result[col] = result[col].astype("Float64")
    result["n_peds_within_10m"] = result["n_peds_within_10m"].astype("Int64")
    for label, _ in _NEIGHBORS:
        result[label] = result[label].astype("string")
    result["is_hard_braking"] = result["is_hard_braking"].astype(bool)
    result["is_night"] = result["is_night"].astype(bool)
    result["is_rain"] = result["is_rain"].astype(bool)
    result["in_curated_set"] = result["in_curated_set"].astype(bool)
    for name in _ALL_PRESETS:
        result[f"preset_rank_{name}"] = result[f"preset_rank_{name}"].astype("Int64")

    logger.info(
        "demo events: %d events tagged (flagship pre-cap %d) from %d CAM_FRONT keyframes",
        len(result),
        flagship_precap,
        len(df),
    )
    return result[list(_COLUMN_ORDER)]


def preset_counts(
    *, processed_dir: Path, staging_dir: Path | None, presets_cfg: dict[str, Any]
) -> dict[str, int]:
    """Pre-cap per-preset tagged-row counts for all six presets, without capping/neighbors.

    Phase 6 (``demo subgraphs``) needs the SQL-side count for every preset -- not just
    the flagship's single asserted number (``build_events``'s ``flagship_expected``) -- to
    compare against the graph's own computed Cypher counts. ``build_events``'s signature
    is frozen (many callers), so this is a standalone function rather than a second return
    value; it reuses every one of ``build_events``'s private helpers (``_load_base``,
    ``_context_features``, ``_load_staging``, ``_load_annotation_distances``,
    ``_model_preset_tags``) so the expensive/complex parts can't drift, and only
    duplicates the short ``tags = {...}`` predicate dict -- kept lock-step with
    ``build_events``'s own copy; if one changes, change both.
    """
    processed_dir = Path(processed_dir)
    near_dist_m = float(presets_cfg["near_dist_m"])
    high_speed_mps = float(presets_cfg["high_speed_mps"])
    model = str(presets_cfg["model_for_results"])

    # Same invariant build_events enforces (see its own comment): the exported
    # n_peds_within_10m column name promises exactly 10.0.
    if near_dist_m != 10.0:
        raise ValueError(
            f"demo events: presets.near_dist_m={near_dist_m} but the exported "
            "n_peds_within_10m column name promises exactly 10.0 — rename the "
            "column everywhere it's read before changing this threshold"
        )

    df = _load_base(processed_dir)
    context = _context_features(processed_dir, near_dist_m)
    df = df.merge(context, on="sample_token", how="left")
    df["n_peds_within_10m"] = df["n_peds_within_10m"].fillna(0).astype("int64")

    staging = _load_staging(staging_dir)
    annotation_distances = (
        _load_annotation_distances(processed_dir)
        if staging is not None
        else pd.DataFrame(columns=["annotation_token", "distance_to_ego_m"])
    )
    (
        _in_curated_set,
        fn_pedestrians_night,
        _fn_ped_min_dist,
        low_conf_braking,
        _low_conf_min_conf,
    ) = _model_preset_tags(df, staging, model, annotation_distances)

    vru_dist = pd.concat([df["min_dist_pedestrian_m"], df["min_dist_cyclist_m"]], axis=1).min(
        axis=1, skipna=True
    )

    # Kept in lockstep with build_events' own `tags` dict above -- same five
    # predicates; only capping/neighbors are skipped (a plain count doesn't need them).
    tags = {
        "hard_braking_near_pedestrians": (
            df["is_hard_braking"] & (df["min_dist_pedestrian_m"] < near_dist_m)
        ),
        "night_pedestrians": df["is_night"] & (df["min_dist_pedestrian_m"] < near_dist_m),
        "fast_cyclists": (
            (df["speed_mps"] >= high_speed_mps) & (df["min_dist_cyclist_m"] < near_dist_m)
        ),
        "rain_vru": df["is_rain"] & (vru_dist < near_dist_m),
        "fn_pedestrians_night": fn_pedestrians_night,
        "low_conf_braking": low_conf_braking,
    }
    return {name: int(tags[name].sum()) for name in _ALL_PRESETS}
