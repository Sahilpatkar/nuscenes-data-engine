"""Pure projections: denormalized Parquet DataFrames -> graph node/edge row dicts.

No I/O and no Neo4j import — every function takes a pandas DataFrame and returns a list
of plain-scalar dicts ready to ``UNWIND $rows`` into a Cypher MERGE. This is the analog
of ``ingestion/parse.py`` (flat transforms) + ``ingestion/categories.py`` (the taxonomy
map), and is where the graph's correctness is unit-tested — no live database required.

Every value is coerced to a Bolt-safe Python scalar (numpy types unwrapped, NaN/None ->
None) so the neo4j driver accepts the batches directly.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from typing import Any

import pandas as pd

from nuscenes_data_engine.ingestion.categories import group_for

# The 10 VLM object-count columns (Phase 6b schema) copied onto Frame nodes.
_VLM_COUNT_FIELDS: tuple[str, ...] = (
    "cars", "trucks", "buses", "trailers", "construction_vehicles",
    "motorcycles", "bicycles", "pedestrians", "traffic_cones", "barriers",
)


def _opt_str(value: Any) -> str | None:
    """``str(value)`` unless the cell is null (NaN/None -> None)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _opt_int(value: Any) -> int | None:
    """``int(value)`` unless the cell is null (NaN/None -> None)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _as_list(value: Any) -> list[Any]:
    """Normalize a list-typed cell to a Python list.

    The VLM list columns land in Parquet as JSON strings (e.g. ``'["glare"]'``); they can
    also arrive as native lists / numpy arrays depending on how the table was written, so
    all three are handled.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return [value]
        return list(parsed) if isinstance(parsed, list) else [parsed]
    if hasattr(value, "tolist"):  # numpy array
        return list(value.tolist())
    return [value]


def location_rows(samples: pd.DataFrame) -> list[dict[str, Any]]:
    """Distinct ``Location`` nodes (one per map area), sorted for determinism."""
    names = sorted(samples["location"].dropna().unique())
    return [{"name": str(name)} for name in names]


def scene_rows(samples: pd.DataFrame) -> list[dict[str, Any]]:
    """Distinct ``Scene`` nodes with their context (and ``location`` for IN_LOCATION)."""
    cols = ["scene_token", "scene_name", "scene_description", "is_night", "is_rain",
            "log_token", "location"]
    distinct = samples[cols].drop_duplicates(subset="scene_token")
    return [
        {
            "token": str(r.scene_token),
            "name": str(r.scene_name),
            "description": str(r.scene_description),
            "is_night": bool(r.is_night),
            "is_rain": bool(r.is_rain),
            "log_token": str(r.log_token),
            "location": str(r.location),
        }
        for r in distinct.itertuples(index=False)
    ]


def _keyframes(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per keyframe: its scene + earliest (representative) camera timestamp."""
    return (
        samples.groupby("sample_token", sort=False)
        .agg(scene_token=("scene_token", "first"), timestamp=("timestamp", "min"))
        .reset_index()
    )


def sample_rows(samples: pd.DataFrame) -> list[dict[str, Any]]:
    """Distinct ``Sample`` (keyframe) nodes; timestamp is the earliest camera time."""
    return [
        {"token": str(r.sample_token), "timestamp": int(r.timestamp),
         "scene_token": str(r.scene_token)}
        for r in _keyframes(samples).itertuples(index=False)
    ]


def frame_rows(samples: pd.DataFrame) -> list[dict[str, Any]]:
    """One ``Frame`` node per (keyframe, camera) image, with edge keys for scene/sample."""
    return [
        {
            "token": str(r.sample_data_token),
            "sample_token": str(r.sample_token),
            "scene_token": str(r.scene_token),
            "channel": str(r.channel),
            "filename": str(r.filename),
            "width": int(r.width),
            "height": int(r.height),
            "timestamp": int(r.timestamp),
            "n_boxes": int(r.n_boxes),
        }
        for r in samples.itertuples(index=False)
    ]


def next_rows(samples: pd.DataFrame) -> list[dict[str, Any]]:
    """``NEXT`` edges between consecutive keyframes within a scene (with time delta)."""
    keyframes = _keyframes(samples)
    rows: list[dict[str, Any]] = []
    for _scene, grp in keyframes.groupby("scene_token", sort=False):
        ordered = grp.sort_values("timestamp")
        prev: Any = None
        for r in ordered.itertuples(index=False):
            if prev is not None:
                rows.append(
                    {"src": str(prev.sample_token), "dst": str(r.sample_token),
                     "dt_us": int(r.timestamp - prev.timestamp)}
                )
            prev = r
    return rows


def contains_rows(annotations: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate the ~1.1M boxes into one ``CONTAINS`` edge per (frame, category)."""
    ann = annotations.copy()
    ann["_vis"] = ann["visibility_token"].astype(int)
    agg = (
        ann.groupby(["sample_data_token", "category_name"], sort=False)
        .agg(
            n=("bbox_area", "size"),
            avg_visibility=("_vis", "mean"),
            min_visibility=("_vis", "min"),
            total_bbox_area=("bbox_area", "sum"),
            max_bbox_area=("bbox_area", "max"),
            num_lidar_pts=("num_lidar_pts", "sum"),
            num_radar_pts=("num_radar_pts", "sum"),
        )
        .reset_index()
    )
    return [
        {
            "token": str(r.sample_data_token),
            "category": str(r.category_name),
            "group": group_for(str(r.category_name)),
            "count": int(r.n),
            "avg_visibility": float(r.avg_visibility),
            "min_visibility": int(r.min_visibility),
            "total_bbox_area": float(r.total_bbox_area),
            "max_bbox_area": float(r.max_bbox_area),
            "num_lidar_pts": int(r.num_lidar_pts),
            "num_radar_pts": int(r.num_radar_pts),
        }
        for r in agg.itertuples(index=False)
    ]


def co_occurs_rows(annotations: pd.DataFrame) -> list[dict[str, Any]]:
    """``CO_OCCURS_WITH`` edges: frames in which each canonical category pair appears."""
    counter: Counter[tuple[str, str]] = Counter()
    for _token, cats in annotations.groupby("sample_data_token", sort=False)["category_name"]:
        distinct = sorted({str(c) for c in cats})
        for a, b in itertools.combinations(distinct, 2):
            counter[(a, b)] += 1
    return [{"a": a, "b": b, "n_frames": int(n)} for (a, b), n in counter.items()]


def vlm_property_rows(labels: pd.DataFrame) -> list[dict[str, Any]]:
    """VLM property updates for ``Frame`` nodes (usable, ``parse_status == 'ok'`` rows)."""
    usable = labels[labels["parse_status"] == "ok"]
    present_counts = [field for field in _VLM_COUNT_FIELDS if field in usable.columns]
    rows: list[dict[str, Any]] = []
    for r in usable.itertuples(index=False):
        row: dict[str, Any] = {
            "token": str(r.sample_data_token),
            "vlm_parse_status": "ok",
            "vlm_time_of_day": _opt_str(r.time_of_day),
            "vlm_weather": _opt_str(r.weather),
            "vlm_label_confidence": _opt_str(r.label_confidence),  # enum: 'high'/'low'
        }
        for field in present_counts:
            row[field] = _opt_int(getattr(r, field))
        rows.append(row)
    return rows


def tag_rows(labels: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """``(Frame)-[:HAS_*]->(tag)`` edges from a VLM list column (usable rows only)."""
    usable = labels[labels["parse_status"] == "ok"]
    rows: list[dict[str, Any]] = []
    for r in usable.itertuples(index=False):
        token = str(r.sample_data_token)
        for tag in _as_list(getattr(r, column)):
            text = str(tag).strip()
            if text:
                rows.append({"token": token, "text": text})
    return rows
