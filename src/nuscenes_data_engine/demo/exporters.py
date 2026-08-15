"""Demo artifact exporters — every published number is derived at export time.

Pure functions over the repo's local artifacts (parquet, results.json, MLflow files,
LanceDB thumbnails). Nothing here talks to a network, a GPU, or the live stack; the
public demo app reads only the files these functions write (design:
docs/superpowers/specs/2026-08-12-demo-phase1-design.md).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

# Verbatim from docs/GRAPH.md — the flagship SQL/Cypher parity query.
FLAGSHIP_SQL = """
    SELECT count(DISTINCT c.sample_token)
    FROM canbus c JOIN annotations_3d a USING (sample_token)
    WHERE c.is_hard_braking AND a.category_group = 'pedestrian'
      AND a.distance_to_ego_m < 10
"""

_PROCESSED_INPUTS = (
    "samples.parquet",
    "annotations.parquet",
    "annotations_3d.parquet",
    "canbus.parquet",
    "ego_pose.parquet",
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required demo input missing: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    """Run a single-row, single-column aggregate query and return its value.

    ``fetchone()`` is typed as returning ``tuple | None`` even though a bare
    aggregate query (``count(*)``, ``corr(...)``) always yields exactly one row;
    guard it explicitly rather than asserting past strict mypy.
    """
    row = con.execute(sql).fetchone()
    if row is None:
        raise ValueError(f"aggregate query returned no rows: {sql}")
    return row[0]


def export_overview(
    *,
    processed_dir: Path,
    al_dir: Path,
    out_dir: Path,
    flagship_cypher: dict[str, Any],
) -> dict[str, Any]:
    """Derive the Overview page's numbers from local artifacts and write the JSON.

    Nothing is hardcoded: scale = parquet row counts, r = live correlation of CAN
    speed vs GT-derived ego speed, flagship = the documented SQL re-run over the
    local tables. The Cypher twin stays *sourced* (docs/GRAPH.md) until Phase 6
    computes it against a live graph — the JSON labels it as such.
    """
    for name in _PROCESSED_INPUTS:
        _require(processed_dir / name)
    results = json.loads(_require(al_dir / "results.json").read_text())

    con = duckdb.connect()
    for name in _PROCESSED_INPUTS:
        table = name.removesuffix(".parquet")
        # duckdb.BinderException: CREATE VIEW (a DDL statement) cannot take a
        # prepared-statement parameter, so bind the path via the relation API
        # instead of `execute(..., [path])` (verified against duckdb 1.5.4).
        con.read_parquet(str(processed_dir / name)).create_view(table)
    scale = {
        "images": _scalar(con, "SELECT count(*) FROM samples"),
        "boxes_2d": _scalar(con, "SELECT count(*) FROM annotations"),
        "objects_3d": _scalar(con, "SELECT count(*) FROM annotations_3d"),
        "canbus_rows": _scalar(con, "SELECT count(*) FROM canbus"),
    }
    can_speed_r = _scalar(
        con,
        "SELECT corr(c.can_vel_mps, e.speed_mps) FROM canbus c "
        "JOIN ego_pose e USING (sample_token) WHERE c.has_canbus",
    )
    flagship_sql = _scalar(con, FLAGSHIP_SQL)

    baseline = results["baseline"]["night"]["mAP50-95"]
    overall_baseline = results["baseline"]["overall"]["mAP50-95"]
    night_deltas = {
        arm: entry["night"]["mAP50-95"] - baseline
        for arm, entry in results.items()
        if arm != "baseline" and not arm.startswith("weak_")
    }
    best_night_arm = max(night_deltas, key=lambda arm: night_deltas[arm])
    weak_retention = None
    if "weak_graph_rate_night" in results and "graph_rate_night" in results:
        gt_gain = results["graph_rate_night"]["overall"]["mAP50-95"] - overall_baseline
        weak_gain = results["weak_graph_rate_night"]["overall"]["mAP50-95"] - overall_baseline
        if gt_gain:
            weak_retention = weak_gain / gt_gain

    metrics: dict[str, Any] = {
        "scale": scale,
        "can_speed_r": round(float(can_speed_r), 3),
        "flagship": {
            "sql": flagship_sql,
            "cypher": flagship_cypher["count"],
            "cypher_source": flagship_cypher["source"],
        },
        "results": {
            "best_night_arm": best_night_arm,
            "best_night_delta": round(night_deltas[best_night_arm], 4),
            "weak_retention_of_gt_gain": (
                None if weak_retention is None else round(weak_retention, 4)
            ),
        },
    }
    _write_json(out_dir / "overview_metrics.json", metrics)
    return metrics


def export_al_results(*, al_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Reshape results.json into the arm-comparison table the demo charts read."""
    results = json.loads(_require(al_dir / "results.json").read_text())
    base_overall = results["baseline"]["overall"]["mAP50-95"]
    base_night = results["baseline"]["night"]["mAP50-95"]
    rows = [
        {
            "arm": arm,
            "n_train_images": entry.get("n_train_images"),
            "overall_map5095": entry["overall"]["mAP50-95"],
            "night_map5095": entry["night"]["mAP50-95"],
            "delta_overall": round(entry["overall"]["mAP50-95"] - base_overall, 4),
            "delta_night": round(entry["night"]["mAP50-95"] - base_night, 4),
        }
        for arm, entry in results.items()
    ]
    df = pd.DataFrame(sorted(rows, key=lambda row: row["arm"])).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "active_learning_results.parquet", index=False)
    return df


def export_weaksup(*, al_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Collect every ``*_pseudo_summary.json`` into one weak-supervision table.

    Arms without a summary are absent rather than an error — the demo shows what
    was actually run.
    """
    rows = []
    for path in sorted(al_dir.glob("*_pseudo_summary.json")):
        summary = json.loads(path.read_text())
        rows.append(
            {
                "arm": summary["arm"],
                "n_candidates": summary["n_candidates"],
                "n_accepted": summary["n_accepted"],
                "verifier_retention": summary["retention"],
                "n_pseudo_boxes": summary["n_boxes"],
                "boxes_per_accepted_frame": summary["mean_boxes_per_accepted_frame"],
                "gt_boxes_per_accepted_frame": summary["mean_gt_boxes_per_accepted_frame"],
            }
        )
    if not rows:
        raise ValueError(f"no *_pseudo_summary.json found under {al_dir}")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "weak_supervision_results.parquet", index=False)
    return df
