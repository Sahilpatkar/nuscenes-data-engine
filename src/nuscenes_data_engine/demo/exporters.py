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
    # docs/DATA.md's documented validation correlates the CAN vehicle_monitor
    # wheel-speed stream (an independent modality from GT pose) against
    # ego-pose-derived speed; `can_vel_mps` comes from the CAN pose/localization
    # stream instead and is a weaker independence claim, even though it happens to
    # round to the same 0.999 today. `WHERE c.has_canbus` is belt-and-braces here —
    # corr() already skips NULL pairs, and only has_canbus=False rows have a NULL
    # can_speed_kmh — but it documents the intent.
    can_speed_r = _scalar(
        con,
        "SELECT corr(c.can_speed_kmh / 3.6, e.speed_mps) FROM canbus c "
        "JOIN ego_pose e USING (sample_token) WHERE c.has_canbus",
    )
    if can_speed_r is None:
        raise ValueError("CAN speed correlation undefined — empty join?")
    flagship_sql = _scalar(con, FLAGSHIP_SQL)

    baseline = results["baseline"]["night"]["mAP50-95"]
    overall_baseline = results["baseline"]["overall"]["mAP50-95"]
    # weak_ arms are excluded here: they are pseudo-label training runs, not
    # independent night-targeting arms, so they can't win "best night gain". Note
    # the margin is narrow — in the real data weak_graph_rate_night_gt is +0.0099
    # against the winning graph_rate_night's +0.0101 — so a future results.json
    # change flipping the winner is a real result change, not a bug here.
    night_deltas = {
        arm: entry["night"]["mAP50-95"] - baseline
        for arm, entry in results.items()
        if arm != "baseline" and not arm.startswith("weak_")
    }
    best_night_arm = max(night_deltas, key=lambda arm: night_deltas[arm])

    # Per-pair weak-supervision retention (weak_<base> vs <base>, both vs baseline),
    # keyed by the GT base arm. `_gt` arms (e.g. weak_random_gt) are a different,
    # separately-documented comparison (GT training on the verifier-kept subset) and
    # are excluded here. The Overview headline is *attributed*, not just the largest
    # or most recent pair: docs/DEMO_PLAN.md and docs/ACTIVE_LEARNING.md's published
    # 18% figure is specifically the weak_random/random pair — other pairs (e.g.
    # weak_graph_rate_night/graph_rate_night, ~39% in the real data) are real but
    # different results, and must never be silently swapped in as "the" headline.
    weak_pairs: dict[str, float] = {}
    for arm, entry in results.items():
        if not arm.startswith("weak_") or arm.endswith("_gt"):
            continue
        base = arm.removeprefix("weak_")
        if base not in results:
            continue
        gt_gain = results[base]["overall"]["mAP50-95"] - overall_baseline
        if gt_gain:
            weak_gain = entry["overall"]["mAP50-95"] - overall_baseline
            weak_pairs[base] = round(weak_gain / gt_gain, 4)

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
            "weak_retention": {
                "headline_arm": "random",  # the documented 18% pair
                "headline": weak_pairs.get("random"),
                "by_base_arm": weak_pairs,
            },
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

    Phase-7 note: the documented 50%/32% loss decomposition (docs/ACTIVE_LEARNING.md)
    needs a 3-way results.json comparison per base arm (e.g. random vs weak_random_gt
    vs weak_random) that these per-arm summaries alone don't carry. The rejected-frame
    GT-box mean (documented 7.61) is also not in the summaries — recompute it from the
    arm's mined-token list minus its accepted tokens, joined to samples.parquet's
    n_boxes per frame.
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


def export_hero(*, mlruns_dir: Path, run_id: str, mosaic: str, out_dir: Path) -> Path:
    """Copy the configured validation mosaic as the interim Overview hero image.

    Replaced by a real curated overlay in Phase 3; until then the hero is still a
    genuine model output (an ultralytics val-batch prediction mosaic), not a mock.
    """
    src = mlruns_dir / "artifacts" / run_id / "artifacts" / "ultralytics_run" / mosaic
    if not src.is_file():
        raise ValueError(f"hero mosaic not found: {src}")
    dest = out_dir / "sample_frames" / "hero.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest


def export_thumbs(
    *, lancedb_path: Path, table: str, tokens: list[str], out_dir: Path
) -> list[Path]:
    """Export LanceDB-embedded thumbnails as ``thumbs/<token>.jpg`` files.

    Reads the store directly (no SearchEngine, no encoder, no torch). Every
    requested token must exist — a silent miss would leave a demo page with a
    broken image, so missing tokens raise naming themselves.

    Tokens are validated up front, before touching the store: they are
    interpolated into a SQL ``IN (...)`` clause via ``repr()``, which breaks on
    quotes (raising a bare ``RuntimeError`` from the query engine, not a
    meaningful error) and is a syntax error for an empty list.
    """
    if not tokens:
        raise ValueError("export_thumbs: empty token list")
    bad = [t for t in tokens if not t.replace("-", "").isalnum()]
    if bad:
        raise ValueError(f"export_thumbs: malformed tokens (expected hex-like): {bad[:5]}")

    import lancedb  # lazy: optional dependency, never needed by the demo app itself

    frames = (
        lancedb.connect(str(lancedb_path))
        .open_table(table)
        .search()
        .where(f"sample_data_token IN ({', '.join(repr(t) for t in tokens)})")
        .select(["sample_data_token", "thumbnail"])
        .limit(len(tokens))
        .to_list()
    )
    by_token = {row["sample_data_token"]: row["thumbnail"] for row in frames}
    missing = [token for token in tokens if token not in by_token]
    if missing:
        raise ValueError(f"tokens missing from the LanceDB store: {missing}")
    thumb_dir = out_dir / "sample_frames" / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for token in sorted(tokens):
        path = thumb_dir / f"{token}.jpg"
        path.write_bytes(by_token[token])
        written.append(path)
    return written
