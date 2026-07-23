"""DuckDB catalog for the chat agent: read-only views + a guarded SQL runner.

The views are lazy Parquet scans (nothing loads until queried), the same tables the
`query` CLI exposes, plus the Phase 6b VLM labels when present. The guard admits
exactly one SELECT statement per call and layers a denylist on top: DuckDB parses
PRAGMA as a SELECT-typed statement and lets a bare SELECT read arbitrary files
(``read_parquet('/any/path')``, ``FROM 'x.parquet'``), so statement type alone is
not enough. COPY / DDL / DML / multi-statement payloads are rejected by type.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")

MAX_ROWS = 50
MAX_CELL_CHARS = 300

TABLES = ("samples", "annotations", "availability")

# Settings/extension escapes and file-reading table functions (word-boundary match).
_DENIED_KEYWORDS = re.compile(
    r"\b(pragma|attach|detach|install|load|call|import|export|"
    r"read_parquet|read_csv(?:_auto)?|read_json(?:_auto)?|read_text|read_blob|"
    r"parquet_scan|glob|getenv)\b",
    re.IGNORECASE,
)
# String literals that look like file paths (DuckDB treats 'x.parquet' as a table).
_FILEISH_LITERAL = re.compile(
    r"'[^']*(?:/|\\|\.(?:parquet|csv|json|jsonl|txt|db|duckdb))[^']*'",
    re.IGNORECASE,
)


def open_catalog(processed_dir: Path, labels_path: Path | None = None) -> Any:
    """In-memory DuckDB connection with views over the processed Parquet tables."""
    import duckdb

    con = duckdb.connect()
    for name in TABLES:
        path = processed_dir / f"{name}.parquet"
        if path.is_file():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    if labels_path is not None and labels_path.is_file():
        con.execute(f"CREATE VIEW labels AS SELECT * FROM read_parquet('{labels_path}')")
    return con


def catalog_tables(con: Any) -> list[str]:
    """Names of the views actually available on this connection."""
    return sorted(row[0] for row in con.execute("SHOW TABLES").fetchall())


def run_sql(con: Any, sql: str, max_rows: int = MAX_ROWS) -> dict[str, Any]:
    """Execute one guarded SELECT; return columns/rows (capped) or an error dict."""
    import duckdb

    try:
        statements = con.extract_statements(sql)
    except duckdb.Error as exc:
        return {"error": f"SQL parse error: {exc}"}
    if len(statements) != 1:
        return {"error": "Exactly one SQL statement per call."}
    if statements[0].type != duckdb.StatementType.SELECT:
        return {"error": "Only SELECT statements are allowed (read-only catalog)."}
    denied = _DENIED_KEYWORDS.search(sql) or _FILEISH_LITERAL.search(sql)
    if denied:
        return {
            "error": f"Disallowed token {denied.group(0)!r}: only the registered views "
            "may be queried (no files, settings, or extensions)."
        }

    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(max_rows + 1)
    except duckdb.Error as exc:
        return {"error": f"SQL error: {exc}"}

    truncated = len(rows) > max_rows
    clipped = [
        [_clip(value) for value in row]
        for row in rows[:max_rows]
    ]
    return {
        "columns": columns,
        "rows": clipped,
        "row_count": len(clipped),
        "truncated": truncated,
    }


def _clip(value: Any) -> Any:
    """JSON-friendly cell: stringify exotic types, clip runaway strings."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    return text[:MAX_CELL_CHARS] + "…" if len(text) > MAX_CELL_CHARS else text


def schema_prompt(tables: list[str]) -> str:
    """The system-prompt fragment describing the queryable tables."""
    sections = {
        "samples": (
            "samples — one row per camera frame (204,894 keyframes):\n"
            "  sample_data_token (PK), sample_token, channel (CAM_FRONT, CAM_BACK, ...),\n"
            "  filename, width, height, timestamp, n_boxes,\n"
            "  scene_token, scene_name, scene_description, log_token,\n"
            "  location (boston-seaport | singapore-onenorth | singapore-queenstown |\n"
            "  singapore-hollandvillage), is_night BOOL, is_rain BOOL"
        ),
        "annotations": (
            "annotations — one row per projected 2D box (~1.1M):\n"
            "  annotation_token, sample_data_token (joins samples), sample_token, channel,\n"
            "  category_name (raw nuScenes, e.g. vehicle.car), category_group (detector\n"
            "  taxonomy: car|truck|bus|pedestrian|bicycle; NULL = outside taxonomy —\n"
            "  use category_name for anything else, e.g. movable_object.trafficcone),\n"
            "  visibility_token ('1'..'4', 4 = fully visible), num_lidar_pts, num_radar_pts,\n"
            "  x_min, y_min, x_max, y_max, bbox_area, plus the same scene columns as samples"
        ),
        "availability": (
            "availability — file-integrity manifest: channel, is_key_frame BOOL,\n"
            "  present BOOL (file exists on disk), one row per referenced file"
        ),
        "labels": (
            "labels — VLM auto-labels for 5,000 CAM_FRONT frames (Phase 6b):\n"
            "  sample_data_token (joins samples), model, parse_status ('ok' = usable),\n"
            "  time_of_day, weather, hazards, notable_conditions, label_confidence,\n"
            "  cars, trucks, buses, trailers, construction_vehicles, motorcycles,\n"
            "  bicycles, pedestrians, traffic_cones, barriers (integer counts)"
        ),
    }
    notes = (
        "Notes: is_night/is_rain are scene-level flags derived from the scene\n"
        "description. Night driving exists only in Singapore; all rain is in Boston.\n"
        "There is no ego-pose or object-distance data — distance questions cannot be\n"
        "answered. Example patterns:\n"
        "  SELECT category_group, count(*) FROM annotations GROUP BY 1 ORDER BY 2 DESC;\n"
        "  SELECT location, count(DISTINCT scene_token) FROM samples\n"
        "    WHERE is_night GROUP BY 1;\n"
        "  SELECT s.sample_data_token, s.n_boxes FROM samples s\n"
        "    WHERE s.channel = 'CAM_FRONT' ORDER BY n_boxes DESC LIMIT 5;"
    )
    listed = [sections[name] for name in tables if name in sections]
    return "Queryable DuckDB tables:\n\n" + "\n\n".join(listed) + "\n\n" + notes
