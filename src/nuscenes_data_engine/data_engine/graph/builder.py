"""Build the knowledge graph from the processed Parquet + LanceDB store.

Streams the denormalized tables, projects them with ``model.py`` (pure), and loads them
via batched ``UNWIND``-MERGE. Every write is a MERGE on a key, so the whole build is
idempotent/resumable — re-running touches nothing that already matches. Passes are
dependency-ordered: nodes are MERGEd before the edges that MATCH their endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.config import Settings
from nuscenes_data_engine.data_engine.graph import connection, knn, model, schema

logger = logging.getLogger("nuscenes_data_engine")

# Derived edge passes gated by the CLI ``--edges`` allowlist. Nodes + structural edges
# (IN_SCENE/IN_SAMPLE/IN_LOCATION/NEXT) are always built — they define the skeleton.
OPTIONAL_PASSES: tuple[str, ...] = ("contains", "co_occurs", "vlm", "similar")

_LOCATIONS = """
UNWIND $rows AS row
MERGE (:Location {name: row.name})
"""

_SCENES = """
UNWIND $rows AS row
MERGE (s:Scene {token: row.token})
  SET s.name = row.name, s.description = row.description,
      s.is_night = row.is_night, s.is_rain = row.is_rain, s.log_token = row.log_token
WITH s, row
MATCH (l:Location {name: row.location})
MERGE (s)-[:IN_LOCATION]->(l)
"""

_SAMPLES = """
UNWIND $rows AS row
MERGE (sm:Sample {token: row.token})
  SET sm.timestamp = row.timestamp
WITH sm, row
MATCH (sc:Scene {token: row.scene_token})
MERGE (sm)-[:IN_SCENE]->(sc)
"""

_NEXT = """
UNWIND $rows AS row
MATCH (a:Sample {token: row.src}), (b:Sample {token: row.dst})
MERGE (a)-[r:NEXT]->(b)
  SET r.dt_us = row.dt_us
"""

_FRAMES = """
UNWIND $rows AS row
MERGE (f:Frame {token: row.token})
  SET f.channel = row.channel, f.filename = row.filename,
      f.width = row.width, f.height = row.height,
      f.timestamp = row.timestamp, f.n_boxes = row.n_boxes
WITH f, row
MATCH (sc:Scene {token: row.scene_token})
MERGE (f)-[:IN_SCENE]->(sc)
WITH f, row
MATCH (sm:Sample {token: row.sample_token})
MERGE (f)-[:IN_SAMPLE]->(sm)
"""

_CONTAINS = """
UNWIND $rows AS row
MATCH (f:Frame {token: row.token})
MERGE (c:Category {name: row.category})
  ON CREATE SET c.group = row.group
  ON MATCH SET c.group = coalesce(c.group, row.group)
MERGE (f)-[rel:CONTAINS]->(c)
  SET rel.count = row.count, rel.avg_visibility = row.avg_visibility,
      rel.min_visibility = row.min_visibility, rel.total_bbox_area = row.total_bbox_area,
      rel.max_bbox_area = row.max_bbox_area, rel.num_lidar_pts = row.num_lidar_pts,
      rel.num_radar_pts = row.num_radar_pts
"""

_CO_OCCURS = """
UNWIND $rows AS row
MATCH (a:Category {name: row.a}), (b:Category {name: row.b})
MERGE (a)-[rel:CO_OCCURS_WITH]->(b)
  SET rel.n_frames = row.n_frames
"""

_VLM_PROPS = """
UNWIND $rows AS row
MATCH (f:Frame {token: row.token})
SET f += row
"""

_HAS_HAZARD = """
UNWIND $rows AS row
MATCH (f:Frame {token: row.token})
MERGE (h:Hazard {text: row.text})
MERGE (f)-[:HAS_HAZARD]->(h)
"""

_HAS_CONDITION = """
UNWIND $rows AS row
MATCH (f:Frame {token: row.token})
MERGE (nc:NotableCondition {text: row.text})
MERGE (f)-[:HAS_CONDITION]->(nc)
"""


def _want(pass_name: str, edges: list[str] | None) -> bool:
    return edges is None or pass_name in edges


def build_graph(
    settings: Settings,
    config_path: Path,
    *,
    edges: list[str] | None = None,
    knn_k: int | None = None,
    channel: str | None = None,
    skip_knn: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build (or extend) the graph; return a summary dict of rows written per pass."""
    processed = Path(settings.processed_dir)
    labels_path = Path(settings.data_dir) / "autolabel" / "labels.parquet"
    database = settings.neo4j_database

    driver = connection.get_driver(settings)
    summary: dict[str, Any] = {}
    try:
        if rebuild:
            logger.info("Rebuild: deleting all nodes/relationships")
            schema.drop_all(driver, database=database)
        schema.apply_schema(driver, database=database)

        samples = pd.read_parquet(processed / "samples.parquet")
        annotations = pd.read_parquet(processed / "annotations.parquet")
        labels = pd.read_parquet(labels_path) if labels_path.is_file() else None

        def load(name: str, cypher: str, rows: list[dict[str, Any]]) -> None:
            written = connection.run_write_batches(driver, cypher, rows, database=database)
            summary[name] = written
            logger.info("  %-14s %8d", name, written)

        # Nodes + structural edges (always).
        load("locations", _LOCATIONS, model.location_rows(samples))
        load("scenes", _SCENES, model.scene_rows(samples))
        load("samples", _SAMPLES, model.sample_rows(samples))
        load("next", _NEXT, model.next_rows(samples))
        load("frames", _FRAMES, model.frame_rows(samples))

        # Derived edges (gated by --edges).
        if _want("contains", edges):
            load("contains", _CONTAINS, model.contains_rows(annotations))
        if _want("co_occurs", edges):
            load("co_occurs", _CO_OCCURS, model.co_occurs_rows(annotations))
        if _want("vlm", edges) and labels is not None:
            load("vlm_props", _VLM_PROPS, model.vlm_property_rows(labels))
            load("hazards", _HAS_HAZARD, model.tag_rows(labels, "hazards"))
            load("conditions", _HAS_CONDITION, model.tag_rows(labels, "notable_conditions"))

        if _want("similar", edges) and not skip_knn:
            summary["similar_to"] = knn.build_similarity_edges(
                driver, settings, config_path, channel=channel, k=knn_k, database=database
            )
            logger.info("  %-14s %8d", "similar_to", summary["similar_to"])
    finally:
        connection.close(driver)

    return summary
