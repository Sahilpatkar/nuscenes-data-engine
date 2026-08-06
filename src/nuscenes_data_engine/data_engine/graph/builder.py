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
  SET s.name = row.name, s.description = row.description, s.location = row.location,
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
      f.timestamp = row.timestamp, f.n_boxes = row.n_boxes,
      f.location = row.location, f.is_night = row.is_night, f.is_rain = row.is_rain
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

# --- Phase B geo-spatial passes ---

_EGO_POSE = """
UNWIND $rows AS row
MATCH (sm:Sample {token: row.sample_token})
MERGE (e:EgoPose {token: row.token})
  SET e.x = row.x, e.y = row.y, e.z = row.z, e.heading = row.heading,
      e.speed_mps = row.speed_mps, e.timestamp = row.timestamp,
      e.location = row.location, e.is_night = row.is_night, e.is_rain = row.is_rain,
      e.point = point({x: row.x, y: row.y})
MERGE (sm)-[:AT_POSE]->(e)
"""

_CANBUS = """
UNWIND $rows AS row
MATCH (:Sample {token: row.sample_token})-[:AT_POSE]->(e:EgoPose)
SET e.has_canbus = row.has_canbus,
    e.can_speed_kmh = row.can_speed_kmh, e.steering_deg = row.steering_deg,
    e.brake_pedal = row.brake_pedal, e.throttle = row.throttle,
    e.yaw_rate = row.yaw_rate,
    e.accel_long_min_mps2 = row.accel_long_min_mps2,
    e.accel_long_max_mps2 = row.accel_long_max_mps2,
    e.is_hard_braking = row.is_hard_braking
"""

_CANBUS_APPLIED = """
UNWIND $tokens AS token
MATCH (:Sample {token: token})-[:AT_POSE]->(e:EgoPose)
WHERE e.has_canbus IS NOT NULL
RETURN count(e) AS n
"""

_INSTANCES = """
UNWIND $rows AS row
MATCH (sc:Scene {token: row.scene_token})
MERGE (i:ObjectInstance {token: row.token})
  SET i.category = row.category, i.n_annotations = row.n_annotations
MERGE (i)-[:IN_SCENE]->(sc)
WITH i, row
MERGE (c:Category {name: row.category})
  ON CREATE SET c.group = row.group
  ON MATCH SET c.group = coalesce(c.group, row.group)
MERGE (i)-[:OF_CATEGORY]->(c)
"""

_OBSERVATIONS = """
UNWIND $rows AS row
MATCH (sm:Sample {token: row.sample_token})
MERGE (o:ObjectObservation {token: row.token})
  SET o.category = row.category, o.x = row.x, o.y = row.y, o.z = row.z,
      o.point = point({x: row.x, y: row.y}),
      o.width = row.width, o.length = row.length, o.height = row.height, o.yaw = row.yaw,
      o.vx = row.vx, o.vy = row.vy, o.speed_mps = row.speed_mps,
      o.num_lidar_pts = row.num_lidar_pts, o.num_radar_pts = row.num_radar_pts,
      o.visibility = row.visibility, o.distance_to_ego_m = row.distance_to_ego_m,
      o.ego_rel_x = row.ego_rel_x, o.ego_rel_y = row.ego_rel_y,
      o.location = row.location, o.is_night = row.is_night, o.is_rain = row.is_rain
MERGE (sm)-[:HAS_OBJECT]->(o)
WITH o, row
MERGE (c:Category {name: row.category})
  ON CREATE SET c.group = row.group
  ON MATCH SET c.group = coalesce(c.group, row.group)
MERGE (o)-[:OF_CATEGORY]->(c)
WITH o, row
MATCH (inst:ObjectInstance {token: row.instance_token})
MERGE (inst)-[:OBSERVED_AS]->(o)
"""

_DONE_OBJECT_SAMPLES = "MATCH (s:Sample)-[:HAS_OBJECT]->() RETURN DISTINCT s.token AS token"


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
    skip_geometry: bool = False,
    rebuild: bool = False,
    limit_scenes: int | None = None,
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
        schema.apply_schema(driver, database=database)  # constraints/indexes before load

        samples = pd.read_parquet(processed / "samples.parquet")
        annotations = pd.read_parquet(processed / "annotations.parquet")
        labels = pd.read_parquet(labels_path) if labels_path.is_file() else None

        keep_scenes: set[str] | None = None
        if limit_scenes is not None:
            keep_scenes = set(samples["scene_token"].drop_duplicates().head(limit_scenes))
            samples = samples[samples["scene_token"].isin(keep_scenes)]
            annotations = annotations[annotations["scene_token"].isin(keep_scenes)]
            if labels is not None:
                labels = labels[labels["sample_data_token"].isin(set(samples["sample_data_token"]))]
            logger.info("Limiting build to %d scenes", len(keep_scenes))

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

        # Phase B geo-spatial passes (ego pose -> instances -> observations).
        ego_path = processed / "ego_pose.parquet"
        if not skip_geometry and ego_path.is_file():
            ego = pd.read_parquet(ego_path)
            annotations_3d = pd.read_parquet(processed / "annotations_3d.parquet")
            instances = pd.read_parquet(processed / "instances.parquet")
            if keep_scenes is not None:
                ego = ego[ego["scene_token"].isin(keep_scenes)]
                annotations_3d = annotations_3d[annotations_3d["scene_token"].isin(keep_scenes)]
                instances = instances[instances["scene_token"].isin(keep_scenes)]

            load("ego_poses", _EGO_POSE, model.ego_pose_rows(ego))
            load("instances", _INSTANCES, model.object_instance_rows(instances))

            # Resumable: skip observations of keyframes already fully loaded.
            done = {
                row["token"]
                for row in connection.read_query(driver, _DONE_OBJECT_SAMPLES, database=database)
            }
            if done:
                annotations_3d = annotations_3d[~annotations_3d["sample_token"].isin(done)]
                logger.info("  observations: skipping %d already-loaded keyframes", len(done))
            written = connection.run_write_grouped(
                driver, _OBSERVATIONS, model.object_observation_rows(annotations_3d),
                group_key="sample_token", database=database,
            )
            summary["observations"] = written
            logger.info("  %-14s %8d", "observations", written)

            canbus_path = processed / "canbus.parquet"
            if canbus_path.is_file():
                canbus = pd.read_parquet(canbus_path)
                if keep_scenes is not None:
                    canbus = canbus[canbus["scene_token"].isin(keep_scenes)]
                canbus_payload = model.canbus_rows(canbus)
                load("canbus", _CANBUS, canbus_payload)
                applied = connection.read_query(
                    driver,
                    _CANBUS_APPLIED,
                    database=database,
                    params={"tokens": [r["sample_token"] for r in canbus_payload]},
                )[0]["n"]
                summary["canbus_applied"] = applied
                if applied < len(canbus_payload):
                    logger.warning(
                        "  canbus: %d/%d rows applied — %d keyframes had no EgoPose "
                        "(stale graph or out-of-sync --limit-scenes?)",
                        applied, len(canbus_payload), len(canbus_payload) - applied,
                    )
    finally:
        connection.close(driver)

    return summary
