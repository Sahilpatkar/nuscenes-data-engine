"""Constraints + indexes for the knowledge graph (idempotent DDL).

Uniqueness constraints double as backing indexes for the node keys the builder MERGEs
on; the secondary indexes speed the property filters the chat agent leans on
(``Frame.channel``, ``Frame.vlm_weather``/``vlm_time_of_day``, ``Scene.is_night``).
Everything is ``... IF NOT EXISTS`` so applying the schema repeatedly is a no-op.
"""

from __future__ import annotations

from typing import Any

from nuscenes_data_engine.data_engine.graph import connection

# (label, key) — uniqueness constraint on the node's primary key.
_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("Scene", "token"),
    ("Sample", "token"),
    ("Frame", "token"),
    ("Location", "name"),
    ("Category", "name"),
    ("Hazard", "text"),
    ("NotableCondition", "text"),
    # Phase B geo-spatial nodes.
    ("EgoPose", "token"),
    ("ObjectObservation", "token"),
    ("ObjectInstance", "token"),
)

# (label, property) — secondary range index for common chat filters.
_INDEXES: tuple[tuple[str, str], ...] = (
    ("Frame", "channel"),
    ("Frame", "location"),
    ("Frame", "vlm_weather"),
    ("Frame", "vlm_time_of_day"),
    ("Scene", "is_night"),
    ("Scene", "location"),
    # Phase B: the distance filter + category lookups on 3D observations.
    ("ObjectObservation", "distance_to_ego_m"),
    ("ObjectObservation", "category"),
)

# (label, property) — Neo4j point index for spatial ops (point.distance, within-region).
_POINT_INDEXES: tuple[tuple[str, str], ...] = (
    ("EgoPose", "point"),
    ("ObjectObservation", "point"),
)


def schema_statements() -> tuple[str, ...]:
    """All ``CREATE CONSTRAINT/INDEX ... IF NOT EXISTS`` statements, in order."""
    stmts: list[str] = []
    for label, key in _CONSTRAINTS:
        name = f"{label.lower()}_{key}"
        stmts.append(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )
    for label, prop in _INDEXES:
        name = f"{label.lower()}_{prop}_idx"
        stmts.append(f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
    for label, prop in _POINT_INDEXES:
        name = f"{label.lower()}_{prop}_pt"
        stmts.append(f"CREATE POINT INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
    return tuple(stmts)


def apply_schema(driver: Any, *, database: str) -> None:
    """Create every constraint and index (idempotent)."""
    connection.run_statements(driver, schema_statements(), database=database)


def drop_all(driver: Any, *, database: str) -> None:
    """Delete all nodes/relationships for a ``--rebuild`` (constraints/indexes kept).

    Uses ``apoc.periodic.iterate`` (batched, memory-managed, its own transactions) — the
    plain ``... IN TRANSACTIONS`` form blows the 2G community instance's transaction memory
    pool DETACH-deleting 1.2M richly-connected ObjectObservation nodes.
    """
    connection.run_autocommit(
        driver,
        "CALL apoc.periodic.iterate("
        "'MATCH (n) RETURN id(n) AS id', "
        "'MATCH (n) WHERE id(n) = id DETACH DELETE n', "
        "{batchSize: 2000, parallel: false})",
        database=database,
    )
