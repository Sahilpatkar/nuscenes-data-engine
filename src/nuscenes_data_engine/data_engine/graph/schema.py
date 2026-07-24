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
)

# (label, property) — secondary index for common chat filters.
_INDEXES: tuple[tuple[str, str], ...] = (
    ("Frame", "channel"),
    ("Frame", "vlm_weather"),
    ("Frame", "vlm_time_of_day"),
    ("Scene", "is_night"),
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
    return tuple(stmts)


def apply_schema(driver: Any, *, database: str) -> None:
    """Create every constraint and index (idempotent)."""
    connection.run_statements(driver, schema_statements(), database=database)


def drop_all(driver: Any, *, database: str) -> None:
    """Delete all nodes/relationships for a ``--rebuild`` (constraints/indexes kept).

    Uses ``CALL { ... } IN TRANSACTIONS`` so a full-graph delete doesn't blow the heap;
    that construct requires an auto-commit transaction.
    """
    connection.run_autocommit(
        driver,
        "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS",
        database=database,
    )
