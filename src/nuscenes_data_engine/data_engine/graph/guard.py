"""Read-only Cypher guard for the chat agent's ``run_cypher`` tool.

Same philosophy as the DuckDB SQL guard (``chat/catalog.py``): the *hard* guarantee is
the database read-only transaction (``execute_read`` — any write clause raises
``ForbiddenInReadOnlyTransaction``); on top of that sits a keyword/procedure denylist as
defense-in-depth (a friendly error before the query ever reaches Neo4j), a row cap, and a
query timeout. The return shape matches ``catalog.run_sql`` so the agent's step summary
and the UI need no changes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from nuscenes_data_engine.data_engine.chat.catalog import _clip

logger = logging.getLogger("nuscenes_data_engine")

MAX_ROWS = 100
QUERY_TIMEOUT_S = 15

# Write clauses — anything that could mutate the graph.
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)
_PERIODIC_COMMIT = re.compile(r"\bUSING\s+PERIODIC\s+COMMIT\b", re.IGNORECASE)
# Every CALL <procedure> is inspected; only read-only procedures are allowed through.
_CALL = re.compile(r"\bCALL\s+([A-Za-z0-9_.]+)", re.IGNORECASE)


def _proc_allowed(proc: str) -> bool:
    """Allowlist of read-only procedures (introspection + GDS streaming algorithms)."""
    low = proc.lower()
    if low.startswith(("db.labels", "db.relationshiptypes", "db.schema", "db.propertykeys")):
        return True
    return low.startswith("gds.") and low.endswith(".stream")


def check_read_only(cypher: str) -> str | None:
    """Return a rejection reason if ``cypher`` looks like a write, else ``None``.

    Pure/DB-free so it is unit-testable and gives the model a fast, explanatory error.
    The read-only transaction is still the authoritative guard behind it.
    """
    if match := _WRITE_KEYWORDS.search(cypher):
        return f"write clause '{match.group(0).upper()}' is not allowed (read-only graph)"
    if _PERIODIC_COMMIT.search(cypher):
        return "USING PERIODIC COMMIT is not allowed"
    for match in _CALL.finditer(cypher):
        proc = match.group(1)
        if not _proc_allowed(proc):
            return f"procedure CALL '{proc}' is not allowed (only read-only procedures)"
    return None


def _consume(result: Any, max_rows: int) -> tuple[list[str], list[list[Any]], bool]:
    """Pull up to ``max_rows`` records into clipped row lists; flag truncation."""
    columns = list(result.keys())
    rows: list[list[Any]] = []
    truncated = False
    for i, record in enumerate(result):
        if i >= max_rows:
            truncated = True
            break
        rows.append([_clip(record.get(col)) for col in columns])
    return columns, rows, truncated


def run_cypher(
    driver: Any,
    cypher: str,
    params: dict[str, Any] | None = None,
    *,
    database: str,
    max_rows: int = MAX_ROWS,
) -> dict[str, Any]:
    """Execute one guarded read-only Cypher query; return columns/rows or an error dict."""
    reason = check_read_only(cypher)
    if reason:
        return {"error": f"Disallowed Cypher: {reason}."}

    import neo4j

    # A READ-mode transaction is the hard guard: the server rejects any write with
    # Neo.ClientError.Statement.AccessMode even if the denylist above were bypassed. The
    # transaction timeout bounds runaway queries; the row cap bounds the result stream.
    try:
        with (
            driver.session(
                database=database, default_access_mode=neo4j.READ_ACCESS
            ) as session,
            session.begin_transaction(timeout=QUERY_TIMEOUT_S) as tx,
        ):
            columns, rows, truncated = _consume(tx.run(cypher, **(params or {})), max_rows)
    except Exception as exc:  # write attempts, syntax errors, timeouts -> model-visible
        return {"error": f"Cypher error: {getattr(exc, 'message', None) or exc}"}
    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}


def graph_schema_prompt() -> str:
    """System-prompt fragment describing the graph for the ``run_cypher`` tool."""
    return (
        "Knowledge graph (Neo4j, read-only via run_cypher) — use it for relationship,\n"
        "path, co-occurrence, similarity, and temporal-next questions that are awkward in\n"
        "SQL. Nodes and relationships:\n"
        "  (:Scene {token, name, description, is_night, is_rain, log_token})\n"
        "  (:Sample {token, timestamp})            -- keyframe (joins the 6 cameras)\n"
        "  (:Frame {token, channel, filename, timestamp, n_boxes, vlm_time_of_day,\n"
        "           vlm_weather, vlm_label_confidence, cars, pedestrians, ...})\n"
        "  (:Location {name})  (:Category {name, group})  (:Hazard {text})\n"
        "  (:NotableCondition {text})\n"
        "  (Frame)-[:IN_SCENE]->(Scene)  (Frame)-[:IN_SAMPLE]->(Sample)\n"
        "  (Sample)-[:IN_SCENE]->(Scene)  (Sample)-[:NEXT {dt_us}]->(Sample)\n"
        "  (Scene)-[:IN_LOCATION]->(Location)\n"
        "  (Frame)-[:CONTAINS {count, avg_visibility, total_bbox_area, ...}]->(Category)\n"
        "  (Frame)-[:HAS_HAZARD]->(Hazard)  (Frame)-[:HAS_CONDITION]->(NotableCondition)\n"
        "  (Category)-[:CO_OCCURS_WITH {n_frames}]->(Category)  -- canonical name order\n"
        "  (Frame)-[:SIMILAR_TO {score, rank}]->(Frame)  -- SigLIP kNN, CAM_FRONT\n"
        "Frame.token is the sample_data_token (joins the SQL tables). category.name is the\n"
        "raw nuScenes name (e.g. vehicle.car); group is the coarse class. Example:\n"
        "  MATCH (:Category {name:'vehicle.bicycle'})<-[:CONTAINS]-(f:Frame)-[:CONTAINS]->(c:Category)\n"
        "  MATCH (f)-[:IN_SCENE]->(s:Scene) WHERE s.is_night\n"
        "  RETURN c.name, count(*) AS n ORDER BY n DESC LIMIT 5\n"
        "Return aggregates or a small LIMIT; results are capped. Still no ego-pose/geometry."
    )
