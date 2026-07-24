"""Neo4j driver + batched write/read helpers for the knowledge graph.

The driver/session are typed ``Any`` at this boundary (the neo4j SDK is dynamically
typed), the same approach ``data_engine/store.py`` takes with lancedb and
``serving/model.py`` with YOLO. Writes go through ``execute_write`` in idempotent
UNWIND-MERGE batches; internal reads through ``execute_read``. The *guarded* read surface
the chat agent uses lives in ``graph/guard.py`` — this module is the low-level plumbing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any, cast

from nuscenes_data_engine.config import Settings

logger = logging.getLogger("nuscenes_data_engine")

# Rows per write transaction. Large enough to amortize round-trips, small enough to keep
# each transaction's memory bounded on the 2G-heap community container.
DEFAULT_BATCH_SIZE = 5000


def get_driver(settings: Settings) -> Any:
    """Open a Neo4j driver and eagerly verify connectivity.

    Eager verification means callers that want graceful degradation (the chat agent) can
    catch ``neo4j.exceptions.ServiceUnavailable`` here and fall back to SQL-only, while
    the builder fails fast.
    """
    import neo4j

    driver = neo4j.GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        # Silence INFORMATION-level notices (e.g. the cartesian-product hint on the
        # indexed two-node CO_OCCURS match) so CLI/chat output stays clean.
        notifications_min_severity="WARNING",
    )
    driver.verify_connectivity()
    return driver


def close(driver: Any) -> None:
    """Close the driver (safe to call on a partially-constructed driver)."""
    if driver is not None:
        driver.close()


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def run_write_batches(
    driver: Any,
    cypher: str,
    rows: Sequence[dict[str, Any]],
    *,
    database: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """``UNWIND $rows`` through ``cypher`` in batches; return the number of rows written.

    ``cypher`` must reference the batch as ``$rows`` (a list of dicts). Every write is a
    ``MERGE`` on a key, so re-running the same batch is a no-op — the whole builder is
    idempotent/resumable.
    """
    if not rows:
        return 0
    written = 0
    with driver.session(database=database) as session:
        for chunk in _chunks(rows, batch_size):
            session.execute_write(lambda tx, c=chunk: tx.run(cypher, rows=c).consume())
            written += len(chunk)
    return written


def run_statements(driver: Any, statements: Sequence[str], *, database: str) -> None:
    """Execute schema/DDL statements, each in its own write transaction (idempotent)."""
    with driver.session(database=database) as session:
        for stmt in statements:
            session.execute_write(lambda tx, s=stmt: tx.run(s).consume())


def run_autocommit(driver: Any, cypher: str, *, database: str) -> None:
    """Run one statement in an auto-commit transaction.

    Needed for ``CALL { ... } IN TRANSACTIONS`` (batched deletes), which Neo4j forbids
    inside an explicit ``execute_write`` transaction.
    """
    with driver.session(database=database) as session:
        session.run(cypher).consume()


def read_query(
    driver: Any, cypher: str, *, database: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Run a read query and return rows as plain dicts (internal use, uncapped).

    This is *not* the model-facing surface — the chat agent goes through
    ``guard.run_cypher`` which enforces read-only, a row cap, and a timeout.
    """
    with driver.session(database=database) as session:
        records = session.execute_read(
            lambda tx: [record.data() for record in tx.run(cypher, **(params or {}))]
        )
    return cast("list[dict[str, Any]]", records)


def write_query(
    driver: Any, cypher: str, *, database: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Run a write-mode query and return rows (for GDS catalog ops that also YIELD)."""
    with driver.session(database=database) as session:
        records = session.execute_write(
            lambda tx: [record.data() for record in tx.run(cypher, **(params or {}))]
        )
    return cast("list[dict[str, Any]]", records)
