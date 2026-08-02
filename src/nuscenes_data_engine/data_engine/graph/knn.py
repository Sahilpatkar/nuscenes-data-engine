"""SIMILAR_TO edges: per-frame kNN over the LanceDB SigLIP vectors.

The frame vectors are already L2-normalized at write time (see ``data_engine/store.py``),
so cosine similarity is a plain dot product and the whole channel's kNN is a blocked
matrix multiply in memory — orders of magnitude faster than one prefiltered LanceDB
search per frame over the 6-camera store. Edges are written in batches as they're
produced, so the pass is genuinely resumable: a re-run skips frames that already have an
out-edge (mirroring ``store.existing_tokens``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from nuscenes_data_engine.config import Settings, load_yaml
from nuscenes_data_engine.data_engine import store
from nuscenes_data_engine.data_engine.graph import connection

logger = logging.getLogger("nuscenes_data_engine")

_SIMILAR_TO_CYPHER = """
UNWIND $rows AS row
MATCH (a:Frame {token: row.src}), (b:Frame {token: row.dst})
MERGE (a)-[r:SIMILAR_TO]->(b)
  SET r.score = row.score, r.rank = row.rank
"""

_DONE_SOURCES_CYPHER = (
    "MATCH (f:Frame)-[:SIMILAR_TO]->() WHERE f.channel = $channel RETURN f.token AS token"
)


def nearest_neighbours(
    vectors: np.ndarray[Any, Any], k: int, *, block: int = 1024
) -> list[list[tuple[int, float]]]:
    """Top-``k`` cosine neighbours per row (excluding self), as ``(index, score)`` lists.

    Pure/DB-free. Assumes rows are L2-normalized (cosine == dot product). Computed in
    row-blocks so the full similarity matrix is never materialized at once.
    """
    n = vectors.shape[0]
    neighbours: list[list[tuple[int, float]]] = []
    take = min(k + 1, n)  # +1 to absorb the self-match before dropping it
    for start in range(0, n, block):
        sims = vectors[start : start + block] @ vectors.T
        for i in range(sims.shape[0]):
            gi = start + i
            row = sims[i]
            cand = np.argpartition(-row, take - 1)[:take]
            cand = cand[np.argsort(-row[cand])]
            neighbours.append([(int(j), float(row[j])) for j in cand if j != gi][:k])
    return neighbours


def _channel_frames(tbl: Any, channel: str) -> Any:
    """All (token, vector) rows for one camera channel, as a DataFrame."""
    n = tbl.count_rows()
    if n == 0:
        return tbl.search().select(["sample_data_token", "vector"]).limit(0).to_pandas()
    return (
        tbl.search()
        .where(f"channel = '{channel}'")
        .select(["sample_data_token", "vector"])
        .limit(n)
        .to_pandas()
    )


def build_similarity_edges(
    driver: Any,
    settings: Settings,
    config_path: Path,
    *,
    channel: str | None = None,
    k: int | None = None,
    database: str | None = None,
    batch_size: int = 5000,
) -> int:
    """Build SIMILAR_TO edges for one channel; return the number of edges written."""
    cfg = load_yaml(config_path).get("lancedb", {})
    channel = channel or settings.graph_knn_channel
    k = k or settings.graph_knn_k
    database = database or settings.neo4j_database

    tbl = store.open_frames_table(
        Path(cfg.get("path", settings.search_lancedb_path)),
        cfg.get("table", settings.search_table),
        dim=0,
    )
    frames = _channel_frames(tbl, channel)
    tokens = [str(token) for token in frames["sample_data_token"]]
    if not tokens:
        logger.info("SIMILAR_TO: no %s frames in the vector store", channel)
        return 0
    vectors = np.asarray(frames["vector"].tolist(), dtype=np.float32)

    done = {
        row["token"]
        for row in connection.read_query(
            driver, _DONE_SOURCES_CYPHER, database=database, params={"channel": channel}
        )
    }
    logger.info(
        "SIMILAR_TO: %d %s frames (%d already linked), k=%d", len(tokens), channel, len(done), k
    )

    neighbours = nearest_neighbours(vectors, k)
    pending: list[dict[str, Any]] = []
    written = 0
    for gi, neigh in enumerate(neighbours):
        src = tokens[gi]
        if src in done:
            continue
        for rank, (j, score) in enumerate(neigh):
            pending.append({"src": src, "dst": tokens[j], "score": score, "rank": rank})
        if len(pending) >= batch_size:
            written += connection.run_write_batches(
                driver, _SIMILAR_TO_CYPHER, pending, database=database
            )
            pending = []
            logger.info("  SIMILAR_TO: %d edges written", written)
    if pending:
        written += connection.run_write_batches(
            driver, _SIMILAR_TO_CYPHER, pending, database=database
        )
    logger.info("SIMILAR_TO: %d edges total", written)
    return written
