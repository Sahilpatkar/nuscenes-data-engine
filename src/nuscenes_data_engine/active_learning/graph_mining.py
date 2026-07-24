"""Graph-diversity active-learning acquisition (Phase 6e x 6d).

The graph-native alternative to embedding-KMeans mining (``active_learning/mining.py``):
use the Neo4j ``SIMILAR_TO`` graph's community structure instead of clustering raw
vectors. GDS Louvain over the *pool's* similarity subgraph yields appearance communities;
the labeling budget is allocated across communities with the same ``allocate`` helper the
stratified sampler uses, and each community contributes its most-connected (representative)
frames. Selection is graph-native and A/B'd against the mined/random arms on the shared
val split — the acquisition function changes, the random-control gate does not.

The GDS steps run on the infra machine (where Neo4j lives); the resulting ``graph.parquet``
feeds ``al run --arm graph`` on the GPU server, exactly like ``mined.parquet``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.data_engine.autolabel.sampling import allocate
from nuscenes_data_engine.data_engine.graph import connection

logger = logging.getLogger("nuscenes_data_engine")

_GRAPH = "al_sim"  # transient GDS in-memory graph name

_MARK_POOL = """
UNWIND $rows AS row
MATCH (f:Frame {token: row.token})
SET f._al_pool = true
"""
_UNMARK = "MATCH (f:Frame) WHERE f._al_pool = true REMOVE f._al_pool"
_DROP = f"CALL gds.graph.drop('{_GRAPH}', false) YIELD graphName RETURN graphName"
_PROJECT = f"""
MATCH (a:Frame)-[:SIMILAR_TO]->(b:Frame)
WHERE a._al_pool = true AND b._al_pool = true
RETURN gds.graph.project('{_GRAPH}', a, b) AS g
"""
_LOUVAIN = f"""
CALL gds.louvain.stream('{_GRAPH}')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).token AS token, communityId AS community
"""
_DEGREE = f"""
CALL gds.degree.stream('{_GRAPH}')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).token AS token, score AS degree
"""


def select_representatives(
    communities: dict[str, int], degrees: dict[str, float], n_mine: int, floor: int
) -> list[str]:
    """Pick ``n_mine`` frames spread across communities, most-connected first.

    Pure/DB-free: the budget is allocated across communities proportional to size (with a
    per-community floor, via the shared ``allocate`` helper); within each community frames
    are ranked by degree (ties broken by token) and the quota taken from the top.
    Deterministic — the same community/degree inputs always yield the same selection.
    """
    members: dict[int, list[str]] = defaultdict(list)
    for token, community in communities.items():
        members[community].append(token)

    sizes = {community: len(tokens) for community, tokens in members.items()}
    quotas = allocate(sizes, n_mine, floor)

    selected: list[str] = []
    for community in sorted(members, key=lambda c: (-sizes[c], c)):
        quota = quotas.get(community, 0)
        ranked = sorted(members[community], key=lambda t: (-degrees.get(t, 0.0), t))
        selected.extend(ranked[:quota])
    return selected


def _louvain_communities(
    driver: Any, pool_tokens: list[str], database: str
) -> tuple[dict[str, int], dict[str, float]]:
    """Mark the pool, project its SIMILAR_TO subgraph, and stream Louvain + degree."""
    connection.run_write_batches(
        driver, _MARK_POOL, [{"token": token} for token in pool_tokens], database=database
    )
    try:
        connection.write_query(driver, _DROP, database=database)  # clear any stale graph
        connection.write_query(driver, _PROJECT, database=database)
        communities = {
            row["token"]: int(row["community"])
            for row in connection.write_query(driver, _LOUVAIN, database=database)
        }
        degrees = {
            row["token"]: float(row["degree"])
            for row in connection.write_query(driver, _DEGREE, database=database)
        }
    finally:
        for cleanup in (_DROP, _UNMARK):
            try:
                connection.write_query(driver, cleanup, database=database)
            except Exception:  # cleanup is best-effort; never mask the real error
                logger.warning("graph-mining cleanup failed for: %s", cleanup.strip()[:40])
    return communities, degrees


def run_graph_mining(config_path: Path, processed_dir: Path | None = None) -> dict[str, Any]:
    """Community-detect the pool's similarity graph and write ``graph.parquet``."""
    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    cfg = load_yaml(config_path)
    settings = get_settings()
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    graph_cfg = cfg.get("graph_mining", {})
    n_mine = int(graph_cfg.get("n_mine", cfg.get("mining", {}).get("n_mine", 1500)))
    floor = int(graph_cfg.get("floor", 1))
    database = settings.neo4j_database

    split = pd.read_parquet(state_dir / "split.parquet")
    baseline_frames = frames_for_scenes(
        processed, set(split[split["role"] == "baseline"]["scene_name"]), channel
    )
    pool_frames = frames_for_scenes(
        processed, set(split[split["role"] == "pool"]["scene_name"]), channel
    )

    driver = connection.get_driver(settings)
    try:
        communities, degrees = _louvain_communities(driver, sorted(pool_frames), database)
    finally:
        connection.close(driver)
    logger.info(
        "Louvain: %d connected pool frames in %d communities",
        len(communities),
        len(set(communities.values())),
    )

    selected = select_representatives(communities, degrees, n_mine, floor)
    selected_set = set(selected)
    assert selected_set <= pool_frames, "graph-mined frames must come from the pool"
    assert not (selected_set & baseline_frames), "graph-mined frames overlap the baseline"

    pd.DataFrame({"sample_data_token": selected}).to_parquet(
        state_dir / "graph.parquet", index=False
    )
    night_share = (
        float(
            pd.read_parquet(
                processed / "samples.parquet", columns=["sample_data_token", "is_night"]
            )
            .set_index("sample_data_token")
            .loc[selected]["is_night"]
            .mean()
        )
        if selected
        else 0.0
    )
    summary = {
        "n_graph": len(selected),
        "n_communities": len(set(communities.values())),
        "n_pool_connected": len(communities),
        "graph_night_share": night_share,
    }
    logger.info(
        "Graph-mined %d frames across %d communities (night share %.2f)",
        summary["n_graph"],
        summary["n_communities"],
        night_share,
    )
    return summary
