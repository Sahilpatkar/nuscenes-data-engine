"""Graph-diversity active-learning acquisition (Phase 6e x 6d).

The graph-native alternative to embedding-KMeans mining (``active_learning/mining.py``):
use the Neo4j ``SIMILAR_TO`` graph's community structure instead of clustering raw
vectors. GDS Louvain over the *pool's* similarity subgraph yields appearance communities;
the labeling budget is allocated across communities — size-proportional via the shared
``allocate`` helper for the round-1 arm, or ∝ smoothed-rate failure mass via
``allocate_by_mass`` for the round-3 arms — and each community contributes its
most-connected (representative) frames. Selection is graph-native and A/B'd against the
mined/random arms on the shared val split — the acquisition function changes, the
random-control gate does not. Round 3 parameterizes the arm (``graph_mining.arms``):
``size`` weighting reproduces the round-1
selection; ``rate_mass`` allocates budget ∝ embedding-routed smoothed-rate failure mass with
a per-community floor and an optional night quota (spec
docs/superpowers/specs/2026-08-04-al-round-3-design.md).

The GDS steps run on the infra machine (where Neo4j lives); the resulting per-arm token set
(``graph.parquet`` for the legacy ``graph`` arm, ``<arm>.parquet`` for every other arm) feeds
``al run --arm <arm>`` on the GPU server, exactly like ``mined.parquet``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nuscenes_data_engine.active_learning.mining import (
    mine_candidates,
    read_vectors,
    score_failures,
)
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


def allocate_by_mass(
    masses: dict[int, float],
    capacities: dict[int, int],
    total: int,
    floors: dict[int, int],
) -> dict[int, int]:
    """Split ``total`` across communities ∝ mass, capped at capacity, with per-key floors.

    Pure/deterministic. Communities missing from ``masses`` weigh 0 and receive only
    their floor unless every active community weighs 0 (then spare capacity is the
    weight). Returns partial totals when capacity is exhausted — callers backfill.
    Masses must be non-negative (negative weights would produce negative shares and
    undercut floors).
    """
    quotas = {c: min(floors.get(c, 0), cap) for c, cap in capacities.items()}
    if sum(quotas.values()) > total:
        raise ValueError(
            f"floors sum to {sum(quotas.values())} > total {total} over {len(capacities)} communities"
        )
    remaining = total - sum(quotas.values())
    while remaining > 0:
        active = {c: capacities[c] - quotas[c] for c in capacities if capacities[c] > quotas[c]}
        if not active:
            break
        weights = {c: masses.get(c, 0.0) for c in active}
        if sum(weights.values()) <= 0.0:
            weights = {c: float(cap) for c, cap in active.items()}
        total_weight = sum(weights.values())
        shares = {c: remaining * weights[c] / total_weight for c in active}
        add = {c: min(int(shares[c]), active[c]) for c in active}
        leftover = remaining - sum(add.values())
        for c in sorted(active, key=lambda k: (-(shares[k] - int(shares[k])), k)):
            if leftover <= 0:
                break
            if add[c] < active[c]:
                add[c] += 1
                leftover -= 1
        for c, extra in add.items():
            quotas[c] += extra
        remaining = total - sum(quotas.values())
    return quotas


def select_by_mass(
    communities: dict[str, int],
    degrees: dict[str, float],
    masses: dict[int, float],
    n_mine: int,
    *,
    night_tokens: set[str] | None = None,
    night_floor: int = 0,
    night_pool: list[str] | None = None,
    seed: int = 64,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Mass-weighted community selection: optional night pass, then floor-1 main pass.

    Pure/DB-free and deterministic. Night pass allocates ``night_floor`` across
    communities ∝ mass over their *night* members (top-degree first, capacity-capped
    with automatic cross-community spill via the allocator); a shortfall backfills
    from ``night_pool`` (which may include disconnected frames) with a seeded draw.
    The main pass fills toward exactly ``n_mine`` (the caller's exact-count guard enforces
    it; connected capacity can under-fill) ∝ mass with a floor of 1 per community
    (communities already holding a night pick need no extra floor frame).

    The community=-1 diagnostics row counts only backfilled tokens outside every
    community; its night_members field repeats that count (sentinel semantics, not a
    community's night membership).
    """
    members: dict[int, list[str]] = defaultdict(list)
    for token, community in communities.items():
        members[community].append(token)
    ranked = {
        c: sorted(tokens, key=lambda t: (-degrees.get(t, 0.0), t))
        for c, tokens in members.items()
    }
    night = night_tokens or set()
    selected: list[str] = []
    picked: dict[int, int] = defaultdict(int)

    if night_floor > 0:
        night_ranked = {c: [t for t in ranked[c] if t in night] for c in ranked}
        night_caps = {c: len(tokens) for c, tokens in night_ranked.items()}
        night_quotas = allocate_by_mass(masses, night_caps, night_floor, floors={})
        for c in sorted(night_quotas):
            take = night_ranked[c][: night_quotas[c]]
            selected.extend(take)
            picked[c] += len(take)
        shortfall = night_floor - len(selected)
        if shortfall > 0:
            pool = sorted(set(night_pool or []) - set(selected))
            if len(pool) < shortfall:
                raise ValueError(
                    f"night floor {night_floor} exceeds available night pool "
                    f"({len(selected) + len(pool)} frames)"
                )
            rng = np.random.default_rng(seed)
            for token in rng.choice(pool, size=shortfall, replace=False):
                token = str(token)
                selected.append(token)
                if token in communities:
                    picked[communities[token]] += 1

    chosen = set(selected)
    capacities = {c: len([t for t in ranked[c] if t not in chosen]) for c in ranked}
    floors = {c: 0 if picked[c] > 0 or capacities[c] == 0 else 1 for c in ranked}
    quotas = allocate_by_mass(masses, capacities, n_mine - len(selected), floors)
    for c in sorted(quotas):
        take = [t for t in ranked[c] if t not in chosen][: quotas[c]]
        selected.extend(take)
        picked[c] += len(take)

    diagnostics = [
        {
            "community": c,
            "size": len(members[c]),
            "mass": round(masses.get(c, 0.0), 4),
            "night_members": len([t for t in members[c] if t in night]),
            "quota": picked[c],
        }
        for c in sorted(members)
    ]
    n_backfilled = len(selected) - sum(picked.values())
    if n_backfilled:
        diagnostics.append(
            {"community": -1, "size": n_backfilled, "mass": 0.0,
             "night_members": n_backfilled, "quota": n_backfilled}
        )
    return selected, diagnostics


def route_failure_mass(
    tbl: Any,
    failures: pd.DataFrame,
    communities: dict[str, int],
    pool_scenes: list[str],
    channel: str,
    top_k: int,
    route_k: int,
) -> dict[int, float]:
    """Distribute each top-``top_k`` rate-ranked failure's score to the communities of
    its ``route_k`` nearest pool frames (LanceDB, prefiltered). Neighbors outside any
    community (disconnected pool frames) are skipped. Failures without a stored
    embedding are skipped with a logged warning (they contribute no mass)."""
    scored = failures.assign(score=score_failures(failures, "smoothed_rate"))
    top = scored.sort_values(["score", "sample_data_token"], ascending=[False, True]).head(top_k)
    vectors = read_vectors(tbl, list(top["sample_data_token"]))
    vector_by_token = dict(
        zip(vectors["sample_data_token"], vectors["vector"], strict=True)
    )
    missing = [t for t in top["sample_data_token"] if t not in vector_by_token]
    if missing:
        logger.warning(
            "route_failure_mass: %d/%d top failures have no stored embedding and "
            "contribute no mass (first: %s)",
            len(missing), len(top), missing[0],
        )
    masses: dict[int, float] = defaultdict(float)
    for token, score in zip(top["sample_data_token"], top["score"], strict=True):
        vector = vector_by_token.get(token)
        if vector is None:
            continue
        neighbors = mine_candidates(
            tbl, np.asarray(vector, dtype=np.float32), pool_scenes, channel, route_k
        )
        for neighbor in neighbors["sample_data_token"]:
            community = communities.get(neighbor)
            if community is not None:
                masses[community] += float(score)
    return dict(masses)


def _louvain_communities(
    driver: Any, pool_tokens: list[str], database: str, seed: int = 64
) -> tuple[dict[str, int], dict[str, float]]:
    """Mark the pool, project its SIMILAR_TO subgraph, and stream Louvain + degree.

    ``concurrency: 1`` pins Louvain to single-threaded execution, removing the
    thread-join nondeterminism that otherwise makes the community partition vary
    run-to-run on an identical graph (round 3 finding — no ``randomSeed`` was set
    previously). ``randomSeed`` itself is deliberately *not* passed: this GDS build
    (2.13.11) rejects it for ``gds.louvain.stream`` with
    ``IllegalArgumentException: Unexpected configuration key: randomSeed``
    (verified empirically against the live instance). ``seed`` is accepted here for
    signature symmetry with ``select_by_mass``'s seed (the same resolved config
    value is threaded through both) and to embed in the config if a future GDS
    version restores support; it is otherwise unused today.
    """
    louvain_query = (
        f"CALL gds.louvain.stream('{_GRAPH}', {{concurrency: 1}}) "
        "YIELD nodeId, communityId "
        f"RETURN gds.util.asNode(nodeId).token AS token, communityId AS community"
    )
    connection.run_write_batches(
        driver, _MARK_POOL, [{"token": token} for token in pool_tokens], database=database
    )
    try:
        connection.write_query(driver, _DROP, database=database)  # clear any stale graph
        connection.write_query(driver, _PROJECT, database=database)
        communities = {
            row["token"]: int(row["community"])
            for row in connection.write_query(driver, louvain_query, database=database)
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


DEFAULT_GRAPH_ARMS: dict[str, dict[str, Any]] = {"graph": {"weighting": "size"}}
VALID_WEIGHTING = ("size", "rate_mass")


def run_graph_mining(
    config_path: Path, processed_dir: Path | None = None, arm: str = "graph"
) -> dict[str, Any]:
    """Community-detect the pool's similarity graph and write the arm's token set."""
    from nuscenes_data_engine.active_learning.split import frames_for_scenes
    from nuscenes_data_engine.data_engine import store

    cfg = load_yaml(config_path)
    settings = get_settings()
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    graph_cfg = cfg.get("graph_mining", {})
    arms_cfg = {**DEFAULT_GRAPH_ARMS, **graph_cfg.get("arms", {})}
    if arm not in arms_cfg:
        raise ValueError(f"Unknown graph-mining arm {arm!r} (expected one of {sorted(arms_cfg)})")
    arm_cfg = arms_cfg[arm] or {}
    weighting = arm_cfg.get("weighting")
    if weighting is None:
        raise ValueError(f"Graph-mining arm {arm!r} has no 'weighting' configured")
    if weighting not in VALID_WEIGHTING:
        raise ValueError(f"Unknown weighting {weighting!r} (expected one of {VALID_WEIGHTING})")
    quotas_cfg = {flag: int(v) for flag, v in (arm_cfg.get("quotas") or {}).items()}
    for flag in quotas_cfg:
        if flag != "is_night":
            raise ValueError(f"Unknown quota flag {flag!r} (round 3 supports is_night only)")
    if any(v < 0 for v in quotas_cfg.values()):
        raise ValueError(f"Quota floors must be >= 0, got {quotas_cfg}")

    n_mine = int(graph_cfg.get("n_mine", cfg.get("mining", {}).get("n_mine", 1500)))
    floor = int(graph_cfg.get("floor", 1))
    route_k = int(graph_cfg.get("route_k", 10))
    night_floor = quotas_cfg.get("is_night", 0)
    if night_floor >= n_mine:
        raise ValueError(
            f"Quota is_night={night_floor} must be < n_mine={n_mine} "
            "(the main pass needs room for community floors)"
        )
    database = settings.neo4j_database
    seed = int(graph_cfg.get("seed", cfg.get("mining", {}).get("seed", 64)))

    split = pd.read_parquet(state_dir / "split.parquet")
    baseline_frames = frames_for_scenes(
        processed, set(split[split["role"] == "baseline"]["scene_name"]), channel
    )
    pool_frames = frames_for_scenes(
        processed, set(split[split["role"] == "pool"]["scene_name"]), channel
    )
    samples = pd.read_parquet(
        processed / "samples.parquet",
        columns=["sample_data_token", "scene_name", "is_night"],
    ).set_index("sample_data_token")
    pool_meta = samples.loc[sorted(pool_frames)]
    if night_floor:
        stratum = int(pool_meta["is_night"].sum())
        if stratum < night_floor:
            raise ValueError(f"Quota is_night={night_floor} exceeds the stratum pool ({stratum} frames)")

    driver = connection.get_driver(settings)
    try:
        communities, degrees = _louvain_communities(driver, sorted(pool_frames), database, seed=seed)
    finally:
        connection.close(driver)
    logger.info(
        "Louvain: %d connected pool frames in %d communities (arm %s, %s weighting)",
        len(communities), len(set(communities.values())), arm, weighting,
    )

    diagnostics: list[dict[str, Any]] = []
    summary_extras: dict[str, Any] = {}
    if weighting == "size":
        selected = select_representatives(communities, degrees, n_mine, floor)
    else:
        engine_cfg = load_yaml(Path(cfg.get("engine_config", "configs/engine.yaml"))).get("lancedb", {})
        tbl = store.open_frames_table(
            Path(engine_cfg.get("path", "data/lancedb")), engine_cfg.get("table", "frames"), dim=0
        )
        failures = pd.read_parquet(state_dir / "failures.parquet")
        top_k = int(cfg.get("sweep", {}).get("top_k_failures", 1000))
        pool_scenes = sorted(split[split["role"] == "pool"]["scene_name"])
        masses = route_failure_mass(
            tbl, failures, communities, pool_scenes, channel, top_k, route_k
        )
        summary_extras = {
            "mass_total": round(float(sum(masses.values())), 4),
            "n_communities_with_mass": sum(1 for m in masses.values() if m > 0),
        }
        night_tokens = set(pool_meta.index[pool_meta["is_night"]])
        selected, diagnostics = select_by_mass(
            communities,
            degrees,
            masses,
            n_mine,
            night_tokens=night_tokens,
            night_floor=night_floor,
            night_pool=sorted(night_tokens),
            seed=seed,
        )

    selected_set = set(selected)
    if len(selected) != n_mine or len(selected_set) != n_mine:
        raise ValueError(f"selected {len(selected)} frames ({len(selected_set)} unique) != n_mine {n_mine}")
    if not selected_set <= pool_frames:
        raise ValueError("graph-mined frames must come from the pool")
    if selected_set & baseline_frames:
        raise ValueError("graph-mined frames overlap the baseline")
    selected_night = int(pool_meta.loc[selected]["is_night"].sum())
    if night_floor and selected_night < night_floor:
        raise ValueError(f"quota is_night: selected {selected_night} < floor {night_floor}")

    out_name = "graph.parquet" if arm == "graph" else f"{arm}.parquet"
    pd.DataFrame({"sample_data_token": selected}).to_parquet(state_dir / out_name, index=False)
    if diagnostics:
        (state_dir / f"communities_{arm}.json").write_text(
            pd.DataFrame(diagnostics).to_json(orient="records", indent=2)
        )

    night_share = float(pool_meta.loc[selected]["is_night"].mean()) if selected else 0.0
    summary = {
        "arm": arm,
        "weighting": weighting,
        "n_mined": len(selected),
        "n_communities": len(set(communities.values())),
        "n_pool_connected": len(communities),
        "mined_night_share": night_share,
        "n_scenes": int(pool_meta.loc[selected]["scene_name"].nunique()),
    }
    if arm == "graph":  # legacy summary keys round 1 consumers/log dashboards used
        summary["n_graph"] = summary["n_mined"]
        summary["graph_night_share"] = night_share
    summary.update(summary_extras)
    logger.info(
        "Arm %s: graph-mined %d frames across %d communities (night %.2f, %d scenes)",
        arm, len(selected), summary["n_communities"], night_share, summary["n_scenes"],
    )
    return summary
