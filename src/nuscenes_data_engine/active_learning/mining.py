"""Cluster baseline failures in embedding space and mine similar frames from the pool.

The failure frames' SigLIP vectors are k-means clustered (interpretable failure
modes); each cluster centroid then queries the LanceDB store, prefiltered to the
train-scene pool, and per-cluster quotas proportional to cluster size select exactly
``n_mine`` frames. A seeded uniform draw from the same pool forms the random control.
Leakage guards assert that no mined/random frame is a baseline or val frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nuscenes_data_engine.config import load_yaml
from nuscenes_data_engine.data_engine import store
from nuscenes_data_engine.data_engine.autolabel.sampling import allocate

logger = logging.getLogger("nuscenes_data_engine")


def read_vectors(tbl: Any, tokens: list[str], chunk: int = 500) -> pd.DataFrame:
    """Bulk-read stored vectors for a token list (chunked SQL IN filters)."""
    parts = []
    for start in range(0, len(tokens), chunk):
        subset = tokens[start : start + chunk]
        quoted = ", ".join(f"'{token}'" for token in subset)
        parts.append(
            tbl.search()
            .where(f"sample_data_token IN ({quoted})")
            .select(["sample_data_token", "vector"])
            .limit(len(subset))
            .to_pandas()
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mine_candidates(
    tbl: Any,
    centroid: np.ndarray[Any, Any],
    pool_scenes: list[str],
    channel: str,
    k: int,
) -> pd.DataFrame:
    """Nearest pool frames to one failure-cluster centroid (prefiltered vector search)."""
    scene_list = ", ".join(f"'{scene}'" for scene in pool_scenes)
    normalized = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    return (
        tbl.search(normalized.tolist())
        .metric("cosine")
        .where(f"channel = '{channel}' AND scene_name IN ({scene_list})", prefilter=True)
        .limit(k)
        .select(["sample_data_token", "scene_name", "is_night", "_distance"])
        .to_pandas()
    )


def run_mining(config_path: Path, processed_dir: Path | None = None) -> dict[str, Any]:
    """Cluster failures, mine the pool, and write mined/random token sets."""
    from sklearn.cluster import KMeans

    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    cfg = load_yaml(config_path)
    mining_cfg = cfg.get("mining", {})
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    seed = int(mining_cfg.get("seed", 64))
    n_mine = int(mining_cfg.get("n_mine", 1500))
    n_clusters = int(mining_cfg.get("n_clusters", 8))

    split = pd.read_parquet(state_dir / "split.parquet")
    pool_scenes = sorted(split[split["role"] == "pool"]["scene_name"])
    baseline_frames = frames_for_scenes(
        processed, set(split[split["role"] == "baseline"]["scene_name"]), channel
    )
    pool_frames = frames_for_scenes(processed, set(pool_scenes), channel)

    failures = pd.read_parquet(state_dir / "failures.parquet")
    top_k = int(cfg.get("sweep", {}).get("top_k_failures", 1000))
    failure_tokens = list(
        failures.sort_values(["failure_score", "sample_data_token"], ascending=[False, True])
        .head(top_k)["sample_data_token"]
    )

    engine_cfg = load_yaml(Path(cfg.get("engine_config", "configs/engine.yaml"))).get("lancedb", {})
    tbl = store.open_frames_table(
        Path(engine_cfg.get("path", "data/lancedb")), engine_cfg.get("table", "frames"), dim=0
    )
    vectors_df = read_vectors(tbl, failure_tokens)
    matrix = np.asarray(vectors_df["vector"].tolist(), dtype=np.float32)
    logger.info("Clustering %d failure vectors into %d clusters", len(matrix), n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(matrix)
    labels = kmeans.labels_

    failure_meta = failures.set_index("sample_data_token").loc[
        list(vectors_df["sample_data_token"])
    ]
    cluster_rows = []
    for cluster_id in range(n_clusters):
        members = failure_meta[labels == cluster_id]
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "size": len(members),
                "night_share": float(members["is_night"].mean()) if len(members) else 0.0,
                "mean_failure_score": float(members["failure_score"].mean())
                if len(members)
                else 0.0,
            }
        )
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_parquet(state_dir / "clusters.parquet", index=False)
    (state_dir / "cluster_summary.json").write_text(clusters.to_json(orient="records", indent=2))
    logger.info("Cluster diagnostics:\n%s", clusters.round(3).to_string(index=False))

    quotas = allocate(
        {row["cluster"]: max(int(row["size"]), 1) for row in cluster_rows}, n_mine, floor=0
    )
    overfetch = int(mining_cfg.get("overfetch", 4))
    mined: dict[str, dict[str, Any]] = {}
    ranked_spare: list[dict[str, Any]] = []
    for cluster_id in sorted(quotas, key=lambda c: -quotas[c]):
        quota = quotas[cluster_id]
        candidates = mine_candidates(
            tbl, kmeans.cluster_centers_[cluster_id], pool_scenes, channel, quota * overfetch
        )
        taken = 0
        for candidate in candidates.to_dict("records"):
            token = candidate["sample_data_token"]
            record = {**candidate, "cluster": cluster_id}
            if token in mined:
                continue
            if taken < quota:
                mined[token] = record
                taken += 1
            else:
                ranked_spare.append(record)
    for record in sorted(ranked_spare, key=lambda r: r["_distance"]):
        if len(mined) >= n_mine:
            break
        mined.setdefault(record["sample_data_token"], record)

    mined_df = pd.DataFrame(list(mined.values())).head(n_mine)
    mined_tokens = set(mined_df["sample_data_token"])
    assert mined_tokens <= pool_frames, "mined frames must come from the pool"
    assert not (mined_tokens & baseline_frames), "mined frames overlap the baseline"

    rng = np.random.default_rng(seed)
    random_tokens = rng.choice(sorted(pool_frames), size=n_mine, replace=False)
    random_df = pd.DataFrame({"sample_data_token": random_tokens})

    mined_df.to_parquet(state_dir / "mined.parquet", index=False)
    random_df.to_parquet(state_dir / "random.parquet", index=False)
    night_share = float(
        pd.read_parquet(processed / "samples.parquet", columns=["sample_data_token", "is_night"])
        .set_index("sample_data_token")
        .loc[list(mined_tokens)]["is_night"]
        .mean()
    )
    summary = {
        "n_mined": len(mined_df),
        "n_random": len(random_df),
        "mined_night_share": night_share,
        "clusters": cluster_rows,
    }
    logger.info("Mined %d frames (night share %.2f); random control %d frames",
                len(mined_df), night_share, len(random_df))
    return summary
