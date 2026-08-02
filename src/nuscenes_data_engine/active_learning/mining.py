"""Cluster baseline failures in embedding space and mine similar frames from the pool.

The failure frames' SigLIP vectors are k-means clustered (interpretable failure
modes); each cluster centroid then queries the LanceDB store, prefiltered to the
train-scene pool, and per-cluster quotas proportional to cluster size select exactly
``n_mine`` frames. Round 2 parameterizes the pipeline per arm (``mining.arms``):
the ranking score (absolute round-1 ``failure_score`` or a smoothed failure rate)
and optional night/rain quota floors filled by stratum-prefiltered passes with
seeded random backfill. A seeded uniform draw from the same pool forms the random
control (written only by the ``mined`` arm — the round-1 control is never
regenerated). Leakage guards assert that no mined/random frame is a baseline or
val frame.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nuscenes_data_engine.config import load_yaml
from nuscenes_data_engine.data_engine import store
from nuscenes_data_engine.data_engine.autolabel.sampling import allocate

logger = logging.getLogger("nuscenes_data_engine")

VALID_SCORING = ("absolute", "smoothed_rate")


def score_failures(failures: pd.DataFrame, scoring: str) -> pd.Series:
    """Acquisition ranking: round-1 absolute score or the round-2 smoothed failure rate."""
    if scoring == "absolute":
        return failures["failure_score"].astype(float)
    if scoring == "smoothed_rate":
        return (failures["n_fn"] + 0.5 * failures["n_low_conf"]) / (failures["n_gt"] + 0.5)
    raise ValueError(f"Unknown scoring {scoring!r} (expected one of {VALID_SCORING})")


def quota_shortfall(records: Iterable[dict[str, Any]], flag: str, floor: int) -> int:
    """Frames still needed to meet a stratum floor; overlap counts toward every flag."""
    return max(floor - sum(1 for record in records if record.get(flag)), 0)


DEFAULT_ARMS: dict[str, dict[str, Any]] = {"mined": {"scoring": "absolute"}}


def _centroid_pass(
    tbl: Any,
    centers: np.ndarray[Any, Any],
    quotas: dict[int, int],
    pool_scenes: list[str],
    channel: str,
    overfetch: int,
    mined: dict[str, dict[str, Any]],
    extra_filter: str | None = None,
) -> list[dict[str, Any]]:
    """One quota-allocation pass over the centroids; mutates ``mined``; returns spares."""
    spare: list[dict[str, Any]] = []
    for cluster_id in sorted(quotas, key=lambda c: -quotas[c]):
        quota = quotas[cluster_id]
        if quota <= 0:
            continue
        candidates = mine_candidates(
            tbl, centers[cluster_id], pool_scenes, channel, quota * overfetch, extra_filter
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
                spare.append(record)
    return spare


def _pool_record(token: str, pool_meta: pd.DataFrame) -> dict[str, Any]:
    """A mined-set record for a frame taken from the pool without a centroid hit."""
    row = pool_meta.loc[token]
    return {
        "sample_data_token": token,
        "scene_name": row["scene_name"],
        "is_night": bool(row["is_night"]),
        "is_rain": bool(row["is_rain"]),
        "_distance": float("nan"),
        "cluster": -1,
    }


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
    extra_filter: str | None = None,
) -> pd.DataFrame:
    """Nearest pool frames to one failure-cluster centroid (prefiltered vector search)."""
    scene_list = ", ".join(f"'{scene}'" for scene in pool_scenes)
    where = f"channel = '{channel}' AND scene_name IN ({scene_list})"
    if extra_filter is not None:
        where += f" AND {extra_filter}"
    normalized = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    return (
        tbl.search(normalized.tolist())
        .metric("cosine")
        .where(where, prefilter=True)
        .limit(k)
        .select(["sample_data_token", "scene_name", "is_night", "is_rain", "_distance"])
        .to_pandas()
    )


def run_mining(
    config_path: Path, processed_dir: Path | None = None, arm: str = "mined"
) -> dict[str, Any]:
    """Cluster failures, mine the pool, and write the arm's token set (+ random control)."""
    from sklearn.cluster import KMeans

    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    cfg = load_yaml(config_path)
    mining_cfg = cfg.get("mining", {})
    arms_cfg = {**DEFAULT_ARMS, **mining_cfg.get("arms", {})}
    if arm not in arms_cfg:
        raise ValueError(f"Unknown mining arm {arm!r} (expected one of {sorted(arms_cfg)})")
    arm_cfg = arms_cfg[arm] or {}
    scoring = arm_cfg.get("scoring", "absolute")
    quotas_cfg = {flag: int(floor) for flag, floor in (arm_cfg.get("quotas") or {}).items()}

    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    seed = int(mining_cfg.get("seed", 64))
    n_mine = int(mining_cfg.get("n_mine", 1500))
    n_clusters = int(mining_cfg.get("n_clusters", 8))
    overfetch = int(mining_cfg.get("overfetch", 4))
    for flag, floor in quotas_cfg.items():
        if floor > n_mine:
            raise ValueError(f"Quota {flag}={floor} exceeds n_mine={n_mine}")

    split = pd.read_parquet(state_dir / "split.parquet")
    pool_scenes = sorted(split[split["role"] == "pool"]["scene_name"])
    baseline_frames = frames_for_scenes(
        processed, set(split[split["role"] == "baseline"]["scene_name"]), channel
    )
    pool_frames = frames_for_scenes(processed, set(pool_scenes), channel)
    samples = pd.read_parquet(
        processed / "samples.parquet",
        columns=["sample_data_token", "scene_name", "is_night", "is_rain"],
    ).set_index("sample_data_token")
    pool_meta = samples.loc[sorted(pool_frames)]
    for flag, floor in quotas_cfg.items():
        stratum_size = int(pool_meta[flag].sum())
        if stratum_size < floor:
            raise ValueError(
                f"Quota {flag}={floor} exceeds the stratum pool ({stratum_size} frames)"
            )

    failures = pd.read_parquet(state_dir / "failures.parquet")
    failures = failures.assign(score=score_failures(failures, scoring))
    top_k = int(cfg.get("sweep", {}).get("top_k_failures", 1000))
    failure_tokens = list(
        failures.sort_values(["score", "sample_data_token"], ascending=[False, True])
        .head(top_k)["sample_data_token"]
    )

    engine_cfg = load_yaml(Path(cfg.get("engine_config", "configs/engine.yaml"))).get("lancedb", {})
    tbl = store.open_frames_table(
        Path(engine_cfg.get("path", "data/lancedb")), engine_cfg.get("table", "frames"), dim=0
    )
    vectors_df = read_vectors(tbl, failure_tokens)
    matrix = np.asarray(vectors_df["vector"].tolist(), dtype=np.float32)
    logger.info(
        "Arm %s (%s scoring): clustering %d failure vectors into %d clusters",
        arm, scoring, len(matrix), n_clusters,
    )

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
    suffix = "" if arm == "mined" else f"_{arm}"
    clusters.to_parquet(state_dir / f"clusters{suffix}.parquet", index=False)
    (state_dir / f"cluster_summary{suffix}.json").write_text(
        clusters.to_json(orient="records", indent=2)
    )
    logger.info("Cluster diagnostics:\n%s", clusters.round(3).to_string(index=False))

    sizes = {row["cluster"]: max(int(row["size"]), 1) for row in cluster_rows}
    rng = np.random.default_rng(seed)
    mined: dict[str, dict[str, Any]] = {}

    # Quota passes in config order: stratum-prefiltered centroid mining, spares next,
    # then a seeded random draw from the stratum pool if candidates ran dry.
    for flag, floor in quotas_cfg.items():
        need = quota_shortfall(mined.values(), flag, floor)
        if need == 0:
            continue
        spare = _centroid_pass(
            tbl, kmeans.cluster_centers_, allocate(sizes, need, floor=0),
            pool_scenes, channel, overfetch, mined, extra_filter=f"{flag} = true",
        )
        for record in sorted(spare, key=lambda r: r["_distance"]):
            if quota_shortfall(mined.values(), flag, floor) == 0:
                break
            mined.setdefault(record["sample_data_token"], record)
        short = quota_shortfall(mined.values(), flag, floor)
        if short:
            stratum = sorted(t for t in pool_meta.index[pool_meta[flag]] if t not in mined)
            for token in rng.choice(stratum, size=short, replace=False):
                mined[str(token)] = _pool_record(str(token), pool_meta)
            logger.info("Quota %s: random-backfilled %d stratum frames", flag, short)

    # Unconstrained fill to exactly n_mine (the round-1 flow).
    remaining = max(n_mine - len(mined), 0)
    spare = _centroid_pass(
        tbl, kmeans.cluster_centers_, allocate(sizes, remaining, floor=0),
        pool_scenes, channel, overfetch, mined,
    )
    for record in sorted(spare, key=lambda r: r["_distance"]):
        if len(mined) >= n_mine:
            break
        mined.setdefault(record["sample_data_token"], record)
    if len(mined) < n_mine:
        leftovers = sorted(t for t in pool_frames if t not in mined)
        for token in rng.choice(leftovers, size=n_mine - len(mined), replace=False):
            mined[str(token)] = _pool_record(str(token), pool_meta)

    mined_df = pd.DataFrame(list(mined.values())).head(n_mine)
    mined_tokens = set(mined_df["sample_data_token"])
    assert len(mined_df) == n_mine, f"selected {len(mined_df)} != n_mine {n_mine}"
    assert mined_tokens <= pool_frames, "mined frames must come from the pool"
    assert not (mined_tokens & baseline_frames), "mined frames overlap the baseline"
    for flag, floor in quotas_cfg.items():
        n_flag = int(mined_df[flag].sum())
        assert n_flag >= floor, f"quota {flag}: selected {n_flag} < floor {floor}"

    out_name = "mined.parquet" if arm == "mined" else f"{arm}.parquet"
    mined_df.to_parquet(state_dir / out_name, index=False)

    summary: dict[str, Any] = {
        "arm": arm,
        "scoring": scoring,
        "n_mined": len(mined_df),
        "mined_night_share": float(mined_df["is_night"].mean()),
        "mined_rain_share": float(mined_df["is_rain"].mean()),
        "n_scenes": int(mined_df["scene_name"].nunique()),
        "clusters": cluster_rows,
    }
    if arm == "mined":
        random_tokens = rng.choice(sorted(pool_frames), size=n_mine, replace=False)
        random_df = pd.DataFrame({"sample_data_token": random_tokens})
        random_df.to_parquet(state_dir / "random.parquet", index=False)
        summary["n_random"] = len(random_df)
    logger.info(
        "Arm %s: mined %d frames (night %.2f, rain %.2f, %d scenes)",
        arm, len(mined_df), summary["mined_night_share"], summary["mined_rain_share"],
        summary["n_scenes"],
    )
    return summary
