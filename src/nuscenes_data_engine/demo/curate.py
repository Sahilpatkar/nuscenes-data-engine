"""``demo curate`` — deterministic bucket selection into a frame manifest + rsync filelist.

Buckets sample deterministically (stable sort, explicit tie-breaks — score desc then
token asc; ascending for ``clean_success``; token asc alone where no score exists) from
the AL failure ledger (``failures.parquet``), hard-braking canbus events, the AL/weak
arms, and an injected semantic-search callable. No RNG anywhere — "seeded" curation is
satisfied by determinism. Buckets are merged into one manifest (deduping tokens that
land in more than one bucket) plus the rsync filelist that ``demo infer`` reads.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

# Failure-ledger columns copied onto the manifest for tokens that have them (i.e. the
# curated token also appears in failures.parquet); train_pool-only tokens get NaN.
_FAILURE_COLUMNS = (
    "scene_name",
    "is_night",
    "n_gt",
    "n_matched",
    "n_fn",
    "n_low_conf",
    "failure_score",
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required curation input missing: {path}")
    return path


def _top_tokens(
    df: pd.DataFrame, *, score_col: str, quota: int, ascending: bool = False
) -> list[str]:
    """Deterministic top-``quota`` tokens: sort by ``score_col`` (desc unless
    ``ascending``), tie-broken by ``sample_data_token`` ascending."""
    if quota <= 0 or df.empty:
        return []
    ordered = df.sort_values(
        [score_col, "sample_data_token"], ascending=[ascending, True], kind="stable"
    )
    return list(ordered["sample_data_token"].head(quota))


def run_curate(
    *,
    processed_dir: Path,
    al_dir: Path,
    staging_dir: Path,
    quotas: dict[str, int],
    al_arm: str,
    weak_arm: str,
    semantic_hits: Callable[[list[str]], dict[str, list[str]]],
    semantic_queries: list[str],
) -> pd.DataFrame:
    """Build the curated frame manifest and write it (+ the rsync filelist) to disk.

    Every returned token resolves to a ``filename`` via ``samples.parquet`` — a token
    that cannot be resolved raises ``ValueError`` naming it, since a manifest row that
    ``demo infer``/rsync cannot locate an image for is worse than not curating it.
    """
    processed_dir = Path(processed_dir)
    al_dir = Path(al_dir)
    staging_dir = Path(staging_dir)

    failures = pd.read_parquet(_require(al_dir / "failures.parquet"))
    samples = pd.read_parquet(_require(processed_dir / "samples.parquet"))
    canbus = pd.read_parquet(_require(processed_dir / "canbus.parquet"))
    al_arm_tokens = pd.read_parquet(_require(al_dir / f"{al_arm}.parquet"))
    weak_arm_tokens = pd.read_parquet(_require(al_dir / f"{weak_arm}.parquet"))
    accepted = pd.read_parquet(_require(al_dir / f"{weak_arm}_accepted.parquet"))

    simple_buckets: dict[str, list[str]] = {}

    simple_buckets["night_failure"] = _top_tokens(
        failures[failures["is_night"]],
        score_col="failure_score",
        quota=quotas.get("night_failure", 0),
    )
    simple_buckets["day_failure"] = _top_tokens(
        failures[~failures["is_night"]],
        score_col="failure_score",
        quota=quotas.get("day_failure", 0),
    )

    # Hard braking: canbus events (keyed by sample_token) joined to their CAM_FRONT
    # sample_data_token — no inherent "score", so tokens sort ascending alone.
    hard_quota = quotas.get("hard_braking", 0)
    if hard_quota > 0:
        cam_front = samples.loc[
            samples["channel"] == "CAM_FRONT", ["sample_token", "sample_data_token"]
        ]
        hard_braking_tokens = (
            canbus[canbus["is_hard_braking"]]
            .merge(cam_front, on="sample_token", how="inner")["sample_data_token"]
            .drop_duplicates()
            .sort_values(kind="stable")
        )
        simple_buckets["hard_braking"] = list(hard_braking_tokens.head(hard_quota))
    else:
        simple_buckets["hard_braking"] = []

    al_quota = quotas.get("al_selected", 0)
    al_pool = sorted(al_arm_tokens["sample_data_token"])
    simple_buckets["al_selected"] = al_pool[:al_quota] if al_quota > 0 else []

    # Weak supervision: accepted/rejected split of the weak arm's mined pool.
    accepted_tokens = set(accepted["sample_data_token"])
    weak_pool = sorted(weak_arm_tokens["sample_data_token"])
    weak_accepted_quota = quotas.get("weak_accepted", 0)
    weak_rejected_quota = quotas.get("weak_rejected", 0)
    accepted_sorted = [t for t in weak_pool if t in accepted_tokens]
    rejected_sorted = [t for t in weak_pool if t not in accepted_tokens]
    simple_buckets["weak_accepted"] = (
        accepted_sorted[:weak_accepted_quota] if weak_accepted_quota > 0 else []
    )
    simple_buckets["weak_rejected"] = (
        rejected_sorted[:weak_rejected_quota] if weak_rejected_quota > 0 else []
    )

    # Clean success: no false negatives, no low-conf hits, a non-trivial GT count —
    # lowest failure_score first (the "cleanest" frames), not highest.
    clean_mask = (
        (failures["n_fn"] == 0) & (failures["n_low_conf"] == 0) & (failures["n_gt"] >= 3)
    )
    simple_buckets["clean_success"] = _top_tokens(
        failures[clean_mask],
        score_col="failure_score",
        quota=quotas.get("clean_success", 0),
        ascending=True,
    )

    token_buckets: dict[str, set[str]] = {}
    for bucket_name, tokens in simple_buckets.items():
        for token in tokens:
            token_buckets.setdefault(token, set()).add(bucket_name)

    # Semantic: an injected callable returning {token: [bucket labels]} — the labels
    # come from the callable itself (not a fixed "semantic" name), sorted by token for
    # determinism (the callable's own dict order is not guaranteed).
    semantic_quota = quotas.get("semantic", 0)
    semantic_map = semantic_hits(semantic_queries) if semantic_quota > 0 else {}
    for token in sorted(semantic_map)[:semantic_quota]:
        token_buckets.setdefault(token, set()).update(semantic_map[token])

    val_tokens = set(failures["sample_data_token"])
    failure_lookup = failures.set_index("sample_data_token")
    filename_lookup = samples.set_index("sample_data_token")["filename"]

    rows: list[dict[str, Any]] = []
    for token in sorted(token_buckets):
        if token not in filename_lookup.index:
            raise ValueError(f"curation: token {token!r} has no filename in samples.parquet")
        row: dict[str, Any] = {
            "sample_data_token": token,
            "split": "val" if token in val_tokens else "train_pool",
            "curation_buckets": sorted(token_buckets[token]),
            "filename": filename_lookup.loc[token],
        }
        if token in failure_lookup.index:
            for col in _FAILURE_COLUMNS:
                row[col] = failure_lookup.loc[token, col]
        rows.append(row)

    manifest = pd.DataFrame(rows)

    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(staging_dir / "frame_manifest.parquet")
    filelist_text = "".join(f"{name}\n" for name in manifest["filename"])
    (staging_dir / "rsync_filelist.txt").write_text(filelist_text)

    logger.info(
        "demo curate: %d tokens (%d val, %d train_pool) -> %s",
        len(manifest),
        int((manifest["split"] == "val").sum()) if len(manifest) else 0,
        int((manifest["split"] == "train_pool").sum()) if len(manifest) else 0,
        staging_dir,
    )
    return manifest
