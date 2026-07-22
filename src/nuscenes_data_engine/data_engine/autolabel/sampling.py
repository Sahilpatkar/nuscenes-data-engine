"""Stratified frame sampling for the auto-labeling run.

Strategy (documented in docs/AUTOLABEL_EVAL.md): CAM_FRONT keyframes, filtered on the
availability manifest, stratified by location x is_night x is_rain. Allocation is
proportional with a per-stratum floor — pure proportional leaves the small night
strata too thin to evaluate (~100 frames), while a balanced design would consume
almost the entire night+rain stratum (642 frames total) and skew overall metrics.

Sampling is deterministic: strata are sorted by token before seeded sampling, so any
machine reproduces the same sample.parquet.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

STRATUM_COLUMNS = ["location", "is_night", "is_rain"]


def allocate(counts: Mapping[Any, int], total: int, floor: int) -> dict[Any, int]:
    """Per-stratum sample sizes: proportional with a floor, largest-remainder rounding.

    Strata smaller than the floor are taken (up to) whole; strata whose proportional
    share falls below the floor are pinned to it; the remainder is split
    proportionally among the rest.
    """
    population = sum(counts.values())
    if total >= population:
        return dict(counts)

    fixed: dict[Any, int] = {}
    active = dict(counts)
    remaining = total
    pinned = True
    while pinned and active:
        pinned = False
        n_active = sum(active.values())
        for key, n in list(active.items()):
            share = remaining * n / n_active
            target = min(floor, n)
            if share < target:
                fixed[key] = target
                remaining -= target
                del active[key]
                pinned = True
    if remaining < 0:
        raise ValueError(f"floor={floor} over {len(counts)} strata cannot fit in total={total}")

    n_active = sum(active.values())
    shares = {key: remaining * n / n_active for key, n in active.items()}
    alloc = {key: min(int(share), active[key]) for key, share in shares.items()}
    leftover = remaining - sum(alloc.values())
    for key in sorted(active, key=lambda k: shares[k] - int(shares[k]), reverse=True):
        if leftover <= 0:
            break
        if alloc[key] < active[key]:
            alloc[key] += 1
            leftover -= 1
    return {**fixed, **alloc}


def _stratified_draw(frames: pd.DataFrame, allocation: dict[Any, int], seed: int) -> pd.DataFrame:
    picked = []
    for key, n in allocation.items():
        stratum = frames[frames["_stratum"] == key].sort_values("sample_data_token")
        picked.append(stratum.sample(n=n, random_state=seed))
    return pd.concat(picked, ignore_index=True)


def build_sample(processed_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Draw the stratified labeling sample (+ the comparison-model subset)."""
    frames = pd.read_parquet(processed_dir / "samples.parquet")
    frames = frames[frames["channel"] == cfg.get("channel", "CAM_FRONT")]

    manifest_path = processed_dir / "availability.parquet"
    if manifest_path.is_file():
        manifest = pd.read_parquet(manifest_path, columns=["sample_data_token", "present"])
        frames = frames.merge(manifest[manifest["present"]], on="sample_data_token")
    else:
        logger.warning("No availability manifest at %s — trusting metadata paths", manifest_path)

    frames = frames.copy()
    frames["_stratum"] = list(zip(*(frames[c] for c in STRATUM_COLUMNS), strict=True))
    counts = frames["_stratum"].value_counts().to_dict()
    seed = int(cfg.get("seed", 6))

    allocation = allocate(counts, int(cfg.get("size", 5000)), int(cfg.get("min_per_stratum", 250)))
    sample = _stratified_draw(frames, allocation, seed)

    sub_counts = sample["_stratum"].value_counts().to_dict()
    sub_allocation = allocate(
        sub_counts, int(cfg.get("opus_subset", 500)), int(cfg.get("opus_min_per_stratum", 25))
    )
    subset_tokens = set(_stratified_draw(sample, sub_allocation, seed + 1)["sample_data_token"])
    sample["in_opus_subset"] = sample["sample_data_token"].isin(subset_tokens)

    sample["stratum"] = sample["_stratum"].map(
        lambda key: f"{key[0]}|{'night' if key[1] else 'day'}|{'rain' if key[2] else 'dry'}"
    )
    sample = sample.drop(columns=["_stratum"]).sort_values("sample_data_token", ignore_index=True)

    for stratum, group in sample.groupby("stratum"):
        logger.info(
            "%-38s %5d frames (%3d comparison-subset)",
            stratum,
            len(group),
            int(group["in_opus_subset"].sum()),
        )
    return sample
