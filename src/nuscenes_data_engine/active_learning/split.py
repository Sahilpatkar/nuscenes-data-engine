"""Baseline/pool scene split for the active-learning experiment.

The baseline trains on a seeded, night-stratified fraction of official-train scenes;
the remainder is the mining pool. Stratification matters: night scenes are 12% of the
corpus, and an unlucky uniform draw would skew the baseline's night exposure and
poison the mined-vs-random comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.config import load_yaml

logger = logging.getLogger("nuscenes_data_engine")


def build_split(processed_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Scene-level table: scene_name, role in {baseline, pool}, is_night, n_frames."""
    from nuscenes.utils.splits import create_splits_scenes

    channel = cfg.get("channel", "CAM_FRONT")
    samples = pd.read_parquet(
        processed_dir / "samples.parquet",
        columns=["sample_data_token", "scene_name", "channel", "is_night"],
    )
    samples = samples[samples["channel"] == channel]
    train_scenes = set(create_splits_scenes()["train"])
    samples = samples[samples["scene_name"].isin(train_scenes)]

    scenes = (
        samples.groupby("scene_name")
        .agg(is_night=("is_night", "first"), n_frames=("sample_data_token", "size"))
        .reset_index()
        .sort_values("scene_name", ignore_index=True)
    )

    frac = float(cfg.get("baseline_frac", 0.25))
    seed = int(cfg.get("seed", 64))
    picked: list[pd.DataFrame] = []
    for _, stratum in scenes.groupby("is_night"):
        n = max(1, round(len(stratum) * frac))
        picked.append(stratum.sort_values("scene_name").sample(n=n, random_state=seed))
    baseline_names = set(pd.concat(picked)["scene_name"])

    scenes["role"] = scenes["scene_name"].map(
        lambda name: "baseline" if name in baseline_names else "pool"
    )
    for role, group in scenes.groupby("role"):
        logger.info(
            "%-8s %3d scenes (%d night), %5d %s frames",
            role,
            len(group),
            int(group["is_night"].sum()),
            int(group["n_frames"].sum()),
            channel,
        )
    return scenes


def frames_for_scenes(processed_dir: Path, scenes: set[str], channel: str) -> set[str]:
    """All sample_data_tokens of the given scenes for one camera channel."""
    samples = pd.read_parquet(
        processed_dir / "samples.parquet",
        columns=["sample_data_token", "scene_name", "channel"],
    )
    mask = samples["scene_name"].isin(scenes) & (samples["channel"] == channel)
    return set(samples[mask]["sample_data_token"])


def run_split(config_path: Path, processed_dir: Path | None = None) -> dict[str, Any]:
    """Build and persist the scene split; return summary counts."""
    cfg = load_yaml(config_path)
    processed = processed_dir or Path("data/processed")
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    scenes = build_split(processed, cfg.get("split", {}))
    state_dir.mkdir(parents=True, exist_ok=True)
    scenes.to_parquet(state_dir / "split.parquet", index=False)
    counts = scenes.groupby("role")["n_frames"].sum().to_dict()
    logger.info("Split written to %s", state_dir / "split.parquet")
    return {"scenes": scenes.groupby("role").size().to_dict(), "frames": counts}
