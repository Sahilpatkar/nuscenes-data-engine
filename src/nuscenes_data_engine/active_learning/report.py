"""Render the arm-comparison report for docs/ACTIVE_LEARNING.md."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.active_learning.experiment import ARMS as ARM_ORDER
from nuscenes_data_engine.config import load_yaml

logger = logging.getLogger("nuscenes_data_engine")


def arm_composition(state_dir: Path, processed_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-arm mined-set diagnostics: scene spread + night/rain share."""
    samples_path = processed_dir / "samples.parquet"
    if not samples_path.is_file():
        return {}
    samples = pd.read_parquet(
        samples_path, columns=["sample_data_token", "scene_name", "is_night", "is_rain"]
    ).set_index("sample_data_token")
    composition: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        path = state_dir / f"{arm}.parquet"
        if arm == "baseline" or not path.is_file():
            continue
        tokens = pd.read_parquet(path, columns=["sample_data_token"])["sample_data_token"]
        rows = samples.reindex(tokens).dropna(subset=["scene_name"])
        if len(rows) < len(tokens):
            logger.warning(
                "Arm %s: only %d/%d mined tokens matched samples.parquet; "
                "composition shares cover the matched subset",
                arm, len(rows), len(tokens),
            )
        if rows.empty:
            continue
        composition[arm] = {
            "n_scenes": int(rows["scene_name"].nunique()),
            "night_share": round(float(rows["is_night"].mean()), 3),
            "rain_share": round(float(rows["is_rain"].mean()), 3),
        }
    return composition


def render_report(
    results: dict[str, Any],
    clusters: pd.DataFrame | None,
    composition: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Markdown comparison of the arms (+ cluster diagnostics when available)."""
    rows = []
    baseline = results.get("baseline", {})
    base_overall = baseline.get("overall", {}).get("mAP50-95")
    base_night = baseline.get("night", {}).get("mAP50-95")
    for arm in ARM_ORDER:
        record = results.get(arm)
        if record is None:
            continue
        overall = record["overall"].get("mAP50-95")
        night = record["night"].get("mAP50-95")
        comp = (composition or {}).get(arm, {})
        rows.append(
            {
                "arm": arm,
                "train_images": record.get("n_train_images"),
                "overall_mAP50": round(record["overall"].get("mAP50", float("nan")), 4),
                "overall_mAP50_95": round(overall, 4) if overall is not None else None,
                "night_mAP50": round(record["night"].get("mAP50", float("nan")), 4),
                "night_mAP50_95": round(night, 4) if night is not None else None,
                "d_overall": f"{overall - base_overall:+.4f}"
                if arm != "baseline" and overall is not None and base_overall is not None
                else "",
                "d_night": f"{night - base_night:+.4f}"
                if arm != "baseline" and night is not None and base_night is not None
                else "",
                "n_scenes": comp.get("n_scenes", ""),
                "night_share": comp.get("night_share", ""),
                "rain_share": comp.get("rain_share", ""),
            }
        )
    fragments = [
        "# Active-learning experiment report\n",
        pd.DataFrame(rows).to_markdown(index=False),
    ]
    if clusters is not None and not clusters.empty:
        fragments += ["\n\n## Failure clusters\n", clusters.round(3).to_markdown(index=False)]
    return "\n".join(fragments) + "\n"


def run_report(config_path: Path, processed_dir: Path | None = None) -> str:
    """Render + persist the report; return the markdown."""
    cfg = load_yaml(config_path)
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    results = json.loads((state_dir / "results.json").read_text())
    clusters_path = state_dir / "clusters.parquet"
    clusters = pd.read_parquet(clusters_path) if clusters_path.is_file() else None
    composition = arm_composition(state_dir, processed_dir or Path("data/processed"))
    markdown = render_report(results, clusters, composition)
    (state_dir / "report.md").write_text(markdown)
    logger.info("\n%s", markdown)
    return markdown
