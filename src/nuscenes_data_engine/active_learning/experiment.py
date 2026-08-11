"""Experiment arms: build the arm's dataset, train, evaluate, record results.

Arms: ``baseline`` (25% train scenes) plus baseline-and-extra-frames arms —
``mined``/``random``/``graph`` from round 1, ``rate``/``strat``/``rate_strat`` from
round 2 (docs/superpowers/specs/2026-08-02-al-round-2-design.md), and
``graph_rate``/``graph_rate_night`` from round 3
(docs/superpowers/specs/2026-08-04-al-round-3-design.md). Plus weak-supervision pairs
(see ``WEAK_ARMS``): each base arm gets a pseudo-labelled arm and a GT twin trained on
the SAME accepted frame set, isolating label quality from frame count
(docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md and
docs/superpowers/specs/2026-08-10-weak-supervise-night-champion-design.md). Each arm
gets its own YOLO dataset dir and run-name suffix; the val split is identical across
arms by construction and asserted at result-merge time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nuscenes_data_engine.config import get_settings, load_yaml

logger = logging.getLogger("nuscenes_data_engine")

ARMS = (
    "baseline", "mined", "random", "graph",
    "rate", "strat", "rate_strat",
    "graph_rate", "graph_rate_night",
    "weak_random", "weak_random_gt",
    "weak_graph_rate_night", "weak_graph_rate_night_gt",
)

# Weak-supervision arms -> (base arm whose accepted set they train on, uses pseudo
# labels). Each base gets exactly two arms: the pseudo one and its GT twin, which
# train on the SAME frames so a delta isolates label quality from frame count.
WEAK_ARMS: dict[str, tuple[str, bool]] = {
    "weak_random": ("random", True),
    "weak_random_gt": ("random", False),
    "weak_graph_rate_night": ("graph_rate_night", True),
    "weak_graph_rate_night_gt": ("graph_rate_night", False),
}

# Per-arm extra-frames parquet, relative to the AL state dir. Module-level (not just a
# local in resolve_arm_frames) so report.arm_composition can resolve the same file —
# the weak arms share their base arm's <base>_accepted.parquet, not f"{arm}.parquet".
ARM_EXTRA_FILE: dict[str, str] = {
    "mined": "mined.parquet",
    "random": "random.parquet",
    "graph": "graph.parquet",
    "rate": "rate.parquet",
    "strat": "strat.parquet",
    "rate_strat": "rate_strat.parquet",
    "graph_rate": "graph_rate.parquet",
    "graph_rate_night": "graph_rate_night.parquet",
    **{arm: f"{base}_accepted.parquet" for arm, (base, _) in WEAK_ARMS.items()},
}


def resolve_arm_frames(state_dir: Path, processed_dir: Path, cfg: dict[str, Any], arm: str) -> set[str]:
    """The train-frame token set for one arm."""
    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    split = pd.read_parquet(state_dir / "split.parquet")
    baseline = frames_for_scenes(
        processed_dir, set(split[split["role"] == "baseline"]["scene_name"]), channel
    )
    if arm == "baseline":
        return baseline
    extra = set(pd.read_parquet(state_dir / ARM_EXTRA_FILE[arm])["sample_data_token"])
    return baseline | extra


def overlay_train_config(base_config_path: Path, al_train: dict[str, Any]) -> dict[str, Any]:
    """train.yaml with the experiment's model/imgsz/epochs/batch overlaid."""
    cfg = load_yaml(base_config_path)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg["model"]["weights"] = al_train.get("model", cfg["model"].get("weights", "yolov8n.pt"))
    cfg["model"]["imgsz"] = int(al_train.get("imgsz", cfg["model"].get("imgsz", 640)))
    cfg["train"]["epochs"] = int(al_train.get("epochs", cfg["train"].get("epochs", 20)))
    cfg["train"]["batch"] = int(al_train.get("batch", cfg["train"].get("batch", 16)))
    return cfg


def merge_results(state_dir: Path, arm: str, record: dict[str, Any]) -> dict[str, Any]:
    """Add one arm's record to results.json, asserting val-set identity across arms."""
    results_path = state_dir / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text()) if results_path.is_file() else {}
    )
    for other, other_record in results.items():
        if other != arm and other_record.get("val_images") != record.get("val_images"):
            raise AssertionError(
                f"val split differs between arms {other} ({other_record.get('val_images')}) "
                f"and {arm} ({record.get('val_images')}) — comparison invalid"
            )
    results[arm] = record
    results_path.write_text(json.dumps(results, indent=2))
    return results


def run_arm(
    config_path: Path,
    *,
    arm: str,
    device: str | None = None,
    epochs: int | None = None,
    wandb_enabled: bool | None = None,
    force_rebuild: bool = False,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the arm's dataset, train, evaluate, and merge its results."""
    from nuscenes_data_engine.evaluation.evaluate import run_evaluation
    from nuscenes_data_engine.training.dataset import build_yolo_dataset, compute_data_version
    from nuscenes_data_engine.training.train import train_model

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r} (expected one of {ARMS})")
    cfg = load_yaml(config_path)
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    settings = get_settings()
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")

    tokens = resolve_arm_frames(state_dir, processed, cfg, arm)
    logger.info("Arm %s: %d train frames", arm, len(tokens))

    arm_dir = state_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    pseudo_labels = None
    pseudo_tokens = None
    base, uses_pseudo = WEAK_ARMS.get(arm, ("", False))
    if uses_pseudo:
        pseudo_path = state_dir / f"{base}_pseudo_labels.parquet"
        if not pseudo_path.is_file():
            raise ValueError(
                f"Arm {arm!r} needs {pseudo_path} — run `al pseudo-label --arm {base}` first"
            )
        pseudo_labels = pd.read_parquet(pseudo_path)
        # Every accepted frame is pseudo-labelled, including those the detector found
        # nothing in — those have no rows in the table, so the token set comes from
        # accepted.parquet or their ground truth would survive into a "no GT" arm.
        pseudo_tokens = set(
            pd.read_parquet(state_dir / f"{base}_accepted.parquet")["sample_data_token"]
        )

    data_yaml, stats = build_yolo_dataset(
        processed,
        Path(settings.nuscenes_dataroot),
        arm_dir / "yolo",
        cameras=[channel],
        train_frames=tokens,
        pseudo_labels=pseudo_labels,
        pseudo_tokens=pseudo_tokens,
        force=force_rebuild,
    )

    train_cfg = overlay_train_config(
        Path(cfg.get("train_base_config", "configs/train.yaml")), cfg.get("train", {})
    )
    if epochs is not None:
        train_cfg["train"]["epochs"] = epochs
    train_cfg["data"] = {"processed_dir": str(processed), "yolo_dir": str(arm_dir / "yolo")}
    arm_train_config = arm_dir / "train_config.yaml"
    arm_train_config.write_text(yaml.safe_dump(train_cfg))

    summary = train_model(
        train_cfg,
        data_yaml,
        data_version=compute_data_version(processed),
        device=device,
        wandb_enabled=wandb_enabled,
        run_suffix=f"al-{arm}",
    )

    report = run_evaluation(
        Path(cfg.get("eval", {}).get("config", "configs/eval.yaml")),
        arm_train_config,
        weights=Path(summary["best_weights"]),
        device=device or "0",
        register=False,
    )
    record = {
        "n_train_images": stats.get("train_images"),
        "val_images": stats.get("val_images"),
        "n_boxes": stats.get("boxes"),
        "run_name": summary["run_name"],
        "best_weights": str(summary["best_weights"]),
        "overall": report["overall"],
        "night": report["slices"].get("time_of_day/night", {}),
        "slices": {k: {m: v[m] for m in ("mAP50", "mAP50-95")} for k, v in report["slices"].items()},
    }
    results = merge_results(state_dir, arm, record)
    logger.info(
        "Arm %s: overall mAP50-95 %.3f, night mAP50-95 %.3f",
        arm,
        record["overall"]["mAP50-95"],
        record["night"].get("mAP50-95", float("nan")),
    )
    return results
