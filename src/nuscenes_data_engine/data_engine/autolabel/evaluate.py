"""Evaluate VLM labels against nuScenes ground truth.

Caveats baked into the design (and documented in AUTOLABEL_EVAL.md): the GT condition
flags are scene-description-derived (binary, imperfect), and GT boxes include heavily
occluded objects a single camera frame cannot show — so count metrics are computed
both against all GT boxes and against a >=40%-visibility subset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.config import load_yaml
from nuscenes_data_engine.data_engine.autolabel.schema import COUNT_FIELDS, GT_COUNT_GROUPS

logger = logging.getLogger("nuscenes_data_engine")


def _prf(pred: pd.Series[bool], gt: pd.Series[bool]) -> dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def gt_counts(
    annotations_path: Path, tokens: list[str], visibility_min: str | None = None
) -> pd.DataFrame:
    """Per-frame GT counts in the 10-class eval taxonomy (zero-filled), indexed by token."""
    annotations = pd.read_parquet(
        annotations_path, columns=["sample_data_token", "category_name", "visibility_token"]
    )
    annotations = annotations[annotations["sample_data_token"].isin(tokens)]
    if visibility_min is not None:
        visibility = pd.to_numeric(annotations["visibility_token"], errors="coerce")
        annotations = annotations[visibility >= int(visibility_min)]
    annotations = annotations.assign(field=annotations["category_name"].map(GT_COUNT_GROUPS))
    annotations = annotations.dropna(subset=["field"])
    counts = annotations.groupby(["sample_data_token", "field"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=tokens, columns=list(COUNT_FIELDS), fill_value=0).fillna(0)
    return counts.astype(int)


def eval_flags(labels: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """time_of_day vs is_night and weather vs is_rain: accuracy + positive-class P/R/F1."""
    joined = labels.merge(
        sample[["sample_data_token", "is_night", "is_rain", "stratum"]], on="sample_data_token"
    )
    rows = []
    for scope, frame in [("overall", joined), *joined.groupby("stratum")]:
        night_pred = frame["time_of_day"] == "night"
        rain_pred = frame["weather"] == "rain"
        rows.append(
            {
                "scope": scope,
                "n": len(frame),
                "night_accuracy": float((night_pred == frame["is_night"]).mean()),
                **{f"night_{k}": v for k, v in _prf(night_pred, frame["is_night"]).items()},
                "dusk_dawn_share": float((frame["time_of_day"] == "dusk_dawn").mean()),
                "rain_accuracy": float((rain_pred == frame["is_rain"]).mean()),
                **{f"rain_{k}": v for k, v in _prf(rain_pred, frame["is_rain"]).items()},
                "overcast_share": float((frame["weather"] == "overcast").mean()),
                "fog_share": float((frame["weather"] == "fog").mean()),
            }
        )
    return pd.DataFrame(rows)


def eval_counts(labels: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Per-class count metrics: MAE, exact, within-1, presence precision/recall."""
    indexed = labels.set_index("sample_data_token")
    common = [token for token in indexed.index if token in gt.index]
    predictions, truth = indexed.loc[common], gt.loc[common]
    rows = []
    for field in COUNT_FIELDS:
        pred = predictions[field].astype(int)
        actual = truth[field]
        error = (pred - actual).abs()
        rows.append(
            {
                "class": field,
                "n": len(common),
                "gt_total": int(actual.sum()),
                "pred_total": int(pred.sum()),
                "mae": float(error.mean()),
                "exact_rate": float((error == 0).mean()),
                "within_1_rate": float((error <= 1).mean()),
                **{f"presence_{k}": v for k, v in _prf(pred > 0, actual > 0).items()},
            }
        )
    return pd.DataFrame(rows)


def eval_count_buckets(labels: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """MAE by GT-count bucket, pooled across classes — the 'crowds are hard' view."""
    indexed = labels.set_index("sample_data_token")
    common = [token for token in indexed.index if token in gt.index]
    pred = indexed.loc[common, list(COUNT_FIELDS)].astype(int).stack()
    actual = gt.loc[common].stack()
    frame = pd.DataFrame({"pred": pred, "gt": actual})
    frame["bucket"] = pd.cut(
        frame["gt"], bins=[-1, 0, 3, 9, float("inf")], labels=["0", "1-3", "4-9", "10+"]
    )
    grouped = frame.groupby("bucket", observed=True)
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "mae": (frame["pred"] - frame["gt"])
            .abs()
            .groupby(frame["bucket"], observed=True)
            .mean(),
        }
    ).reset_index()


def model_agreement(primary: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, float]:
    """How often the two models agree with each other on the shared subset."""
    joined = primary.merge(comparison, on="sample_data_token", suffixes=("_a", "_b"))
    if joined.empty:
        return {"n": 0}
    count_mae = pd.concat(
        [(joined[f"{f}_a"].astype(int) - joined[f"{f}_b"].astype(int)).abs() for f in COUNT_FIELDS]
    ).mean()
    return {
        "n": len(joined),
        "time_of_day_agreement": float((joined["time_of_day_a"] == joined["time_of_day_b"]).mean()),
        "weather_agreement": float((joined["weather_a"] == joined["weather_b"]).mean()),
        "count_mae_between_models": float(count_mae),
    }


def run_eval(config_path: Path, processed_dir: Path | None = None) -> dict[str, Any]:
    """Compute all metrics and write parquet tables + an eval summary fragment."""
    cfg = load_yaml(config_path)
    state_dir = Path(cfg.get("state", {}).get("dir", "data/autolabel"))
    out_dir = Path(cfg.get("eval", {}).get("out_dir", state_dir / "eval"))
    processed = processed_dir or Path("data/processed")
    visibility_min = cfg.get("eval", {}).get("visibility_min")

    sample = pd.read_parquet(state_dir / "sample.parquet")
    labels = pd.read_parquet(state_dir / "labels.parquet")
    ok = labels[labels["parse_status"] == "ok"]
    primary_model = cfg["models"]["primary"]
    comparison_model = cfg["models"]["comparison"]
    by_model = {model: frame for model, frame in ok.groupby("model")}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "parse_status": labels.groupby(["model", "parse_status"]).size().to_dict()
    }
    fragments = ["# Auto-label evaluation summary\n"]

    for model, frame in by_model.items():
        slug = str(model).replace("/", "__")  # model ids may contain slashes
        tokens = list(frame["sample_data_token"])
        flags = eval_flags(frame, sample)
        flags.to_parquet(out_dir / f"flags_{slug}.parquet", index=False)
        counts_all = eval_counts(frame, gt_counts(processed / "annotations.parquet", tokens))
        counts_vis = eval_counts(
            frame, gt_counts(processed / "annotations.parquet", tokens, visibility_min)
        )
        buckets = eval_count_buckets(frame, gt_counts(processed / "annotations.parquet", tokens))
        counts_all.to_parquet(out_dir / f"counts_all_{slug}.parquet", index=False)
        counts_vis.to_parquet(out_dir / f"counts_vis_{slug}.parquet", index=False)
        overall = flags[flags["scope"] == "overall"].iloc[0]
        summary[model] = {
            "n_ok": len(frame),
            "night_f1": overall["night_f1"],
            "rain_f1": overall["rain_f1"],
        }
        fragments += [
            f"\n## {model} (n={len(frame)})\n",
            f"- Night: acc {overall['night_accuracy']:.3f}, F1 {overall['night_f1']:.3f} "
            f"(dusk_dawn share {overall['dusk_dawn_share']:.3f})",
            f"- Rain: acc {overall['rain_accuracy']:.3f}, F1 {overall['rain_f1']:.3f}",
            "\nCounts vs all GT boxes:\n",
            counts_all.round(3).to_markdown(index=False),
            f"\nCounts vs GT boxes with visibility >= {visibility_min}:\n",
            counts_vis.round(3).to_markdown(index=False),
            "\nMAE by GT-count bucket (all classes pooled):\n",
            buckets.round(3).to_markdown(index=False),
        ]

    if primary_model in by_model and comparison_model in by_model:
        agreement = model_agreement(by_model[primary_model], by_model[comparison_model])
        summary["model_agreement"] = agreement
        fragments += ["\n## Haiku vs Opus agreement (shared subset)\n", str(agreement)]

    (out_dir / "eval_summary.md").write_text("\n".join(fragments) + "\n")
    logger.info("Eval written to %s (summary: eval_summary.md)", out_dir)
    return summary
