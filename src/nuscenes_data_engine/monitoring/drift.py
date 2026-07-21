"""Data-drift monitoring with Evidently.

Tracks input-image statistics (brightness, resolution, detection-count distribution)
against the training reference and flags drift.

Written against the locked Evidently 0.7.x API (`Report`/`Dataset`/`DataDefinition` +
`presets.DataDriftPreset`); most online examples show the older 0.4 or newer 0.8
syntax — don't "fix" the imports from those.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nuscenes_data_engine.monitoring.features import FEATURE_COLUMNS, load_feature_table

logger = logging.getLogger("nuscenes_data_engine")


def build_drift_report(reference: Path, current: Path, *, drift_share: float = 0.25) -> Any:
    """Build an Evidently drift report comparing current inputs to the reference.

    Args:
        reference: Feature table computed from the training data.
        current: Feature table computed from recent serving inputs (parquet or JSONL).
        drift_share: Fraction of drifted columns that flags dataset-level drift.

    Returns:
        An Evidently snapshot (renderable to HTML/JSON).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    ref_df, cur_df = load_feature_table(reference), load_feature_table(current)
    columns = [c for c in FEATURE_COLUMNS if c in ref_df.columns and c in cur_df.columns]
    usable = [c for c in columns if ref_df[c].notna().any() and cur_df[c].notna().any()]
    if dropped := sorted(set(columns) - set(usable)):
        logger.warning("Skipping all-NaN drift columns: %s", ", ".join(dropped))
    if not usable:
        raise ValueError("No usable feature columns shared by reference and current tables.")

    definition = DataDefinition(numerical_columns=usable)
    ref_ds = Dataset.from_pandas(ref_df[usable], data_definition=definition)
    cur_ds = Dataset.from_pandas(cur_df[usable], data_definition=definition)
    report = Report([DataDriftPreset(columns=usable, drift_share=drift_share)], include_tests=True)
    return report.run(cur_ds, ref_ds)


def summarize_drift(snapshot: Any) -> dict[str, Any]:
    """Reduce an Evidently snapshot to a compact machine-readable verdict."""
    data = snapshot.dict()
    columns: dict[str, dict[str, Any]] = {}
    n_drifted, share_drifted, dataset_drift = 0, 0.0, False

    for metric in data.get("metrics", []):
        name = str(metric.get("metric_id", metric.get("metric_name", "")))
        if name.startswith("ValueDrift(column="):
            column = name.split("column=")[1].split(",")[0].rstrip(")")
            columns.setdefault(column, {})["score"] = float(metric["value"])
        elif name.startswith("DriftedColumnsCount("):
            value = metric["value"]
            n_drifted, share_drifted = int(value["count"]), float(value["share"])

    for test in data.get("tests", []):
        # status is a str-Enum: == "FAIL" is True, but str() gives "TestStatus.FAIL".
        name, failed = str(test.get("name", "")), test.get("status") == "FAIL"
        if name.startswith("Value Drift for column "):
            column = name.removeprefix("Value Drift for column ")
            columns.setdefault(column, {})["drift_detected"] = failed
        elif name.startswith("Share of Drifted Columns"):
            dataset_drift = failed

    return {
        "columns": columns,
        "n_drifted": n_drifted,
        "share_drifted": share_drifted,
        "dataset_drift": dataset_drift,
    }


def save_drift_report(snapshot: Any, out_dir: Path) -> tuple[Path, Path]:
    """Write the HTML report + JSON summary into ``out_dir``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path, json_path = out_dir / "drift_report.html", out_dir / "drift_summary.json"
    # Evidently silently writes nothing when handed a pathlib.Path — must be str.
    snapshot.save_html(str(html_path))
    json_path.write_text(json.dumps(summarize_drift(snapshot), indent=2), encoding="utf-8")
    logger.info("Drift report: %s (summary: %s)", html_path, json_path)
    return html_path, json_path
