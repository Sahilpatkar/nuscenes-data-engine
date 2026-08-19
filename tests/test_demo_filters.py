"""Pure filter-logic tests for the Failure Explorer (no Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

APP_DEMO = Path(__file__).resolve().parents[1] / "app" / "demo"
sys.path.insert(0, str(APP_DEMO))

from filters import failure_flags, filter_frames  # noqa: E402


def _manifest() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_data_token": ["v0", "v1", "v2", "t0"],
        "split": ["val", "val", "val", "train_pool"],
        "is_night": [True, False, True, False],
        "is_rain": [False, True, False, False],
        "curation_buckets": [["night_failure"], ["day_failure"], ["clean_success"], ["weak_accepted"]],
        "n_preds_baseline": pd.array([2, 1, 0, None], dtype="Int64"),
    })


def _gt() -> pd.DataFrame:
    return pd.DataFrame({
        "annotation_token": ["a", "b", "c"],
        "sample_data_token": ["v0", "v0", "v1"],
        "category_group": ["pedestrian", "car", "car"],
        "size_bucket": ["small", "large", "medium"],
        "distance_to_ego_m": [4.0, 22.0, 9.0],
        "matched_baseline": pd.array([False, True, True], dtype="boolean"),
    })


def _preds() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_data_token": ["v0", "v1", "v1"],
        "model": ["baseline", "baseline", "baseline"],
        "category_group": ["car", "car", "car"],
        "conf": [0.9, 0.3, 0.9],
        "status": ["fp", "low_conf", "tp"],
    })


def test_failure_flags_per_model() -> None:
    flags = failure_flags(_manifest(), _gt(), _preds(), model="baseline")
    row = flags.set_index("sample_data_token")
    assert bool(row.loc["v0", "has_fn"]) and bool(row.loc["v0", "has_fp"])
    assert not bool(row.loc["v1", "has_fn"])
    assert bool(row.loc["v1", "has_low_conf"])
    assert bool(row.loc["v2", "clean"])          # 0 preds, no gt rows -> clean
    assert "t0" not in row.index                  # val only


def test_filter_frames_criteria() -> None:
    kwargs = dict(manifest=_manifest(), gt=_gt(), preds=_preds(), model="baseline")
    assert set(filter_frames(**kwargs, lighting="night")["sample_data_token"]) == {"v0", "v2"}
    assert set(filter_frames(**kwargs, rain=True)["sample_data_token"]) == {"v1"}
    assert set(filter_frames(**kwargs, category="pedestrian")["sample_data_token"]) == {"v0"}
    assert set(filter_frames(**kwargs, size_bucket="medium")["sample_data_token"]) == {"v1"}
    assert set(filter_frames(**kwargs, distance_range=(0.0, 10.0))["sample_data_token"]) == {"v0", "v1"}
    assert set(filter_frames(**kwargs, failure_type="has_fn")["sample_data_token"]) == {"v0"}
    assert set(filter_frames(**kwargs, failure_type="clean")["sample_data_token"]) == {"v2"}
    assert set(filter_frames(**kwargs, bucket="day_failure")["sample_data_token"]) == {"v1"}
    everything = filter_frames(**kwargs)
    assert set(everything["sample_data_token"]) == {"v0", "v1", "v2"}
