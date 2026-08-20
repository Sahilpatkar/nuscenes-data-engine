"""Pure filter-logic tests for the Failure Explorer (no Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DEMO = Path(__file__).resolve().parents[1] / "app" / "demo"
sys.path.insert(0, str(APP_DEMO))

from filters import (  # noqa: E402
    failure_counts,
    failure_flags,
    filter_frames,
    rank_events,
    sort_frames,
)


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


def test_failure_flags_na_matched_is_not_fn() -> None:
    """NA in matched_<model> means "not evaluated", never a miss -- pins the claim
    ``failure_flags``'s docstring makes (and that ``render.draw_overlay`` also
    depends on for its own NA-safe rendering) with a dedicated case, not just as
    an implicit side effect of the other fixtures never happening to hit NA."""
    manifest = pd.DataFrame({
        "sample_data_token": ["v9"],
        "split": ["val"],
        "is_night": [True],
        "is_rain": [False],
        "curation_buckets": [[]],
        "n_preds_baseline": pd.array([0], dtype="Int64"),
    })
    gt = pd.DataFrame({
        "annotation_token": ["z"],
        "sample_data_token": ["v9"],
        "category_group": ["car"],
        "size_bucket": ["small"],
        "distance_to_ego_m": [5.0],
        "matched_baseline": pd.array([pd.NA], dtype="boolean"),
    })
    preds = pd.DataFrame({
        "sample_data_token": pd.array([], dtype="object"),
        "model": pd.array([], dtype="object"),
        "category_group": pd.array([], dtype="object"),
        "conf": pd.array([], dtype="float64"),
        "status": pd.array([], dtype="object"),
    })
    flags = failure_flags(manifest, gt, preds, model="baseline")
    row = flags.set_index("sample_data_token").loc["v9"]
    assert not bool(row["has_fn"])
    assert bool(row["clean"])


def test_filter_frames_criteria() -> None:
    kwargs = dict(manifest=_manifest(), gt=_gt(), preds=_preds(), model="baseline")
    assert set(filter_frames(**kwargs, lighting="night")["sample_data_token"]) == {"v0", "v2"}
    assert set(filter_frames(**kwargs, rain=True)["sample_data_token"]) == {"v1"}
    assert set(filter_frames(**kwargs, category="pedestrian")["sample_data_token"]) == {"v0"}
    assert set(filter_frames(**kwargs, size_bucket="medium")["sample_data_token"]) == {"v1"}
    assert set(filter_frames(**kwargs, distance_range=(0.0, 10.0))["sample_data_token"]) == {"v0", "v1"}
    assert set(filter_frames(**kwargs, failure_type="has_fn")["sample_data_token"]) == {"v0"}
    assert set(filter_frames(**kwargs, failure_type="clean")["sample_data_token"]) == {"v2"}
    everything = filter_frames(**kwargs)
    assert set(everything["sample_data_token"]) == {"v0", "v1", "v2"}


def test_filter_frames_category_size_distance_intersect_on_one_box() -> None:
    """category/size_bucket/distance_range are conjunctive over the SAME GT row,
    not independently against possibly-different rows on the same frame -- the
    real bug the 81k-combination behavioral sweep caught: "pedestrian + 80-123m"
    was returning frames with some pedestrian and some distant box, zero of which
    actually had a pedestrian at that distance."""
    manifest = pd.DataFrame({
        "sample_data_token": ["v0", "v1"],
        "split": ["val", "val"],
        "is_night": [True, True],
        "is_rain": [False, False],
        "curation_buckets": [[], []],
        "n_preds_baseline": pd.array([0, 0], dtype="Int64"),
    })
    gt = pd.DataFrame({
        # v0: a NEAR pedestrian + a FAR car -- no single box is both a
        # pedestrian AND far, so v0 must NOT match "pedestrian + far".
        # v1: a FAR pedestrian -- one box satisfies both, so v1 must match.
        "annotation_token": ["a", "b", "c"],
        "sample_data_token": ["v0", "v0", "v1"],
        "category_group": ["pedestrian", "car", "pedestrian"],
        "size_bucket": ["small", "large", "small"],
        "distance_to_ego_m": [5.0, 90.0, 90.0],
        "matched_baseline": pd.array([True, True, True], dtype="boolean"),
    })
    preds = pd.DataFrame({
        "sample_data_token": pd.array([], dtype="object"),
        "model": pd.array([], dtype="object"),
        "category_group": pd.array([], dtype="object"),
        "conf": pd.array([], dtype="float64"),
        "status": pd.array([], dtype="object"),
    })
    out = filter_frames(
        manifest=manifest, gt=gt, preds=preds, model="baseline",
        category="pedestrian", distance_range=(80.0, 100.0),
    )
    assert set(out["sample_data_token"]) == {"v1"}


def test_filter_frames_buckets_any_overlap() -> None:
    kwargs = dict(manifest=_manifest(), gt=_gt(), preds=_preds(), model="baseline")
    # Single bucket -- same as the old singular `bucket` param's one case.
    assert set(filter_frames(**kwargs, buckets=["day_failure"])["sample_data_token"]) == {"v1"}
    # Multiple buckets: any-overlap semantics -- a frame matches if ANY of its own
    # curation_buckets appears in the requested list (this is what the sidebar's
    # multiselect needs -- OR across selections, not AND).
    assert set(
        filter_frames(**kwargs, buckets=["day_failure", "clean_success"])["sample_data_token"]
    ) == {"v1", "v2"}
    # Empty/None list is a no-op, same as leaving the multiselect untouched.
    assert set(filter_frames(**kwargs, buckets=[])["sample_data_token"]) == {"v0", "v1", "v2"}
    assert set(filter_frames(**kwargs, buckets=None)["sample_data_token"]) == {"v0", "v1", "v2"}


# --- sort_frames: dedicated fixtures with three mutually-distinguishing orderings
# (failure count desc / distance asc-NA-last / scene name alphabetical must each
# produce a DIFFERENT token order, or a broken key could pass by coincidence). ---


def _sort_manifest() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_data_token": ["v0", "v1", "v2"],
        "split": ["val", "val", "val"],
        "is_night": [True, True, True],
        "is_rain": [False, False, False],
        "curation_buckets": [[], [], []],
        "scene_name": ["scene-charlie", "scene-alpha", "scene-bravo"],
        "n_preds_baseline": pd.array([0, 0, 0], dtype="Int64"),
    })


def _sort_gt() -> pd.DataFrame:
    # v0: 1 FN, far away (20m). v1: 0 FN, nearest GT box (2m). v2: no GT rows at
    # all -- NA distance, must sort last regardless of key.
    return pd.DataFrame({
        "annotation_token": ["a", "b"],
        "sample_data_token": ["v0", "v1"],
        "category_group": ["car", "car"],
        "size_bucket": ["small", "small"],
        "distance_to_ego_m": [20.0, 2.0],
        "matched_baseline": pd.array([False, True], dtype="boolean"),
    })


def _sort_preds() -> pd.DataFrame:
    # v0: 2 FP (+ its 1 FN above = 3 total, the worst). v2: 1 low_conf (no GT
    # rows, so 1 total). v1: none (0 total, the cleanest).
    return pd.DataFrame({
        "sample_data_token": ["v0", "v0", "v2"],
        "model": ["baseline", "baseline", "baseline"],
        "category_group": ["car", "car", "car"],
        "conf": [0.9, 0.9, 0.3],
        "status": ["fp", "fp", "low_conf"],
    })


def _sort_kwargs() -> dict[str, object]:
    return dict(gt=_sort_gt(), preds=_sort_preds(), model="baseline")


def test_sort_frames_by_failure_count() -> None:
    frames = filter_frames(manifest=_sort_manifest(), gt=_sort_gt(), preds=_sort_preds(), model="baseline")
    out = sort_frames(frames, key="failure_count", **_sort_kwargs())
    assert list(out["sample_data_token"]) == ["v0", "v2", "v1"]   # 3, 1, 0 -- worst first


def test_sort_frames_by_distance() -> None:
    frames = filter_frames(manifest=_sort_manifest(), gt=_sort_gt(), preds=_sort_preds(), model="baseline")
    out = sort_frames(frames, key="distance", **_sort_kwargs())
    assert list(out["sample_data_token"]) == ["v1", "v0", "v2"]   # 2m, 20m, NA-last


def test_sort_frames_by_scene_name() -> None:
    frames = filter_frames(manifest=_sort_manifest(), gt=_sort_gt(), preds=_sort_preds(), model="baseline")
    out = sort_frames(frames, key="scene_name", **_sort_kwargs())
    assert list(out["sample_data_token"]) == ["v1", "v2", "v0"]   # alpha, bravo, charlie


def test_failure_counts_empty_tokens_has_object_dtype_column() -> None:
    """A zero-length token list (e.g. a filter combo with zero matching frames)
    must still produce a result with sample_data_token as object dtype, not
    float64 -- pandas defaults an empty Python list to float64 when building a
    DataFrame column from it, which then breaks sort_frames's merge against a
    real (always object-dtype) token column ("merge on object and float64
    columns"). Caught by the new empty-state page path (final-review round)."""
    counts = failure_counts(_gt(), _preds(), [], model="baseline")
    assert counts.empty
    assert counts["sample_data_token"].dtype == object


def test_sort_frames_on_an_empty_filter_result_does_not_raise() -> None:
    empty = filter_frames(
        manifest=_manifest(), gt=_gt(), preds=_preds(), model="baseline",
        failure_type="has_fp", lighting="day",   # has_fp -> only v0; day -> only v1; no overlap
    )
    assert empty.empty
    out = sort_frames(empty, gt=_gt(), preds=_preds(), model="baseline", key="failure_count")
    assert out.empty


def test_sort_frames_invalid_key_raises() -> None:
    frames = filter_frames(manifest=_sort_manifest(), gt=_sort_gt(), preds=_sort_preds(), model="baseline")
    with pytest.raises(ValueError, match="unknown key"):
        sort_frames(frames, key="bogus", **_sort_kwargs())


# --- rank_events: Scenario Search's pure filter/sort helper (Phase 5) --------------


def _events() -> pd.DataFrame:
    """Three events: two tagged `preset_a` (ranks 2, 1 -- deliberately NOT in row
    order, so a bug that just returns rows as-is instead of sorting by rank would
    still be caught), one untagged for `preset_a` (NA rank) but tagged `preset_b`.
    """
    return pd.DataFrame(
        {
            "sample_data_token": ["e0", "e1", "e2"],
            "preset_rank_preset_a": pd.array([2, pd.NA, 1], dtype="Int64"),
            "preset_rank_preset_b": pd.array([pd.NA, 1, pd.NA], dtype="Int64"),
        }
    )


def test_rank_events_tag_filter() -> None:
    """Only rows with a non-null preset_rank_<preset> come back -- e1 is untagged
    for preset_a (NA rank) and must be excluded even though it's tagged preset_b."""
    out = rank_events(_events(), "preset_a")
    assert set(out["sample_data_token"]) == {"e0", "e2"}
    assert "e1" not in set(out["sample_data_token"])


def test_rank_events_rank_order() -> None:
    """Sorted by the preset's own rank column ascending (1 = most severe first),
    not by input row order -- e2 (rank 1) must come before e0 (rank 2), even
    though e0 appears first in the input frame."""
    out = rank_events(_events(), "preset_a")
    assert list(out["sample_data_token"]) == ["e2", "e0"]

    out_b = rank_events(_events(), "preset_b")
    assert list(out_b["sample_data_token"]) == ["e1"]


def test_rank_events_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        rank_events(_events(), "not_a_real_preset")
