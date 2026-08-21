"""Pure filter-logic tests for the Failure Explorer (no Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DEMO = Path(__file__).resolve().parents[1] / "app" / "demo"
sys.path.insert(0, str(APP_DEMO))

from filters import (  # noqa: E402
    braking_caption,
    confidence_caption,
    distance_caption,
    failure_counts,
    failure_flags,
    filter_frames,
    graph_node_label,
    rank_events,
    severity_caption,
    sort_frames,
    speed_caption,
    subgraph_narrative,
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


# --- card/panel caption formatting (consolidated review, items 1 & 6) --------------


def test_braking_caption_negative_accel_shows_g_force() -> None:
    # -9.80665 m/s^2 == exactly 1.00g
    assert braking_caption(-9.80665) == "1.00g braking"
    assert braking_caption(-4.903325) == "0.50g braking"


def test_braking_caption_positive_or_zero_accel_is_not_braking() -> None:
    """27/126 real flagship-adjacent events have a POSITIVE accel_long_min_mps2
    (accelerating, not braking, at their most extreme longitudinal sample) --
    labeling that a braking-g figure misrepresented the frame (item 1)."""
    assert braking_caption(2.5) == "no braking data"
    assert braking_caption(0.0) == "no braking data"


def test_braking_caption_na_is_no_braking_data() -> None:
    assert braking_caption(float("nan")) == "no braking data"


def test_distance_speed_confidence_captions() -> None:
    assert distance_caption(7.25, label="ped") == "7.2m ped"
    assert distance_caption(float("nan"), label="ped") == "no ped"
    assert speed_caption(10.1) == "10.1 m/s"
    assert speed_caption(float("nan")) == "no speed"
    assert confidence_caption(0.234) == "0.23 conf"
    assert confidence_caption(float("nan")) == "no low-conf detection"


def test_severity_caption_uses_each_presets_own_ranking_quantity() -> None:
    """Cards must show the SAME quantity the preset ranks by -- not always
    min-pedestrian distance regardless of which preset produced the card (item 6).
    One row carrying every field a preset might read, so each assertion below
    proves the dispatcher picks the field that specific preset actually uses."""
    row = {
        "accel_long_min_mps2": -7.5,
        "min_dist_pedestrian_m": 5.0,
        "min_dist_cyclist_m": 3.0,
        "speed_mps": 12.0,
        "fn_ped_min_dist_m": 40.0,
        "low_conf_min_conf": 0.21,
    }
    assert severity_caption("hard_braking_near_pedestrians", row) == "0.76g braking"
    assert severity_caption("night_pedestrians", row) == "5.0m ped"
    assert severity_caption("fast_cyclists", row) == "12.0 m/s"
    assert severity_caption("rain_vru", row) == "3.0m VRU"   # nearer of ped(5)/cyc(3)
    assert severity_caption("fn_pedestrians_night", row) == "40.0m missed ped"
    assert severity_caption("low_conf_braking", row) == "0.21 conf"


def test_severity_caption_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        severity_caption("not_a_real_preset", {})


# --- subgraph_narrative (Phase 6, Task 3) ------------------------------------------
#
# subgraph_export.assemble_subgraph's {nodes, edges, path} shape (see
# tests/test_demo_subgraphs.py::test_event_subgraph_assembly_from_fake_records for
# the real node/meta contract) -- built by hand here, not via assemble_subgraph
# itself, since filters.py stays pure/backend-free (no nuscenes_data_engine import).


def _node(node_id: str, label: str, on_path: bool, meta: dict[str, object]) -> dict[str, object]:
    return {"id": node_id, "label": label, "group": label.lower(), "on_path": on_path, "meta": meta}


def _full_subgraph(
    *,
    scene_name: str | None = "scene-0123",
    accel: float | None = -4.5,
    is_hard_braking: bool = True,
    obj_group: str | None = "pedestrian",
    obj_distance: float | None = 3.8,
    include_scene: bool = True,
    include_ego: bool = True,
    include_object: bool = True,
) -> dict[str, object]:
    nodes = []
    path: list[list[str]] = []

    sample = _node("sample:smp1", "Sample", True, {"timestamp": 1000, "scene": scene_name})
    nodes.append(sample)

    scene_id = "scene:scn1"
    ego_id = "egopose:ego1"
    if include_scene:
        nodes.append(_node(scene_id, "Scene", True, {"name": scene_name, "location": "boston-seaport"}))
    if include_ego:
        nodes.append(
            _node(
                ego_id, "EgoPose", True,
                {"speed_mps": 12.0, "accel_long_min_mps2": accel, "is_hard_braking": is_hard_braking},
            )
        )
    # assemble_subgraph appends the [scene, sample, ego] backbone chain ONLY when
    # BOTH the Scene and the EgoPose exist (subgraph_export.py) -- with either
    # missing, path holds object chains alone. Emitting a backbone-shaped chain
    # here regardless was what hid item I3: the narrative read path[0][2] as the
    # EgoPose even when path[0] was really an object chain, and reported the
    # Category node as an EgoPose.
    if include_scene and include_ego:
        path.append([scene_id, "sample:smp1", ego_id])

    if include_object:
        obj_id = "object:ped1"
        cat_id = "category:human.pedestrian.adult"
        nodes.append(
            _node(
                obj_id, "ObjectObservation", True,
                {"category": "human.pedestrian.adult", "group": obj_group, "distance_to_ego_m": obj_distance},
            )
        )
        nodes.append(_node(cat_id, "Category", True, {"name": "human.pedestrian.adult", "group": obj_group}))
        path.append(["sample:smp1", obj_id, cat_id])

    return {"nodes": nodes, "edges": [], "path": path}


def test_subgraph_narrative_full_sentence() -> None:
    narrative = subgraph_narrative(_full_subgraph())
    assert narrative == "Scene scene-0123 → Sample → EgoPose (hard braking -4.50 m/s²) → pedestrian at 3.80 m"


def test_subgraph_narrative_not_hard_braking_omits_the_words() -> None:
    narrative = subgraph_narrative(_full_subgraph(accel=-2.0, is_hard_braking=False))
    assert "hard braking" not in narrative
    assert "EgoPose (-2.00 m/s²)" in narrative


def test_subgraph_narrative_missing_ego_degrades_gracefully() -> None:
    """No EgoPose means no backbone chain at all in assemble_subgraph's output --
    the Scene segment still has to survive that (each piece drops independently)."""
    narrative = subgraph_narrative(_full_subgraph(include_ego=False))
    assert "EgoPose" not in narrative
    assert narrative == "Scene scene-0123 → Sample → pedestrian at 3.80 m"


def test_subgraph_narrative_object_only_path_is_not_read_as_an_egopose() -> None:
    """(item I3) A subgraph with NO backbone chain -- only a [sample, object,
    category] chain, exactly what assemble_subgraph emits when the Scene or EgoPose
    is absent -- used to render as "Sample → EgoPose": the chain's third element
    (the Category node) was read as the EgoPose slot purely because it sat at
    path[0][2]. The backbone is now identified by its node LABEL, so this reads as
    what it is: a sample with a matching pedestrian and no ego pose."""
    subgraph = {
        "nodes": [
            _node("sample:s1", "Sample", True, {"timestamp": 1000}),
            _node(
                "object:p1", "ObjectObservation", True,
                {"category": "human.pedestrian.adult", "group": "pedestrian",
                 "distance_to_ego_m": 3.8},
            ),
            _node(
                "category:human.pedestrian.adult", "Category", True,
                {"name": "human.pedestrian.adult", "group": "pedestrian"},
            ),
        ],
        "edges": [],
        "path": [["sample:s1", "object:p1", "category:human.pedestrian.adult"]],
    }
    assert subgraph_narrative(subgraph) == "Sample → pedestrian at 3.80 m"


def test_subgraph_narrative_missing_scene_degrades_gracefully() -> None:
    narrative = subgraph_narrative(_full_subgraph(include_scene=False, scene_name=None))
    assert not narrative.startswith("Scene")
    assert narrative == "Sample → EgoPose (hard braking -4.50 m/s²) → pedestrian at 3.80 m"


def test_subgraph_narrative_no_matching_object_degrades_gracefully() -> None:
    narrative = subgraph_narrative(_full_subgraph(include_object=False))
    assert "at" not in narrative.split("EgoPose")[-1].split("m/s²)")[-1]
    assert narrative == "Scene scene-0123 → Sample → EgoPose (hard braking -4.50 m/s²)"


def test_subgraph_narrative_empty_subgraph_is_never_an_exception() -> None:
    assert subgraph_narrative({}) == "Sample"
    assert subgraph_narrative({"nodes": [], "edges": [], "path": []}) == "Sample"


# --- graph_node_label (Phase 6 fix round: the agraph panel's node labels) ----------
#
# The panel used to label every node "<type>: <first 10 chars of its id>", which
# says nothing for a hex token and actively misleads for a Category
# ("human.pedestrian.adult" and "human.pedestrian.construction_worker" BOTH rendered
# as "Category: human.pede...", duplicated labels on 30/126 events -- item I4).
# These labels carry the quantity a viewer actually reads the graph for.


def test_graph_node_label_object_observation_shows_category_tail_and_distance() -> None:
    node = _node(
        "object:abc123", "ObjectObservation", True,
        {"category": "human.pedestrian.adult", "group": "pedestrian", "distance_to_ego_m": 9.94},
    )
    assert graph_node_label(node) == "adult 9.9 m"
    car = _node(
        "object:def456", "ObjectObservation", False,
        {"category": "vehicle.car", "group": "car", "distance_to_ego_m": 5.28},
    )
    assert graph_node_label(car) == "car 5.3 m"


def test_graph_node_label_object_observation_without_a_numeric_distance() -> None:
    """A missing/NA/non-numeric distance drops the figure rather than rendering
    "nan m" or raising."""
    for distance in (None, float("nan"), "unknown"):
        node = _node(
            "object:abc123", "ObjectObservation", True,
            {"category": "human.pedestrian.adult", "distance_to_ego_m": distance},
        )
        assert graph_node_label(node) == "adult"
    assert graph_node_label(_node("object:abc123", "ObjectObservation", True, {})) == "object"


def test_graph_node_label_egopose_shows_signed_longitudinal_accel() -> None:
    braking = _node(
        "egopose:abc123", "EgoPose", True,
        {"accel_long_min_mps2": -4.52, "is_hard_braking": True, "speed_mps": 12.0},
    )
    assert graph_node_label(braking) == "ego -4.5 m/s²"
    accelerating = _node("egopose:abc123", "EgoPose", True, {"accel_long_min_mps2": 1.2})
    assert graph_node_label(accelerating) == "ego +1.2 m/s²"
    assert graph_node_label(_node("egopose:abc123", "EgoPose", True, {})) == "ego pose"
    assert graph_node_label(
        _node("egopose:abc123", "EgoPose", True, {"accel_long_min_mps2": None})
    ) == "ego pose"


def test_graph_node_label_scene_uses_its_name() -> None:
    assert graph_node_label(_node("scene:abc123", "Scene", True, {"name": "scene-1084"})) == "scene-1084"
    assert graph_node_label(_node("scene:abc123", "Scene", True, {})) == "scene"


def test_graph_node_label_sample_marks_the_event_frame() -> None:
    """Three Sample nodes share a subgraph (prev/current/next) -- the on_path one is
    THE event's frame, and the only one a viewer needs to pick out."""
    assert graph_node_label(_node("sample:abc123", "Sample", True, {})) == "sample (t)"
    assert graph_node_label(_node("sample:def456", "Sample", False, {})) == "sample"


def test_graph_node_label_category_and_location_keep_their_full_tail() -> None:
    """(item I4) The id tail IS the name for these two -- truncating its HEAD made
    every human.pedestrian.* category render identically."""
    adult = _node("category:human.pedestrian.adult", "Category", True, {})
    worker = _node("category:human.pedestrian.construction_worker", "Category", False, {})
    assert graph_node_label(adult) == "human.pedestrian.adult"
    assert graph_node_label(worker) == "human.pedestrian.construction_worker"
    assert graph_node_label(adult) != graph_node_label(worker)
    location = _node("location:singapore-hollandvillage", "Location", False, {})
    assert graph_node_label(location) == "singapore-hollandvillage"


def test_graph_node_label_unknown_type_falls_back_to_the_label() -> None:
    assert graph_node_label(_node("thing:abc123", "SomethingElse", False, {})) == "SomethingElse"
