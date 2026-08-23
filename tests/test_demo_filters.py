"""Pure filter-logic tests for the Failure Explorer (no Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DEMO = Path(__file__).resolve().parents[1] / "app" / "demo"
sys.path.insert(0, str(APP_DEMO))

from filters import (  # noqa: E402
    FILMSTRIP_STEPS,
    FilmstripStep,
    braking_caption,
    community_jump,
    confidence_caption,
    crowding_long,
    distance_caption,
    failure_counts,
    failure_flags,
    filmstrip_steps,
    filter_frames,
    fixed_boxes,
    frame_caption,
    graph_node_label,
    loss_long,
    model_label,
    parity_caption,
    parity_short,
    rank_events,
    replay_tool_counts,
    selection_factors,
    severity_caption,
    sort_frames,
    speed_caption,
    step_detail,
    subgraph_narrative,
    tour_frame_candidates,
    verdict_line,
    weak_frame_summary,
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


def test_parity_caption_pluralizes_keyframe_on_count() -> None:
    """(item 6, Phase 6 follow-up review) "keyframe" singular only for
    sql_count == 1; plural for every other count, including 0."""
    assert parity_caption(1, 1, True, 1) == "1 matching keyframe dataset-wide — Cypher 1 · SQL 1 ✓"
    assert (
        parity_caption(5, 5, True, 1)
        == "5 matching keyframes dataset-wide — Cypher 5 · SQL 5 ✓ · showing the top 1"
    )
    assert parity_caption(0, 0, True, 0) == "0 matching keyframes dataset-wide — Cypher 0 · SQL 0 ✓"


def test_parity_caption_omits_showing_the_top_clause_when_nothing_was_cut() -> None:
    """The " · showing the top N" clause appears only when the grid actually
    truncated the full population (n_shown < sql_count) -- suppressed when
    n_shown == sql_count (nothing cut) and, defensively, when n_shown somehow
    exceeds sql_count too."""
    assert parity_caption(30, 30, True, 30) == "30 matching keyframes dataset-wide — Cypher 30 · SQL 30 ✓"
    assert (
        parity_caption(786, 786, True, 30)
        == "786 matching keyframes dataset-wide — Cypher 786 · SQL 786 ✓ · showing the top 30"
    )


def test_parity_caption_mismatch_text_replaces_the_symbol() -> None:
    """A recorded mismatch (parity is False) folds "✗ mismatch recorded" into the
    string ahead of the "showing the top" clause, rather than drawing a ✓/✗
    symbol (item M4, Phase 6 review: a finding, not small print)."""
    assert (
        parity_caption(5, 4, False, 1)
        == "5 matching keyframes dataset-wide — Cypher 4 · SQL 5 ✗ mismatch recorded · showing the top 1"
    )
    assert (
        parity_caption(1, 2, False, 1)
        == "1 matching keyframe dataset-wide — Cypher 2 · SQL 1 ✗ mismatch recorded"
    )


def test_parity_caption_parity_none_draws_no_symbol() -> None:
    """parity is None (never reached for a dynamics preset in practice -- model
    presets never call this) draws no ✓/✗ symbol at all, rather than being
    called a mismatch by default."""
    assert parity_caption(3, None, None, 1) == "3 matching keyframes dataset-wide — Cypher None · SQL 3 · showing the top 1"


# --- Phase 7 (Task 4): Active Learning page helpers ------------------------------


def _exemplar_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One val frame's visible/invisible GT boxes and both models' predictions.

    Five GT boxes, one per case ``fixed_boxes`` has to decide:
      g1  baseline never claims it, the arm detects it (tp 0.61)      -> a row
      g2  baseline only claims it at low confidence (0.22), arm tp    -> a row
      g3  baseline already detects it (tp)                            -> excluded
      g4  the arm itself only claims it at low confidence             -> excluded
      g5  the arm never claims it at all                              -> excluded
      g6  the arm detects it, but the box is below the visibility floor -> excluded
    """
    gt = pd.DataFrame({
        "annotation_token": ["g1", "g2", "g3", "g4", "g5", "g6"],
        "category_group": ["pedestrian", "car", "car", "truck", "bus", "pedestrian"],
        "distance_to_ego_m": [26.3, 11.2, 5.0, 40.0, 60.0, 8.0],
        "below_visibility_min": [False, False, False, False, False, True],
    })
    preds = pd.DataFrame({
        "model": [
            "baseline", "graph_rate_night",
            "baseline", "graph_rate_night",
            "baseline", "graph_rate_night",
            "baseline", "graph_rate_night",
            "baseline",
            "graph_rate_night",
        ],
        "status": [
            "fp", "tp",              # g1: baseline has no matched claim at all
            "low_conf", "tp",        # g2
            "tp", "tp",              # g3
            "low_conf", "low_conf",  # g4
            "low_conf",              # g5: arm never claims it
            "tp",                    # g6: below the visibility floor
        ],
        "conf": [0.51, 0.61, 0.22, 0.58, 0.9, 0.92, 0.11, 0.19, 0.15, 0.77],
        "matched_annotation_token": [
            None, "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g6",
        ],
    })
    return gt, preds


def test_fixed_boxes_rows_baseline_none_or_low_conf_to_arm_tp() -> None:
    """Only a real upgrade is a row: the arm detects the box (``tp``) where the
    baseline either never claimed it ("none") or only claimed it below the
    confidence floor ("low-conf 0.22"). A box both models detect, a box the arm
    itself is unsure about, and a box below the visibility floor are all excluded
    -- the page must never present those as "fixed"."""
    gt, preds = _exemplar_frame()

    table = fixed_boxes(gt, preds, baseline="baseline", arm="graph_rate_night")

    assert list(table.columns) == [
        "annotation_token", "category_group", "distance_to_ego_m",
        "baseline_claim", "arm_claim",
    ]
    assert list(table["annotation_token"]) == ["g2", "g1"]         # sorted by distance
    assert list(table["distance_to_ego_m"]) == [11.2, 26.3]
    assert list(table["category_group"]) == ["car", "pedestrian"]
    assert list(table["baseline_claim"]) == ["low-conf 0.22", "none"]
    assert list(table["arm_claim"]) == ["0.58", "0.61"]


def test_fixed_boxes_is_empty_when_nothing_was_upgraded() -> None:
    """A frame where the baseline already caught everything yields an empty table
    with the same columns -- the page renders "no upgraded boxes", not a crash."""
    gt, preds = _exemplar_frame()
    # Give the baseline a confident claim on every box the arm claims (not just a
    # status flip: g1's baseline row carries no matched_annotation_token at all, so
    # flipping its status would leave g1 unclaimed and still an upgrade).
    arm_rows = preds.loc[preds["model"] == "graph_rate_night"]
    caught = pd.concat([preds, arm_rows.assign(model="baseline", status="tp")], ignore_index=True)

    table = fixed_boxes(gt, caught, baseline="baseline", arm="graph_rate_night")

    assert table.empty
    assert "arm_claim" in table.columns


def _communities() -> pd.DataFrame:
    return pd.DataFrame({
        "community": [-1, 10301, 205, 77],
        "size": [12, 916, 916, 40],
        "night_members": [12, 916, 300, 40],
        "mass": [0.0, 395.19, 500.0, 12.0],
        "quota_graph_rate": [0, 83, 120, 9],
        "quota_graph_rate_night": [0, 323, 90, 30],
        "is_backfill": [True, False, False, False],
    })


def test_community_jump_finds_largest_all_night_community() -> None:
    """The caption's number: the LARGEST community whose members are all night
    frames, and what the night floor did to its quota. Community 205 is heavier
    (mass 500 vs 395) and the same size but only 300/916 night members (not
    all-night); community 77 IS all-night but smaller; the backfill sentinel (-1) is
    all-night by construction and is never a community.
    """
    jump = community_jump(
        _communities(), before="quota_graph_rate", after="quota_graph_rate_night"
    )

    assert jump is not None
    assert jump["community"] == 10301
    assert jump["size"] == 916
    assert jump["quota_before"] == 83
    assert jump["quota_after"] == 323
    assert jump["ratio"] == pytest.approx(323 / 83)


def test_community_jump_returns_none_without_an_all_night_community() -> None:
    mixed = _communities()
    real = mixed["community"] >= 0
    mixed.loc[real, "night_members"] = mixed.loc[real, "size"] - 1

    assert community_jump(
        mixed, before="quota_graph_rate", after="quota_graph_rate_night"
    ) is None


def _explain_row() -> dict[str, object]:
    return {
        "sample_data_token": "t0",
        "arm": "graph_rate_night",
        "is_night": True,
        "scene_name": "scene-1071",
        "community": 10301,
        "community_size": 916,
        "community_night_members": 916,
        "community_mass": 395.19,
        "community_mass_rank": 2,
        "community_quota": 323,
        "degree": 41.0,
        "degree_rank_in_community": 3,
        "pick_pass": "night",
        "n_failures_routed": 0,
        "mass_routed": 0.0,
    }


def test_selection_factors_from_explain_row() -> None:
    """The "why was this frame selected?" panel, in the order the mechanism runs:
    night, community, that community's failure mass and quota, which pass took the
    frame, its rank inside the community -- and, LAST and explicitly not the
    reason, whether any failure mass was routed to this frame itself."""
    factors = selection_factors(_explain_row(), n_communities=97, night_floor=375)

    assert factors == [
        ("Night frame", "yes", True),
        ("Community", "#10301 · 916 frames · 916 at night", None),
        ("Community failure mass", "395.19 (rank 2 of 97)", None),
        ("Community quota", "323 frames", None),
        ("Picked in", "night pass (night floor 375)", None),
        ("Similarity-degree rank", "3rd of 916", None),
        ("Routed failures", "none — not itself a routing target", None),
    ]


def test_selection_factors_zero_mass_community_names_the_floor() -> None:
    """A community with no routed failure mass still mines one frame -- that quota
    came from select_by_mass's per-community FLOOR, not from mass. Printed as a
    bare "1 frame" under "Community failure mass: 0.00 (rank 81 of 97)" it reads as
    a quota the mass bought (consolidated review I3: 6 of the 78 gallery frames sit
    in zero-mass communities).
    """
    row = {
        **_explain_row(),
        "community_mass": 0.0,
        "community_mass_rank": 81,
        "community_quota": 1,
        "pick_pass": "main",
    }
    factors = dict((label, value) for label, value, _flag in selection_factors(row, n_communities=97))

    assert factors["Community failure mass"] == "0.00 (rank 81 of 97)"
    assert factors["Community quota"] == (
        "1 frame (per-community floor — this community drew no routed failure mass)"
    )
    # a community that DID draw mass still just states its quota
    assert dict(
        (label, value) for label, value, _flag in selection_factors(
            {**row, "community_mass": 12.5}, n_communities=97
        )
    )["Community quota"] == "1 frame"


def test_ordinal_suffixes_including_the_teens() -> None:
    """1st/2nd/3rd/4th, the 11-13 exception, and the same exception a century up."""
    from filters import _ordinal

    assert [_ordinal(n) for n in (1, 2, 3, 4)] == ["1st", "2nd", "3rd", "4th"]
    assert [_ordinal(n) for n in (11, 12, 13)] == ["11th", "12th", "13th"]
    assert [_ordinal(n) for n in (21, 101, 111)] == ["21st", "101st", "111th"]


def test_visible_gt_and_gt_for_render_are_shared_by_both_phase_7_pages() -> None:
    """One frame's GT rows, visibility-floor rows dropped, and the ``matched``
    column draw_overlay reads -- shared helpers now, not a copy per page
    (consolidated review M6).

    ``model=None`` (the Weak Supervision page: train-pool frames nothing evaluated)
    yields an all-NA ``matched``, which draw_overlay renders as plain GT rather than
    as misses.
    """
    from filters import gt_for_render, visible_gt

    gt = pd.DataFrame({
        "sample_data_token": ["v0", "v0", "v1"],
        "annotation_token": ["a1", "a2", "a3"],
        "below_visibility_min": [False, True, False],
        "matched_baseline": pd.array([True, False, True], dtype="boolean"),
    })

    rows = visible_gt(gt, "v0")
    assert list(rows["annotation_token"]) == ["a1"]        # a2 is under the floor
    # an older package without the column keeps every row rather than raising
    assert len(visible_gt(gt.drop(columns=["below_visibility_min"]), "v0")) == 2

    assert list(gt_for_render(rows, "baseline")["matched"]) == [True]
    unevaluated = gt_for_render(rows)
    assert unevaluated["matched"].dtype == "boolean"
    assert unevaluated["matched"].isna().all()
    # a model this package has no column for is "not evaluated", not "missed"
    assert gt_for_render(rows, "champion")["matched"].isna().all()


def test_selection_factors_main_pass_day_frame_with_routed_mass() -> None:
    """The other side of every branch: a day frame taken by the main pass, with
    failure mass actually routed to it (the sparse case -- 251 of the 1500 selected
    frames), and no night floor known (the explain group's validation JSON absent)."""
    row = {
        **_explain_row(),
        "is_night": False,
        "pick_pass": "main",
        "degree_rank_in_community": 21,
        "n_failures_routed": 4,
        "mass_routed": 12.34,
    }

    factors = dict((label, value) for label, value, _flag in selection_factors(row, n_communities=97))

    assert factors["Night frame"] == "no"
    assert factors["Picked in"] == "main pass"
    assert factors["Similarity-degree rank"] == "21st of 916"
    assert factors["Routed failures"] == "4 failures, mass 12.3"


def test_selection_factors_backfill_and_missing_ranks() -> None:
    """A frame the night backfill drew from outside every community: community -1,
    NA mass/degree ranks. The panel says so rather than printing "<NA>"."""
    row = {
        **_explain_row(),
        "community": -1,
        "community_size": 12,
        "community_night_members": 12,
        "community_mass": 0.0,
        "community_mass_rank": pd.NA,
        "community_quota": 0,
        "degree": float("nan"),
        "degree_rank_in_community": pd.NA,
        "pick_pass": "backfill",
        "n_failures_routed": 1,
        "mass_routed": 0.5,
    }

    factors = dict((label, value) for label, value, _flag in selection_factors(row, n_communities=97))

    assert factors["Picked in"] == "seeded backfill"
    assert factors["Community failure mass"] == "0.00 (rank n/a of 97)"
    assert factors["Similarity-degree rank"] == "n/a"
    assert factors["Routed failures"] == "1 failure, mass 0.5"


def test_model_label_names_the_champion_checkpoint_honestly() -> None:
    """"champion" is the yolov8m @960 checkpoint from the Phase-3 model comparison,
    not an active-learning arm and not the best AL result (that is `graph`,
    +0.0344 overall) -- wherever it appears it is labelled as the model-size
    champion it actually is. Every other model keeps its own name."""
    assert model_label("champion") == "champion (yolov8m @960)"
    assert model_label("baseline") == "baseline"
    assert model_label("graph_rate_night") == "graph_rate_night"


def test_model_label_distinguishes_the_two_weak_supervision_arms() -> None:
    """Phase 9b puts both weak-supervision checkpoints on screen at once, and their
    raw names differ by a bare ``_gt`` suffix -- which is precisely the distinction
    the page exists to make. Each carries what it was trained on (pseudo labels vs
    the human-labelled control) and the architecture, so the two are never confused
    for one another, nor for the yolov8m ``champion``."""
    assert model_label("weak_graph_rate_night") == (
        "weak_graph_rate_night (pseudo labels, yolov8n)"
    )
    assert model_label("weak_graph_rate_night_gt") == (
        "weak_graph_rate_night_gt (GT-labelled twin, yolov8n)"
    )
    assert model_label("some_unlabelled_model") == "some_unlabelled_model"


# --- Phase 7 (Task 5): Weak Supervision page helpers ------------------------------


def _vlm_rows(n: int) -> pd.DataFrame:
    """``n`` of one token's weak_labels.parquet rows."""
    return pd.DataFrame({
        "sample_data_token": ["wA"] * n,
        "category_group": ["car"] * n,
        "x_min": [50.0] * n, "y_min": [50.0] * n,
        "x_max": [150.0] * n, "y_max": [150.0] * n,
        "score": [0.73] * n,
    })


def _counts_row() -> dict[str, object]:
    """One vlm_counts.parquet row (exporters.export_vlm_counts' own columns)."""
    return {
        # label_confidence is the VLM's own STRING enum ("high"/"low"), never a
        # float (consolidated review C1).
        "sample_data_token": "wA", "parse_status": "ok", "label_confidence": "high",
        "vlm_time_of_day": "night", "vlm_weather": "clear",
        "vlm_car": 2.0, "vlm_truck": 0.0, "vlm_bus": 0.0,
        "vlm_pedestrian": 1.0, "vlm_bicycle": 0.0,
        "gt_car": 2, "gt_truck": 0, "gt_bus": 1, "gt_pedestrian": 1, "gt_bicycle": 0,
    }


def test_weak_frame_summary_flags_mutual_zero_and_counts() -> None:
    """The verdict badge and the VLM-vs-GT count table for one curated frame.

    "accepted" with zero pseudo boxes is the MUTUAL-ZERO case (the verifier
    agreed with the detector that the frame holds none of the five classes), a
    materially different claim from "accepted, here are its boxes" -- and
    "rejected" never has boxes at all, by construction.
    """
    accepted = weak_frame_summary("accepted", _vlm_rows(2), _counts_row())
    assert accepted["mutual_zero"] is False
    assert accepted["verdict_label"] == "accepted — 2 pseudo boxes"
    assert weak_frame_summary("accepted", _vlm_rows(1), _counts_row())["verdict_label"] == (
        "accepted — 1 pseudo box"
    )

    table = accepted["table"]
    assert list(table.columns) == ["class", "vlm_count", "gt_count"]
    assert list(table["class"]) == ["car", "truck", "bus", "pedestrian", "bicycle"]
    assert list(table["vlm_count"]) == ["2", "0", "0", "1", "0"]
    assert list(table["gt_count"]) == ["2", "0", "1", "1", "0"]

    mutual = weak_frame_summary("accepted", _vlm_rows(0), _counts_row())
    assert mutual["mutual_zero"] is True
    assert mutual["verdict_label"] == "accepted — 0 pseudo boxes (mutual zero)"

    rejected = weak_frame_summary("rejected", _vlm_rows(0), _counts_row())
    assert rejected["mutual_zero"] is False
    assert rejected["verdict_label"] == "rejected — no pseudo boxes by construction"

    # A curated frame with no vlm_counts row at all (the VLM never labelled it):
    # "n/a" strings, never a NaN rendered into the table.
    absent = weak_frame_summary("accepted", _vlm_rows(1), None)
    assert list(absent["table"]["vlm_count"]) == ["n/a"] * 5
    assert list(absent["table"]["gt_count"]) == ["n/a"] * 5

    partial = weak_frame_summary("accepted", _vlm_rows(1), {**_counts_row(), "vlm_car": None})
    car = partial["table"].loc[partial["table"]["class"] == "car"]
    assert list(car["vlm_count"]) == ["n/a"]
    assert list(car["gt_count"]) == ["2"]
    assert not partial["table"].isna().to_numpy().any()


def test_loss_decomposition_long_format() -> None:
    """One weak_loss_decomposition row -> the three stacked-chart components, in
    the order they are stacked (what weak supervision kept, then each way it lost
    the rest). The retained share is the RAW ``weak_gain / gt_gain`` ratio, not
    the stored (4-dp rounded) ``retention``, so the three shares sum to 1."""
    row = {
        "base_arm": "random", "gt_gain": 0.0362, "weak_gt_gain": 0.0182,
        "weak_gain": 0.0066, "retention": 0.1824,
        "dropped_frame_cost": 0.018, "dropped_frame_share": 0.4972,
        "label_cost": 0.0116, "label_share": 0.3204, "headline": True,
    }
    long = loss_long(row)

    assert list(long.columns) == ["base_arm", "component", "value", "share"]
    assert list(long["base_arm"]) == ["random"] * 3
    assert list(long["component"]) == ["retained", "dropped-frame cost", "label cost"]
    assert list(long["value"]) == pytest.approx([0.0066, 0.018, 0.0116])
    assert list(long["share"]) == pytest.approx([0.0066 / 0.0362, 0.4972, 0.3204])
    assert sum(long["share"]) == pytest.approx(1.0, abs=1e-3)


def test_crowding_long_format() -> None:
    """weak_supervision_results -> the paired accepted/rejected bars, two rows per
    arm, accepted first (the verifier keeps the SPARSE frames -- the rejected side
    is the crowded one, which is the whole point of the chart)."""
    df = pd.DataFrame({
        "arm": ["random", "graph_rate_night"],
        "gt_boxes_per_accepted_frame": [3.8674, 3.4114],
        "gt_boxes_per_rejected_frame": [7.6052, 6.6817],
    })
    long = crowding_long(df)

    assert list(long.columns) == ["arm", "side", "gt_boxes_per_frame"]
    assert list(long["arm"]) == ["random", "random", "graph_rate_night", "graph_rate_night"]
    assert list(long["side"]) == ["accepted", "rejected", "accepted", "rejected"]
    assert list(long["gt_boxes_per_frame"]) == pytest.approx(
        [3.8674, 7.6052, 3.4114, 6.6817]
    )


# --- the recorded chat replay helpers ---------------------------------------------
#
# Shapes come from `demo chat-record`'s own records (src/nuscenes_data_engine/demo/
# chat_record.py::RECORD_KEYS / FRAME_COLUMNS) and the agent's step summaries
# (data_engine/chat/agent.py::_summarize) -- these tests pin how the page turns them
# into on-screen text, with no Streamlit runtime involved.


def _steps_replay(*tools: str) -> dict[str, object]:
    return {"steps": [{"tool": tool, "input": {}, "output": "ok"} for tool in tools], "charts": []}


def test_replay_tool_counts_counts_tools_and_charts() -> None:
    """The "tools exercised" card: one count per agent tool across every replay,
    plus the total number of charts recorded. An ERRORED replay's steps still
    count -- the agent really did run those tools before the session broke."""
    replays = [
        {
            "steps": [
                {"tool": "run_sql", "input": {"sql": "SELECT 1"}, "output": "1 rows"},
                {"tool": "show_frames", "input": {"sample_data_tokens": ["a"]}, "output": "1 frames attached"},
                {"tool": "make_chart", "input": {}, "output": "charted: Night scenes"},
            ],
            "charts": [{"kind": "bar", "title": "Night scenes", "columns": [], "rows": []}],
        },
        _steps_replay("run_cypher", "search_frames", "run_sql"),
        {
            "steps": [{"tool": "run_sql", "input": {"sql": "SELECT 1"}, "output": "error: boom"}],
            "charts": [],
            "error": "boom",
        },
    ]

    assert replay_tool_counts(replays) == {
        "run_sql": 3, "run_cypher": 1, "search_frames": 1, "show_frames": 1,
        "make_chart": 1, "charts": 1,
    }

    # Every key is present even with nothing recorded -- the card reads the dict
    # straight, so a missing key would be a KeyError on an empty package.
    assert replay_tool_counts([]) == {
        "run_sql": 0, "run_cypher": 0, "search_frames": 0, "show_frames": 0,
        "make_chart": 0, "charts": 0,
    }


def test_verdict_line_passed_failed_reference_and_showcase() -> None:
    """The graded verdict line: which checks applied, which failed by name, and the
    reference value the numeric check was graded against.

    The reference is stated; the model's own number is NOT extracted from its prose
    (the answer itself is on screen in full above the line -- guessing which number
    in a sentence was "the" answer would be the page inventing a fact).
    """
    passed = {
        "kind": "eval", "error": None,
        "checks": {
            "english": True, "tool_use": True, "grounded": True, "numeric": True,
            "expected": 66, "passed": True,
        },
    }
    assert verdict_line(passed) == "✓ passed (english, tool_use, grounded, numeric) · reference 66"

    failed = {
        "kind": "eval", "error": None,
        "checks": {
            "english": True, "tool_use": True, "grounded": True, "numeric": False,
            "expected": 2.91, "passed": False,
        },
    }
    assert verdict_line(failed) == "✗ failed: numeric · reference 2.91"

    two_failures = {
        "kind": "eval", "error": None,
        "checks": {
            "english": True, "tool_use": True, "grounded": False, "numeric": False,
            "expected": 12, "passed": False,
        },
    }
    assert verdict_line(two_failures) == "✗ failed: grounded, numeric · reference 12"

    # A case with no reference_sql carries no `expected` key -- the clause is
    # omitted rather than printed as "reference None".
    no_reference = {
        "kind": "eval", "error": None,
        "checks": {"english": True, "tool_use": True, "grounded": True, "frames": True, "passed": True},
    }
    assert verdict_line(no_reference) == "✓ passed (english, tool_use, grounded, frames)"

    # An integral reference reads as an int whichever JSON type it came back as;
    # a fractional one is printed as recorded, never re-rounded.
    integral = {
        "kind": "eval", "error": None,
        "checks": {"english": True, "tool_use": True, "grounded": True, "numeric": True,
                   "expected": 66.0, "passed": True},
    }
    assert verdict_line(integral).endswith("· reference 66")

    errored = {
        "kind": "eval", "checks": {"passed": False},
        "error": "overloaded_error: the provider dropped the session",
    }
    assert verdict_line(errored) == "✗ errored — overloaded_error: the provider dropped the session"

    showcase = {"kind": "showcase", "checks": None, "error": None}
    assert verdict_line(showcase) == "showcase — not graded"


def test_frame_caption() -> None:
    """A retrieved frame's caption: scene, location, and the conditions the package
    recorded for it (rain only when it rained -- a "no rain" label on every day
    frame is noise)."""
    night_rain = {
        "sample_data_token": "abc", "scene_name": "scene-0916",
        "location": "singapore-onenorth", "is_night": True, "is_rain": True,
        "channel": "CAM_FRONT", "score": 0.87,
    }
    assert frame_caption(night_rain) == "scene-0916 · singapore-onenorth · night, rain"

    day = {**night_rain, "scene_name": "scene-0001", "location": "boston-seaport",
           "is_night": False, "is_rain": False}
    assert frame_caption(day) == "scene-0001 · boston-seaport · day"


def test_step_detail() -> None:
    """One agent step -> (outcome line, code block, code language). The outcome
    passes the recorder's own summary through verbatim (agent.py::_summarize wrote
    it), with the step's input adding what the summary alone does not say."""
    sql = {"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM scenes"}, "output": "1 rows"}
    assert step_detail(sql) == ("`run_sql` → 1 rows", "SELECT count(*) FROM scenes", "sql")

    cypher = {"tool": "run_cypher", "input": {"cypher": "MATCH (s:Scene) RETURN s"}, "output": "3 rows"}
    assert step_detail(cypher) == ("`run_cypher` → 3 rows", "MATCH (s:Scene) RETURN s", "cypher")

    search = {"tool": "search_frames", "input": {"query": "foggy road", "k": 8}, "output": "8 frames found"}
    assert step_detail(search) == ("`search_frames` → 8 frames found · query: foggy road", None, None)

    show = {
        "tool": "show_frames",
        "input": {"sample_data_tokens": ["a", "b", "c", "d", "e", "f"]},
        "output": "6 frames attached",
    }
    assert step_detail(show) == ("`show_frames` → 6 frames attached · 6 tokens", None, None)

    chart = {"tool": "make_chart", "input": {"kind": "bar", "title": "Night scenes"},
             "output": "charted: Night scenes"}
    assert step_detail(chart) == ("`make_chart` → charted: Night scenes", None, None)

    unknown = {"tool": "tool", "input": {}, "output": "output"}
    assert step_detail(unknown) == ("`tool` → output", None, None)

    # An errored step still shows its query and its error verbatim -- a failed SQL
    # attempt is part of the recorded session, not something to hide.
    errored = {
        "tool": "run_sql", "input": {"sql": "SELECT * FROM nope"},
        "output": "error: Catalog Error: Table with name nope does not exist!",
    }
    assert step_detail(errored) == (
        "`run_sql` → error: Catalog Error: Table with name nope does not exist!",
        "SELECT * FROM nope",
        "sql",
    )


# --- Phase 9a (Task 1): trust chrome primitives -----------------------------------


def test_parity_short_wording() -> None:
    """The Scenario viewer tour step's one-line parity summary: a match, a recorded
    mismatch, a GT-only preset (no Cypher count at all), and singular-noun
    agreement at a count of 1."""
    assert parity_short(30, 30, True) == "30 events found · SQL 30 / Graph 30 ✓"
    assert (
        parity_short(30, 29, False)
        == "30 events found · SQL 30 / Graph 29 ✗ mismatch recorded"
    )
    assert parity_short(4, None, None) == "4 events found · SQL 4 / Graph n/a (GT-only preset)"
    assert parity_short(1, 1, True, noun="event") == "1 event found · SQL 1 / Graph 1 ✓"


def _tour_manifest() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_data_token": [
            "day2", "night0", "night1", "night2", "night3", "noexplain", "otherarm",
        ],
        "al_selected_by": ["graph_rate_night"] * 6 + ["weak_random"],
        "is_night": [False, True, True, True, True, True, True],
    })


def _tour_explain() -> pd.DataFrame:
    # "noexplain" has no row here on purpose -- it must be dropped even though it
    # is otherwise arm-selected.
    #
    # community_mass_rank is the SECOND sort key (after night), so the ranks here
    # are deliberately at odds with the pedestrian counts in _tour_gt below: the
    # highest-pedestrian night frame ("night3", 3 pedestrians) sits in the
    # worst-ranked community (5) and must therefore sort LAST among the night
    # frames, which no pedestrian-first ordering could produce.
    return pd.DataFrame({
        "sample_data_token": ["day2", "night0", "night1", "night2", "night3", "otherarm"],
        "community_mass_rank": pd.array([1, 2, 2, 2, 5, 1], dtype="Int64"),
    })


def _tour_gt() -> pd.DataFrame:
    tokens = (
        ["day2"] * 2
        + ["night1"] * 1
        + ["night2"] * 1
        + ["night3"] * 3
        # generously seeded with pedestrians on the two frames that must be
        # EXCLUDED regardless -- proves it's the arm/explain filter dropping
        # them, not a coincidental pedestrian count of zero.
        + ["noexplain"] * 10
        + ["otherarm"] * 10
    )
    return pd.DataFrame({
        "sample_data_token": tokens,
        "category_group": ["pedestrian"] * len(tokens),
    })


def test_tour_frame_candidates_orders_night_then_mass_rank_then_pedestrians_then_token() -> None:
    """Ranked for the guided tour (Phase 9a review I3): night frames before day
    frames (the arm is night-targeted), then by the frame's community's failure-mass
    rank ascending (the step explains the mass → quota mechanism, so it leads with
    the frame whose community best exemplifies it), then by visible pedestrian GT
    count descending, then token -- dropping any frame the arm didn't select or that
    has no explain row.

    All four keys are exercised: "night3" carries the most pedestrians of any night
    frame and still sorts last among them (mass rank 5 beats its pedestrian count);
    "night1"/"night2" tie "night0" on mass rank 2 and win on pedestrians; and
    "night1"/"night2" tie on both and separate on token."""
    candidates = tour_frame_candidates(
        _tour_manifest(), _tour_explain(), _tour_gt(), arm="graph_rate_night"
    )
    assert candidates == ["night1", "night2", "night0", "night3", "day2"]


def test_tour_frame_candidates_sorts_unranked_frames_after_ranked_ones() -> None:
    """A frame whose explain row carries no ``community_mass_rank`` -- an NA cell,
    or a package whose explain group predates the column -- sorts after every
    ranked frame instead of ahead of them (rank 0 would have been the effect of
    a naive ``fillna(0)``)."""
    explain = _tour_explain()
    explain.loc[explain["sample_data_token"] == "night1", "community_mass_rank"] = pd.NA
    candidates = tour_frame_candidates(
        _tour_manifest(), explain, _tour_gt(), arm="graph_rate_night"
    )
    assert candidates == ["night2", "night0", "night3", "night1", "day2"]

    without_column = _tour_explain().drop(columns=["community_mass_rank"])
    # No ranks at all -> every night frame is unranked and the old pedestrian-first
    # ordering is what remains.
    assert tour_frame_candidates(
        _tour_manifest(), without_column, _tour_gt(), arm="graph_rate_night"
    ) == ["night3", "night1", "night2", "night0", "day2"]


def test_tour_frame_candidates_na_is_night_sorts_as_day() -> None:
    """``is_night`` NA (``pd.NA``/``None``, e.g. a frame the enrichment pass never
    reached) must not crash the sort -- ``bool(pd.NA)`` raises. It sorts as day
    (the arm is night-targeted, so an unknown-night frame does not get to lead),
    matching the NA-safe ``.fillna(False)`` pattern used elsewhere in this
    module."""
    manifest = pd.concat(
        [
            _tour_manifest(),
            pd.DataFrame({
                "sample_data_token": ["nanight"],
                "al_selected_by": ["graph_rate_night"],
                "is_night": pd.array([None], dtype="boolean"),
            }),
        ],
        ignore_index=True,
    )
    explain = pd.concat(
        [
            _tour_explain(),
            pd.DataFrame({
                "sample_data_token": ["nanight"],
                "community_mass_rank": pd.array([1], dtype="Int64"),
            }),
        ],
        ignore_index=True,
    )
    gt = _tour_gt()
    candidates = tour_frame_candidates(manifest, explain, gt, arm="graph_rate_night")
    assert candidates == ["night1", "night2", "night0", "night3", "day2", "nanight"]


def test_tour_frame_candidates_empty_manifest_or_explain_is_empty_list() -> None:
    manifest, explain, gt = _tour_manifest(), _tour_explain(), _tour_gt()
    assert tour_frame_candidates(manifest.iloc[0:0], explain, gt, arm="graph_rate_night") == []
    assert tour_frame_candidates(manifest, explain.iloc[0:0], gt, arm="graph_rate_night") == []


def test_tour_frame_candidates_without_al_selected_by_is_empty_list() -> None:
    """A package whose ``frame_manifest`` predates ``al_selected_by`` has no way to
    say which arm curated a frame, so the tour's arm-selected steps get no
    candidates at all (and say so) rather than falling back to every frame."""
    manifest = _tour_manifest().drop(columns=["al_selected_by"])
    assert tour_frame_candidates(
        manifest, _tour_explain(), _tour_gt(), arm="graph_rate_night"
    ) == []


# --- Phase 9b (Task 2): the filmstrip's steps -------------------------------------


def _event_row(**overrides: object) -> pd.Series:
    """One ``scenario_events.parquet`` row as the pages read it (``.iloc[0]`` of a
    ranked frame), v0.8 schema: CAN speed for the event frame and for each of the
    four t-2..t+2 neighbours, alongside the ego-pose ``speed_*`` columns the older
    readouts still use."""
    row = {
        "sample_data_token": "e0",
        "speed_mps": 10.0,
        "can_speed_kmh": 36.0,
        "accel_long_min_mps2": -7.5,
        "t_minus2": "m2", "speed_t_minus2": 12.0, "can_speed_t_minus2": 44.0,
        "accel_t_minus2": -0.5,
        "t_minus1": "m1", "speed_t_minus1": 11.0, "can_speed_t_minus1": 41.0,
        "accel_t_minus1": -1.5,
        "t_plus1": "p1", "speed_t_plus1": 8.0, "can_speed_t_plus1": 28.0,
        "accel_t_plus1": -6.0,
        "t_plus2": "p2", "speed_t_plus2": 6.0, "can_speed_t_plus2": 20.0,
        "accel_t_plus2": -2.0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_filmstrip_steps_pins_the_five_column_label_pairs() -> None:
    """The single definition both the Scenario page and the tour derive their own
    strip order from -- plain ASCII hyphens, the event itself (column None) in the
    middle."""
    assert FILMSTRIP_STEPS == (
        ("t_minus2", "t-2"),
        ("t_minus1", "t-1"),
        (None, "current"),
        ("t_plus1", "t+1"),
        ("t_plus2", "t+2"),
    )


def test_filmstrip_steps_reads_can_speed_in_time_order() -> None:
    curve = filmstrip_steps(_event_row())

    assert curve.speed_is_can is True
    assert [step.label for step in curve.steps] == ["t-2", "t-1", "current", "t+1", "t+2"]
    assert [step.token for step in curve.steps] == ["m2", "m1", "e0", "p1", "p2"]
    # the CAN column, never the ego-pose speed_* one (which would read 43.2 here)
    assert [step.can_speed_kmh for step in curve.steps] == [44.0, 41.0, 36.0, 28.0, 20.0]
    assert [step.accel_mps2 for step in curve.steps] == [-0.5, -1.5, -7.5, -6.0, -2.0]
    assert [step.is_current for step in curve.steps] == [False, False, True, False, False]
    assert curve.steps[2] == FilmstripStep(
        label="current", token="e0", can_speed_kmh=36.0, accel_mps2=-7.5, is_current=True
    )


def test_filmstrip_steps_drops_na_neighbours_at_a_scene_edge() -> None:
    """A scene-edge event has NA neighbour tokens -- those steps are left out of the
    strip entirely (the current frame is always there), exactly as the pages do."""
    curve = filmstrip_steps(
        _event_row(t_minus2=None, speed_t_minus2=None, can_speed_t_minus2=None,
                   accel_t_minus2=None, t_plus2=float("nan"))
    )

    assert [step.label for step in curve.steps] == ["t-1", "current", "t+1"]
    assert curve.speed_is_can is True


def test_filmstrip_steps_falls_back_to_ego_speed_on_a_pre_0_8_package() -> None:
    """A package built before v0.8 carries no ``can_speed_*`` columns at all: the
    helper falls back to ``speed_mps`` x 3.6 and says so, so the caller titles the
    axis "ego speed" rather than mislabelling an ego-pose figure as CAN."""
    row = _event_row()
    old_package = row.drop(
        ["can_speed_kmh", *(f"can_speed_{column}" for column, _ in FILMSTRIP_STEPS if column)]
    )

    curve = filmstrip_steps(old_package)

    assert curve.speed_is_can is False
    assert [step.can_speed_kmh for step in curve.steps] == pytest.approx(
        [43.2, 39.6, 36.0, 28.8, 21.6]
    )
    assert [step.accel_mps2 for step in curve.steps] == [-0.5, -1.5, -7.5, -6.0, -2.0]


def test_filmstrip_steps_missing_readouts_are_none_not_nan() -> None:
    curve = filmstrip_steps(_event_row(can_speed_t_plus1=None, accel_t_plus1=float("nan")))

    plus_one = curve.steps[3]
    assert plus_one.label == "t+1" and plus_one.token == "p1"
    assert plus_one.can_speed_kmh is None and plus_one.accel_mps2 is None


def test_filmstrip_steps_accepts_a_dict_and_an_itertuples_row() -> None:
    """The two pages hold an event row in different shapes -- a ``pd.Series`` read
    with ``getattr``/``[]`` and (via ``itertuples``) an attribute-only namedtuple --
    so the shared helper reads both, plus a plain mapping for tests."""
    frame = pd.DataFrame([_event_row().to_dict()])
    from_tuple = filmstrip_steps(next(iter(frame.itertuples(index=False))))
    from_dict = filmstrip_steps(_event_row().to_dict())

    assert from_tuple == from_dict == filmstrip_steps(_event_row())
