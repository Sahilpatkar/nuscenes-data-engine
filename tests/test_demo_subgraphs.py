"""Tests for demo/subgraph_export.py (pure fixtures; the live test self-skips).

See docs/superpowers/plans/2026-08-20-demo-phase6.md Task 1 + spec §1
(docs/superpowers/specs/2026-08-20-demo-phase6-design.md). Layer-by-layer:

- ``count_cypher``: the flagship must equal docs/GRAPH.md's verbatim fenced Cypher
  (modulo whitespace/``//`` comments); the other three dynamics presets use
  ``$near``/``$speed`` parameters (never interpolated values); model presets have
  none.
- ``assemble_subgraph``: pure assembly from canned "records" (the shape
  ``event_subgraph_cypher()`` returns via an injected ``run_query``) — dedup, the
  12-nearest cap, on_path marking, path ordering, per-type meta.
- ``export_subgraphs``: parity semantics (flagship asserts, others flag+warn) and
  the on-disk JSON contract.
- One ``@pytest.mark.integration`` test against the real local Neo4j — self-skips
  (tests/test_graph.py's own precedent, ~:588-606) on any driver/connection
  exception, so it's safe to leave enabled in the default test run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nuscenes_data_engine.demo.subgraph_export import (
    assemble_subgraph,
    count_cypher,
    event_subgraph_cypher,
    export_subgraphs,
)

PRESETS_THRESHOLDS = {"near_dist_m": 10.0, "high_speed_mps": 10.0}

_DYNAMICS_PRESETS = ("hard_braking_near_pedestrians", "night_pedestrians", "fast_cyclists", "rain_vru")
_MODEL_PRESETS = ("fn_pedestrians_night", "low_conf_braking")
_ALL_PRESETS = (*_DYNAMICS_PRESETS, *_MODEL_PRESETS)


def _normalize_cypher(text: str) -> str:
    """Whitespace- and ``//``-comment-insensitive normal form, for text-equality asserts."""
    no_comments = re.sub(r"//.*", "", text)
    return re.sub(r"\s+", " ", no_comments).strip()


# ---------------------------------------------------------------------------
# count_cypher
# ---------------------------------------------------------------------------


def test_flagship_count_cypher_matches_graph_md_verbatim() -> None:
    doc = Path("docs/GRAPH.md").read_text()
    pattern = r"```cypher\n(MATCH \(e:EgoPose \{is_hard_braking: true\}\).*?)```"
    match = re.search(pattern, doc, re.DOTALL)
    assert match is not None, "docs/GRAPH.md: flagship fenced ```cypher block not found"
    expected = _normalize_cypher(match.group(1))

    actual = count_cypher("hard_braking_near_pedestrians", PRESETS_THRESHOLDS)
    assert actual is not None
    assert _normalize_cypher(actual) == expected


def test_count_cypher_for_other_dynamics_presets_uses_thresholds() -> None:
    # Distinct, non-10 values so a stray literal "10" anywhere in the query text
    # would be caught -- these must NEVER appear; only $near/$speed placeholders.
    thresholds = {"near_dist_m": 12.5, "high_speed_mps": 9.5}

    night = count_cypher("night_pedestrians", thresholds)
    assert night is not None
    assert "Scene" in night and "is_night: true" in night
    assert "pedestrian" in night
    assert "$near" in night
    assert "12.5" not in night

    fast = count_cypher("fast_cyclists", thresholds)
    assert fast is not None
    assert "speed_mps" in fast
    assert "bicycle" in fast
    assert "$speed" in fast and "$near" in fast
    assert "12.5" not in fast and "9.5" not in fast

    rain = count_cypher("rain_vru", thresholds)
    assert rain is not None
    assert "is_rain: true" in rain
    assert "pedestrian" in rain and "bicycle" in rain
    assert "$near" in rain
    assert "12.5" not in rain


def test_model_presets_have_no_count_cypher() -> None:
    assert count_cypher("fn_pedestrians_night", PRESETS_THRESHOLDS) is None
    assert count_cypher("low_conf_braking", PRESETS_THRESHOLDS) is None


# ---------------------------------------------------------------------------
# assemble_subgraph
# ---------------------------------------------------------------------------


def _obj(token: str, category: str, group: str, distance: float) -> dict[str, Any]:
    return {
        "token": token,
        "category": category,
        "distance_to_ego_m": distance,
        "ego_rel_x": distance * 0.6,
        "ego_rel_y": distance * -0.2,
        "visibility": "4",
    }


def _cat(name: str, group: str) -> dict[str, Any]:
    return {"name": name, "group": group}


def test_event_subgraph_assembly_from_fake_records() -> None:
    sample = {"token": "smp1", "timestamp": 1000}
    scene = {
        "token": "scn1", "name": "scene-0001", "location": "boston-seaport",
        "is_night": True, "is_rain": False,
    }
    location = {"name": "boston-seaport"}
    ego = {
        "token": "ego1", "speed_mps": 12.0, "accel_long_min_mps2": -4.5,
        "is_hard_braking": True, "can_speed_kmh": 43.2,
    }
    prev_token = "smp0"
    next_token = "smp2"

    # 2 matching pedestrians (<10m, distinct categories -> 2 distinct Category nodes).
    matching = [
        (_obj("ped-near-1", "human.pedestrian.adult", "pedestrian", 3.8),
         _cat("human.pedestrian.adult", "pedestrian")),
        (_obj("ped-near-2", "human.pedestrian.child", "pedestrian", 7.2),
         _cat("human.pedestrian.child", "pedestrian")),
    ]
    # 13 non-matching: 5 cars, 3 bicycles, 5 pedestrians >= 10m away. Sorted
    # distances: 4,5,10.5,12,15,15,20,22,35,40,45,50,60 -- nearest 12 keeps
    # everything except the farthest (60).
    others = [
        (_obj("car-1", "vehicle.car", "car", 5), _cat("vehicle.car", "car")),
        (_obj("car-2", "vehicle.car", "car", 12), _cat("vehicle.car", "car")),
        (_obj("car-3", "vehicle.car", "car", 20), _cat("vehicle.car", "car")),
        (_obj("car-4", "vehicle.car", "car", 35), _cat("vehicle.car", "car")),
        (_obj("car-5", "vehicle.car", "car", 50), _cat("vehicle.car", "car")),
        (_obj("bike-1", "vehicle.bicycle", "bicycle", 4), _cat("vehicle.bicycle", "bicycle")),
        (_obj("bike-2", "vehicle.bicycle", "bicycle", 15), _cat("vehicle.bicycle", "bicycle")),
        (_obj("bike-3", "vehicle.bicycle", "bicycle", 45), _cat("vehicle.bicycle", "bicycle")),
        (_obj("ped-far-1", "human.pedestrian.adult", "pedestrian", 10.5),
         _cat("human.pedestrian.adult", "pedestrian")),
        (_obj("ped-far-2", "human.pedestrian.adult", "pedestrian", 15),
         _cat("human.pedestrian.adult", "pedestrian")),
        (_obj("ped-far-3", "human.pedestrian.adult", "pedestrian", 22),
         _cat("human.pedestrian.adult", "pedestrian")),
        (_obj("ped-far-4", "human.pedestrian.adult", "pedestrian", 40),
         _cat("human.pedestrian.adult", "pedestrian")),
        (_obj("ped-far-5", "human.pedestrian.adult", "pedestrian", 60),
         _cat("human.pedestrian.adult", "pedestrian")),
    ]

    records = [
        {
            "sample": sample, "scene": scene, "location": location, "ego": ego,
            "prev_token": prev_token, "next_token": next_token,
            "object": obj, "category": cat,
        }
        for obj, cat in (*matching, *others)
    ]

    result = assemble_subgraph(
        records, preset="hard_braking_near_pedestrians", thresholds=PRESETS_THRESHOLDS,
    )
    nodes = result["nodes"]
    edges = result["edges"]
    path = result["path"]

    # --- dedup ---
    node_ids = [n["id"] for n in nodes]
    assert len(node_ids) == len(set(node_ids))

    by_id = {n["id"]: n for n in nodes}

    # --- 2 matching + 12-nearest-capped-13 = 14 ObjectObservation nodes ---
    obj_nodes = [n for n in nodes if n["label"] == "ObjectObservation"]
    assert len(obj_nodes) == 14
    assert "object:ped-far-5" not in by_id  # the farthest (60m) non-matching, capped out

    # --- on_path: Scene/Sample/EgoPose + matching objects + their Category ---
    sample_id, scene_id, ego_id = "sample:smp1", "scene:scn1", "egopose:ego1"
    assert by_id[sample_id]["on_path"] is True
    assert by_id[scene_id]["on_path"] is True
    assert by_id[ego_id]["on_path"] is True
    assert by_id["object:ped-near-1"]["on_path"] is True
    assert by_id["object:ped-near-2"]["on_path"] is True
    assert by_id["category:human.pedestrian.adult"]["on_path"] is True
    assert by_id["category:human.pedestrian.child"]["on_path"] is True

    # off-path context: Location, non-matching objects, prev/next samples
    assert by_id["location:boston-seaport"]["on_path"] is False
    assert by_id["object:car-1"]["on_path"] is False
    assert by_id["category:vehicle.car"]["on_path"] is False
    assert by_id["sample:smp0"]["on_path"] is False
    assert by_id["sample:smp2"]["on_path"] is False

    # --- path: Scene->Sample->EgoPose backbone, then each matching Sample->Obj->Category ---
    assert path[0] == [scene_id, sample_id, ego_id]
    assert path[1] == [sample_id, "object:ped-near-1", "category:human.pedestrian.adult"]
    assert path[2] == [sample_id, "object:ped-near-2", "category:human.pedestrian.child"]
    assert len(path) == 3

    # --- meta keys per node type ---
    assert set(by_id[scene_id]["meta"]) == {"name", "location", "is_night", "is_rain"}
    assert set(by_id[sample_id]["meta"]) == {"timestamp", "scene", "is_night", "is_rain"}
    assert set(by_id[ego_id]["meta"]) == {
        "speed_mps", "accel_long_min_mps2", "is_hard_braking", "can_speed_kmh",
    }
    assert set(by_id["object:ped-near-1"]["meta"]) == {
        "category", "group", "distance_to_ego_m", "ego_rel_x", "ego_rel_y", "visibility",
    }
    assert set(by_id["category:human.pedestrian.adult"]["meta"]) == {"name", "group"}
    assert set(by_id["location:boston-seaport"]["meta"]) == {"name"}

    # --- edges reference only ids that exist ---
    for edge in edges:
        assert edge["source"] in by_id
        assert edge["target"] in by_id
    edge_labels = {(e["source"], e["target"], e["label"]) for e in edges}
    assert (sample_id, scene_id, "IN_SCENE") in edge_labels
    assert (scene_id, "location:boston-seaport", "IN_LOCATION") in edge_labels
    assert (sample_id, ego_id, "AT_POSE") in edge_labels
    assert ("sample:smp0", sample_id, "NEXT") in edge_labels
    assert (sample_id, "sample:smp2", "NEXT") in edge_labels
    assert (sample_id, "object:ped-near-1", "HAS_OBJECT") in edge_labels
    assert ("object:ped-near-1", "category:human.pedestrian.adult", "OF_CATEGORY") in edge_labels


def test_matching_objects_are_distance_ordered_in_path_and_edges() -> None:
    """Matching (on_path) objects must be NEAREST-first in ``path``/``edges``,
    whatever order the driver's rows arrived in.

    The page reads ``path[1]`` for its "why this event" narrative and each card's
    caption quotes the NEAREST matching object -- so a Neo4j row order that isn't
    distance order makes the two disagree (31/120 real cards did). Non-matching
    objects were already distance-sorted (the 12-nearest cap needs it); matching
    ones were appended in raw row order.
    """
    sample = {"token": "smp1", "timestamp": 1000}
    scene = {
        "token": "scn1", "name": "scene-0001", "location": "boston-seaport",
        "is_night": True, "is_rain": False,
    }
    ego = {
        "token": "ego1", "speed_mps": 12.0, "accel_long_min_mps2": -4.5,
        "is_hard_braking": True, "can_speed_kmh": 43.2,
    }
    # Deliberately NOT distance-ordered, and each with its own Category so every
    # matching object contributes its own path chain.
    arrivals = [
        (_obj("ped-mid", "human.pedestrian.adult", "pedestrian", 5.7),
         _cat("human.pedestrian.adult", "pedestrian")),
        (_obj("ped-far", "human.pedestrian.child", "pedestrian", 8.4),
         _cat("human.pedestrian.child", "pedestrian")),
        (_obj("ped-near", "human.pedestrian.police_officer", "pedestrian", 2.1),
         _cat("human.pedestrian.police_officer", "pedestrian")),
    ]
    records = [
        {
            "sample": sample, "scene": scene, "location": {"name": "boston-seaport"},
            "ego": ego, "prev_token": None, "next_token": None,
            "object": obj, "category": cat,
        }
        for obj, cat in arrivals
    ]

    result = assemble_subgraph(
        records, preset="hard_braking_near_pedestrians", thresholds=PRESETS_THRESHOLDS,
    )
    by_id = {n["id"]: n for n in result["nodes"]}
    object_chains = result["path"][1:]

    # path[1] is the NEAREST matching object -- the one the card caption quotes.
    assert object_chains[0][1] == "object:ped-near"
    distances = [by_id[chain[1]]["meta"]["distance_to_ego_m"] for chain in object_chains]
    assert distances == [2.1, 5.7, 8.4]
    assert distances == sorted(distances)

    # ...and the edge list follows the same order (deterministic by construction).
    has_object_targets = [
        edge["target"] for edge in result["edges"] if edge["label"] == "HAS_OBJECT"
    ]
    assert has_object_targets == ["object:ped-near", "object:ped-mid", "object:ped-far"]


def test_assemble_subgraph_empty_records_returns_empty_shape() -> None:
    assert assemble_subgraph([], preset="hard_braking_near_pedestrians", thresholds=PRESETS_THRESHOLDS) == {
        "nodes": [], "edges": [], "path": [],
    }


# ---------------------------------------------------------------------------
# export_subgraphs
# ---------------------------------------------------------------------------


def _cypher_to_preset_map(thresholds: dict[str, float]) -> dict[str, str]:
    result: dict[str, str] = {}
    for preset in _DYNAMICS_PRESETS:
        cypher = count_cypher(preset, thresholds)
        assert cypher is not None
        result[cypher] = preset
    return result


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["sample_data_token", "sample_token", "preset_tags"])


def test_parity_assert_for_flagship_and_flag_for_others(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cypher_to_preset = _cypher_to_preset_map(PRESETS_THRESHOLDS)

    # --- (a) flagship mismatch: sql=30, cypher=29 -> raises naming both ---
    def fake_run_query_mismatch(cypher: str, **params: Any) -> list[dict[str, Any]]:
        preset = cypher_to_preset[cypher]
        assert preset == "hard_braking_near_pedestrians"  # flagship raises before any other query
        return [{"n": 29}]

    sql_counts_a = dict.fromkeys(_ALL_PRESETS, 0)
    sql_counts_a["hard_braking_near_pedestrians"] = 30

    with pytest.raises(ValueError) as excinfo:
        export_subgraphs(
            run_query=fake_run_query_mismatch,
            events=_empty_events(),
            thresholds=PRESETS_THRESHOLDS,
            sql_counts=sql_counts_a,
            flagship_expected=30,
            out_dir=tmp_path / "a",
        )
    assert "30" in str(excinfo.value)
    assert "29" in str(excinfo.value)

    # --- (b) flagship matches; night_pedestrians diverges -> flagged + warned, no raise ---
    returns = {
        "hard_braking_near_pedestrians": 30,
        "night_pedestrians": 780,
        "fast_cyclists": 0,
        "rain_vru": 0,
    }

    def fake_run_query_ok(cypher: str, **params: Any) -> list[dict[str, Any]]:
        preset = cypher_to_preset[cypher]
        return [{"n": returns[preset]}]

    sql_counts_b = dict.fromkeys(_ALL_PRESETS, 0)
    sql_counts_b["hard_braking_near_pedestrians"] = 30
    sql_counts_b["night_pedestrians"] = 786

    caplog.clear()
    with caplog.at_level("WARNING"):
        summary = export_subgraphs(
            run_query=fake_run_query_ok,
            events=_empty_events(),
            thresholds=PRESETS_THRESHOLDS,
            sql_counts=sql_counts_b,
            flagship_expected=30,
            out_dir=tmp_path / "b",
        )

    assert summary["hard_braking_near_pedestrians"]["parity"] is True
    assert summary["night_pedestrians"]["parity"] is False
    assert summary["night_pedestrians"]["sql_count"] == 786
    assert summary["night_pedestrians"]["cypher_count"] == 780
    assert any("night_pedestrians" in record.message for record in caplog.records)


def test_export_writes_one_json_per_preset(tmp_path: Path) -> None:
    subgraph_cypher = event_subgraph_cypher()
    cypher_to_preset = _cypher_to_preset_map(PRESETS_THRESHOLDS)

    def fake_run_query(cypher: str, **params: Any) -> list[dict[str, Any]]:
        if cypher == subgraph_cypher:
            token = params["sample_token"]
            return [
                {
                    "sample": {"token": token, "timestamp": 500},
                    "scene": {
                        "token": f"scn-{token}", "name": "scene-x",
                        "location": "boston-seaport", "is_night": True, "is_rain": False,
                    },
                    "location": {"name": "boston-seaport"},
                    "ego": {
                        "token": f"ego-{token}", "speed_mps": 5.0,
                        "accel_long_min_mps2": -1.0, "is_hard_braking": False,
                        "can_speed_kmh": 18.0,
                    },
                    "prev_token": None,
                    "next_token": None,
                    "object": None,
                    "category": None,
                }
            ]
        preset = cypher_to_preset[cypher]
        return [{"n": 30 if preset == "hard_braking_near_pedestrians" else 0}]

    events = pd.DataFrame(
        {
            "sample_data_token": ["sdtA", "sdtB"],
            "sample_token": ["stA", "stB"],
            "preset_tags": [
                ["hard_braking_near_pedestrians", "night_pedestrians"],
                ["fn_pedestrians_night"],
            ],
        }
    )
    sql_counts = dict.fromkeys(_ALL_PRESETS, 0)
    sql_counts["hard_braking_near_pedestrians"] = 30

    def run(out_dir: Path) -> dict[str, dict[str, Any]]:
        return export_subgraphs(
            run_query=fake_run_query,
            events=events,
            thresholds=PRESETS_THRESHOLDS,
            sql_counts=sql_counts,
            flagship_expected=30,
            out_dir=out_dir,
        )

    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"
    summary = run(out_dir_1)
    run(out_dir_2)

    written = sorted(p.name for p in out_dir_1.iterdir())
    assert written == sorted(f"{preset}.json" for preset in _ALL_PRESETS)

    for preset in _ALL_PRESETS:
        path_1 = out_dir_1 / f"{preset}.json"
        path_2 = out_dir_2 / f"{preset}.json"
        # deterministic bytes: same inputs -> byte-identical output
        assert path_1.read_bytes() == path_2.read_bytes()

        payload = json.loads(path_1.read_text())
        assert payload["preset"] == preset
        required_keys = {"preset", "count_cypher", "sql_count", "cypher_count", "parity", "events"}
        assert required_keys <= set(payload)
        assert set(payload) - required_keys <= {"note"}

        if preset in _MODEL_PRESETS:
            assert payload["cypher_count"] is None
            assert payload["note"] == "the model verdict is not in the graph"
        else:
            assert "note" not in payload

    flagship_events = json.loads(
        (out_dir_1 / "hard_braking_near_pedestrians.json").read_text()
    )["events"]
    assert set(flagship_events) == {"sdtA"}
    fn_events = json.loads((out_dir_1 / "fn_pedestrians_night.json").read_text())["events"]
    assert set(fn_events) == {"sdtB"}
    assert summary["hard_braking_near_pedestrians"]["n_events"] == 1
    assert summary["fn_pedestrians_night"]["n_events"] == 1
    assert summary["fast_cyclists"]["n_events"] == 0


# ---------------------------------------------------------------------------
# live Neo4j integration (self-skips when unreachable)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_flagship_parity_and_one_event() -> None:
    pytest.importorskip("neo4j")
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.data_engine.graph import connection
    from nuscenes_data_engine.demo.events import build_events

    settings = get_settings()
    try:
        driver = connection.get_driver(settings)
    except Exception as exc:  # no compose Neo4j running -> not a failure, just skipped
        pytest.skip(f"Neo4j not reachable: {exc}")

    try:

        def run_query(cypher: str, **params: Any) -> list[dict[str, Any]]:
            return connection.read_query(
                driver, cypher, database=settings.neo4j_database, params=params
            )

        flagship_cypher = count_cypher("hard_braking_near_pedestrians", PRESETS_THRESHOLDS)
        assert flagship_cypher is not None
        rows = run_query(flagship_cypher)
        assert next(iter(rows[0].values())) == 30

        events = build_events(
            processed_dir=Path(settings.processed_dir),
            staging_dir=None,
            presets_cfg={
                "cap_per_preset": 30, "near_dist_m": 10.0, "high_speed_mps": 10.0,
                "model_for_results": "baseline",
            },
            flagship_expected=30,
        )
        flagship_events = events[
            events["preset_tags"].apply(lambda tags: "hard_braking_near_pedestrians" in tags)
        ]
        assert len(flagship_events) > 0
        sample_token = str(flagship_events.iloc[0]["sample_token"])

        records = run_query(event_subgraph_cypher(), sample_token=sample_token)
        subgraph = assemble_subgraph(
            records, preset="hard_braking_near_pedestrians", thresholds=PRESETS_THRESHOLDS,
        )
        ego_nodes = [n for n in subgraph["nodes"] if n["label"] == "EgoPose"]
        assert len(ego_nodes) == 1
        assert ego_nodes[0]["meta"]["is_hard_braking"] is True
    finally:
        connection.close(driver)
