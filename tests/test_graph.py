"""Tests for Phase 6e knowledge graph (offline: pure projections + guard, no live DB).

The graph builder's correctness lives in the pure ``model.py`` projections (aggregation,
co-occurrence pairing, temporal ordering) — tested here directly against small in-memory
DataFrames, no Neo4j required. The live end-to-end build is a ``graph_smoke`` test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nuscenes_data_engine.data_engine.graph import guard, model

# ---------------------------------------------------------------------------
# fixtures — tiny denormalized frames/annotations/labels shaped like the Parquet
# ---------------------------------------------------------------------------


def _samples() -> pd.DataFrame:
    """Two scenes: sA (Boston, day) with 2 keyframes x 2 cameras, sB (Singapore, night)."""
    scene_a = {
        "scene_token": "sA",
        "scene_name": "scene-a",
        "scene_description": "day drive",
        "log_token": "logA",
        "location": "boston-seaport",
        "is_night": False,
        "is_rain": False,
    }
    scene_b = {
        "scene_token": "sB",
        "scene_name": "scene-b",
        "scene_description": "night drive",
        "log_token": "logB",
        "location": "singapore-onenorth",
        "is_night": True,
        "is_rain": False,
    }
    rows = [
        {"sample_data_token": "fd1", "sample_token": "sm1", "channel": "CAM_FRONT",
         "filename": "a.jpg", "width": 1600, "height": 900, "timestamp": 1000, "n_boxes": 3, **scene_a},
        {"sample_data_token": "fd2", "sample_token": "sm1", "channel": "CAM_BACK",
         "filename": "b.jpg", "width": 1600, "height": 900, "timestamp": 1005, "n_boxes": 1, **scene_a},
        {"sample_data_token": "fd3", "sample_token": "sm2", "channel": "CAM_FRONT",
         "filename": "c.jpg", "width": 1600, "height": 900, "timestamp": 1500, "n_boxes": 2, **scene_a},
        {"sample_data_token": "fd4", "sample_token": "sm2", "channel": "CAM_BACK",
         "filename": "d.jpg", "width": 1600, "height": 900, "timestamp": 1502, "n_boxes": 0, **scene_a},
        {"sample_data_token": "fd5", "sample_token": "sm3", "channel": "CAM_FRONT",
         "filename": "e.jpg", "width": 1600, "height": 900, "timestamp": 2000, "n_boxes": 2, **scene_b},
    ]
    return pd.DataFrame(rows)


def _annotations() -> pd.DataFrame:
    rows = [
        # fd1: two cars (vis 4 and 2) + one pedestrian
        {"sample_data_token": "fd1", "category_name": "vehicle.car", "visibility_token": "4",
         "bbox_area": 100.0, "num_lidar_pts": 10, "num_radar_pts": 2},
        {"sample_data_token": "fd1", "category_name": "vehicle.car", "visibility_token": "2",
         "bbox_area": 50.0, "num_lidar_pts": 5, "num_radar_pts": 1},
        {"sample_data_token": "fd1", "category_name": "human.pedestrian.adult", "visibility_token": "3",
         "bbox_area": 30.0, "num_lidar_pts": 3, "num_radar_pts": 0},
        # fd2: a truck
        {"sample_data_token": "fd2", "category_name": "vehicle.truck", "visibility_token": "4",
         "bbox_area": 200.0, "num_lidar_pts": 20, "num_radar_pts": 5},
        # fd3: car + pedestrian (same pair as fd1 -> co-occurrence count 2)
        {"sample_data_token": "fd3", "category_name": "vehicle.car", "visibility_token": "3",
         "bbox_area": 40.0, "num_lidar_pts": 4, "num_radar_pts": 1},
        {"sample_data_token": "fd3", "category_name": "human.pedestrian.adult", "visibility_token": "4",
         "bbox_area": 25.0, "num_lidar_pts": 2, "num_radar_pts": 0},
        # fd5: car + bicycle
        {"sample_data_token": "fd5", "category_name": "vehicle.car", "visibility_token": "1",
         "bbox_area": 10.0, "num_lidar_pts": 1, "num_radar_pts": 0},
        {"sample_data_token": "fd5", "category_name": "vehicle.bicycle", "visibility_token": "4",
         "bbox_area": 15.0, "num_lidar_pts": 2, "num_radar_pts": 1},
    ]
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    # Mirrors the real labels.parquet: label_confidence is a string enum, the list
    # columns are JSON strings, and the object counts are floats (with NaN when unusable).
    rows = [
        {"sample_data_token": "fd1", "parse_status": "ok", "time_of_day": "day", "weather": "clear",
         "label_confidence": "high", "cars": 2.0, "pedestrians": 1.0, "bicycles": 0.0,
         "hazards": '["pedestrian near crosswalk"]', "notable_conditions": '["glare"]'},
        {"sample_data_token": "fd5", "parse_status": "ok", "time_of_day": "night", "weather": "rain",
         "label_confidence": "low", "cars": 1.0, "pedestrians": 0.0, "bicycles": 1.0,
         "hazards": '["cyclist in lane"]', "notable_conditions": "[]"},
        # unusable row (parse truncated) -> excluded everywhere; NaN counts must not crash
        {"sample_data_token": "fd3", "parse_status": "truncated", "time_of_day": None,
         "weather": None, "label_confidence": None, "cars": None, "pedestrians": None,
         "bicycles": None, "hazards": '["should be ignored"]', "notable_conditions": '["ignored"]'},
    ]
    return pd.DataFrame(rows)


def _by(rows: list[dict], *keys: str) -> list[dict]:
    return sorted(rows, key=lambda r: tuple(r[k] for k in keys))


# ---------------------------------------------------------------------------
# node projections
# ---------------------------------------------------------------------------


def test_location_rows_distinct_sorted() -> None:
    assert model.location_rows(_samples()) == [
        {"name": "boston-seaport"},
        {"name": "singapore-onenorth"},
    ]


def test_scene_rows_distinct_with_context() -> None:
    rows = _by(model.scene_rows(_samples()), "token")
    assert [r["token"] for r in rows] == ["sA", "sB"]
    assert rows[1] == {
        "token": "sB",
        "name": "scene-b",
        "description": "night drive",
        "is_night": True,
        "is_rain": False,
        "log_token": "logB",
        "location": "singapore-onenorth",
    }


def test_sample_rows_timestamp_is_min_over_cameras() -> None:
    rows = {r["token"]: r for r in model.sample_rows(_samples())}
    # sm1 spans cameras at 1000/1005 -> keyframe timestamp is the earliest (1000).
    assert rows["sm1"] == {"token": "sm1", "timestamp": 1000, "scene_token": "sA"}
    assert rows["sm2"]["timestamp"] == 1500
    assert rows["sm3"] == {"token": "sm3", "timestamp": 2000, "scene_token": "sB"}


def test_frame_rows_carry_props_and_edge_keys() -> None:
    rows = {r["token"]: r for r in model.frame_rows(_samples())}
    assert len(rows) == 5
    assert rows["fd1"] == {
        "token": "fd1",
        "sample_token": "sm1",
        "scene_token": "sA",
        "channel": "CAM_FRONT",
        "filename": "a.jpg",
        "width": 1600,
        "height": 900,
        "timestamp": 1000,
        "n_boxes": 3,
    }


def test_next_rows_link_consecutive_keyframes_within_scene() -> None:
    # sA has sm1(1000) -> sm2(1500); sB has a single keyframe (no NEXT).
    assert model.next_rows(_samples()) == [{"src": "sm1", "dst": "sm2", "dt_us": 500}]


# ---------------------------------------------------------------------------
# edge projections
# ---------------------------------------------------------------------------


def test_contains_rows_aggregate_boxes_per_frame_category() -> None:
    rows = {(r["token"], r["category"]): r for r in model.contains_rows(_annotations())}
    car = rows[("fd1", "vehicle.car")]
    assert car == {
        "token": "fd1",
        "category": "vehicle.car",
        "group": "car",
        "count": 2,
        "avg_visibility": 3.0,   # (4 + 2) / 2
        "min_visibility": 2,
        "total_bbox_area": 150.0,
        "max_bbox_area": 100.0,
        "num_lidar_pts": 15,
        "num_radar_pts": 3,
    }
    # A category outside the detector taxonomy keeps a null group, not a crash.
    assert rows[("fd1", "human.pedestrian.adult")]["group"] == "pedestrian"


def test_co_occurs_rows_count_frames_per_canonical_pair() -> None:
    rows = _by(model.co_occurs_rows(_annotations()), "a", "b")
    assert rows == [
        {"a": "human.pedestrian.adult", "b": "vehicle.car", "n_frames": 2},  # fd1 + fd3
        {"a": "vehicle.bicycle", "b": "vehicle.car", "n_frames": 1},          # fd5
    ]


# ---------------------------------------------------------------------------
# VLM projections (usable rows only)
# ---------------------------------------------------------------------------


def test_vlm_property_rows_skip_unusable_and_prefix_props() -> None:
    rows = {r["token"]: r for r in model.vlm_property_rows(_labels())}
    assert set(rows) == {"fd1", "fd5"}  # fd3 parse_status != ok -> excluded
    assert rows["fd1"]["vlm_time_of_day"] == "day"
    assert rows["fd1"]["vlm_weather"] == "clear"
    assert rows["fd1"]["vlm_label_confidence"] == "high"  # string enum, not a float
    assert rows["fd1"]["cars"] == 2 and rows["fd1"]["pedestrians"] == 1


def test_tag_rows_explode_list_columns_for_usable_rows() -> None:
    hazards = _by(model.tag_rows(_labels(), "hazards"), "token")
    assert hazards == [
        {"token": "fd1", "text": "pedestrian near crosswalk"},
        {"token": "fd5", "text": "cyclist in lane"},
    ]
    # notable_conditions: fd1 has one, fd5 is empty, fd3 is unusable.
    assert model.tag_rows(_labels(), "notable_conditions") == [{"token": "fd1", "text": "glare"}]


# ---------------------------------------------------------------------------
# guard.py — read-only Cypher guard (no DB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (:X)",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) DELETE n",
        "MATCH (n) SET n.x = 1",
        "MATCH (n) REMOVE n.x",
        "MERGE (:X {id: 1})",
        "DROP CONSTRAINT foo IF EXISTS",
        "MATCH (n) FOREACH (x IN n.items | SET x.y = 1)",
        "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
        "USING PERIODIC COMMIT 500 MATCH (n) RETURN n",
        "CALL apoc.create.node(['X'], {}) YIELD node RETURN node",
        "CALL gds.graph.drop('g') YIELD graphName RETURN graphName",
        "CALL gds.louvain.write('g', {}) YIELD nodePropertiesWritten RETURN 1",
        "CALL dbms.components() YIELD name RETURN name",
        "CALL db.createLabel('X')",
    ],
)
def test_check_read_only_rejects_writes(cypher: str) -> None:
    assert guard.check_read_only(cypher) is not None, cypher


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (f:Frame)-[:CONTAINS]->(c:Category {name: 'vehicle.car'}) RETURN c LIMIT 5",
        "MATCH (s:Scene) WHERE s.is_night RETURN count(*) AS n",
        "MATCH (n) RETURN n.created_at, n.reset_flag LIMIT 1",
        "CALL gds.louvain.stream('sim') YIELD nodeId, communityId RETURN communityId LIMIT 5",
        "CALL db.labels() YIELD label RETURN label",
        "CALL db.schema.visualization()",
    ],
)
def test_check_read_only_allows_reads(cypher: str) -> None:
    assert guard.check_read_only(cypher) is None, cypher


class _FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str) -> Any:
        return self._data.get(key)


class _FakeResult:
    """Minimal stand-in for a neo4j Result: keys() + iteration over records."""

    def __init__(self, keys: list[str], records: list[dict[str, Any]]) -> None:
        self._keys = keys
        self._records = records

    def keys(self) -> list[str]:
        return self._keys

    def __iter__(self) -> Any:
        return iter(_FakeRecord(record) for record in self._records)


def test_consume_caps_rows_and_clips_cells() -> None:
    result = _FakeResult(["id", "blob"], [{"id": i, "blob": "x" * 400} for i in range(150)])
    columns, rows, truncated = guard._consume(result, max_rows=100)
    assert columns == ["id", "blob"]
    assert len(rows) == 100 and truncated
    assert rows[0][1].endswith("…")  # runaway cell clipped like the SQL guard


def test_run_cypher_rejects_write_before_touching_db() -> None:
    # driver=None proves the guard short-circuits before any DB access.
    out = guard.run_cypher(None, "MATCH (n) DELETE n", database="neo4j")
    assert set(out) == {"error"} and "Disallowed" in out["error"]


def test_graph_schema_prompt_describes_the_model() -> None:
    prompt = guard.graph_schema_prompt()
    assert "Frame" in prompt and "Category" in prompt
    assert "CO_OCCURS_WITH" in prompt and "SIMILAR_TO" in prompt
    assert "run_cypher" in prompt or "read-only" in prompt


# ---------------------------------------------------------------------------
# agent wiring — run_cypher tool composition + dispatch (no live DB)
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """Replays scripted replies; records the tools and messages it was handed."""

    model = "fake-model"

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = replies
        self.seen_tools: list[list[dict[str, Any]]] = []
        self.seen_msgs: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.seen_msgs.append([dict(m) for m in messages])
        self.seen_tools.append(tools)
        return self._replies[len(self.seen_msgs) - 1]


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import json

    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.fixture()
def graph_con(tmp_path: Any) -> Any:
    pytest.importorskip("duckdb")
    from nuscenes_data_engine.data_engine.chat import catalog

    pd.DataFrame(
        {"sample_data_token": ["t1"], "scene_name": ["s"], "channel": ["CAM_FRONT"],
         "n_boxes": [1], "is_night": [False]}
    ).to_parquet(tmp_path / "samples.parquet")
    return catalog.open_catalog(tmp_path)


def test_agent_offers_and_dispatches_cypher_when_graph_present(
    graph_con: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    called: dict[str, Any] = {}

    def fake_run_cypher(driver: Any, cypher: str, params: Any, *, database: str,
                        max_rows: int = guard.MAX_ROWS) -> dict[str, Any]:
        called.update(cypher=cypher, database=database, driver=driver)
        return {"columns": ["n"], "rows": [[5]], "row_count": 1, "truncated": False}

    monkeypatch.setattr(guard, "run_cypher", fake_run_cypher)
    transport = _RecordingTransport(
        [
            {"role": "assistant", "content": None,
             "tool_calls": [_tool_call("c1", "run_cypher", {"cypher": "MATCH (n) RETURN count(n) AS n"})]},
            {"role": "assistant", "content": "There are 5 nodes."},
        ]
    )
    driver = object()
    result = agent.answer(
        "how many nodes?", transport=transport, con=graph_con, search_engine=None,
        graph_driver=driver, graph_database="graphdb",
    )
    assert result.answer == "There are 5 nodes."
    # run_cypher is offered as a tool, and the graph schema is injected into the system prompt.
    assert "run_cypher" in [t["function"]["name"] for t in transport.seen_tools[0]]
    assert "CO_OCCURS_WITH" in transport.seen_msgs[0][0]["content"]
    # the guard was invoked with the model's cypher, this driver, and the configured database.
    assert called["cypher"].startswith("MATCH") and called["database"] == "graphdb"
    assert called["driver"] is driver
    assert result.steps[0]["tool"] == "run_cypher"


def test_agent_hides_cypher_tool_when_graph_absent(graph_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    transport = _RecordingTransport([{"role": "assistant", "content": "ok"}])
    agent.answer("hi", transport=transport, con=graph_con, search_engine=None, graph_driver=None)
    assert "run_cypher" not in [t["function"]["name"] for t in transport.seen_tools[0]]
    assert "CO_OCCURS_WITH" not in transport.seen_msgs[0][0]["content"]


def test_nearest_neighbours_topk_by_cosine() -> None:
    import numpy as np

    from nuscenes_data_engine.data_engine.graph import knn

    # Unit vectors; v0 is closest to v1 (dot 0.8), then v2 (0.0), then v3 (-1.0).
    vectors = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    neigh = knn.nearest_neighbours(vectors, k=2)
    assert len(neigh) == 4
    idx0 = [j for j, _score in neigh[0]]
    assert idx0 == [1, 2]  # self (0) excluded; ordered by descending cosine
    assert neigh[0][0][1] == pytest.approx(0.8, abs=1e-6)


def test_run_tool_cypher_reports_unavailable_without_driver() -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    out = agent._run_tool(
        "run_cypher", {"cypher": "MATCH (n) RETURN n"}, con=None, search_engine=None,
        result=agent.ChatResult(answer="", model="m"), graph_driver=None,
    )
    assert set(out) == {"error"} and "not available" in out["error"]


# ---------------------------------------------------------------------------
# graph_smoke — end-to-end build against a live Neo4j (opt-in, excluded by default)
# ---------------------------------------------------------------------------


@pytest.mark.graph_smoke
def test_graph_build_end_to_end(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    neo4j = pytest.importorskip("neo4j")
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.data_engine.graph import builder, connection, guard, schema

    processed = tmp_path / "processed"
    processed.mkdir()
    _samples().to_parquet(processed / "samples.parquet")
    _annotations().to_parquet(processed / "annotations.parquet")
    monkeypatch.setenv("PROCESSED_DIR", str(processed))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))  # no labels.parquet -> VLM pass skipped
    settings = get_settings()

    try:
        driver = connection.get_driver(settings)
    except Exception as exc:  # no compose Neo4j running -> not a failure, just skipped
        pytest.skip(f"Neo4j not reachable: {exc}")

    try:
        schema.drop_all(driver, database=settings.neo4j_database)
        summary = builder.build_graph(settings, Path("configs/engine.yaml"), skip_knn=True)
        assert summary["scenes"] == 2 and summary["frames"] == 5
        assert summary["contains"] >= 5

        counts = {
            row["label"]: row["n"]
            for row in connection.read_query(
                driver,
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n",
                database=settings.neo4j_database,
            )
        }
        assert counts["Scene"] == 2 and counts["Frame"] == 5 and counts["Category"] >= 4

        # The guarded read surface actually returns rows against a live DB (regression
        # guard for the managed-transaction / timeout wiring).
        out = guard.run_cypher(
            driver, "MATCH (n:Scene) RETURN count(n) AS n", database=settings.neo4j_database
        )
        assert out["columns"] == ["n"] and out["rows"] == [[2]]

        # The read-only transaction is the *hard* guard: a write raises even if the
        # denylist were bypassed.
        with (
            driver.session(database=settings.neo4j_database) as session,
            pytest.raises(neo4j.exceptions.Neo4jError),
        ):
            session.execute_read(lambda tx: tx.run("CREATE (:Sneaky)").consume())
    finally:
        schema.drop_all(driver, database=settings.neo4j_database)
        connection.close(driver)
