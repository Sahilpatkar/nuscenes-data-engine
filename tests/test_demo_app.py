"""The demo app must stay deployable on Streamlit Cloud: no src/, no heavy deps."""

from __future__ import annotations

import ast
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

FORBIDDEN = ("nuscenes_data_engine", "requests", "torch", "lancedb", "neo4j", "duckdb")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "app" / "demo"

# Phase 6 (Task 3): the six scenario presets subgraph_export.py exports one JSON
# per -- mirrors tests/test_demo_export.py's own module-level tuple of the same
# name/shape (that file exercises build.py's copy/validate side; this file's own
# copy exercises the Scenario page's rendering side, no cross-import needed for
# six short strings).
_SUBGRAPH_PRESETS = (
    "hard_braking_near_pedestrians",
    "night_pedestrians",
    "fast_cyclists",
    "rain_vru",
    "fn_pedestrians_night",
    "low_conf_braking",
)
_SUBGRAPH_MODEL_PRESETS = ("fn_pedestrians_night", "low_conf_braking")


def test_demo_app_never_imports_the_backend() -> None:
    # Static ast.Import/ImportFrom check only: dynamic imports (importlib, __import__)
    # slip past this. It's a contributor guardrail against the obvious mistake, not a
    # sandbox — don't rely on it to block a determined attempt to reach the backend.
    offenders = []
    for path in sorted(DEMO_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"demo app imports forbidden modules: {offenders}"


def test_demo_requirements_stay_minimal() -> None:
    raw_lines = [
        line
        for line in (DEMO_DIR / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    lines = [line.split("==")[0].split(">=")[0].strip() for line in raw_lines]
    # altair (Phase 7) is streamlit's OWN hard dependency -- declaring it adds no
    # wheel to the deployment, it just pins the version the charts are written
    # against (app/demo/render.py::bar_chart imports it at module level).
    assert set(lines) <= {
        "streamlit", "pandas", "pyarrow", "pillow", "streamlit-agraph", "altair"
    }
    # The subset check above would still pass if streamlit-agraph were dropped
    # entirely -- the interactive graph panel (Phase 6) needs it present, not just
    # not-disallowed. Same for altair and the Phase-7 charts.
    assert any(line.startswith("streamlit-agraph") for line in raw_lines)
    assert any(line.startswith("altair") for line in raw_lines)


def test_streamlit_config_and_requirements_for_cloud() -> None:
    """The Community-Cloud deployment scaffolding, as Cloud actually reads it.

    Cloud looks for a dependency file in the ENTRYPOINT's directory first and only
    then at the repository root, recognising ``uv.lock``, ``Pipfile``,
    ``environment.yml``, ``requirements.txt``, ``pyproject.toml`` in that order --
    and installs exactly one of them. ``app/demo/requirements.txt`` sits next to the
    entrypoint ``app/demo/main.py``, so that is the file Cloud installs. Two
    consequences this test pins:

    - the demo requirements file must EXIST next to main.py -- if it ever vanished,
      the search would fall through to the repo root and Cloud would install the
      root ``uv.lock``, i.e. the whole project including torch, on a free tier that
      cannot hold it;
    - the repo root must have NO ``requirements.txt`` -- a second dependency file
      that Cloud never reads (the entrypoint directory wins) but that a contributor
      would reasonably believe is the deployed one.

    ``.streamlit/config.toml`` is shared by Cloud and a local ``streamlit run``.
    """
    import tomllib

    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["headless"] is True
    assert config["browser"]["gatherUsageStats"] is False

    assert (DEMO_DIR / "main.py").is_file(), "the Cloud entrypoint moved"
    assert (DEMO_DIR / "requirements.txt").is_file(), (
        "app/demo/requirements.txt is the file Streamlit Cloud installs -- without it "
        "the dependency search falls through to the root uv.lock (the full project)"
    )
    assert not (REPO_ROOT / "requirements.txt").exists(), (
        "a root requirements.txt would never be read (Cloud takes the entrypoint "
        "directory's file first) but would read as the deployed dependency set"
    )


def _stage_subgraphs_with_event(staging_dir: Path) -> None:
    """Stage six ``graph_subgraphs/<preset>.json`` files under ``staging_dir`` --
    adapted from tests/test_demo_export.py's own ``_stage_subgraphs`` helper (Task
    2), and deliberately NOT named the same (item M7, Phase 6 review: the two
    differ), which mirrors ``demo subgraph_export.export_subgraphs``'s real output shape
    closely enough for ``run_build``'s ``_include_subgraphs`` copy/hash/flagship-
    read logic to exercise, without a live Neo4j connection.

    Unlike that helper, the flagship preset's one event ("s1" -- this fixture's own
    flagship-tagged CAM_FRONT sample, see the Task-3 comments below) gets a REAL
    subgraph built through the actual ``assemble_subgraph`` (not a synthetic ``{}``
    events dict): Task 3's page tests render the graph panel, the path narrative,
    and the legend against it, so it needs real nodes/meta/path, not a stub.
    ``assemble_subgraph`` is pure and takes plain dicts ("records", the shape
    ``event_subgraph_cypher()`` rows would have) -- no live Neo4j needed here
    either. The other four presets' JSONs stay minimal (``events: {}``, which is
    also exactly what this fixture's events frame ranks for them): no test below
    opens their graph panel, only their header's parity line/GT-only note, which
    reads straight off this file's top-level ``sql_count``/``cypher_count``/
    ``parity``/``note`` fields.

    ``flagship_records``' shared sample/scene/ego/prev/next columns (repeated
    across the two ObjectObservation rows) mirror this fixture's own staged
    samples/canbus/ego_pose/annotations_3d for "s1": is_hard_braking=True,
    accel_long_min_mps2=-7.5, speed_mps=10.1, can_speed_kmh=36.0 (from
    canbus.is_hard_braking/accel_long_min_mps2 and ego_pose.speed_mps/canbus.
    can_speed_kmh below), a matching pedestrian at 5m (f1, < near_dist_m=10) and a
    non-matching one at 20m (f2) -- the SAME f1/f2 annotations_3d rows already
    staged for events.py's own flagship tagging, so the narrative's "pedestrian at
    5.00 m" is the SAME 5m distance the card grid/ego panel show elsewhere on this
    page, not a disconnected number.

    "night_pedestrians" gets its own real subgraph for "v1" (the fixture's other
    tagged event, pedestrian f4 at 7m) for the same reason the flagship does, and
    because ``build.py::_include_subgraphs``' stale-staging guard (item M2, Phase 6
    review) requires every preset's staged ``events`` keys to be EXACTLY the tokens
    this build's events frame ranks for it -- "v1" for night_pedestrians, "s1" for
    the flagship, nothing for the other four.
    """
    from nuscenes_data_engine.demo.subgraph_export import assemble_subgraph

    out = staging_dir / "graph_subgraphs"
    out.mkdir(parents=True, exist_ok=True)

    common = {
        "sample": {"token": "s1", "timestamp": 1000},
        "scene": {
            "token": "sceneX", "name": "scene-X", "location": "boston-seaport",
            "is_night": False, "is_rain": False,
        },
        "location": {"name": "boston-seaport"},
        "ego": {
            "token": "s1", "speed_mps": 10.1, "accel_long_min_mps2": -7.5,
            "is_hard_braking": True, "can_speed_kmh": 36.0,
        },
        "prev_token": None,
        "next_token": "v1",
    }
    flagship_records = [
        {
            **common,
            "object": {
                "token": "f1", "category": "human.pedestrian.adult",
                "distance_to_ego_m": 5.0, "ego_rel_x": 3.0, "ego_rel_y": -1.0,
                "visibility": "4",
            },
            "category": {"name": "human.pedestrian.adult", "group": "pedestrian"},
        },
        {
            **common,
            "object": {
                "token": "f2", "category": "human.pedestrian.adult",
                "distance_to_ego_m": 20.0, "ego_rel_x": 18.0, "ego_rel_y": -4.0,
                "visibility": "4",
            },
            "category": {"name": "human.pedestrian.adult", "group": "pedestrian"},
        },
    ]
    flagship_subgraph = assemble_subgraph(
        flagship_records,
        preset="hard_braking_near_pedestrians",
        thresholds={"near_dist_m": 10.0, "high_speed_mps": 10.0},
    )

    night_records = [
        {
            "sample": {"token": "v1", "timestamp": 1001},
            "scene": {
                "token": "sceneX", "name": "scene-X", "location": "boston-seaport",
                "is_night": True, "is_rain": False,
            },
            "location": {"name": "boston-seaport"},
            "ego": {
                "token": "v1", "speed_mps": 4.9, "accel_long_min_mps2": -0.5,
                "is_hard_braking": False, "can_speed_kmh": 18.0,
            },
            "prev_token": "s1",
            "next_token": None,
            "object": {
                "token": "f4", "category": "human.pedestrian.adult",
                "distance_to_ego_m": 7.0, "ego_rel_x": 6.0, "ego_rel_y": -2.0,
                "visibility": "4",
            },
            "category": {"name": "human.pedestrian.adult", "group": "pedestrian"},
        }
    ]
    night_subgraph = assemble_subgraph(
        night_records,
        preset="night_pedestrians",
        thresholds={"near_dist_m": 10.0, "high_speed_mps": 10.0},
    )
    staged_events = {
        "hard_braking_near_pedestrians": {"s1": flagship_subgraph},
        "night_pedestrians": {"v1": night_subgraph},
    }

    # sql_count/cypher_count are 1/1 for every preset except night_pedestrians,
    # which gets 5/5 (item 1, Phase 6 follow-up review): the flagship MUST stay
    # at 1 -- run_build's sql==cypher assertion (Task 2) checks it against this
    # fixture's flagship expected_sql_count (also 1, see built_demo_data below).
    # night_pedestrians has no such constraint (only its `events` dict's KEYS are
    # checked by build.py::_include_subgraphs' stale-staging guard, never the
    # sql_count/cypher_count values), so giving it a distinct count from n_shown
    # (this fixture ranks exactly one event per preset) is what lets a test tell
    # sql_count and n_shown apart in the rendered parity line -- a swap of the
    # two in _render_parity_line would otherwise pass every existing assertion.
    counts_by_preset = dict.fromkeys(_SUBGRAPH_PRESETS, 1)
    counts_by_preset["night_pedestrians"] = 5

    for preset in _SUBGRAPH_PRESETS:
        is_model = preset in _SUBGRAPH_MODEL_PRESETS
        count = counts_by_preset[preset]
        payload: dict[str, Any] = {
            "preset": preset,
            "count_cypher": None if is_model else "MATCH (n) RETURN count(n)",
            "sql_count": count,
            "cypher_count": None if is_model else count,
            "parity": None if is_model else True,
            "events": staged_events.get(preset, {}),
        }
        if is_model:
            payload["note"] = "the model verdict is not in the graph"
        (out / f"{preset}.json").write_text(json.dumps(payload, sort_keys=True, indent=2))


def _stage_al_explain(staging_dir: Path) -> None:
    """Stage the optional `demo al-explain` group (spec §1) under ``staging_dir``.

    Phase 7 (Task 4): the Active Learning page's "why was this frame selected?"
    panel reads ``al_selection_explain.parquet`` (per-frame community/pick-pass/
    degree facts) and the night floor out of ``al_explain_validation.json``. The
    real command needs Neo4j + GDS + LanceDB, so this stages its OUTPUT directly --
    the same idiom `_stage_subgraphs_with_event` uses for `demo subgraphs`, and what
    tests/test_demo_export.py's own al_explain build test does.

    One row per token in this fixture's ``graph_rate_night.parquet`` (the arm's
    selected set), so every gallery card has a row -- exactly the real package's
    relationship (1500 selected frames, 1500 explain rows). The two train_pool
    tokens carry the two branches the panel has to render: "wA" was taken by the
    NIGHT pass with no failure mass routed to it (the ordinary case -- 1249 of the
    1500 real selected frames have none), "wR" by the main pass with mass actually
    routed to it.
    """
    out = staging_dir / "al_explain"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_data_token": ["v0", "v1", "wA", "wR"],
        "arm": ["graph_rate_night"] * 4,
        "is_night": [True, False, True, False],
        "scene_name": ["scene-v0", "scene-v1", "scene-wA", "scene-wR"],
        "community": [0, 1, 0, 1],
        "community_size": [10, 8, 10, 8],
        "community_night_members": [2, 1, 2, 1],
        "community_mass": [5.0, 3.0, 5.0, 3.0],
        "community_mass_rank": pd.array([1, 2, 1, 2], dtype="Int64"),
        "community_quota": [1, 2, 1, 2],
        "degree": [4.0, 2.0, 3.0, 1.0],
        "degree_rank_in_community": pd.array([1, 1, 2, 2], dtype="Int64"),
        "pick_pass": ["night", "main", "night", "main"],
        "n_failures_routed": [0, 0, 0, 2],
        "mass_routed": [0.0, 0.0, 0.0, 4.25],
    }).to_parquet(out / "al_selection_explain.parquet", index=False)
    (out / "al_explain_validation.json").write_text(json.dumps({
        "arm": "graph_rate_night",
        "selected_match": True,
        "communities_match": True,
        "n_selected": 4,
        "n_communities": 2,
        "mass_total": 8.0,
        "gds_version": "2.13.11",
        "config": {
            "channel": "CAM_FRONT", "n_mine": 3, "night_floor": 1,
            "route_k": 10, "seed": 64, "top_k": 1000,
        },
    }))


# The two frame tokens the staged showcase replay retrieved -- thumbs for both are
# written by hand after the build (this fixture stages no LanceDB store, so the
# builder's own thumbnail export skips with a warning, exactly as it does for the
# curation/event/semsearch galleries).
_CHAT_REPLAY_TOKENS = ("cr1", "cr2")


def _chat_replay_frame(token: str, **overrides: Any) -> dict[str, Any]:
    """One retrieved frame as `demo chat-record` stores it.

    `demo build`'s ``_validate_chat_replays`` requires EXACTLY the keys of
    ``chat_record.FRAME_COLUMNS``, so this mirrors tests/test_demo_export.py's own
    ``_chat_frame`` helper (Task 2) rather than importing it -- pytest fixtures and
    helpers are not shared across modules, and that file exercises the builder's
    validation side while this one exercises the page's rendering side.
    """
    frame: dict[str, Any] = {
        "sample_data_token": token,
        "scene_name": "scene-0916",
        "location": "singapore-onenorth",
        "is_night": True,
        "is_rain": True,
        "channel": "CAM_FRONT",
        "score": 0.87,
    }
    frame.update(overrides)
    return frame


def _stage_chat_replays(staging_dir: Path) -> None:
    """Stage a recorded chat session (``chat_replays.json`` +
    ``chat_replay_summary.json``) under ``staging_dir``, mirroring `demo
    chat-record`'s output (Task 1) closely enough for `demo build`'s copy +
    validation and for the "Ask the Dataset" page's own rendering.

    Four replays, one per state the page has to render:

    - one SHOWCASE replay (array order puts showcase first, as the recorder writes
      it) with a bar chart, two retrieved frames and three steps -- a SQL query
      (code block), a show_frames call and a make_chart call;
    - one PASSED eval replay with a numeric check graded against reference 66;
    - one FAILED eval replay whose numeric check missed reference 2.91, carrying a
      Cypher step and a semantic-search step (so the tool-count card and
      ``step_detail``'s cypher/search branches are exercised on a real package);
    - one ERRORED eval replay: no answer, ``checks == {"passed": False}``, and the
      provider error the recorder stored (`demo chat-record` records the failure
      and continues, exactly as `chat-eval`'s ``run_eval`` does).

    The summary's ``n_eval``/``n_passed``/``n_showcase`` must agree with the
    records or the build refuses them.
    """
    records: list[dict[str, Any]] = [
        {
            "id": "show_night_locations",
            "kind": "showcase",
            "question": "Which locations have the most night frames?",
            "answer": "Singapore-onenorth has the most night frames, five of them.",
            "model": "claude-test",
            "provider": "anthropic",
            "steps": [
                {
                    "tool": "run_sql",
                    "input": {"sql": "SELECT location, count(*) FROM scenes GROUP BY location"},
                    "output": "2 rows",
                },
                {
                    "tool": "show_frames",
                    "input": {"sample_data_tokens": list(_CHAT_REPLAY_TOKENS)},
                    "output": "2 frames attached",
                },
                {
                    "tool": "make_chart",
                    "input": {"kind": "bar", "title": "Night frames per location"},
                    "output": "charted: Night frames per location",
                },
            ],
            "frames": [
                _chat_replay_frame(_CHAT_REPLAY_TOKENS[0]),
                _chat_replay_frame(
                    _CHAT_REPLAY_TOKENS[1], scene_name="scene-0001",
                    location="boston-seaport", is_night=False, is_rain=False, score=0.71,
                ),
            ],
            "charts": [{
                "kind": "bar",
                "title": "Night frames per location",
                "columns": ["location", "n"],
                "rows": [["boston-seaport", 3], ["singapore-onenorth", 5]],
            }],
            "checks": None,
            "latency_s": 4.2,
            "error": None,
        },
        {
            "id": "eval_scene_count",
            "kind": "eval",
            "question": "How many scenes are in the dataset?",
            "answer": "There are 66 scenes.",
            "model": "claude-test",
            "provider": "anthropic",
            "steps": [
                {"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM scenes"},
                 "output": "1 rows"},
            ],
            "frames": [],
            "charts": [],
            "checks": {
                "english": True, "tool_use": True, "grounded": True, "numeric": True,
                "expected": 66, "passed": True,
            },
            "latency_s": 3.1,
            "error": None,
        },
        {
            "id": "eval_mean_speed",
            "kind": "eval",
            "question": "What is the mean ego speed?",
            "answer": "About 3.1 m/s across the logged keyframes.",
            "model": "claude-test",
            "provider": "anthropic",
            "steps": [
                {"tool": "run_cypher", "input": {"cypher": "MATCH (e:EgoState) RETURN avg(e.speed_mps)"},
                 "output": "1 rows"},
                {"tool": "search_frames", "input": {"query": "fast highway driving", "k": 6},
                 "output": "6 frames found"},
            ],
            "frames": [],
            "charts": [],
            "checks": {
                "english": True, "tool_use": True, "grounded": True, "numeric": False,
                "expected": 2.91, "passed": False,
            },
            "latency_s": 5.4,
            "error": None,
        },
        {
            "id": "eval_night_pedestrians",
            "kind": "eval",
            "question": "Which scenes have the most pedestrians at night?",
            "answer": None,
            "model": "claude-test",
            "provider": "anthropic",
            "steps": [],
            "frames": [],
            "charts": [],
            "checks": {"passed": False},
            "latency_s": 0.4,
            "error": "overloaded_error: the provider dropped the session",
        },
    ]
    summary = {
        "model": "claude-test",
        "provider": "anthropic",
        "recorded_at": "2026-08-21T09:15:00+00:00",
        "git_sha": "deadbeef",
        "n_eval": 3,
        "n_passed": 1,
        "n_showcase": 1,
        "search_available": True,
        "graph_available": True,
        "max_turns": 8,
    }
    out = staging_dir / "chat_replays"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chat_replays.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (out / "chat_replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


@pytest.fixture()
def built_demo_data(tmp_path: Path) -> Path:
    """A real (tiny) demo_data/ package, built through the actual exporters/build path.

    Adapted from tests/test_demo_export.py's tiny_inputs/build_config fixtures — kept
    minimal here since this test only needs a package to exist and render, not to
    exercise every exporter edge case (that is test_demo_export.py's job). Importing
    nuscenes_data_engine.demo.build here is fine: the forbidden-import rule is about
    app/demo/*.py, not the tests that exercise the builder.

    PIL is only a transitive dep (via streamlit, not declared in any repo extra
    itself), and this fixture runs before the test body's own importorskip — so the
    guard has to live here, not there, or a serve-less environment would fail
    collection of the whole module instead of skipping cleanly.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    from nuscenes_data_engine.demo.build import run_build

    processed = tmp_path / "processed"
    al = tmp_path / "active_learning"
    processed.mkdir()
    al.mkdir()
    pd.DataFrame(
        {
            "sample_data_token": ["s1", "v1", "s3"],
            # Phase 5: demo/events.py::build_events reads samples.parquet too
            # (CAM_FRONT-filtered, joined to canbus/ego_pose by sample_token) --
            # s1/v1 double as their own sample_token here (this fixture's tiny
            # scale never needs the real per-channel token distinction) and are
            # CAM_FRONT so build_events sees exactly canbus/ego_pose's covered set
            # (2 rows, both already present below); s3 is a different channel so
            # it's invisible to build_events without needing a matching canbus/
            # ego_pose row of its own.
            #
            # Task 3 (Scenario Search page): the second CAM_FRONT token is
            # deliberately "v1" -- the SAME token already staged below as a
            # curated val frame (frame_manifest split="val") -- rather than a
            # fresh "s2", so build_events's in_curated_set actually has overlap to
            # find: a real curated token that also has processed-table dynamics
            # features, exactly as in the real pipeline (curated tokens ARE
            # CAM_FRONT sample_data_tokens). This keeps the total samples.parquet
            # row count at 3 (test_overview_page_renders_from_a_built_package's
            # "Camera keyframes" == "3" is unaffected) while giving the Scenario
            # Search page fixture one curated event (v1) and one non-curated event
            # (s1, see canbus/annotations_3d below).
            "sample_token": ["s1", "v1", "s3"],
            "channel": ["CAM_FRONT", "CAM_FRONT", "CAM_BACK"],
            "scene_token": ["sceneX", "sceneX", "sceneX"],
            "scene_name": ["scene-X", "scene-X", "scene-X"],
            "timestamp": [1000, 1001, 1002],
            # v1 is_night=True (+ its own near pedestrian, added to
            # annotations_3d below) tags it "night_pedestrians" -- a dynamics
            # preset distinct from s1's flagship tag, so the two Task-3 fixture
            # events aren't both flagship rows.
            "is_night": [False, True, False],
            "is_rain": [False, False, False],
        }
    ).to_parquet(processed / "samples.parquet")
    # Phase 7 (Task 2): sample_data_token/category_group -- export_weaksup's
    # rejected-side recipe reads exactly these two columns. All 4 rows are s1's; 3
    # carry a detector category_group (pedestrian/car/pedestrian) and the 4th is
    # None (ignored) -- s1's detector count is 3, matching random_pseudo_summary.
    # json's mean_gt_boxes_per_accepted_frame=3.0 below (random_accepted=["s1"]).
    pd.DataFrame(
        {
            "sample_token": ["s1"] * 4,
            "sample_data_token": ["s1"] * 4,
            "category_group": ["pedestrian", "car", "pedestrian", None],
        }
    ).to_parquet(processed / "annotations.parquet")
    pd.DataFrame(
        {
            # annotation_token: required since Task 1 (Phase 3) -- gt_boxes.parquet
            # (staged below) is joined onto this table by annotation_token to add
            # distance_to_ego_m; f1..f4 are otherwise unused by the flagship SQL.
            # f4: a pedestrian at 7m for v1 -- Task 3's "night_pedestrians" tag
            # (is_night & min_dist_pedestrian_m < near_dist_m) needs a real near
            # pedestrian on v1, distinct from s1's own (f1/f2).
            "annotation_token": ["f1", "f2", "f3", "f4"],
            "sample_token": ["s1", "s1", "v1", "v1"],
            "category_group": ["pedestrian", "pedestrian", "car", "pedestrian"],
            "distance_to_ego_m": [5.0, 20.0, 3.0, 7.0],
        }
    ).to_parquet(processed / "annotations_3d.parquet")
    pd.DataFrame(
        {
            "sample_token": ["s1", "v1"],
            "has_canbus": [True, True],
            "can_vel_mps": [10.0, 5.0],
            "can_speed_kmh": [36.0, 18.0],
            # v1 is deliberately NOT hard-braking -- it must tag only
            # night_pedestrians, not also the flagship hard_braking_near_
            # pedestrians preset (that would make both fixture events flagship
            # rows, defeating the "one curated, one not" / "one flagship, one
            # not" split Task 3's page tests rely on).
            "is_hard_braking": [True, False],
            # Phase 5: demo/events.py::build_events also reads accel_long_min_mps2
            # (the flagship preset's severity key) -- s1's magnitude is deliberately
            # a strong-braking value, consistent with is_hard_braking=True.
            "accel_long_min_mps2": [-7.5, -0.5],
        }
    ).to_parquet(processed / "canbus.parquet")
    pd.DataFrame({"sample_token": ["s1", "v1"], "speed_mps": [10.1, 4.9]}).to_parquet(
        processed / "ego_pose.parquet"
    )
    (al / "results.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "n_train_images": 100,
                    "overall": {"mAP50-95": 0.20},
                    # night.per_class (Phase 9a): the real baseline row carries a
                    # night PEDESTRIAN mAP (0.0826) and the guided tour's first step
                    # shows it as its own card -- without a per_class entry here the
                    # exporter writes NA and the fixture would only ever exercise
                    # that card's absent path.
                    "night": {"mAP50-95": 0.10, "per_class": {"pedestrian": 0.05}},
                },
                "random": {
                    "n_train_images": 115,
                    "overall": {"mAP50-95": 0.21},
                    "night": {"mAP50-95": 0.11},
                },
                # Phase 7 (Task 4): configs' al.arm -- the arm the Active Learning
                # page tells the story of, so the fixture package must actually
                # carry a row for it (delta_night +0.0300, delta_overall +0.0300
                # against baseline, and the best night gain of the four arms, as
                # graph_rate_night is in the real results.json). night.per_class
                # gives the expander table a real night_ped_map5095 to show.
                "graph_rate_night": {
                    "n_train_images": 120,
                    "overall": {"mAP50-95": 0.23},
                    "night": {"mAP50-95": 0.13, "per_class": {"pedestrian": 0.09}},
                },
                "weak_random": {
                    "n_train_images": 110,
                    "overall": {"mAP50-95": 0.2018},
                    "night": {"mAP50-95": 0.105},
                },
                # Phase 7 (Task 2): weak_random's GT-trained twin -- gives
                # export_weak_loss_decomposition a "random" row (gt_gain=0.01,
                # weak_gt_gain=0.005, weak_gain=0.0018), which run_build asserts
                # against overview_metrics.json's own (independently, but
                # identically, computed) weak_retention.by_base_arm["random"].
                "weak_random_gt": {
                    "n_train_images": 112,
                    "overall": {"mAP50-95": 0.205},
                    "night": {"mAP50-95": 0.103},
                },
            }
        )
    )
    (al / "random_pseudo_summary.json").write_text(
        json.dumps(
            {
                "arm": "random", "n_candidates": 10, "n_accepted": 6, "retention": 0.6,
                "n_boxes": 12, "mean_boxes_per_accepted_frame": 2.0,
                "mean_gt_boxes_per_accepted_frame": 3.0,
                "conf": 0.5, "tolerance": 1, "n_no_label": 1, "n_unparsed": 0,
                "rejected_by_class": {"pedestrian": 1}, "accepted_mutual_zero_by_class": {"car": 1},
            }
        )
    )
    # Phase 7 (Task 2): export_weaksup's own per-arm inputs -- "random" candidates
    # (s1 accepted, s3 rejected -- s3 has no annotations.parquet rows at all -> 0
    # detector GT boxes by the reindex-fill-0 rule).
    pd.DataFrame({"sample_data_token": ["s1", "s3"]}).to_parquet(al / "random.parquet")
    pd.DataFrame({"sample_data_token": ["s1"]}).to_parquet(al / "random_accepted.parquet")
    # Phase 3: the hero is a hand-picked exemplar crop from the curated-frames group,
    # not an mlruns mosaic -- stage a minimal two-token curation group so
    # run_build's now-mandatory hero-token resolution has a real crop to copy. A
    # genuine (if tiny) JPEG: st.image() in the AppTest run below actually decodes
    # it. Two val tokens across two models (review round, quality pass on Task 4):
    #   - "v0": baseline misses GT box a2 (pedestrian) that graph_rate_night
    #     catches -- fixes_fn_vs_baseline_graph_rate_night=True. This is the only
    #     shape that exercises the exemplar-badge column (fixes_fn_vs_<A>_<B>
    #     never existed with a single model staged, so the inverted-badge bug
    #     from the first review round was untestable until now).
    #   - "v1": zero GT rows, all-FP predictions from both models (a hallucination
    #     frame during a notionally hard-braking moment) -- this is exactly the
    #     shape the distance-slider-default bug hid: a frame with no GT boxes can
    #     never be "in range" under ANY concrete distance_range, so it must only
    #     ever be excluded by an explicit user choice, never by the slider's
    #     full-extent default.
    staging = tmp_path / "curation_staging"
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        # Phase 7 (Task 2): "wA"/"wR" are train_pool weak-supervision frames (no
        # predictions, so they're excluded from the val-coverage check below) --
        # "wA" carries curation_buckets=["weak_accepted"] and gets one pseudo box
        # in graph_rate_night_pseudo_labels.parquet; "wR" carries
        # ["weak_rejected"] and gets none (rejected frames have no pseudo boxes by
        # construction). Task 5's Weak Supervision page tests can render off
        # these two tokens directly.
        "sample_data_token": ["v0", "v1", "wA", "wR"],
        "split": ["val", "val", "train_pool", "train_pool"],
        "filename": ["images/v0.jpg", "images/v1.jpg", "images/wA.jpg", "images/wR.jpg"],
        # scene_name: the grid loop's caption reads this straight off each row
        # (row.scene_name) -- absent here, that line never ran in any test before
        # this review round, because bug #2 (the distance-slider default) also
        # happened to filter v0 itself out of every prior single-model fixture
        # (its only GT row had a NaN distance), leaving `frames` empty and the
        # grid loop body dead code from the page's very first test.
        "scene_name": ["scene-v0", "scene-v1", "scene-wA", "scene-wR"],
        # Two buckets on v0, not one: curation_buckets round-trips through parquet
        # as a numpy array (pyarrow's list dtype) -- a single-element array is
        # falsy-safe by accident (`bool()` of a length-1 array just returns that
        # element's truthiness), so this needs >= 2 entries to actually exercise
        # `array or []`-style bugs in the page (`ValueError: truth value of an
        # array with more than one element is ambiguous`).
        "curation_buckets": [
            ["night_failure", "al_selected"], ["hard_braking"],
            ["weak_accepted"], ["weak_rejected"],
        ],
        # is_night/is_rain: the Failure Explorer sidebar builds its lighting/rain
        # filter options straight off these columns' actual values -- absent here,
        # the page would KeyError before an AppTest ever gets to render.
        "is_night": [True, False, True, False], "is_rain": [False, False, False, False],
        "n_preds_baseline": pd.array([1, 2, pd.NA, pd.NA], dtype="Int64"),
        "n_preds_graph_rate_night": pd.array([2, 2, pd.NA, pd.NA], dtype="Int64"),
        "fixes_fn_vs_baseline_graph_rate_night": pd.array(
            [True, False, pd.NA, pd.NA], dtype="boolean"
        ),
        "fixes_fn_vs_graph_rate_night_baseline": pd.array(
            [False, False, pd.NA, pd.NA], dtype="boolean"
        ),
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame({
        "sample_data_token": ["v0", "v0", "v0", "v1", "v1", "v1", "v1"],
        "model": [
            "baseline", "graph_rate_night", "graph_rate_night",
            "baseline", "baseline", "graph_rate_night", "graph_rate_night",
        ],
        "category_group": ["car", "car", "pedestrian", "car", "car", "car", "car"],
        "x_min": [1.0, 1.0, 10.0, 5.0, 6.0, 5.0, 6.0],
        "y_min": [1.0, 1.0, 10.0, 5.0, 6.0, 5.0, 6.0],
        "x_max": [2.0, 2.0, 30.0, 7.0, 8.0, 7.0, 8.0],
        "y_max": [2.0, 2.0, 30.0, 7.0, 8.0, 7.0, 8.0],
        "conf": [0.9, 0.9, 0.8, 0.4, 0.5, 0.4, 0.5],
        # v0: baseline only ever claims a1 (misses a2); graph_rate_night claims
        # both a1 and a2 (a genuine catch). v1 has zero GT rows, so every claim
        # from either model is necessarily a false positive.
        "status": ["tp", "tp", "tp", "fp", "fp", "fp", "fp"],
        "matched_annotation_token": ["a1", "a1", "a2", None, None, None, None],
    }).to_parquet(staging / "predictions.parquet")
    pd.DataFrame({
        "annotation_token": ["a1", "a2"], "sample_data_token": ["v0", "v0"],
        "category_group": ["car", "pedestrian"],
        "x_min": [1.0, 10.0], "y_min": [1.0, 10.0],
        "x_max": [2.0, 30.0], "y_max": [2.0, 30.0],
        "matched_baseline": pd.array([True, False], dtype="boolean"),
        "matched_graph_rate_night": pd.array([True, True], dtype="boolean"),
        # below_visibility_min: real gt_boxes always carries this column: the
        # Failure Explorer drops such rows everywhere (rendering + the box table),
        # so it must be present for the page to even read gt_boxes.parquet. v1 has
        # no gt_boxes rows at all (the zero-GT case both review-fix regression
        # tests below depend on).
        "below_visibility_min": [False, False],
    }).to_parquet(staging / "gt_boxes.parquet")
    hero_bytes = io.BytesIO()
    Image.new("RGB", (2, 2), color=(120, 120, 120)).save(hero_bytes, format="JPEG")
    (staging / "crops" / "v0.jpg").write_bytes(hero_bytes.getvalue())
    v1_bytes = io.BytesIO()
    Image.new("RGB", (2, 2), color=(60, 60, 60)).save(v1_bytes, format="JPEG")
    (staging / "crops" / "v1.jpg").write_bytes(v1_bytes.getvalue())
    for token, color in (("wA", (90, 90, 90)), ("wR", (30, 30, 30))):
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=color).save(buf, format="JPEG")
        (staging / "crops" / f"{token}.jpg").write_bytes(buf.getvalue())

    # Task 3 (Scenario Search page): a tiny staged semantic-search result so
    # build.py::_include_semsearch has something to copy (`demo semsearch` itself
    # is torch-local and not exercised here -- this stages its OUTPUT directly,
    # same idea as staging curation's parquets above instead of running `demo
    # curate`/`demo infer`). Two rows across two distinct queries -- the page
    # groups the gallery by query, so a single-query fixture would leave that
    # grouping untested. Both tokens ("v1", "v0") already get thumbs (below).
    pd.DataFrame({
        "query": ["crowded nighttime pedestrian crossing", "foggy road with heavy lens glare"],
        "rank": [1, 1],
        "sample_data_token": ["v1", "v0"],
        "score": [0.91, 0.77],
        # k=2 while only 1 hit landed per query -- exercises the gallery's "N of
        # k front-camera hits" caption (item 7c, consolidated review) actually
        # distinguishing N from k, not just echoing the row count back.
        "k": [2, 2],
    }).to_parquet(staging / "semantic_search_results.parquet")

    # Phase 7 (Task 1): al_communities.parquet's inputs (always required -- see
    # run_build) -- 2 communities whose quotas each sum to n_mine=3, mirroring
    # tests/test_demo_export.py's tiny_inputs fixture.
    for arm, quotas in (("graph_rate", [2, 1]), ("graph_rate_night", [1, 2])):
        records = [
            {"community": 0, "size": 10, "mass": 5.0, "night_members": 2, "quota": quotas[0]},
            {"community": 1, "size": 8, "mass": 3.0, "night_members": 1, "quota": quotas[1]},
        ]
        (al / f"communities_{arm}.json").write_text(json.dumps(records))
    al_config_path = tmp_path / "active_learning.yaml"
    al_config_path.write_text(yaml.safe_dump({"mining": {"n_mine": 3}}))
    # weak_verdict/al_selected_by's arm parquets: v0/wA are weak-arm candidates
    # AND accepted ("accepted"); v1/wR are candidates but not accepted
    # ("rejected"). Both are in the al_arm's selected set (same file, weak_arm ==
    # al_arm here) -- matches wA/wR's own curation_buckets above.
    pd.DataFrame({"sample_data_token": ["v0", "v1", "wA", "wR"]}).to_parquet(
        al / "graph_rate_night.parquet"
    )
    pd.DataFrame({"sample_data_token": ["v0", "wA"]}).to_parquet(
        al / "graph_rate_night_accepted.parquet"
    )
    # Phase 7 (Task 2): weak_labels/vlm_counts' own inputs. wA gets one pseudo box
    # (its curation_buckets=["weak_accepted"]); wR gets none (rejected frames have
    # no pseudo boxes by construction) -- both get a VLM row so vlm_counts can
    # compare its counts to GT for both tabs.
    pd.DataFrame({
        "sample_data_token": ["wA"],
        "category_group": ["car"],
        "x_min": [50.0], "y_min": [50.0], "x_max": [150.0], "y_max": [150.0],
        "score": [0.73],
    }).to_parquet(al / "graph_rate_night_pseudo_labels.parquet")
    (al / "autolabel_weak").mkdir()
    pd.DataFrame({
        "sample_data_token": ["wA", "wR"],
        "model": ["qwen2.5-vl", "qwen2.5-vl"],
        "parse_status": ["ok", "ok"],
        "time_of_day": ["night", "day"],
        "weather": ["clear", "rain"],
        "hazards": ["[]", "[]"],
        "notable_conditions": ["[]", "[]"],
        # The VLM's own STRING enum -- rendering it as a float crashed the page on
        # every frame it had actually labelled (consolidated review C1).
        "label_confidence": ["high", "low"],
        "cars": [1.0, 0.0], "trucks": [0.0, 0.0], "buses": [0.0, 0.0],
        "trailers": [0.0, 0.0], "construction_vehicles": [0.0, 0.0],
        "motorcycles": [0.0, 0.0], "bicycles": [0.0, 0.0],
        "pedestrians": [0.0, 1.0], "traffic_cones": [0.0, 0.0], "barriers": [0.0, 0.0],
    }).to_parquet(al / "autolabel_weak" / "labels.parquet")

    out = tmp_path / "demo_data"
    config = {
        "paths": {
            "processed_dir": str(processed),
            "active_learning_dir": str(al),
            "active_learning_config": str(al_config_path),
            "mlruns_dir": str(tmp_path / "mlruns"),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(out),
        },
        "models": {
            "baseline": {"run": "runX", "imgsz": 640},
            "graph_rate_night": {"run": "runY", "imgsz": 640},
        },
        "hero": {"token": "v0"},
        "budgets": {"max_package_mb": 100},
        "flagship": {"expected_sql_count": 1, "cypher_count": 30, "cypher_source": "docs/GRAPH.md"},
        "presets": {
            "cap_per_preset": 30,
            "near_dist_m": 10.0,
            "high_speed_mps": 10.0,
            "model_for_results": "baseline",
        },
        "curation": {
            "staging_dir": str(staging),
            "weak_arm": "graph_rate_night",
            "al_arm": "graph_rate_night",
        },
        # Phase 7 (Task 1): "v0" already carries fixes_fn_vs_baseline_graph_rate_
        # night=True, split="val", and predictions for both configured models
        # above, so it's a valid exemplar token.
        "al": {
            "arm": "graph_rate_night",
            "baseline": "baseline",
            "community_arms": ["graph_rate", "graph_rate_night"],
            "exemplar_tokens": ["v0"],
        },
    }
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # Task 3 (Phase 6): stage graph_subgraphs/ BEFORE run_build so the Scenario
    # page's tests below get a package with the interactive graph panel included
    # by default (mirrors curation/semsearch already being staged above) --
    # _stage_subgraphs_with_event's flagship cypher_count=1 matches this fixture's
    # flagship expected_sql_count (also 1), so run_build's sql==cypher assertion
    # (Task 2) passes. Only the flagship's count is constrained this way; the
    # other five presets' sql_count/cypher_count are free (see that helper).
    _stage_subgraphs_with_event(staging)
    _stage_al_explain(staging)
    # Staged BEFORE run_build so the package this fixture returns carries the
    # recorded chat replays the "Ask the Dataset" page reads (the builder copies +
    # re-validates them; a package without them is the
    # built_demo_data_without_chat_replay variant below).
    _stage_chat_replays(staging)

    run_build(config_path)

    # No real LanceDB store is staged in this fixture, so run_build's thumbnail
    # export skips with a warning (see build.py's _include_curation) -- the
    # Failure Explorer grid and the Overview hero both prefer a thumb over a crop
    # when one exists, so write one directly for the page tests below to exercise
    # that path too (crops/v0.jpg alone would leave it untested here).
    thumbs_dir = out / "sample_frames" / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    thumb_bytes = io.BytesIO()
    Image.new("RGB", (2, 2), color=(80, 80, 80)).save(thumb_bytes, format="JPEG")
    (thumbs_dir / "v0.jpg").write_bytes(thumb_bytes.getvalue())

    # Task 3 (Scenario Search page): thumbs for both scenario_events tokens ("s1"
    # the non-curated flagship event, "v1" the curated night_pedestrians event --
    # each is also the other's only filmstrip neighbor, scene sceneX's two
    # CAM_FRONT keyframes). Same skip-then-write-directly rationale as v0.jpg
    # above -- _include_events's own thumbnail export skips (no LanceDB store
    # staged here), so the filmstrip/card-grid images are written by hand.
    for token, color in (("s1", (40, 40, 40)), ("v1", (200, 200, 200))):
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=color).save(buf, format="JPEG")
        (thumbs_dir / f"{token}.jpg").write_bytes(buf.getvalue())

    # The staged showcase replay's two retrieved frames -- same skip-then-write-by-
    # hand rationale as the tokens above (no LanceDB store here, so
    # _include_chat_replay's own thumb export logged a warning and copied nothing).
    for token in _CHAT_REPLAY_TOKENS:
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=(120, 60, 60)).save(buf, format="JPEG")
        (thumbs_dir / f"{token}.jpg").write_bytes(buf.getvalue())

    return out


@pytest.fixture()
def built_demo_data_without_explain(built_demo_data: Path) -> Path:
    """The same package with the optional al_explain group removed -- the ordinary
    state of a fresh clone (no Neo4j, so `demo al-explain` never ran).

    Deletes the two files from an otherwise-normal built package rather than
    rebuilding one without the staging: the page's own absent branch keys off
    ``data.al_explain_available()``, i.e. the file's presence, and the builder's
    absent/included recording is already pinned on the builder side
    (tests/test_demo_export.py::test_build_records_al_explain_group_absent_then_
    included). Same idiom as test_scenario_graph_panel_absent_when_subgraphs_not_
    staged.
    """
    for name in ("al_selection_explain.parquet", "al_explain_validation.json"):
        (built_demo_data / name).unlink()
    manifest_path = built_demo_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["validation"]["al_explain"] = "absent"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return built_demo_data


@pytest.fixture()
def built_demo_data_low_conf_hero(built_demo_data: Path) -> Path:
    """The same package, with the arm's recovery of the hero frame's pedestrian
    rewritten to the SHIPPED package's own shape: ``conf=0.135``, ``status
    ="low_conf"`` -- a claim the matching rule counts as a hit and the tour's
    confident-detection rule does not.

    The base fixture makes that recovery a confident ``tp`` (conf 0.80), which is
    the branch ``test_tour_result_screen`` pins: no honesty line, because the claim
    isn't actually unsure. That left the TRUE side of every honesty branch
    (``_hero_recovery_is_low_conf`` / ``_hero_recovery_conf`` / the step-1 fact
    line's low_conf wording) untested even though it is what the real package
    renders (Phase 9a review I4).

    The rewritten row is found through the package's own tables -- the hero token
    from ``overview_metrics.json``, the arm from ``al_exemplars.json``, and the
    recovered box from ``gt_boxes``' matched columns -- rather than by hardcoding
    "v0"/"a2", so this fixture keeps describing the same claim if the base fixture's
    tokens ever change. Same edit-the-built-package idiom as
    ``_add_weak_night_twin_arm``.
    """
    hero = str(json.loads((built_demo_data / "overview_metrics.json").read_text())["hero_token"])
    arm = str(json.loads((built_demo_data / "al_exemplars.json").read_text())["arm"])

    gt = pd.read_parquet(built_demo_data / "gt_boxes.parquet")
    recovered = gt.loc[
        (gt["sample_data_token"] == hero)
        & (gt["category_group"] == "pedestrian")
        & gt["matched_baseline"].eq(False).fillna(False)
        & gt[f"matched_{arm}"].eq(True).fillna(False)
    ]
    assert not recovered.empty, "the fixture hero no longer carries a recovered pedestrian"
    annotation = str(recovered.iloc[0]["annotation_token"])

    path = built_demo_data / "predictions.parquet"
    preds = pd.read_parquet(path)
    claim = (
        (preds["sample_data_token"] == hero)
        & (preds["model"] == arm)
        & (preds["matched_annotation_token"] == annotation)
    )
    assert bool(claim.any()), "no arm prediction claims the hero's recovered pedestrian"
    preds.loc[claim, "conf"] = 0.135
    preds.loc[claim, "status"] = "low_conf"
    preds.to_parquet(path, index=False)
    return built_demo_data


@pytest.fixture()
def built_demo_data_without_chat_replay(built_demo_data: Path) -> Path:
    """The same package with the recorded chat group removed -- the ordinary state
    of a fresh clone (`demo chat-record` is a paid, one-off run against a live
    provider, so it never ran there).

    Deletes the two files from an otherwise-normal built package rather than
    rebuilding one without the staging, for the same reason
    ``built_demo_data_without_explain`` does: the page's absent branch keys off
    ``data.chat_replay_available()``, i.e. the files' presence, and the builder's
    own absent/included recording is pinned on the builder side
    (tests/test_demo_export.py::test_include_chat_replay_absent_returns_absent_
    and_copies_nothing).
    """
    for name in ("chat_replays.json", "chat_replay_summary.json"):
        (built_demo_data / name).unlink()
    manifest_path = built_demo_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["validation"]["chat_replay"] = "absent"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return built_demo_data


def test_overview_page_renders_from_a_built_package(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AppTest smoke test locking the app<->data.py loader seam.

    Runs the real app/demo/main.py through Streamlit's in-process AppTest runner
    (streamlit.testing.v1) against a tiny package built by the real builder, and
    checks it renders without exception with a known metric on the page.

    Two things AppTest does NOT do for us, verified by first trying without them:
    - It does not add the script's own directory to sys.path the way `streamlit run`
      does (that fixup lives in the CLI bootstrap path, not the ScriptRunner AppTest
      drives) — so main.py's plain `from data import ...` / `from views import ...`
      need it prepended by hand, exactly as `streamlit run app/demo/main.py` would.
    - It does not know about DEMO_DATA_DIR — that's app/demo/data.py's own env-var
      override (added for this test), pointing the loaders at the tmp package instead
      of the committed demo_data/.
    monkeypatch undoes both after the test.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    at = AppTest.from_file(str(DEMO_DIR / "main.py")).run()

    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values["Camera keyframes"] == "3"

    # item I5 (Phase 6 review): this fixture stages graph_subgraphs/ before
    # run_build, so overview_metrics.json's flagship.cypher_source is the COMPUTED
    # one ("computed (neo4j, demo subgraphs)") -- the caption must not still promise
    # a live-graph export that already landed (and must not read "...sourced from
    # computed (neo4j, demo subgraphs) until the live-graph export lands").
    captions = [str(c.value) for c in at.caption]
    assert any("computed live against Neo4j at build time" in c for c in captions)
    assert not any("until the live-graph export lands" in c for c in captions)


# docs/DEMO_PLAN.md:697's credibility statement, verbatim. The public app serves
# artifacts, not a live pipeline, and Overview's footer has to say so in the plan's
# own words -- paraphrasing it would quietly weaken the claim it is making.
CREDIBILITY_STATEMENT = (
    "Results shown here were generated by the full offline pipeline. The public "
    "application serves curated experiment outputs for reproducibility and "
    "demonstration."
)


def test_overview_footer_carries_the_credibility_statement(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "how to read this demo" footer states where the results came from.

    This is the one sentence a visitor needs to read the whole app correctly: the
    numbers are the offline pipeline's, the deployment is only serving them.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    at = AppTest.from_file(str(DEMO_DIR / "main.py")).run()

    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any(CREDIBILITY_STATEMENT in caption for caption in captions), captions
    # The footer's existing sentence stays: the credibility statement is added to it,
    # not swapped in for it.
    assert any("label recorded outputs as recorded" in caption for caption in captions)


def test_overview_flagship_caption_when_the_cypher_twin_is_only_sourced(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(item I5) The other branch: a package built WITHOUT `demo subgraphs` staging
    keeps overview_metrics.json's sourced flagship.cypher_source (the Phase 1-5
    contract, pinned by tests/test_demo_export.py::test_build_notes_absent_
    subgraphs), and the caption stays the honest "sourced from ... until the
    live-graph export lands" sentence.

    Patches that one field on the built package rather than rebuilding without the
    staging: overview_metrics.json IS the page's whole input here, and the build's
    own absent-subgraphs behaviour is already pinned on the builder side.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    overview_path = built_demo_data / "overview_metrics.json"
    overview = json.loads(overview_path.read_text())
    overview["flagship"]["cypher_source"] = "docs/GRAPH.md"
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True))

    _reset_demo_app_modules()
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    at = AppTest.from_file(str(DEMO_DIR / "main.py")).run()

    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any(
        "sourced from docs/GRAPH.md until the live-graph export lands" in c
        for c in captions
    )
    assert not any("computed live against Neo4j" in c for c in captions)


def _failures_apptest(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    # url_path="failures" in main.py, same rationale as the other pages' explicit
    # url_paths (switch_page resolves by hashing the filename-derived name).
    at.switch_page("views/failures.py").run(timeout=30)
    return at


def test_failure_explorer_renders_grid_and_detail(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    assert not at.exception
    # Navigate to the Failure Explorer page. The plan's assumed `at.navigation`
    # page list does not exist on this installed streamlit (1.59.2) -- AppTest
    # only exposes `switch_page(page_path)`, which resolves the target page by
    # hashing a name derived from ``page_path``'s filename (no leading digits/
    # emoji, underscores kept) and matching it against each `st.Page`'s
    # `url_path` hash. main.py gives the Failure Explorer page an explicit
    # `url_path="failures"` so that hash matches "views/failures.py"'s derived
    # name exactly (verified empirically against this streamlit version).
    at.switch_page("views/failures.py").run(timeout=30)
    assert not at.exception
    assert any("val frames" in str(m.value) for m in at.caption)   # val-only caption present

    # Review fix #2 regression: the distance slider's default (full-extent) value
    # must be a true no-op. filter_frames only keeps a frame if >= 1 of its GT
    # rows falls inside a concrete distance_range, so "v1" (zero GT rows, an
    # all-FP hallucination frame) can never be "in range" under ANY concrete
    # range -- if the page ever passes the slider's default value straight
    # through instead of treating full-extent as None, v1 silently vanishes from
    # even the completely unfiltered grid. Pin: default state shows every val
    # frame (2 here), not just the ones with GT boxes.
    default_caption = next(str(m.value) for m in at.caption if "val frames" in str(m.value))
    matched, total = (int(n) for n in default_caption.split(" val frames")[0].split(" / "))
    assert matched == total == 2

    # Review fix #1 regression: the exemplar badge direction. "v0" is staged so
    # baseline misses GT box a2 (pedestrian) that graph_rate_night catches --
    # fixes_fn_vs_baseline_graph_rate_night=True means "baseline (A) missed it,
    # graph_rate_night (B) fixed it" (infer.py's fixes_fn_vs_<A>_<B> convention).
    # Selecting baseline (the misser) must show NO badge; selecting
    # graph_rate_night (the fixer) must show the badge, crediting the right
    # model on each side.
    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception
    assert not any("Exemplar" in str(s.value) for s in at.success)   # baseline: misser, no badge
    # Final-review fix #3: the frequency caption is shown regardless of whether
    # THIS model+frame combo currently has a badge -- it's global context ("fix-
    # pairs are common"), not per-badge decoration. Fixture: only "v0" has any
    # fixes_fn_vs_ column True, out of 2 val frames -- "1 of 2".
    assert any(
        "Fix-pairs are common across the curated set (1 of 2 val frames" in str(c.value)
        for c in at.caption
    )

    at.radio(key="failure_model").set_value("graph_rate_night").run(timeout=30)
    assert not at.exception
    assert any(
        "Exemplar: `graph_rate_night` catches a box `baseline` misses" in str(s.value)
        for s in at.success
    )
    assert any(
        "Fix-pairs are common across the curated set (1 of 2 val frames" in str(c.value)
        for c in at.caption
    )

    # Final-review fix #2: the empty state. graph_rate_night (now selected) has
    # zero FN across both fixture frames (v0's GT is fully matched_graph_rate_
    # night, v1 has no GT rows at all) -- "Has FN" must show 0 results AND the
    # st.info nudge, not just a "0 / 2" caption with an otherwise-blank page.
    at.radio(key="failure_type_select").set_value("Has FN").run(timeout=30)
    assert not at.exception
    zero_caption = next(str(m.value) for m in at.caption if "val frames" in str(m.value))
    assert zero_caption.startswith("0 / 2")
    assert any("No frames match these filters" in str(m.value) for m in at.info)

    # Beyond the plan's floor: also drive the detail view (grid buttons write
    # st.session_state["failure_token"], which a plain AppTest.run() never
    # clicks) -- this is the only path that exercises draw_overlay, the per-box
    # table, and the multi-bucket metadata panel, and it is where a real bug
    # was caught during self-review (curation_buckets round-trips through
    # parquet as a numpy array, and `array or []` raises ValueError for any
    # frame in more than one bucket -- see the fixture's two-bucket comment).
    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception


# --- Phase 9b (Task 3): the Failure Explorer as a visual hook ------------------------
#
# Spec docs/superpowers/specs/2026-08-22-demo-phase9b-design.md sec1: the page stops
# reading as a filter dashboard with a grid. The model radio moves out of the sidebar
# to sit beside the overlay it changes, the two "tuning" controls fold away into an
# "Advanced filters" expander, and the detail comes FIRST -- auto-selected from the
# sorted filtered set so the page never opens on a wall of thumbnails. Every string and
# key the earlier phases pinned (asserted in the test above) survives verbatim.


def test_failure_explorer_model_radio_sits_beside_the_image(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model choice is no longer a sidebar filter: it is instantiated in the detail
    header, next to the overlay it changes, while ``render()`` reads the value out of
    session state before filtering so the grid follows it on the rerun."""
    pytest.importorskip("streamlit")
    at = _failures_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    # The only radio left in the sidebar is the failure-type filter.
    assert [r.key for r in at.sidebar.radio] == ["failure_type_select"]
    model_radio = at.radio(key="failure_model")   # ... but the widget still exists
    # AppTest reports the option list as the widget sends it (format_func applied).
    # filters.model_label leaves any name it has no entry for untouched, so the
    # fixture's two models read the same either way -- this pins the raw values the
    # page filters on, not the champion/weak_* display labels.
    assert model_radio.options == ["baseline", "graph_rate_night"]
    assert model_radio.value == "baseline"

    # The radio really drives the detail beside it: v0's per-box table is its 2 GT
    # rows plus the selected model's claims -- baseline claims a1 only (3 rows),
    # graph_rate_night claims a1 and a2 (4 rows).
    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception
    assert len(at.dataframe[0].value) == 3
    at.radio(key="failure_model").set_value("graph_rate_night").run(timeout=30)
    assert not at.exception
    assert len(at.dataframe[0].value) == 4


def test_failure_explorer_advanced_filters_fold(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Curation bucket and Sort by fold into ``st.sidebar.expander("Advanced
    filters")``; the primary condition filters keep the spec's order above it."""
    pytest.importorskip("streamlit")
    at = _failures_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    folds = [e for e in at.sidebar.expander if e.label == "Advanced filters"]
    assert len(folds) == 1
    fold = folds[0]
    assert fold.multiselect(key="failure_bucket_select").label == "Curation bucket"
    assert fold.selectbox(key="failure_sort_select").label == "Sort by"

    # AppTest walks into the expander and flattens it into the same document order,
    # so this single list pins both the primary order and the fact that the two
    # advanced controls come last (inside the fold) -- and that the model radio is
    # not among them.
    sidebar_widgets = [
        node.label
        for node in at.sidebar
        if node.type in {"selectbox", "slider", "radio", "multiselect"}
    ]
    assert sidebar_widgets == [
        "Lighting",
        "Rain",
        "Category",
        "Size",
        "Distance to ego (m)",
        "Failure type",
        "Curation bucket",
        "Sort by",
    ]


def test_failure_explorer_auto_selects_the_first_frame(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detail renders on the very first load -- no click needed -- for the first
    frame of the sorted filtered set, saying so; an explicit View click (which writes
    ``failure_token``) wins and drops the auto-selected note."""
    pytest.importorskip("streamlit")
    at = _failures_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    assert "Detail" in [str(s.value) for s in at.subheader]
    assert len(at.image) > 0
    assert any("auto-selected" in str(c.value) for c in at.caption)
    # Not merely "some frame": the default sort is failure count, worst first, and
    # under baseline v1 (2 FPs) outranks v0 (1 FN) -- so the auto-selected frame is
    # v1. A page that just fell back to the manifest's first row would show v0.
    assert {m.label: m.value for m in at.metric}["Scene"] == "scene-v1"

    view_radio = at.radio(key="failure_view_mode")
    assert view_radio.options == ["Overlay", "GT only", "Predictions only"]

    per_box = [e for e in at.get("expander") if e.label == "Per-box detail"]
    assert len(per_box) == 1
    assert len(per_box[0].dataframe) == 1

    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Scene"] == "scene-v0"
    assert not any("auto-selected" in str(c.value) for c in at.caption)


def test_failure_explorer_empty_filter_shows_no_detail(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detail-first must not mean "detail always": a filter combination that empties
    the set still shows only the nudge -- no auto-selected frame, no per-box table."""
    pytest.importorskip("streamlit")
    at = _failures_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    # Same combination the grid/detail test uses for the empty state:
    # graph_rate_night has zero FN across both fixture frames.
    at.radio(key="failure_model").set_value("graph_rate_night").run(timeout=30)
    assert not at.exception
    at.radio(key="failure_type_select").set_value("Has FN").run(timeout=30)
    assert not at.exception

    caption = next(str(c.value) for c in at.caption if "val frames" in str(c.value))
    assert caption.startswith("0 / 2")
    assert any("No frames match these filters" in str(m.value) for m in at.info)
    assert "Detail" not in [str(s.value) for s in at.subheader]
    assert not any("auto-selected" in str(c.value) for c in at.caption)
    assert len(at.dataframe) == 0
    assert len(at.image) == 0


def test_overview_hero_renders_overlay(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    assert not at.exception   # hero overlay path must not raise even with tiny fixture boxes
    # Final-review fix: the old caption text was generic enough to read
    # identically whether the overlay path or the plain-image fallback rendered,
    # so this assertion previously passed on vibes. This fixture's hero token
    # ("v0") has real GT+pred boxes staged for it (built_demo_data), so
    # _hero_overlay must take the live-overlay branch, not the fallback -- pin a
    # substring unique to the corrected caption so a regression to the fallback
    # (or a wrong caption) fails loudly instead of silently matching either path.
    hero_captions = [caption for img in at.image for caption in img.captions]
    assert any("defeats all three models" in caption for caption in hero_captions)


def test_overview_hero_overlay_keeps_a_gt_box_whose_visibility_flag_is_na(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 9a final review item 6: ``_hero_overlay`` reads visibility through
    the shared ``filters.visible_gt`` helper, not an ad-hoc
    ``~gt_token["below_visibility_min"]`` -- a plain ``~`` on the nullable-boolean
    column propagates NA into the mask and silently drops the row, instead of
    keeping it the way ``visible_gt_boxes``'s own NA-is-not-below-the-floor rule
    says it should.

    Proven by the strongest signal this function exposes: with the hero token's
    only GT box's visibility flag left NA and no baseline predictions staged for
    it, the ad-hoc filter would leave BOTH the GT rows and the predictions empty
    and fall back to ``None`` (the plain-crop path, per the function's own
    ``if gt_token.empty and preds_token.empty`` check) -- the shared helper keeps
    the NA row, so an Image must come back instead.
    """
    pytest.importorskip("streamlit")
    gt_path = built_demo_data / "gt_boxes.parquet"
    gt = pd.read_parquet(gt_path)
    gt = gt.loc[gt["sample_data_token"] != "v0"].copy()
    gt = pd.concat(
        [
            gt,
            pd.DataFrame({
                "annotation_token": ["a2"], "sample_data_token": ["v0"],
                "category_group": ["pedestrian"],
                "x_min": [10.0], "y_min": [10.0], "x_max": [30.0], "y_max": [30.0],
                "matched_baseline": pd.array([False], dtype="boolean"),
                "matched_graph_rate_night": pd.array([True], dtype="boolean"),
                "below_visibility_min": pd.array([None], dtype="boolean"),
            }),
        ],
        ignore_index=True,
    )
    gt.to_parquet(gt_path, index=False)

    preds_path = built_demo_data / "predictions.parquet"
    preds = pd.read_parquet(preds_path)
    preds = preds.loc[
        ~((preds["sample_data_token"] == "v0") & (preds["model"] == "baseline"))
    ].copy()
    preds.to_parquet(preds_path, index=False)

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    from views import overview

    assert overview._hero_overlay("v0") is not None


# --- Scenario Search page (Task 3, Phase 5) -----------------------------------------
#
# built_demo_data's scenario_events.parquet ends up with exactly two events (see the
# fixture's Task-3 comments on the samples/canbus/annotations_3d blocks above):
#   - "s1": hard_braking_near_pedestrians (the flagship preset) -- NOT in the curated
#     val set, and its only filmstrip neighbor is v1 at t_plus1 (scene sceneX's other
#     CAM_FRONT keyframe).
#   - "v1": night_pedestrians -- IS in the curated val set (frame_manifest split=val),
#     with a real crop (copied by _include_curation) but zero GT boxes and two
#     baseline false-positive predictions staged. Its only filmstrip neighbor is s1 at
#     t_minus1.
# The staged semantic_search_results.parquet carries two distinct queries, one hit
# each ("v1", "v0").

PRESETS_UNDER_TEST = (
    "hard_braking_near_pedestrians", "night_pedestrians", "fast_cyclists",
    "rain_vru", "fn_pedestrians_night", "low_conf_braking",
)


def _reset_demo_app_modules() -> None:
    """Force every app/demo module to be re-imported fresh on the next AppTest run.

    Streamlit's AppTest re-executes main.py's top-level code on each .run(), but
    modules main.py (or a page module) merely `import`s -- data, filters, render,
    views.* -- are cached in sys.modules like any normal Python import and are
    NOT re-executed across separate `AppTest.from_file(...)` instances within the
    same pytest process. Left alone, data.py's module-level `DEMO_DATA =
    Path(os.environ.get("DEMO_DATA_DIR", ...))` stays pinned to whichever test's
    monkeypatched env var was active the FIRST time "data" was ever imported this
    process, silently ignoring every later test's own built_demo_data (verified
    directly: a second AppTest instance, with DEMO_DATA_DIR repointed at a
    different tmp package, still read the FIRST instance's directory). Most
    existing assertions never noticed, since every built_demo_data fixture
    produces the same token shape (same "v0"/"v1"/"s1" names) regardless of which
    test built it -- but a test that checks for a file's ABSENCE needs the
    CURRENT test's own directory actually read.

    data.py's loaders are ``@st.cache_data``-decorated and keyed by their arguments
    alone (most take none), so the same staleness applies to their CACHED RETURN
    VALUES across tests -- cleared here for the same reason, so a test that edits
    its own package's contents actually sees the edit.
    """
    import streamlit as st

    st.cache_data.clear()
    for name in list(sys.modules):
        module = sys.modules[name]
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).is_relative_to(DEMO_DIR):
            del sys.modules[name]


def _scenarios_apptest(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch):
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    at.switch_page("views/scenarios.py").run(timeout=30)
    return at


def test_scenario_preset_buttons_and_card_grid(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) All six preset buttons render; selecting one shows the family's
    scope caption and that preset's ranked event cards."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    for name in PRESETS_UNDER_TEST:
        assert at.button(key=f"scenario_preset_{name}")   # all six buttons present

    at.button(key="scenario_preset_night_pedestrians").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["scenario_preset"] == "night_pedestrians"
    # Dynamics-family scope caption -- the fixture's night_pedestrians event ("v1")
    # comes from the dataset-wide dynamics family, not the curated-val model family.
    assert any("Dataset-wide" in str(c.value) for c in at.caption)
    assert at.button(key="scenario_select_v1")   # the one ranked card for this preset


def test_scenario_event_viewer_curated_vs_not_curated(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) Selecting an event renders the viewer: crop+overlay for the curated
    event ("v1"), thumb + the honest "not in the curated prediction set" caption
    for the non-curated one ("s1")."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "night_pedestrians"
    at.session_state["scenario_token"] = "v1"
    at.run(timeout=30)
    assert not at.exception
    assert len(at.image) >= 1   # crop+draw_overlay rendered without raising
    assert not any("not in the curated prediction set" in str(c.value) for c in at.caption)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception
    assert any(
        "not in the curated prediction set" in str(c.value) for c in at.caption
    )


def test_scenario_filmstrip_slider_changes_readout(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) The filmstrip select_slider changes the displayed speed/accel readout."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    slider = at.select_slider(key="scenario_filmstrip")
    # s1's only non-NA filmstrip neighbor is v1 at t_plus1 (scene sceneX's other
    # CAM_FRONT keyframe) -- t_minus1/t_minus2/t_plus2 are all NA at this scene-edge.
    assert set(slider.options) == {"current", "t+1"}

    before_captions = [str(c.value) for c in at.caption]
    slider.set_value("t+1").run(timeout=30)
    assert not at.exception
    after_captions = [str(c.value) for c in at.caption]
    assert before_captions != after_captions   # the speed/accel readout changed


def test_scenario_flagship_badge(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) The flagship preset's header shows the SQL/Cypher parity line.

    Phase 6 (Task 3) replaces the old flagship-only, overview_metrics-sourced
    badge ("N events -- identical count ... Cypher sourced from ... until Phase
    6") with the generic per-preset parity line every dynamics preset now gets,
    read straight off that preset's own graph_subgraphs/<preset>.json --
    built_demo_data's _stage_subgraphs_with_event (Task 3) stages
    cypher_count=sql_count=1 (parity True) for the flagship.

    The line says WHAT the number counts and that the grid below it is capped
    (item I2, Phase 6 review): rendering a bare "Cypher: 786 · SQL: 786 ✓" above a
    30-card grid left the viewer to guess whether 786 or 30 was the answer. Here,
    with n_shown (1) >= sql_count (1), nothing was actually cut, so the "showing
    the top N" clause is suppressed and "keyframe" is singular (item 6, Phase 6
    follow-up review) -- see test_scenario_night_pedestrians_parity_line_states_
    the_full_population_count_separately_from_n_shown just below for the plural,
    clause-included case.
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.run(timeout=30)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any(c == "1 matching keyframe dataset-wide — Cypher 1 · SQL 1 ✓" for c in captions)


def test_scenario_night_pedestrians_parity_line_states_the_full_population_count_separately_from_n_shown(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(item 1, Phase 6 follow-up review) Pins sql_count/n_shown apart: the
    flagship's own fixture has sql_count == n_shown == 1, which a swap of the two
    numbers in _render_parity_line would pass unnoticed. night_pedestrians stages
    sql_count=cypher_count=5 (built_demo_data's _stage_subgraphs_with_event)
    while this fixture still ranks exactly one event ("v1") for it, so n_shown
    stays 1 -- a caption reading "5 ... showing the top 1" only assembles
    correctly if sql_count and n_shown are read from the right places.
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "night_pedestrians"
    at.run(timeout=30)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any(
        c == "5 matching keyframes dataset-wide — Cypher 5 · SQL 5 ✓ · showing the top 1"
        for c in captions
    )


def test_scenario_graph_panel_shows_parity_narrative_and_legend(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(f) Selecting the flagship event ("s1") renders the interactive graph
    panel's surrounding Streamlit elements: the header's Cypher/SQL parity line,
    the path narrative caption, and the six-node-type legend.

    ``streamlit_agraph.agraph`` is a custom component AppTest does not render (no
    real browser round-trip happens, so ``clicked`` is always its default/initial
    value) -- this only proves the surrounding page renders without raising and
    with the right text, not that the graph itself draws correctly or that a
    click updates the metadata panel. That's covered by the plan's Task 4 live
    `streamlit run` check (a documented limitation, spec
    docs/superpowers/specs/2026-08-20-demo-phase6-design.md §3).
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    captions = [str(c.value) for c in at.caption]
    # n_shown (1) >= sql_count (1) here -- singular, no "showing the top" clause
    # (item 6, Phase 6 follow-up review).
    assert any(c == "1 matching keyframe dataset-wide — Cypher 1 · SQL 1 ✓" for c in captions)
    # The narrative is built by filters.subgraph_narrative from
    # _stage_subgraphs_with_event's real assemble_subgraph output for "s1" -- see
    # that helper's docstring for why 5.00m (not 3.80m) is the right distance here.
    assert any(
        "Scene scene-X" in c and "EgoPose (hard braking -7.50 m/s²)" in c
        and "pedestrian at 5.00 m" in c
        for c in captions
    )
    assert any(
        "Scene" in c and "Sample" in c and "EgoPose" in c and "ObjectObservation" in c
        and "Category" in c and "Location" in c
        for c in captions
    )


def test_scenario_parity_mismatch_is_a_warning_not_grey_small_print(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(item M4, Phase 6 review) A recorded Cypher/SQL mismatch is a FINDING -- the
    honest reporting the exporter deliberately doesn't raise on for the three
    non-flagship dynamics presets. It must surface as st.warning, not as the same
    grey caption a clean parity gets.

    Patches the built package's own night_pedestrians JSON (the fixture stages
    parity True for every dynamics preset, and the flagship's mismatch can never
    reach the page at all -- `demo build` fails on it).
    """
    pytest.importorskip("streamlit")
    path = built_demo_data / "graph_subgraphs" / "night_pedestrians.json"
    payload = json.loads(path.read_text())
    payload["cypher_count"] = 4
    payload["parity"] = False
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))

    at = _scenarios_apptest(built_demo_data, monkeypatch)
    at.session_state["scenario_preset"] = "night_pedestrians"
    at.run(timeout=30)
    assert not at.exception

    assert any(
        "mismatch recorded" in str(w.value) and "Cypher 4" in str(w.value)
        for w in at.warning
    )
    assert not any("mismatch recorded" in str(c.value) for c in at.caption)


def test_scenario_graph_panel_absent_when_subgraphs_not_staged(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(g) A package built without `demo subgraphs` staging (the Phase 1-5
    contract, still valid in Phase 6 -- `demo build` never requires Neo4j) shows
    the honest absent note in the graph slot instead of crashing or silently
    rendering nothing. Deletes the staged graph_subgraphs/ from an otherwise-
    normal built package rather than rebuilding one from scratch without it --
    same idiom as the existing scenario_events.parquet/crop-file absence tests
    just above.
    """
    pytest.importorskip("streamlit")
    shutil.rmtree(built_demo_data / "graph_subgraphs")

    at = _scenarios_apptest(built_demo_data, monkeypatch)
    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception
    assert any("graph export not included in this package" in str(i.value) for i in at.info)
    # The header's parity line is silent (nothing to report), not a stale claim.
    captions = [str(c.value) for c in at.caption]
    assert not any("matching keyframes dataset-wide" in c for c in captions)


def test_scenario_graph_panel_shows_per_event_absent_note_not_whole_package_note(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(item M3 / item 4, Phase 6 follow-up review) A staged graph_subgraphs/
    package that just doesn't hold THIS event's subgraph is a DIFFERENT case from
    the whole-package absence covered by
    test_scenario_graph_panel_absent_when_subgraphs_not_staged just above: the
    package does carry a graph export, this one event isn't in it. Deletes the
    flagship preset's only staged event ("s1") from its own JSON (rather than the
    whole graph_subgraphs/ directory) so the panel must fall through to
    _GRAPH_EVENT_ABSENT_NOTE, and the two notes stay distinguishable on screen --
    a viewer must not read "no subgraph for this event" as "no graph in this
    package at all", or vice versa.

    Unreachable for a package actually built by `demo build` since
    build.py::_include_subgraphs' stale-staging guard, per that note's own
    comment -- this test edits the built package's JSON directly, after the
    build already succeeded, the same way test_scenario_parity_mismatch_is_a_
    warning_not_grey_small_print edits night_pedestrians' JSON just above.
    """
    pytest.importorskip("streamlit")
    path = built_demo_data / "graph_subgraphs" / "hard_braking_near_pedestrians.json"
    payload = json.loads(path.read_text())
    del payload["events"]["s1"]
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))

    at = _scenarios_apptest(built_demo_data, monkeypatch)
    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    infos = [str(i.value) for i in at.info]
    assert any("no subgraph staged for this event" in i for i in infos)
    assert not any("graph export not included in this package" in i for i in infos)


def test_scenario_model_preset_shows_gt_only_note(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(h) A model-result preset's header shows the graph-holds-GT-only note, not
    a Cypher/SQL parity line -- the model verdict lives in the prediction set,
    never the graph (subgraph_export.py's own model-preset ``note``)."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.button(key="scenario_preset_fn_pedestrians_night").click().run(timeout=30)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any("graph holds GT only" in c and "prediction set" in c for c in captions)
    assert not any("matching keyframes dataset-wide" in c for c in captions)


def test_scenario_semantic_gallery_renders(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(e) The semantic section renders under the recorded-not-live banner."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    assert any("Recorded semantic search" in str(c.value) for c in at.caption)
    all_text = [str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption]
    assert any("crowded nighttime pedestrian crossing" in t for t in all_text)
    assert any("foggy road with heavy lens glare" in t for t in all_text)
    # item 7c, consolidated review: "N of k front-camera hits" -- the fixture
    # stages k=2 with exactly 1 hit per query, so N != k here (not just echoing
    # the row count back as both numbers).
    assert any("1 of 2 front-camera hits" in t for t in all_text)


def test_scenario_every_preset_renders_without_exception(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four of the six presets have zero matching events in this tiny fixture
    (fast_cyclists, rain_vru, fn_pedestrians_night, low_conf_braking) -- the
    empty-card-grid path (and a stale `scenario_token` selected under a
    different, non-empty preset) must render cleanly rather than raising."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    at.session_state["scenario_token"] = "s1"   # stale selection from another preset

    for name in PRESETS_UNDER_TEST:
        at.button(key=f"scenario_preset_{name}").click().run(timeout=30)
        assert not at.exception, f"preset {name!r} raised: {at.exception}"
        assert at.session_state["scenario_preset"] == name


def test_scenario_page_errors_on_stale_package_missing_events(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(item 4, consolidated review) A package built by an older `demo build` (no
    scenario_events.parquet, package_version < 0.4) must show a directive
    st.error, not a bare FileNotFoundError/ValueError traceback."""
    pytest.importorskip("streamlit")
    (built_demo_data / "scenario_events.parquet").unlink()

    at = _scenarios_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert any("needs demo_data >= 0.4" in str(e.value) for e in at.error)


def test_scenario_curated_event_falls_back_to_thumb_when_crop_missing(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(item 5, consolidated review) A curated event whose crop file is missing
    (a partial/corrupted package) must still render via its thumb, the same
    resilience the non-curated branch already had -- not render nothing."""
    pytest.importorskip("streamlit")
    (built_demo_data / "sample_frames" / "crops" / "v1.jpg").unlink()

    at = _scenarios_apptest(built_demo_data, monkeypatch)
    at.session_state["scenario_preset"] = "night_pedestrians"
    at.session_state["scenario_token"] = "v1"
    at.run(timeout=30)
    assert not at.exception
    assert len(at.image) >= 1   # the thumb-fallback overlay still rendered
    assert not any("no image available" in str(c.value) for c in at.caption)


# --- Phase 7 (Task 4): the Active Learning page --------------------------------


def _active_learning_apptest(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    # url_path="active_learning" in main.py, same rationale as the Failure Explorer's
    # and Scenario Search's explicit url_paths (see those pages' comments).
    at.switch_page("views/active_learning.py").run(timeout=30)
    return at


def test_active_learning_page_renders_story_chart_and_exemplar_table(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a/b/e) The page's spine: the six story beats, the arm chart, and the
    before/after exemplar controls + per-box table -- every number read from the
    package (the fixture's graph_rate_night arm is +0.0300 night against baseline).
    """
    pytest.importorskip("streamlit")
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    # (a) story arrows: render.story_arrows writes "**<title>**  \n<body>" markdown
    markdowns = [str(m.value) for m in at.markdown]
    assert any("Result" in text for text in markdowns)
    assert any("Problem" in text for text in markdowns)

    # the derived night delta, from active_learning_results.parquet (0.13 - 0.10)
    numbers = [str(m.value) for m in at.metric] + [str(c.value) for c in at.caption] + markdowns
    assert any("+0.0300" in text for text in numbers)

    # (b) the arm chart itself (st.altair_chart -> a "vega_lite_chart" element;
    # AppTest has no typed accessor for charts, so it is fetched by element type)
    assert len(at.get("vega_lite_chart")) >= 1

    # (e) exemplar controls: the selectbox over al_exemplars.json's tokens and the
    # model radio, whose options are the package's own model labels.
    assert at.selectbox(key="al_exemplar")
    assert at.radio(key="al_model").options == ["baseline", "graph_rate_night"]

    # (e) the per-box table: v0's GT box a2 is caught by graph_rate_night (tp 0.8)
    # and never claimed by baseline -- exactly one upgraded row.
    tables = [d.value for d in at.dataframe if "arm_claim" in getattr(d.value, "columns", [])]
    assert len(tables) == 1
    assert len(tables[0]) >= 1
    assert list(tables[0]["baseline_claim"]) == ["none"]


def test_active_learning_page_explain_absent_note(
    built_demo_data_without_explain: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) Without `demo al-explain` the gallery still works, and the page says
    exactly what is missing instead of inventing a per-frame reason."""
    pytest.importorskip("streamlit")
    at = _active_learning_apptest(built_demo_data_without_explain, monkeypatch)
    assert not at.exception

    notes = [str(i.value) for i in at.info]
    assert any(
        "per-frame community and routed mass are not included in this package "
        "(demo al-explain)" in note
        for note in notes
    )
    # the gallery is still there (a "View" button per AL-selected frame)
    assert at.button(key="al_frame_select_wA")


def test_active_learning_page_why_selected_panel_with_explain(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) With the group staged, picking a selected frame shows the derived factor
    panel -- community/quota/pass/degree rank -- and says plainly that a train-pool
    frame carries no predictions. Routed failure mass is CONTEXT, never the reason:
    "wA" has none, and the panel says so rather than claiming a high failure rate.
    """
    pytest.importorskip("streamlit")
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    at.session_state["al_frame_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception

    panel = [str(m.value) for m in at.markdown]
    assert any("Community quota" in text for text in panel)
    assert any("#0 · 10 frames · 2 at night" in text for text in panel)
    assert any("night pass (night floor 1)" in text for text in panel)
    assert any("none — not itself a routing target" in text for text in panel)
    assert not any(
        "per-frame community and routed mass are not included" in str(i.value) for i in at.info
    )
    assert any(
        "train-pool frame — no predictions" in str(c.value) for c in at.caption
    )


def test_active_learning_page_on_a_pre_phase_7_package(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package built before Phase 7 carries none of this page's tables --
    al_exemplars.json, al_communities.parquet, the al_explain group -- and a
    frame_manifest without ``al_selected_by``. The page must say what is missing
    and stop, not raise (consolidated review M11).

    Stripped from a built package rather than rebuilt, the same idiom
    test_weak_supervision_page_on_a_pre_phase_7_package uses.
    """
    pytest.importorskip("streamlit")
    for name in (
        "al_exemplars.json", "al_communities.parquet",
        "al_selection_explain.parquet", "al_explain_validation.json",
    ):
        (built_demo_data / name).unlink()
    manifest_path = built_demo_data / "frame_manifest.parquet"
    pd.read_parquet(manifest_path).drop(columns=["al_selected_by"]).to_parquet(
        manifest_path, index=False
    )

    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    notes = [str(e.value) for e in at.error] + [str(i.value) for i in at.info]
    assert any("needs demo_data >= 0.6" in note for note in notes)


def test_active_learning_page_with_an_empty_exemplar_list(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``al_exemplars.json`` with no tokens is a real, allowed state (a package built
    with curation absent) -- the before/after section says so instead of raising on
    an empty selectbox (consolidated review M11)."""
    pytest.importorskip("streamlit")
    path = built_demo_data / "al_exemplars.json"
    package = json.loads(path.read_text())
    package["tokens"] = []
    path.write_text(json.dumps(package, indent=2, sort_keys=True))

    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert any("no exemplar frames in this package" in str(i.value) for i in at.info)
    # the rest of the page is unaffected -- the arm chart is still there
    assert len(at.get("vega_lite_chart")) >= 1


def test_active_learning_gallery_pages_night_frames_first(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """78 selected frames rendered as 78 images and 78 buttons in one grid is a wall
    (consolidated review M8): the gallery shows the first 24, night frames first
    (this arm is night-targeted), behind a "show all" checkbox.

    The built package's manifest is widened with synthetic selected frames rather
    than rebuilt -- the page reads ``al_selected_by``/``is_night``/``scene_name``
    and nothing else about them, and no crop needs to exist for a thumb-less frame.
    """
    pytest.importorskip("streamlit")
    manifest_path = built_demo_data / "frame_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    extra = pd.DataFrame({
        "sample_data_token": [f"g{i}" for i in range(30)],
        "scene_name": [f"scene-{i:03d}" for i in range(30)],
        # the last one is the ONLY night frame, so night-first ordering must lift it
        # out of the tail and into the first page
        "is_night": [False] * 29 + [True],
        "al_selected_by": pd.array(["graph_rate_night"] * 30, dtype="string"),
        "split": ["train_pool"] * 30,
    })
    pd.concat([manifest, extra], ignore_index=True).to_parquet(manifest_path, index=False)

    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert any("night frames first" in str(c.value) for c in at.caption)
    # night frame first, day frame 29 (alphabetically last) past the first page
    assert at.button(key="al_frame_select_g29")
    assert not [b for b in at.button if b.key == "al_frame_select_g28"]

    at.checkbox(key="al_gallery_show_all").set_value(True).run(timeout=30)
    assert not at.exception
    assert at.button(key="al_frame_select_g28")


# --- Phase 7 (Task 5): the Weak Supervision page -------------------------------


def _weak_supervision_apptest(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    # url_path="weak_supervision" in main.py, same rationale as the other pages'
    # explicit url_paths (switch_page resolves by hashing the filename-derived name).
    at.switch_page("views/weak_supervision.py").run(timeout=30)
    return at


def _add_second_weak_pair(built_demo_data: Path) -> None:
    """Give the built package a SECOND weak/GT pair (the real package carries two:
    the documented ``random`` headline and the ``graph_rate_night`` pair Overview
    promises). This fixture's results.json only has ``weak_random``/
    ``weak_random_gt``, so the non-headline card/caption/arrow would otherwise be
    untestable -- the row is appended to the built parquet directly, the same
    idiom the scenario tests use to edit a built package after the fact.

    Deliberately NO ``weak_graph_rate_night`` row is added to
    active_learning_results.parquet: the page must then say plainly that the
    trained arm isn't in this package rather than inventing a delta for it.
    """
    path = built_demo_data / "weak_loss_decomposition.parquet"
    existing = pd.read_parquet(path)
    second = pd.DataFrame([{
        "base_arm": "graph_rate_night", "gt_gain": 0.03, "weak_gt_gain": 0.02,
        "weak_gain": 0.01181, "retention": 0.3937,
        "dropped_frame_cost": 0.01, "dropped_frame_share": 0.01 / 0.03,
        "label_cost": 0.00819, "label_share": 0.00819 / 0.03, "headline": False,
    }])
    pd.concat([existing, second], ignore_index=True).to_parquet(path, index=False)


def test_weak_supervision_page_cards_decomposition_bias_and_tabs(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a)-(e) The page's spine: both retention pairs as cards, the loss split, the
    crowded-frame bias, and the accepted/rejected gallery with the VLM-vs-GT
    counts -- every number read from the package (the fixture's headline pair is
    0.0018/0.0100 = 18.0% retained, 50.0% dropped-frame, 32.0% label cost).
    """
    pytest.importorskip("streamlit")
    _add_second_weak_pair(built_demo_data)
    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    # (a) both pairs, the documented headline attributed as such
    # Short, backtick-free card labels: st.metric breaks its label mid-TOKEN in a
    # narrow column ("graph_ra te_night"), so the arm names carry no markup and the
    # night card's arm pair moved to a caption (real-browser finding 4).
    labels = {m.label: m.value for m in at.metric}
    assert labels["GT gain retained — random (headline)"] == "18.0%"
    assert labels["GT gain retained — graph_rate_night"] == "39.4%"
    assert "`" not in "".join(labels)
    # verifier retention (a DIFFERENT quantity: what the verifier kept, not what
    # the training run retained) and the night pedestrian slice
    assert labels["Verifier retention — random"] == "60%"
    assert "Night pedestrian mAP50-95" in labels

    # (b)/(c) the stacked loss decomposition and the paired crowding chart
    assert len(at.get("vega_lite_chart")) >= 2
    captions = [str(c.value) for c in at.caption]
    # the "(headline)" card suffix carries its own attribution caption, naming
    # docs/ACTIVE_LEARNING.md and clarifying it marks the LOWER share, not the
    # better one (real-browser finding: a bare "(headline)" read as "best").
    assert any(
        "(headline) marks the pair docs/ACTIVE_LEARNING.md publishes" in c
        and "the lower of the two shares, not the better one" in c
        for c in captions
    )
    assert any("18.0%" in c and "50.0%" in c and "32.0%" in c for c in captions)
    assert any("accepted frames average 3.00 GT boxes/frame" in c for c in captions)
    by_class = [
        d.value for d in at.dataframe
        if "mutual_zero_share" in list(getattr(d.value, "columns", []))
    ]
    assert len(by_class) == 1
    assert "16.7%" in list(by_class[0]["mutual_zero_share"])

    # (d) the two tabs over frame_manifest.weak_verdict
    assert {"accepted", "rejected"} <= {str(tab.label) for tab in at.tabs}

    at.session_state["ws_accepted_token"] = "wA"
    at.session_state["ws_rejected_token"] = "wR"
    at.run(timeout=30)
    assert not at.exception

    text = [str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption]
    assert any("accepted — 1 pseudo box" in t for t in text)
    assert any("rejected — no pseudo boxes by construction" in t for t in text)
    assert any(
        "no pseudo boxes exist for rejected frames by construction" in t for t in text
    )
    assert len(at.image) >= 1                                   # wA's crop + pseudo box
    # label_confidence is the VLM's string enum, rendered as text: formatting it as
    # a float raised ValueError and took the page down on every frame the VLM had
    # labelled (consolidated review C1) -- 32 of the 78 real gallery frames.
    assert any("label confidence high" in t for t in text)
    counts = [
        d.value for d in at.dataframe
        if list(getattr(d.value, "columns", [])) == ["class", "vlm_count", "gt_count"]
    ]
    assert len(counts) == 2                                     # one per tab
    accepted_counts = counts[0]
    assert list(accepted_counts.loc[accepted_counts["class"] == "car", "vlm_count"]) == ["1"]

    # the downstream result stated at ARM level (train-pool frames have no
    # predictions), from active_learning_results: weak_random is +0.0050 night
    assert any(
        "`weak_random`" in t and "+0.0050" in t and "+0.0018" in t for t in text
    )
    # ... and the pair whose trained arm this package doesn't carry says so
    assert any("`weak_graph_rate_night`" in t and "not in this package" in t for t in text)


def test_weak_supervision_mutual_zero_flag(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted frame with zero pseudo boxes is the MUTUAL-ZERO case, and the
    badge says so -- "accepted" alone would read as "the VLM labelled it", when
    what actually happened is that the verifier agreed with the detector that the
    frame holds none of the five classes.

    Empties wA's rows out of the built package's weak_labels.parquet (the fixture
    stages exactly one pseudo box, for wA) rather than rebuilding: the badge keys
    off the frame's pseudo-box count, and the exporter's own "accepted frames with
    no rows are simply absent" contract is pinned on the builder side.
    """
    pytest.importorskip("streamlit")
    labels_path = built_demo_data / "weak_labels.parquet"
    pd.read_parquet(labels_path).iloc[0:0].to_parquet(labels_path, index=False)

    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    at.session_state["ws_accepted_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception

    text = [str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption]
    assert any("accepted — 0 pseudo boxes (mutual zero)" in t for t in text)


def test_weak_supervision_gallery_pages_its_tabs(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accepted tab holds 47 frames in the real package -- one grid of 47 images
    and 47 buttons is the same wall the AL gallery was (consolidated review M8), so
    each tab shows the first 24 behind its own "show all" checkbox."""
    pytest.importorskip("streamlit")
    manifest_path = built_demo_data / "frame_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    extra = pd.DataFrame({
        "sample_data_token": [f"a{i}" for i in range(30)],
        "scene_name": [f"scene-{i:03d}" for i in range(30)],
        "is_night": [False] * 30,
        "weak_verdict": pd.array(["accepted"] * 30, dtype="string"),
        "split": ["train_pool"] * 30,
    })
    pd.concat([manifest, extra], ignore_index=True).to_parquet(manifest_path, index=False)

    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert any(
        str(c.value).startswith("The first 24 of ") and "verifier accepted" in str(c.value)
        for c in at.caption
    )
    assert not [b for b in at.button if b.key == "ws_accepted_select_a29"]

    at.checkbox(key="ws_accepted_show_all").set_value(True).run(timeout=30)
    assert not at.exception
    assert at.button(key="ws_accepted_select_a29")
    # the rejected tab has 1 frame, well under the page size -- no checkbox for it
    assert not [c for c in at.checkbox if c.key == "ws_rejected_show_all"]


def test_overview_no_longer_promises_phase_7(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overview's live promise is fulfilled: the non-headline pair's sentence names
    the page in the present tense, and no caption still says "Phase 7".

    The fixture's results.json carries a single weak/GT pair, so a second
    by_base_arm entry is patched into the built overview_metrics.json (the page's
    whole input for this sentence) -- same idiom as
    test_overview_flagship_caption_when_the_cypher_twin_is_only_sourced.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    overview_path = built_demo_data / "overview_metrics.json"
    overview = json.loads(overview_path.read_text())
    overview["results"]["weak_retention"]["by_base_arm"]["graph_rate_night"] = 0.3937
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True))

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    assert not at.exception

    captions = [str(c.value) for c in at.caption]
    assert not any("Phase 7" in c for c in captions)
    assert any(
        "`graph_rate_night` (39.4%)" in c and "The Weak Supervision page presents it." in c
        for c in captions
    )


def test_overview_outcome_first(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 9a (spec §4): the Overview leads with the CTA into the guided tour,
    the loop strip, and four flagship metrics; the Scale cards and the
    architecture restatement move into collapsed expanders rather than
    disappearing -- every pre-existing pinned string/metric must still be present
    (AppTest walks into expanders, so moving content there keeps it reachable)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    at = AppTest.from_file(str(DEMO_DIR / "main.py")).run(timeout=30)

    assert not at.exception
    assert at.button(key="overview_start_tour")

    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        ":orange-badge[" in text or ":gray-badge[Diagnose]" in text for text in markdowns
    )
    captions = [str(c.value) for c in at.caption]
    assert any("System loop" in c for c in captions)

    # The headline row IS the first four metrics on the page, in the documented
    # order (Phase 9a review M8): a set/superset check passed just as happily when
    # the row had five cards, or when the scale expander's cards came first.
    # Everything after index 3 belongs to the "Dataset scale & data checks"
    # expander, which AppTest walks into and flattens into the same list.
    assert [str(m.label) for m in at.metric][:4] == [
        "Best night gain",
        "Night pedestrian mAP50-95",
        "Weak-sup share of GT gain",
        "Graph = SQL flagship",
    ]
    metric_values = {str(m.label): str(m.value) for m in at.metric}
    assert metric_values["Camera keyframes"] == "3"   # still reachable, inside the expander

    expander_labels = [str(e.label) for e in at.get("expander")]
    # "& data checks" (review M2): the CAN-speed correlation card in this drawer is
    # a sanity check on the CAN join, not a scale figure.
    assert "Dataset scale & data checks" in expander_labels
    assert "Architecture (for technical reviewers)" in expander_labels
    assert any("TRINITY" in text for text in markdowns)

    # the flagship/retention captions -- test_overview_page_renders_from_a_built_
    # package's and test_overview_footer_carries_the_credibility_statement's own
    # pinned substrings. They are provenance about the headline cards, not
    # architecture, so review M1 moved them into their own expander directly under
    # those cards; AppTest walks into expanders, so both are still found here.
    assert "Where these numbers come from" in expander_labels
    assert any("computed live against Neo4j at build time" in c for c in captions)
    assert any(
        "of the ground-truth mAP gain for the" in c and "headline)." in c for c in captions
    )
    # named, not positional: the caption no longer sits under the card it is about.
    assert any("The Weak-sup share of GT gain card is" in c for c in captions)

    # both footer sentences, unmoved.
    assert any(CREDIBILITY_STATEMENT in c for c in captions)
    assert any("label recorded outputs as recorded" in c for c in captions)


def test_overview_cta_switches_to_the_tour(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking the CTA sends the visitor into the guided tour at step 0."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    at = AppTest.from_file(str(DEMO_DIR / "main.py")).run(timeout=30)
    assert not at.exception

    at.button(key="overview_start_tour").click().run(timeout=30)
    assert not at.exception
    assert [str(title.value) for title in at.title] == ["Guided tour"]
    assert at.session_state["tour_step"] == 0


def test_weak_supervision_page_on_a_pre_phase_7_package(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package built before Phase 7 (the committed demo_data/ is still
    package_version 0.5) carries weak_supervision_results.parquet in its OLD,
    narrower schema -- no ``tolerance``/``conf``/``gt_boxes_per_rejected_frame``/
    ``gt_boxes_per_candidate_frame`` -- no decomposition/by-class/labels/counts
    tables at all, and a frame_manifest without ``weak_verdict``. The page must
    then show what the package does carry and say what is missing, rather than
    raising an AttributeError on a column that didn't exist yet.

    Rebuilt from the built package by dropping exactly those columns/files (the
    builder's own absent/included recording is pinned on the builder side) --
    same idiom as test_scenario_page_errors_on_stale_package_missing_events.
    """
    pytest.importorskip("streamlit")
    for name in (
        "weak_loss_decomposition.parquet", "weak_verifier_by_class.parquet",
        "weak_labels.parquet", "vlm_counts.parquet",
    ):
        (built_demo_data / name).unlink()
    weaksup_path = built_demo_data / "weak_supervision_results.parquet"
    pd.read_parquet(weaksup_path)[[
        "arm", "n_candidates", "n_accepted", "verifier_retention", "n_pseudo_boxes",
        "boxes_per_accepted_frame", "gt_boxes_per_accepted_frame",
    ]].to_parquet(weaksup_path, index=False)
    manifest_path = built_demo_data / "frame_manifest.parquet"
    pd.read_parquet(manifest_path).drop(columns=["weak_verdict"]).to_parquet(
        manifest_path, index=False
    )

    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    assert any("needs demo_data >= 0.6" in str(i.value) for i in at.info)
    # the one thing the old package does carry is still on screen
    assert any(m.label == "Verifier retention — random" for m in at.metric)


def _chat_replay_apptest(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    # url_path="chat_replay" in main.py, same rationale as the other pages' explicit
    # url_paths (switch_page resolves by hashing the filename-derived name).
    at.switch_page("views/chat_replay.py").run(timeout=30)
    return at


def test_chat_replay_page_renders_showcase_and_graded(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page's spine: the recorded banner and cards (every figure derived from
    the staged session), the showcase replay in full (answer, chart, frames, steps),
    and the graded selectbox -- including the failing case's verdict and the errored
    case's provider error, both shown rather than hidden."""
    pytest.importorskip("streamlit")
    at = _chat_replay_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    captions = [str(c.value) for c in at.caption]
    assert any(
        "Recorded answers from the dataset chat agent (claude-test, recorded "
        "2026-08-21)" in caption
        and "the live agent runs in the local stack" in caption
        for caption in captions
    )

    # (1) cards: two cards only (a wider "Model"/"Tools exercised" value would
    # truncate inside st.metric's large font -- consolidated review, real-browser
    # finding) + counts derived from the staged records (4 replays: 1 showcase + 3
    # eval, of which 1 passed; steps: 2 SQL, 1 Cypher, 1 search, 1 show_frames, 1
    # chart).
    metrics = {(str(m.label), str(m.value)) for m in at.metric}
    assert ("Questions recorded", "4") in metrics
    assert ("Graded pass rate", "1/3") in metrics
    assert not any(label == "Model" for label, _ in metrics)
    assert not any(label == "Tools exercised" for label, _ in metrics)
    assert (
        "Model `claude-test` · tools exercised: SQL 2 · Cypher 1 · "
        "semantic search 1 · frames shown 1 · charts 1"
    ) in captions

    # (2) the showcase replay: question, answer text, its chart and both frames.
    markdowns = [str(m.value) for m in at.markdown]
    assert any("Which locations have the most night frames?" in text for text in markdowns)
    assert any("Singapore-onenorth has the most night frames" in text for text in markdowns)
    assert len(at.get("vega_lite_chart")) >= 1
    assert len(at.image) >= 2
    assert "scene-0916 · singapore-onenorth · night, rain" in captions
    assert "scene-0001 · boston-seaport · day" in captions

    # (2) the steps expander, its SQL code block, and the honesty line about what a
    # step does NOT carry.
    assert "Agent steps (3)" in [str(e.label) for e in at.get("expander")]
    codes = {(str(c.value), c.language) for c in at.code}
    assert ("SELECT location, count(*) FROM scenes GROUP BY location", "sql") in codes
    assert any("Raw SQL result rows are not stored" in caption for caption in captions)

    # (3) the graded selectbox: the eval replays, labelled by their own verdicts.
    graded = at.selectbox(key="chat_replay_graded")
    assert graded.options == [
        "✓ How many scenes are in the dataset?",
        "✗ What is the mean ego speed?",
        "✗ Which scenes have the most pedestrians at night?",
    ]
    assert any(
        "✓ passed (english, tool_use, grounded, numeric) · reference 66" in caption
        for caption in captions
    )

    # (3) the failing case names its failing check and the reference it missed.
    at.selectbox(key="chat_replay_graded").select("eval_mean_speed").run(timeout=30)
    assert not at.exception
    assert any(
        "✗ failed: numeric · reference 2.91" in str(c.value) for c in at.caption
    )
    assert any("MATCH (e:EgoState)" in str(c.value) for c in at.code)
    assert any(
        "query: fast highway driving" in str(m.value) for m in at.markdown
    )

    # (4) the errored case renders the provider's error, not a blank answer.
    at.selectbox(key="chat_replay_graded").select("eval_night_pedestrians").run(timeout=30)
    assert not at.exception
    assert any(
        "overloaded_error: the provider dropped the session" in str(e.value) for e in at.error
    )
    assert any("✗ errored — overloaded_error" in str(c.value) for c in at.caption)


def test_chat_replay_question_with_trailing_newline_renders_without_a_broken_bold_span(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several of the real recorded questions are YAML folded scalars and end in a
    literal trailing ``\\n``. Unstripped, that newline lands INSIDE the showcase
    heading's bold span (``**question**``) and CommonMark never closes the ``**``
    across it, so the page showed literal asterisks; the same trailing newline
    also left the graded selectbox's option label multi-line. The fix is
    display-only -- mutate the BUILT package's own `chat_replays.json` (as the
    real recorder's YAML-folded-scalar output would already look) rather than the
    staging fixture, so the record on disk stays exactly what `demo chat-record`
    would have written."""
    pytest.importorskip("streamlit")
    replays_path = built_demo_data / "chat_replays.json"
    records = json.loads(replays_path.read_text())
    for record in records:
        if record["id"] in ("show_night_locations", "eval_scene_count"):
            record["question"] = record["question"] + "\n"
    replays_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    at = _chat_replay_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    # The showcase heading's bold span closes right after the (stripped)
    # question text -- no newline snuck inside it to leave the "**" unclosed.
    markdowns = [str(m.value) for m in at.markdown]
    assert "**Which locations have the most night frames?**" in markdowns

    # The graded selectbox's label is a single line too.
    graded = at.selectbox(key="chat_replay_graded")
    assert "✓ How many scenes are in the dataset?" in graded.options
    assert all("\n" not in option for option in graded.options)


def test_chat_replay_page_absent_note(
    built_demo_data_without_chat_replay: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No recorded session in the package (the fresh-clone state, since `demo
    chat-record` is a paid run) -- one honest note naming the command that produces
    one, and nothing else claimed."""
    pytest.importorskip("streamlit")
    at = _chat_replay_apptest(built_demo_data_without_chat_replay, monkeypatch)
    assert not at.exception

    assert "no recorded sessions in this package (demo chat-record)" in [
        str(info.value) for info in at.info
    ]
    assert not at.get("vega_lite_chart")
    assert not at.selectbox


def test_chat_replay_page_on_a_pre_0_7_package(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package built before package_version 0.7 carries no chat group at all --
    and no manifest key for it either. The page keys on the FILES' presence, not on
    the manifest, so it draws the same note (here the manifest is deliberately left
    claiming "included")."""
    pytest.importorskip("streamlit")
    for name in ("chat_replays.json", "chat_replay_summary.json"):
        (built_demo_data / name).unlink()

    at = _chat_replay_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert "no recorded sessions in this package (demo chat-record)" in [
        str(info.value) for info in at.info
    ]
    manifest = json.loads((built_demo_data / "manifest.json").read_text())
    assert manifest["validation"]["chat_replay"] == "included"


# --- the guided tour (Phase 9a) -----------------------------------------------------
#
# views/tour.py is one page walking seven steps out of st.session_state["tour_step"];
# Task 2 implements steps 0 (the night weakness) and 1 (the hero frame's missed
# pedestrian) and registers 2-6 as placeholders the next tasks fill in. It is reached
# with switch_page("views/tour.py") -- an in-script st.switch_page is not sticky
# across at.run() -- which resolves the page by hashing the filename-derived name
# ("tour") against each st.Page's url_path, hence url_path="tour" in main.py.

# Every page of the sectioned navigation, and the title each one renders. The
# Overview is absent on purpose: it is the DEFAULT page (url_path "", script hash
# calc_hash("render") from the callable's name), so switch_page("views/overview.py")
# cannot address it -- the initial at.run() lands there instead, which is what
# test_navigation_sections_keep_every_page_reachable asserts first.
_SECTIONED_PAGES = (
    ("views/tour.py", "Guided tour"),
    ("views/failures.py", "Failure Explorer"),
    ("views/scenarios.py", "Scenario Search"),
    ("views/active_learning.py", "Active Learning"),
    ("views/weak_supervision.py", "Weak Supervision"),
    ("views/chat_replay.py", "Ask the Dataset (recorded)"),
)


def _tour_apptest(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    at.switch_page("views/tour.py").run(timeout=30)
    return at


def test_tour_walks_steps_0_and_1(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tour opens on step 0 (baseline's night weakness: recorded mAP cards plus
    one recomputed frame count) and Next walks to step 1 (the hero frame's missed
    pedestrian, with the model toggle) -- every number derived from the fixture
    package, never written into the page."""
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    assert [str(title.value) for title in at.title] == ["Guided tour"]
    captions = [str(caption.value) for caption in at.caption]
    assert any(text.startswith("Step 1 of 7 · The weakness: night") for text in captions)

    # The nav row is drawn FIRST, and Back is dead on the first step.
    assert at.button(key="tour_back").disabled is True
    assert at.button(key="tour_next").disabled is False

    # (step 0) the baseline arm's three cards, straight off
    # active_learning_results.parquet (fixture baseline: 0.20 / 0.10 / ped 0.05).
    assert [str(metric.label) for metric in at.metric] == [
        "Overall mAP50-95", "Night mAP50-95", "Night pedestrian mAP50-95",
    ]
    assert [str(metric.value) for metric in at.metric] == ["0.2000", "0.1000", "0.0500"]

    markdowns = [str(block.value) for block in at.markdown]
    # (step 0) the recomputed line: the fixture's only night val frame ("v0")
    # carries baseline's miss of GT box a2 -> 1 of 1.
    assert any(
        "1 of 1 night validation frames carry at least one baseline miss" in text
        for text in markdowns
    )
    # the breadcrumb lights this step's stage only
    assert any(
        ":orange-badge[Diagnose]" in text and ":gray-badge[Mine]" in text for text in markdowns
    )
    # both provenance claims, one per kind of number on the step
    assert any("recorded experiment output" in text for text in captions)
    assert any("recomputed in this app" in text for text in captions)
    # "Go deeper" -> the Failure Explorer, resolved through the nav registry
    # (AppTest has no typed accessor for page links, so they are fetched by element
    # type and read off the proto).
    links = [str(link.proto.label) for link in at.get("page_link")]
    assert any("Failure Explorer" in label for label in links)

    # --- Next -> step 1 ------------------------------------------------------------
    at.button(key="tour_next").click().run(timeout=30)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    assert any(text.startswith("Step 2 of 7 · One missed pedestrian") for text in captions)
    # Back is live in the SAME run the click landed in: the buttons move tour_step in
    # an on_click callback, which streamlit runs BEFORE the script redraws the nav row
    # (mutating it inline after the row is drawn would leave Back greyed out on the
    # step the viewer just walked into).
    assert at.button(key="tour_back").disabled is False

    assert at.radio(key="tour_hero_model").options == ["baseline", "graph_rate_night"]
    assert len(at.image) == 1
    assert any("defeats all three models" in text for text in captions)

    markdowns = [str(block.value) for block in at.markdown]
    # the derived fact line: baseline never claims v0's pedestrian (a2);
    # graph_rate_night claims it at conf 0.80 (a plain tp in this fixture).
    assert any(
        "Pedestrian" in text and "baseline: no claim" in text
        and "graph_rate_night: 0.800" in text
        for text in markdowns
    )
    # and the counted (never assumed) miss -> hit claim
    assert any("only night frame" in text for text in markdowns)
    assert any("recomputed in this app" in text for text in captions)

    # --- Back -> step 0 ------------------------------------------------------------
    at.button(key="tour_back").click().run(timeout=30)
    assert not at.exception
    assert any(
        str(caption.value).startswith("Step 1 of 7 · The weakness: night")
        for caption in at.caption
    )


def _walk_to_step(at: Any, index: int) -> Any:
    """Click "Next →" ``index`` times from the tour's CURRENT step -- not to an
    absolute 0-based target. On a fresh ``at`` (the common case, step 0), that is
    the same thing: ``index`` clicks land on the screen captioned "Step {index +
    1} of 7". But a second call on the same ``at`` (e.g.
    ``test_tour_hero_honesty_lines_render_when_the_recovery_is_low_conf``, which
    calls this three times on one ``at`` with 1/4/1) advances ``index`` steps
    PAST wherever the walk already was, so its calls land on steps 1, 5 and 6
    (0-based) -- not 1, 4 and 1.

    The walk is deliberately a real click sequence rather than a session_state
    poke: the Back/Next callbacks are the only thing that moves ``tour_step``, and
    a test that set the key directly would stop covering them.
    """
    for _ in range(index):
        at.button(key="tour_next").click().run(timeout=30)
    return at


def test_tour_walks_steps_2_to_5(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steps 2-5 (screens 3-6): the mined event, why one mined frame was picked, the
    retrain, and the same kind of frame after -- every number read from the fixture
    package (its flagship event "s1", its night-pass selected frame, its arm table,
    its one exemplar "v0"), never written into the page."""
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data, monkeypatch)

    # --- step 2: "Find more like it" -----------------------------------------------
    _walk_to_step(at, 2)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert any(text.startswith("Step 3 of 7 · Find more like it") for text in captions)
    # the fixture's only hard_braking_near_pedestrians event: "s1", scene-X, its own
    # severity figure (accel_long_min_mps2 = -7.5) and its near-pedestrian facts
    # (f1 at 5m within 10m, f2 at 20m outside it).
    assert any("scene-X" in text and "0.76g braking" in text for text in markdowns)
    assert any("within 10 m: 1 pedestrian · nearest at 5.0 m" in text for text in markdowns)
    # the event frame itself plus its filmstrip thumbs (s1 and its t+1 neighbour v1)
    assert len(at.image) >= 2
    # parity read from graph_subgraphs/hard_braking_near_pedestrians.json (1/1, True)
    assert any("found · SQL 1 / Graph 1 ✓" in text for text in captions)
    # the counted night line: s1 is a day event, so 0 of the 1 matching events
    assert any("0 of 1 matching events are at night" in text for text in markdowns)
    assert "Scenario Search" in str(at.button(key="tour_open_event").label)

    # --- step 3: "Why this frame was picked" ---------------------------------------
    at.button(key="tour_next").click().run(timeout=30)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert any(text.startswith("Step 4 of 7 · Why this frame was picked") for text in captions)
    # selection_factors, rendered exactly as the Active Learning page renders them --
    # the fixture's first candidate frame was taken by the night pass (floor 1).
    assert any(text.startswith("**Night frame:** yes ✓") for text in markdowns)
    assert any("**Picked in:** night pass (night floor 1)" in text for text in markdowns)
    # the selection is REPRODUCED (demo al-explain), not recomputed or recorded
    assert any("reproduced selection" in text for text in captions)
    assert "Active Learning" in str(at.button(key="tour_open_al_frame").label)

    # --- step 4: "Retrain on what was found" ---------------------------------------
    at.button(key="tour_next").click().run(timeout=30)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    assert any(text.startswith("Step 5 of 7 · Retrain on what was found") for text in captions)
    # the fixture arm table: baseline 100 -> graph_rate_night 120 train images, and
    # the arm's own mined-set composition (1 scene, all-night).
    assert [str(metric.label) for metric in at.metric] == [
        "Frames mined", "Scenes covered", "Night share", "Training images",
    ]
    # "Training images" carries the noun, so the VALUE is just the after-count --
    # the "100 → 120" arrow is 13 characters on the real package
    # ("7,035 → 8,535") and truncates in a 1-of-4 st.metric card at <=1200px
    # (Phase 9a final review), so the mined total moves to the delta instead.
    assert [str(metric.value) for metric in at.metric] == ["20", "1", "100%", "120"]
    assert str(at.metric[3].delta) == "+20 mined frames"
    assert len(at.get("vega_lite_chart")) == 1
    assert any("recorded experiment output" in text for text in captions)

    # --- step 5: "Same kind of frame, after" ---------------------------------------
    at.button(key="tour_next").click().run(timeout=30)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert any(text.startswith("Step 6 of 7 · Same kind of frame, after") for text in captions)
    assert at.radio(key="tour_exemplar_model").options == ["baseline", "graph_rate_night"]
    assert len(at.image) == 1
    # the exemplar's upgraded box: baseline never claims v0's pedestrian (a2),
    # graph_rate_night claims it at 0.80.
    tables = [d.value for d in at.dataframe if "arm_claim" in getattr(d.value, "columns", [])]
    assert len(tables) == 1
    # the result is the arm, not the frame: fixture baseline night 0.10 -> arm 0.13.
    assert any(
        "Night mAP50-95 0.1000 → 0.1300 (+0.0300) on the held-out split" in text
        for text in markdowns
    )
    assert any("recomputed in this app" in text for text in captions)
    assert "Active Learning" in str(at.button(key="tour_open_exemplar").label)


def test_tour_result_screen(built_demo_data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 6 (screen 7): four bordered answers -- the weakness, what was added,
    whether the model improved, and what failed -- every number read straight off
    the fixture package (results.json's baseline/graph_rate_night rows,
    weak_loss_decomposition's headline pair, weak_supervision_results' one arm),
    six "Go deeper" links to every other page, and Restart back to step 0."""
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 6)
    assert not at.exception

    captions = [str(caption.value) for caption in at.caption]
    assert any(
        text.startswith("Step 7 of 7 · What we found, added, gained") for text in captions
    )
    assert at.button(key="tour_next").disabled is True

    markdowns = [str(block.value) for block in at.markdown]
    for heading in (
        "**What weakness did we find?**",
        "**What data did we add?**",
        "**Did the model improve?**",
        "**What failed along the way?**",
    ):
        assert any(heading in text for text in markdowns), heading

    # the breadcrumb lights every stage on the result screen
    breadcrumb = next(text for text in markdowns if ":orange-badge[Diagnose]" in text)
    for stage in ("Diagnose", "Mine", "Train", "Evaluate"):
        assert f":orange-badge[{stage}]" in breadcrumb

    # (1) the weakness -- baseline's night vs overall, straight off results.json,
    # plus the recomputed miss count step 0 already pins (1 of 1)
    assert any("Night mAP50-95 0.1000 vs overall 0.2000." in text for text in markdowns)
    assert any(
        "1 of 1 night validation frames carry at least one baseline miss" in text
        for text in markdowns
    )

    # (2) what was added -- the arm's mined-set composition vs the random comparator
    # (this fixture has no "mined" arm, so that comparator clause is skipped)
    assert any(
        "mined 20 frames across 1 scenes, 100% at night" in text
        and "random covered 1 scenes at 0% night" in text
        for text in markdowns
    )
    assert not any("similarity mining" in text for text in markdowns)

    # (3) did it improve -- the same night-mAP sentence step 5 shows, the night
    # pedestrian slice, and the overall delta (the fixture's best-overall arm IS
    # the tour's own arm, so the "best overall" sentence is correctly skipped)
    assert any(
        "Night mAP50-95 0.1000 → 0.1300 (+0.0300) on the held-out split" in text
        for text in markdowns
    )
    assert any("Night pedestrian mAP50-95 0.0500 → 0.0900." in text for text in markdowns)
    assert any("Overall mAP50-95 +0.0300 against baseline." in text for text in markdowns)
    assert not any("best overall arm" in text for text in markdowns)

    # (4) what failed -- the headline weak/GT pair's loss split (0.0018/0.01 =
    # 18.0% retained, 50.0% dropped-frame, 32.0% label cost) and the same pair's
    # sparse-frame bias (3.00 vs 0.00 GT boxes/frame for "random")
    assert any(
        "Weak (VLM-verified) labels kept 18.0% of the ground-truth gain for the "
        "`random` pair: 50.0% was lost with the frames the verifier dropped, "
        "32.0% to label noise on the ones it kept." in text
        for text in markdowns
    )
    # ONE sparse-frames sentence, about the headline pair only (review I7) -- the
    # real package carries a weaksup row per weak arm, and the loop wrote the
    # sentence once per row on a screen that answers one question in a few lines.
    sparse = [text for text in markdowns if "The verifier kept sparse frames:" in text]
    assert sparse == [
        "The verifier kept sparse frames: 3.00 vs 0.00 GT boxes per accepted vs "
        "rejected frame (random)."
    ]
    # No worst-night-arm sentence on THIS package (review I2): the superlative is
    # computed over every arm, and the arm with the lowest delta_night here is
    # `baseline` itself (+0.0000) -- not a weak-supervision result, so there is no
    # honest "worst of the 5 arms" claim to make. The old code took the min over
    # the weak arms alone and then called it worst of all five.
    assert not any("worst night result" in text for text in markdowns)
    # the hero's own recovery is a confident tp in this fixture (conf 0.800), not
    # the low-confidence claim the shipped package's hero is -- the honesty line
    # must not fire on a claim that isn't actually low-confidence.
    assert not any("low-confidence claim" in text for text in markdowns)

    provenance_captions = " ".join(captions)
    assert "recorded experiment output" in provenance_captions
    assert "recomputed in this app" in provenance_captions

    # "Go deeper" -> every page but the tour itself
    links = [str(link.proto.label) for link in at.get("page_link")]
    assert len(links) >= 6

    # --- Restart -> step 0 ------------------------------------------------------
    at.button(key="tour_restart").click().run(timeout=30)
    assert not at.exception
    assert any(
        str(caption.value).startswith("Step 1 of 7 · The weakness: night")
        for caption in at.caption
    )


def test_tour_result_screen_names_the_worst_night_arm_only_when_it_is_a_weak_one(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of review I2: give the package the real one's shape -- the
    best night arm's own weak twin holding the LOWEST delta_night of every arm --
    and the superlative sentence is written, naming that arm and the full arm count.

    ``test_tour_result_screen`` pins the skip branch on the base fixture (whose
    lowest delta_night is `baseline`'s own +0.0000), so between them both sides of
    the check are covered.
    """
    pytest.importorskip("streamlit")
    _add_weak_night_twin_arm(built_demo_data)
    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 6)
    assert not at.exception

    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        "`weak_graph_rate_night` is the worst night result of the 6 arms (-0.0100)." in text
        for text in markdowns
    )


def test_tour_hero_honesty_lines_render_when_the_recovery_is_low_conf(
    built_demo_data_low_conf_hero: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I4: on a package whose hero recovery is the shipped package's own
    low-confidence claim, all three honesty branches fire -- step 2's fact line
    names the status and the floor, step 6 carries ``_HERO_HONESTY_LINE``, and step
    7's fourth answer states the confidence itself.

    (Steps are 1-based here, as the viewer reads them: ``_walk_to_step`` takes the
    0-based index.)
    """
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data_low_conf_hero, monkeypatch)

    # --- step 2: the per-model fact line says what kind of hit it is -------------
    _walk_to_step(at, 1)
    assert not at.exception
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        "graph_rate_night: 0.135 (low_conf — below the 0.40 hit floor; the matching "
        "rule still counts it as a hit)" in text
        for text in markdowns
    )

    # --- step 6: the before/after table's own honesty line -----------------------
    _walk_to_step(at, 4)
    assert not at.exception
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        "is a low-confidence claim and does not pass this table's "
        "confident-detection rule" in text
        for text in markdowns
    )
    # ... and, because that rule finds nothing to show, the empty-table note
    assert any("no upgraded boxes on this frame" in str(c.value) for c in at.caption)

    # --- step 7: "what failed along the way?" states the confidence ---------------
    _walk_to_step(at, 1)
    assert not at.exception
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        "The hero frame's pedestrian recovery is a low-confidence claim (conf 0.135) "
        "— counted as a hit by the matching rule, not a confident detection." in text
        for text in markdowns
    )


def test_tour_step_4_says_flagship_and_rejected_when_the_frame_really_is(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I3: the "why this frame was picked" step carries two CONDITIONAL
    sentences -- the frame is itself a flagship event, and the weak verifier later
    rejected it -- but on the real v0.7 package the frame the new ordering lands
    on (`00740c25baf64f12a411963aadb5c81c`, community #10301, 916 frames all at
    night, failure mass 395.19 at rank 3 of 97, quota 323, degree rank 1 of 916,
    a night-pass pick) fires NEITHER: it is not one of the flagship preset's
    events, and its `weak_verdict` is `accepted`. The base fixture also fires
    neither, so the package is given both properties here instead: its chosen
    frame is added to the flagship preset's event rows, and its weak verdict is
    flipped to "rejected".
    """
    pytest.importorskip("streamlit")
    events_path = built_demo_data / "scenario_events.parquet"
    events = pd.read_parquet(events_path)
    flagship = events.loc[events["preset_rank_hard_braking_near_pedestrians"].notna()]
    assert not flagship.empty
    twin = flagship.iloc[[0]].copy()
    twin["sample_data_token"] = "v0"
    # rank 2, so the fixture's own s1 stays the top event step 3 opens on and this
    # frame is genuinely the SECOND flagship event, not a duplicate of the first.
    twin["preset_rank_hard_braking_near_pedestrians"] = 2
    pd.concat([events, twin], ignore_index=True).to_parquet(events_path, index=False)

    manifest_path = built_demo_data / "frame_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest.loc[manifest["sample_data_token"] == "v0", "weak_verdict"] = "rejected"
    manifest.to_parquet(manifest_path, index=False)

    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 3)
    assert not at.exception
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        "This frame is itself flagship event #2 — the scenario query and the "
        "selection agree on it." in text
        for text in markdowns
    )
    assert any(
        "Later, the weak-supervision verifier rejected this frame as too crowded to "
        "label automatically — step 7 shows why that matters." in text
        for text in markdowns
    )


def test_tour_mined_event_caption_says_train_pool_when_the_package_carries_the_frame(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 5c: "no predictions on this frame" has two different reasons, and the
    mined-event step now says which one applies.

    The base fixture's flagship event ("s1") has no ``frame_manifest`` row at all --
    the package carries the event and its thumbs, not the frame -- so it draws the
    not-curated caption. Given a train-pool manifest row for the same token, the
    frame IS in the package and was simply never a test image, which is the
    train-pool note the Active Learning page shows for that case.
    """
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 2)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any("not in the curated prediction set" in text for text in captions)
    assert not any("train-pool frame" in text for text in captions)

    manifest_path = built_demo_data / "frame_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    # is_night is set (rather than left to concat's NaN fill) only to keep the
    # column a real bool: an object-dtype is_night makes every .fillna(False) in the
    # app emit a pandas downcasting FutureWarning that has nothing to do with this
    # test. Everything else this row doesn't set is genuinely absent, as it would be
    # for a frame the package carries no predictions or curation facts for.
    extra = pd.DataFrame({
        "sample_data_token": ["s1"], "split": ["train_pool"], "is_night": [False],
    })
    pd.concat([manifest, extra], ignore_index=True).to_parquet(manifest_path, index=False)

    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 2)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any(
        "train-pool frame — no predictions (models never saw it as a test image)" in text
        for text in captions
    )
    assert not any("not in the curated prediction set" in text for text in captions)


def _string_literals(path: Path) -> set[str]:
    """Every string constant in one module's source, implicit concatenation folded
    (the parser folds ``"a " "b"`` into one ``ast.Constant``) -- so a copied string
    can be checked against a page even where the page spells it inline rather than
    as a module constant."""
    tree = ast.parse(path.read_text())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_tour_copies_deep_page_wording_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review I6: ``views/tour.py`` deliberately COPIES nine strings from the two
    pages its steps condense (it cannot import them -- they are module-private over
    there, or inline) and its own header comment promises "a tour step must say
    exactly what the deep page behind it says". Nothing enforced that promise, so a
    reword on either side would silently drift the two apart. This test is the
    enforcement: every copied string is compared to its source, by equality.
    """
    pytest.importorskip("streamlit")
    import importlib

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    tour = importlib.import_module("views.tour")
    scenarios = importlib.import_module("views.scenarios")
    active_learning = importlib.import_module("views.active_learning")
    filters = importlib.import_module("filters")
    try:
        # (a) the strings the pages DO name -- compared attribute to attribute
        assert tour._NOT_CURATED_CAPTION == scenarios._NOT_CURATED_CAPTION
        assert tour._EXPLAIN_ABSENT_NOTE == active_learning._EXPLAIN_ABSENT_NOTE
        assert tour._EXPLAIN_FRAME_ABSENT_NOTE == active_learning._EXPLAIN_FRAME_ABSENT_NOTE
        assert tour._TRAIN_POOL_NOTE == active_learning._TRAIN_POOL_NOTE

        # (b) the ones the pages write inline -- compared against every string
        # literal in the page's own source, which is the same equality check
        # (`in` over a set of exact values, not a substring search).
        scenario_strings = _string_literals(DEMO_DIR / "views" / "scenarios.py")
        al_strings = _string_literals(DEMO_DIR / "views" / "active_learning.py")
        assert tour._EVENTS_ABSENT_NOTE in scenario_strings
        assert tour._OVERLAY_LEGEND in al_strings
        assert tour._NO_EXEMPLAR_NOTE in al_strings
        assert tour._NO_UPGRADED_BOXES_NOTE in al_strings

        # (c) the filmstrip's own step columns and labels, in strip order: the
        # Scenario page splits them either side of the current frame, which the
        # tour flattens into one tuple with the event itself (column None) between.
        # Phase 9b: both are now DERIVED from filters.FILMSTRIP_STEPS (one
        # definition, two shapes), and this is the check that they stay derived.
        page_steps = (
            *scenarios._BEFORE_STEPS,
            (None, scenarios._CURRENT_STEP),
            *scenarios._AFTER_STEPS,
        )
        assert page_steps == tour._FILMSTRIP_STEPS == filters.FILMSTRIP_STEPS
    finally:
        _reset_demo_app_modules()


def test_tour_deep_link_buttons_set_state(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step's deep link pre-selects what the viewer was looking at on the page
    it opens -- the same event, frame or exemplar, not that page's default.

    Only the state is asserted: an in-script ``st.switch_page`` reruns into the
    target page but is not sticky across AppTest's own ``at.run()`` (the plan's
    verified facts), so each button gets its own walk from a fresh app.
    """
    pytest.importorskip("streamlit")
    for step_index, key, expected in (
        (2, "tour_open_event", {"scenario_preset": "hard_braking_near_pedestrians",
                                "scenario_token": "s1"}),
        (3, "tour_open_al_frame", {"al_frame_token": "v0"}),
        (5, "tour_open_exemplar", {"al_exemplar": "v0"}),
    ):
        at = _tour_apptest(built_demo_data, monkeypatch)
        _walk_to_step(at, step_index)
        at.button(key=key).click().run(timeout=30)
        assert not at.exception, key
        for state_key, value in expected.items():
            assert at.session_state[state_key] == value, key


def test_tour_degrades_without_optional_groups(
    built_demo_data_without_explain: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two optional groups the mining steps read: without `demo al-explain`'s
    per-frame facts step 3 says so instead of inventing a reason, and without
    scenario_events.parquet step 2 says so instead of raising."""
    pytest.importorskip("streamlit")
    package = built_demo_data_without_explain

    at = _tour_apptest(package, monkeypatch)
    _walk_to_step(at, 3)
    assert not at.exception
    assert any("not included in this package" in str(e.value) for e in at.info)

    (package / "scenario_events.parquet").unlink()
    at = _tour_apptest(package, monkeypatch)
    _walk_to_step(at, 2)
    assert not at.exception
    assert any("needs demo_data >= 0.4" in str(e.value) for e in at.info)


def test_navigation_sections_keep_every_page_reachable(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouping the pages into sections must not change how any of them resolves:
    the Overview is still the default page, and every other page (the new tour
    included) is still reachable by its own pinned url_path."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    at = AppTest.from_file(str(DEMO_DIR / "main.py"))
    at.run(timeout=30)
    assert not at.exception
    assert [str(title.value) for title in at.title] == ["nuScenes Perception Data Engine"]

    for page_path, title in _SECTIONED_PAGES:
        at.switch_page(page_path).run(timeout=30)
        assert not at.exception, f"{page_path} raised"
        assert [str(element.value) for element in at.title] == [title], page_path


def test_nav_registry_names_the_path_it_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nav.page's failure mode is a ValueError naming the path asked for -- a
    deep link to a page that was renamed must fail loudly at the link, not render a
    silent no-op."""
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    _reset_demo_app_modules()
    import nav

    with pytest.raises(ValueError, match="not_a_page"):
        nav.page("not_a_page")


def test_demo_no_longer_promises_phase_8() -> None:
    """The last stub is gone: no placeholder module, no import of one, and no page
    still promising a future phase."""
    assert not (DEMO_DIR / "views" / "stubs.py").exists()
    assert "stubs" not in (DEMO_DIR / "main.py").read_text()
    promises = [
        path.name for path in sorted(DEMO_DIR.rglob("*.py")) if "Phase 8" in path.read_text()
    ]
    assert not promises


# --- Phase 9a (Task 6): trust chrome on every page ------------------------------
#
# The tour and Overview already carry a loop breadcrumb and provenance captions
# (Tasks 1-5). This task puts the same chrome — the breadcrumb, provenance labels,
# the Scenario event viewer's compact SQL<->Graph parity line, and the "What we
# learned" callouts — on the five existing loop pages (spec
# docs/superpowers/specs/2026-08-22-demo-phase9a-design.md sec3).

_ALL_LOOP_STAGES = ("Diagnose", "Mine", "Train", "Evaluate")


def test_every_page_carries_the_loop_breadcrumb(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every page below the tour/Overview carries the breadcrumb directly under
    its title, with exactly the stages the spec's "Page -> stages" table pins lit
    orange and the rest grey."""
    pytest.importorskip("streamlit")
    cases = (
        (_failures_apptest, ("Diagnose",)),
        (_scenarios_apptest, ("Mine",)),
        (_active_learning_apptest, ("Mine", "Train", "Evaluate")),
        (_weak_supervision_apptest, ("Train", "Evaluate")),
        (_chat_replay_apptest, ("Diagnose",)),
    )
    for apptest_fn, lit in cases:
        at = apptest_fn(built_demo_data, monkeypatch)
        assert not at.exception, apptest_fn.__name__
        markdowns = [str(m.value) for m in at.markdown]
        breadcrumb = next(
            (text for text in markdowns if ":orange-badge[" in text or ":gray-badge[" in text),
            None,
        )
        assert breadcrumb is not None, apptest_fn.__name__
        for stage in _ALL_LOOP_STAGES:
            badge = f":orange-badge[{stage}]" if stage in lit else f":gray-badge[{stage}]"
            assert badge in breadcrumb, (apptest_fn.__name__, stage, breadcrumb)


def test_provenance_captions_present(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every page's provenance captions read the fixed PROVENANCE wording (spec
    sec3) at the spot the "Provenance placement" paragraph pins."""
    pytest.importorskip("streamlit")

    # Failure Explorer: the detail overlay, once a frame is selected.
    at = _failures_apptest(built_demo_data, monkeypatch)
    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception
    assert any("recomputed in this app" in str(c.value) for c in at.caption)

    # Scenario Search: the flagship (dynamics) preset header's parity caption,
    # present on the very first load (the flagship is the default preset).
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert any("recorded experiment output" in str(c.value) for c in at.caption)

    # Active Learning: "reproduced selection" needs the why-selected panel open on
    # a frame the al-explain group covers ("wA"); the arm chart's "recorded" is
    # present on the default load already.
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    at.session_state["al_frame_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any("reproduced selection" in c for c in captions)
    assert any("recorded experiment output" in c for c in captions)

    # Weak Supervision: the cards'/crowding chart's "recorded" is on the default
    # load; the frame panel's "recomputed" needs a frame selected.
    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    at.session_state["ws_accepted_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any("recorded experiment output" in c for c in captions)
    assert any("recomputed in this app" in c for c in captions)

    # Ask the Dataset: the header caption's "recorded".
    at = _chat_replay_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    assert any("recorded experiment output" in str(c.value) for c in at.caption)


def test_scenario_viewer_shows_compact_parity_line(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The event viewer header gets its own compact SQL<->Graph parity line
    (filters.parity_short) for a dynamics preset with subgraphs staged — the
    preset header's own pinned parity_caption line
    (test_scenario_flagship_badge) is unaffected. The flagship fixture's own
    subgraph payload has sql_count == cypher_count == 1 (parity True), so the
    noun is singular: "event", not "events"."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    captions = [str(c.value) for c in at.caption]
    assert any("1 event found · SQL 1 / Graph 1 ✓" in c for c in captions)


def _add_weak_night_twin_arm(built_demo_data: Path) -> None:
    """Give the built package a ``weak_graph_rate_night`` row in
    ``active_learning_results.parquet`` — deliberately absent from the base
    fixture (see ``_add_second_weak_pair``'s own docstring, just above): the
    real package's own weak twin of the best-night arm, set here to the worst
    night result in the table, mirroring the real package's shape (spec's
    honesty rules: "weak_graph_rate_night is the worst night arm").

    Only the columns the AL page's night-inversion callout and arm chart
    actually read are set; the rest come back NaN for this row via
    ``pd.concat``'s own column-align fill, the same idiom
    ``test_active_learning_gallery_pages_night_frames_first`` uses to widen
    ``frame_manifest.parquet`` with synthetic rows.
    """
    path = built_demo_data / "active_learning_results.parquet"
    arms = pd.read_parquet(path)
    extra = pd.DataFrame({
        "arm": ["weak_graph_rate_night"],
        "family": ["weak"],
        "round_order": [int(arms["round_order"].max()) + 1],
        "n_train_images": [120],
        "overall_map5095": [0.19],
        "night_map5095": [0.09],
        "delta_overall": [-0.01],
        "delta_night": [-0.01],
    })
    pd.concat([arms, extra], ignore_index=True).to_parquet(path, index=False)


def test_learned_callouts_interpolate_numbers(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "What we learned" callouts on the Active Learning and Weak Supervision
    pages, every number interpolated from the package tables (spec sec3, "What
    we learned" callouts)."""
    pytest.importorskip("streamlit")

    _add_weak_night_twin_arm(built_demo_data)
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    text = [str(m.value) for m in at.markdown]
    # graph_rate_night is the fixture's own best-night arm (+0.0300 against
    # baseline, active_learning_results.parquet) -- test_active_learning_page_
    # renders_story_chart_and_exemplar_table pins the same figure. With the weak
    # twin added it also holds the table's LOWEST delta_night, so both superlatives
    # the sentence claims ("the best night arm of 6 ... is the worst") are true of
    # this package and the sentence is written in full (review I1).
    assert any(
        "What we learned" in t and "+0.0300" in t and "`weak_graph_rate_night`" in t
        and "is the best night arm of 6" in t and "(-0.0100) is the worst" in t
        for t in text
    )

    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    text = [str(m.value) for m in at.markdown]
    # the fixture's headline pair ("random"): 0.0018/0.0100 = 18.0% retained --
    # test_weak_supervision_page_cards_decomposition_bias_and_tabs pins the same
    # figure.
    assert any("What we learned" in t and "18.0%" in t for t in text)


def test_night_inversion_callout_drops_its_superlatives_when_the_table_disagrees(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I1: the night-inversion callout claims TWO superlatives ("the best
    night arm of N ... is the worst"). Both are checked against this package's own
    arm table before they are written; when either is false the same two numbers
    are contrasted with no rank claim at all, rather than a sentence the table
    contradicts.

    Here the weak twin is still the best-night arm's twin, but a third arm
    (`random`, pushed to -0.0200) holds the table's lowest night delta -- so
    "is the worst" would be false.
    """
    pytest.importorskip("streamlit")
    _add_weak_night_twin_arm(built_demo_data)
    path = built_demo_data / "active_learning_results.parquet"
    arms = pd.read_parquet(path)
    arms.loc[arms["arm"] == "random", "delta_night"] = -0.02
    arms.to_parquet(path, index=False)

    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception
    text = [str(m.value) for m in at.markdown]
    learned = [t for t in text if "What we learned" in t and "Targeting night" in t]
    assert len(learned) == 1
    # the same two numbers, and the same point
    assert "`graph_rate_night` posts +0.0300 night mAP50-95 against the baseline" in learned[0]
    assert "`weak_graph_rate_night` posts -0.0100" in learned[0]
    assert "the night gain came from the frames, not from cheaper labels" in learned[0]
    # ... and neither superlative
    assert "best night arm" not in learned[0]
    assert "is the worst" not in learned[0]


def test_night_inversion_callout_counts_only_ranked_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 9a final review item 7: "of N arms" must count the arms actually
    RANKED by delta_night (``ranked = arms.dropna(subset=["delta_night"])``,
    already used to compute best/worst just above), not every row in the table
    -- an arm this package never evaluated for night delta (NaN) is not one of
    "the N arms" the superlative names. A direct call, not an AppTest: the
    function under test is a pure ``learned(text)`` builder once ``load_overview``
    is stubbed."""
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    from views import active_learning

    calls: list[str] = []
    monkeypatch.setattr(active_learning, "learned", calls.append)
    monkeypatch.setattr(
        active_learning, "load_overview",
        lambda: {"results": {"best_night_arm": "graph_rate_night"}},
    )
    arms = pd.DataFrame({
        "arm": ["baseline", "graph_rate_night", "weak_graph_rate_night", "unranked"],
        "delta_night": [0.0, 0.03, -0.01, float("nan")],
    })
    active_learning._render_night_inversion_callout(arms)
    assert len(calls) == 1
    assert "is the best night arm of 3" in calls[0]


def test_night_rank_caption_counts_only_ranked_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 9a final review item 7, Weak Supervision's own version: ``total =
    len(ordered)`` must count the arms the ranking actually has an opinion on. A
    direct call: ``_night_rank_caption`` is a pure function of ``(arms,
    weak_arm) -> str``."""
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    from views import weak_supervision

    arms = pd.DataFrame({
        "arm": ["baseline", "random", "weak_random", "unranked"],
        "delta_night": [0.0, 0.01, -0.02, float("nan")],
    })
    text = weak_supervision._night_rank_caption(arms, "weak_random")
    assert "is the worst night result of the 3 arms" in text


def test_loss_learned_callout_counts_only_ranked_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 9a final review item 7, the third site: the loss-split "what we
    learned" sentence's own "worst night result of the N arms" clause."""
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    from views import weak_supervision

    calls: list[str] = []
    monkeypatch.setattr(weak_supervision, "learned", calls.append)
    loss = pd.DataFrame({
        "base_arm": ["random", "graph_rate_night"],
        "retention": [0.18, 0.394],
        "dropped_frame_share": [0.5, 0.1],
        "label_share": [0.32, 0.2],
        "headline": [True, False],
    })
    arms = pd.DataFrame({
        "arm": ["baseline", "weak_random", "weak_graph_rate_night", "unranked"],
        "delta_night": [0.0, -0.005, -0.02, float("nan")],
    })
    weak_supervision._render_loss_learned_callout(loss, arms)
    assert len(calls) == 1
    assert "posted the worst night result of the 3 arms" in calls[0]
