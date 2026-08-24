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


# Phase 9b (Task 8): four staged models, so `demo infer` writes twelve ordered
# fixes_fn_vs_<a>_<b> exemplar columns rather than the two a two-model package had.
# Generated from the frames' own matched_<model> flags (below) with infer.py's own
# rule -- "A missed a GT box that B caught" -- so the fixture cannot drift out of
# agreement with the gt_boxes/predictions rows it is derived from.
_FIXTURE_MODELS = (
    "baseline", "graph_rate_night", "weak_graph_rate_night", "weak_graph_rate_night_gt",
)
_V0_MATCHED = {
    "baseline": {"a1": True, "a2": False},
    "graph_rate_night": {"a1": True, "a2": True},
    "weak_graph_rate_night": {"a1": True, "a2": False},
    "weak_graph_rate_night_gt": {"a1": True, "a2": True},
}


def _fixes_fn_columns() -> dict[str, Any]:
    """``fixes_fn_vs_<a>_<b>`` for every ordered model pair, over the fixture's four
    manifest rows (v0, v1, wA, wR): computed on v0, False on v1 (no GT rows at all,
    so there is no false negative to fix) and NA on the two train-pool frames (no
    model ever ran on them)."""
    return {
        f"fixes_fn_vs_{model_a}_{model_b}": pd.array(
            [
                any(
                    not _V0_MATCHED[model_a][box] and _V0_MATCHED[model_b][box]
                    for box in ("a1", "a2")
                ),
                False, pd.NA, pd.NA,
            ],
            dtype="boolean",
        )
        for model_a in _FIXTURE_MODELS
        for model_b in _FIXTURE_MODELS
        if model_a != model_b
    }


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
    #
    # Phase 9b (Task 7): the same table is also autolabel's gt_counts input (the
    # recomputed crowding buckets), which reads category_name/visibility_token --
    # both carried here, mapping through GT_COUNT_GROUPS to the same coarse classes
    # category_group already names. "b1"/"b2" are two extra tokens whose GT counts
    # deliberately land in ALL FOUR count buckets (b1: cars 2 -> "1-3",
    # pedestrians 5 -> "4-9", traffic_cones 10 -> "10+", the other seven fields 0;
    # b2: no rows at all, so all ten are 0), so the fixture package's
    # vlm_count_buckets.parquet carries a row per bucket. They are in no arm, no
    # candidate set and no curated frame, so every other exporter is untouched.
    bucket_gt = (
        [("b1", "vehicle.car", "car")] * 2
        + [("b1", "human.pedestrian.adult", "pedestrian")] * 5
        + [("b1", "movable_object.trafficcone", None)] * 10
    )
    pd.DataFrame(
        {
            "sample_token": ["s1"] * 4 + [token for token, _, _ in bucket_gt],
            "sample_data_token": ["s1"] * 4 + [token for token, _, _ in bucket_gt],
            "category_group": (
                ["pedestrian", "car", "pedestrian", None]
                + [group for _, _, group in bucket_gt]
            ),
            "category_name": (
                [
                    "human.pedestrian.adult", "vehicle.car",
                    "human.pedestrian.adult", "movable_object.debris",
                ]
                + [name for _, name, _ in bucket_gt]
            ),
            "visibility_token": ["4"] * (4 + len(bucket_gt)),
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
        # Phase 9b (Task 8): the two weak checkpoints found 1/2 boxes on v0 and
        # nothing on v1 -- 0 is a recorded finding, NA is "never ran" (the two
        # train-pool frames).
        "n_preds_weak_graph_rate_night": pd.array([1, 0, pd.NA, pd.NA], dtype="Int64"),
        "n_preds_weak_graph_rate_night_gt": pd.array([2, 0, pd.NA, pd.NA], dtype="Int64"),
        **_fixes_fn_columns(),
    }).to_parquet(staging / "frame_manifest.parquet")
    # Phase 9b (Task 8): `demo infer` now runs the two weak-supervision checkpoints
    # over the held-out val frames as well -- `weak_graph_rate_night` (trained on the
    # VLM-verified pseudo labels) and `weak_graph_rate_night_gt` (the GT-labelled
    # twin over the same frames). On v0 the twin catches the pedestrian a2 that the
    # weak arm misses, which is exactly the before/after the Weak Supervision page's
    # "One frame, three views" row 2 draws. Neither weak checkpoint claims anything
    # on v1 -- a RECORDED finding of zero (n_preds 0 below), not "never ran".
    pd.DataFrame({
        "sample_data_token": ["v0"] * 6 + ["v1"] * 4,
        "model": [
            "baseline", "graph_rate_night", "graph_rate_night",
            "weak_graph_rate_night", "weak_graph_rate_night_gt",
            "weak_graph_rate_night_gt",
            "baseline", "baseline", "graph_rate_night", "graph_rate_night",
        ],
        "category_group": [
            "car", "car", "pedestrian", "car", "car", "pedestrian",
            "car", "car", "car", "car",
        ],
        "x_min": [1.0, 1.0, 10.0, 1.0, 1.0, 10.0, 5.0, 6.0, 5.0, 6.0],
        "y_min": [1.0, 1.0, 10.0, 1.0, 1.0, 10.0, 5.0, 6.0, 5.0, 6.0],
        "x_max": [2.0, 2.0, 30.0, 2.0, 2.0, 30.0, 7.0, 8.0, 7.0, 8.0],
        "y_max": [2.0, 2.0, 30.0, 2.0, 2.0, 30.0, 7.0, 8.0, 7.0, 8.0],
        "conf": [0.9, 0.9, 0.8, 0.85, 0.86, 0.77, 0.4, 0.5, 0.4, 0.5],
        # v0: baseline only ever claims a1 (misses a2); graph_rate_night claims
        # both a1 and a2 (a genuine catch); the weak arm claims a1 only and its
        # GT-labelled twin claims both. v1 has zero GT rows, so every claim
        # from either of the first two models is necessarily a false positive.
        "status": ["tp", "tp", "tp", "tp", "tp", "tp", "fp", "fp", "fp", "fp"],
        "matched_annotation_token": [
            "a1", "a1", "a2", "a1", "a1", "a2", None, None, None, None,
        ],
    }).to_parquet(staging / "predictions.parquet")
    # a3 (Phase 9b, Task 8): one visible PEDESTRIAN GT box on the accepted weak
    # frame "wA" -- the Weak Supervision page's row-1 showcase frame has to carry
    # both a pseudo box and a visible pedestrian, and wA carried no GT row at all
    # before. Every matched_<model> is NA on it: it is a train-pool frame, so no
    # model ever evaluated it ("not evaluated" is not "missed").
    pd.DataFrame({
        "annotation_token": ["a1", "a2", "a3"],
        "sample_data_token": ["v0", "v0", "wA"],
        "category_group": ["car", "pedestrian", "pedestrian"],
        "x_min": [1.0, 10.0, 20.0], "y_min": [1.0, 10.0, 20.0],
        "x_max": [2.0, 30.0, 60.0], "y_max": [2.0, 30.0, 60.0],
        "matched_baseline": pd.array([True, False, None], dtype="boolean"),
        "matched_graph_rate_night": pd.array([True, True, None], dtype="boolean"),
        # The two weak checkpoints, same shape as their predictions above: the
        # pseudo-labelled arm misses a2, its GT-labelled twin catches it.
        "matched_weak_graph_rate_night": pd.array([True, False, None], dtype="boolean"),
        "matched_weak_graph_rate_night_gt": pd.array([True, True, None], dtype="boolean"),
        # below_visibility_min: real gt_boxes always carries this column: the
        # Failure Explorer drops such rows everywhere (rendering + the box table),
        # so it must be present for the page to even read gt_boxes.parquet. v1 has
        # no gt_boxes rows at all (the zero-GT case both review-fix regression
        # tests below depend on).
        "below_visibility_min": [False, False, False],
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
    # Phase 9b (Task 7): the Phase-6b VLM label table -- the ONLY source the
    # crowding buckets are recomputed from (the weak run's table just above is a
    # pseudo-labelling input, not a count-accuracy eval). Written inside tmp_path
    # and wired through paths.autolabel_dir below so this fixture can never reach
    # for the machine's real data/autolabel/ (5,000 rows).
    #
    # Against b1/b2's GT above these give: "0" n=17 (b2's cars off by one, the rest
    # exact), "1-3" n=1 MAE 1, "4-9" n=1 MAE 1, "10+" n=1 MAE 2 -- 20 = 2 ok rows x
    # the ten count fields, the pairs invariant the exporter asserts.
    autolabel = tmp_path / "autolabel"
    autolabel.mkdir()
    pd.DataFrame({
        "sample_data_token": ["b1", "b2"],
        "model": ["qwen2.5-vl", "qwen2.5-vl"],
        "parse_status": ["ok", "ok"],
        "time_of_day": ["day", "day"],
        "weather": ["clear", "clear"],
        "hazards": ["[]", "[]"],
        "notable_conditions": ["[]", "[]"],
        "label_confidence": ["high", "medium"],
        "cars": [3.0, 1.0], "trucks": [0.0, 0.0], "buses": [0.0, 0.0],
        "trailers": [0.0, 0.0], "construction_vehicles": [0.0, 0.0],
        "motorcycles": [0.0, 0.0], "bicycles": [0.0, 0.0],
        "pedestrians": [4.0, 0.0], "traffic_cones": [8.0, 0.0], "barriers": [0.0, 0.0],
    }).to_parquet(autolabel / "labels.parquet")

    out = tmp_path / "demo_data"
    config = {
        "paths": {
            "processed_dir": str(processed),
            "active_learning_dir": str(al),
            "active_learning_config": str(al_config_path),
            "autolabel_dir": str(autolabel),
            "mlruns_dir": str(tmp_path / "mlruns"),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(out),
        },
        "models": {
            "baseline": {"run": "runX", "imgsz": 640},
            "graph_rate_night": {"run": "runY", "imgsz": 640},
            # Phase 9b (Task 8): configs/demo.yaml's two weak-supervision
            # checkpoints -- configured models, so run_build validates that every
            # val token has recorded coverage from all four.
            "weak_graph_rate_night": {"run": "runW", "imgsz": 640},
            "weak_graph_rate_night_gt": {"run": "runWG", "imgsz": 640},
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


@pytest.fixture()
def built_demo_data_without_weak_models(built_demo_data: Path) -> Path:
    """The same package as a pre-0.8 `demo infer` wrote it: the three original
    models only, with no held-out predictions from the two weak-supervision
    checkpoints.

    Same edit-the-built-package idiom as ``built_demo_data_without_explain`` -- the
    page's absent branch keys off the ``matched_<model>`` columns being absent
    (``filters.weak_result_token``), which is exactly what an older `demo infer`
    leaves behind.
    """
    gt_path = built_demo_data / "gt_boxes.parquet"
    gt = pd.read_parquet(gt_path)
    gt.drop(
        columns=[column for column in gt.columns if column.startswith("matched_weak_")]
    ).to_parquet(gt_path, index=False)

    preds_path = built_demo_data / "predictions.parquet"
    preds = pd.read_parquet(preds_path)
    preds.loc[~preds["model"].str.startswith("weak_")].to_parquet(preds_path, index=False)

    manifest_path = built_demo_data / "frame_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest.drop(
        columns=[column for column in manifest.columns if "weak_graph_rate_night" in column]
    ).to_parquet(manifest_path, index=False)
    return built_demo_data


@pytest.fixture()
def built_demo_data_without_buckets(built_demo_data: Path) -> Path:
    """The same package as a pre-0.8 `demo build` wrote it: no recomputed crowding
    buckets (and the same state a machine with no ``data/autolabel/`` builds today).

    Deletes the file from an otherwise-normal built package rather than rebuilding
    without the staging, for the same reason ``built_demo_data_without_explain``
    does: the loader's absent branch keys off the file's presence, and the
    builder's own absent/included recording is pinned on the builder side
    (tests/test_demo_export.py::test_build_records_vlm_count_buckets_absent_
    without_a_phase6b_table).
    """
    (built_demo_data / "vlm_count_buckets.parquet").unlink()
    manifest_path = built_demo_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["validation"]["vlm_count_buckets"] = "absent"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return built_demo_data


_VLM_COUNT_BUCKET_COLUMNS = ["model", "bucket", "n", "mae"]


def test_load_vlm_count_buckets_reads_the_packages_table(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader hands the page the builder's own four-column table, buckets in
    count order -- the fixture's b1/b2 labels put a row in every bucket."""
    pytest.importorskip("streamlit")
    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    import data as demo_data  # type: ignore[import-not-found]

    buckets = demo_data.load_vlm_count_buckets()
    assert list(buckets.columns) == _VLM_COUNT_BUCKET_COLUMNS
    assert list(buckets["bucket"]) == ["0", "1-3", "4-9", "10+"]
    # frame x class PAIRS, not frames: two ok rows x the ten count fields
    assert int(buckets["n"].sum()) == 20


def test_load_vlm_count_buckets_is_empty_on_a_pre_0_8_package(
    built_demo_data_without_buckets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older package simply has no such file -- the loader returns the empty
    frame with the four columns (the graceful-absence shape every optional table
    uses), so the page can draw its own absent note instead of raising."""
    pytest.importorskip("streamlit")
    _reset_demo_app_modules()
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data_without_buckets))
    import data as demo_data  # type: ignore[import-not-found]

    buckets = demo_data.load_vlm_count_buckets()
    assert buckets.empty
    assert list(buckets.columns) == _VLM_COUNT_BUCKET_COLUMNS


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
    # filters.model_label leaves any name it has no entry for untouched, so the two
    # detection models read as their raw names, while the two weak-supervision
    # checkpoints (Phase 9b) carry the labels that keep them apart -- the arm
    # trained on pseudo labels and its GT-labelled twin differ by a `_gt` suffix
    # alone, which is exactly the distinction the page must not blur.
    assert model_radio.options == [
        "baseline",
        "graph_rate_night",
        "weak_graph_rate_night (pseudo labels, yolov8n)",
        "weak_graph_rate_night_gt (GT-labelled twin, yolov8n)",
    ]
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


def test_failure_explorer_n_preds_cards_name_the_two_weak_arms_apart(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Visual walk: the per-model n_preds cards sit in the detail's 2/5 column, and
    the two weak checkpoints' raw names differ only by a trailing ``_gt`` -- both
    labels truncated to the identical "n_preds (weak_graph_rate..." and the reader
    could not tell the pseudo-labelled arm from its GT twin.

    The cards now carry short display names (``failures._N_PREDS_SHORT``); the model
    RADIO is untouched and keeps ``filters.model_label``'s full names, which it has
    the width for.
    """
    pytest.importorskip("streamlit")
    at = _failures_apptest(built_demo_data, monkeypatch)
    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception

    labels = [str(m.label) for m in at.metric]
    n_preds = [label for label in labels if label.startswith("n_preds")]
    assert n_preds == [
        "n_preds (baseline)",
        "n_preds (graph_rate_night)",
        "n_preds (weak)",
        "n_preds (weak GT twin)",
    ]
    assert len(set(n_preds)) == len(n_preds)          # ... and no two read alike
    # Short LABELS, unchanged VALUES: v0's two weak arms found 1 and 2 boxes.
    values = {str(m.label): str(m.value) for m in at.metric}
    assert values["n_preds (weak)"] == "1"
    assert values["n_preds (weak GT twin)"] == "2"

    # The radio is untouched: its options are still `filters.model_label`'s full,
    # self-describing names (AppTest's `options` are the FORMATTED labels), so the
    # short names are a card-width fix, not a rename.
    options = at.radio(key="failure_model").options
    assert "weak_graph_rate_night_gt (GT-labelled twin, yolov8n)" in options
    assert "weak GT twin" not in " ".join(options)


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


def test_failure_explorer_keeps_the_model_choice_across_an_empty_filter(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item M6, Phase 9b consolidated review: a filter combination that matches
    nothing renders no detail, so the model radio is never instantiated on that run
    and streamlit drops its widget state -- the page silently fell back to
    `baseline` once the filters were widened again, discarding a choice the viewer
    made and never told them.

    The choice is now shadow-persisted in a key no widget owns, so the round trip
    (choose -> empty -> widen) comes back to the model the viewer picked.
    """
    pytest.importorskip("streamlit")
    at = _failures_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    at.radio(key="failure_model").set_value("graph_rate_night").run(timeout=30)
    assert not at.exception
    assert at.radio(key="failure_model").value == "graph_rate_night"

    # graph_rate_night has zero FN across both fixture frames -- the same empty
    # combination the two tests above use.
    at.radio(key="failure_type_select").set_value("Has FN").run(timeout=30)
    assert not at.exception
    assert any("No frames match these filters" in str(m.value) for m in at.info)

    at.radio(key="failure_type_select").set_value("All").run(timeout=30)
    assert not at.exception
    assert at.radio(key="failure_model").value == "graph_rate_night"
    # ... and the choice really is driving the detail again: v0's per-box table is
    # its 2 GT rows plus graph_rate_night's two claims (baseline claims a1 alone).
    at.session_state["failure_token"] = "v0"
    at.run(timeout=30)
    assert not at.exception
    assert len(at.dataframe[0].value) == 4


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
    assert any("defeats all five models" in caption for caption in hero_captions)


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
    # Once, not twice: the viewer's caption beside the frame is the only place the
    # page spells the sentence out (the metric row shows the short card value).
    assert sum(
        "not in the curated prediction set" in str(c.value) for c in at.caption
    ) == 1


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
    # Item I3, Phase 9b consolidated review: the readout is the GT EGO-POSE speed in
    # m/s while the metric card above it is the CAN speed in km/h -- the same
    # keyframe, ~25 % apart. Both are labelled now, so the two figures cannot read
    # as one contradicting itself.
    assert any(
        caption.startswith("current: ego ") and ", accel " in caption
        for caption in before_captions
    ), before_captions
    slider.set_value("t+1").run(timeout=30)
    assert not at.exception
    after_captions = [str(c.value) for c in at.caption]
    assert before_captions != after_captions   # the speed/accel readout changed
    assert any(caption.startswith("t+1: ego ") for caption in after_captions)


@pytest.fixture()
def built_demo_data_with_a_fast_cyclist(built_demo_data: Path) -> Path:
    """The flagship event, additionally tagged as a ``fast_cyclists`` hit with a
    cyclist distance of its own.

    Four of the six presets have no event at all in this tiny package, so the
    preset whose VRU card differs from every other one's is staged on the built
    package -- the same edit-the-built-package idiom
    ``built_demo_data_without_can_speed`` uses.
    """
    path = built_demo_data / "scenario_events.parquet"
    events = pd.read_parquet(path)
    hit = events["sample_data_token"] == "s1"
    events.loc[hit, "preset_rank_fast_cyclists"] = 1
    events.loc[hit, "min_dist_cyclist_m"] = 4.3
    events.to_parquet(path, index=False)
    return built_demo_data


def test_scenario_metric_row_names_the_vru_the_preset_is_about(
    built_demo_data_with_a_fast_cyclist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item M2, Phase 9b consolidated review: the merged metric row showed the
    PEDESTRIAN distance under every preset, including the one whose whole subject is
    a cyclist. The distance card switches with the preset (it does not duplicate --
    the row stays six cards), so a fast_cyclists event reads its own VRU's distance.
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data_with_a_fast_cyclist, monkeypatch)

    at.session_state["scenario_preset"] = "fast_cyclists"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    assert len(at.metric) == 6
    labels = [str(m.label) for m in at.metric]
    assert labels == [
        "CAN speed", "Min long. accel", "Min cyclist dist", "Peds within 10m",
        "Lighting / rain", "Model result",
    ]
    assert {str(m.label): str(m.value) for m in at.metric}["Min cyclist dist"] == "4.3 m"

    # Every other preset keeps the pedestrian distance -- the same six cards.
    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.run(timeout=30)
    assert not at.exception
    assert len(at.metric) == 6
    assert "Min ped dist" in [str(m.label) for m in at.metric]
    assert "Min cyclist dist" not in [str(m.label) for m in at.metric]


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


# Phase 9b (Task 5): the graph panel's progressive path reveal. The Phase-6 path
# colour doubles as the reveal's "amber" (views/scenarios.py's _NODE_ON_PATH_COLOR
# / _AMBER, so the legend's "Orange = the matched path" stays true) -- pinned here
# as a literal, like every other page string this file asserts on.
_AMBER = "#FF851B"


def _capture_agraph(
    monkeypatch: pytest.MonkeyPatch, *, clicked: str | None = None
) -> list[dict[str, Any]]:
    """Capture the Node/Edge/Config objects the graph panel hands to
    ``streamlit_agraph.agraph``, and decide what a click returns.

    ``_render_graph_panel`` imports ``agraph`` INSIDE the function (so a missing
    streamlit-agraph degrades to a warning instead of a fatal module import),
    which is exactly what makes this patch bite: the name is resolved on the
    module at call time. AppTest never round-trips a custom component, so without
    this the real ``agraph`` returns its default (None) and everything the panel
    computed -- which nodes it drew, in which colour -- stays unobservable.
    """
    streamlit_agraph = pytest.importorskip("streamlit_agraph")
    calls: list[dict[str, Any]] = []

    def _fake_agraph(nodes: Any, edges: Any, config: Any) -> str | None:
        calls.append({"nodes": list(nodes), "edges": list(edges), "config": config})
        return clicked

    monkeypatch.setattr(streamlit_agraph, "agraph", _fake_agraph)
    return calls


def test_scenario_graph_path_reveal(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Phase 9b Task 5) The select_slider walks the matched path one step at a
    time and the toggle folds the context nodes away.

    The fixture's flagship subgraph ("s1", built by _stage_subgraphs_with_event
    through the real assemble_subgraph) has 5 on-path nodes -- scene:sceneX,
    sample:s1, object:f1 (+ its Category) and egopose:s1 -- and 3 context ones:
    the 20 m pedestrian object:f2, the next Sample sample:v1, and
    location:boston-seaport.
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    calls = _capture_agraph(monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    slider = at.select_slider(key="scenario_path_step")
    assert list(slider.options) == [
        "1 · Scene scene-X", "2 · Keyframe", "3 · Pedestrian · 5.0 m", "4 · Ego pose",
    ]
    assert slider.value == slider.options[-1]   # the whole path is revealed by default

    # --- context nodes off (the default): the on-path 5, and no dangling edge ----
    drawn = calls[-1]
    ids = [node.id for node in drawn["nodes"]]
    assert len(ids) == 5
    assert set(ids) == {
        "scene:sceneX", "sample:s1", "egopose:s1", "object:f1",
        "category:human.pedestrian.adult",
    }
    # object:f2 (context) hangs off the SAME Category node as the matched
    # pedestrian, so an edge filter that only checked its source would leave that
    # OF_CATEGORY edge pointing at a node the panel never drew.
    assert all(edge.source in set(ids) and edge.to in set(ids) for edge in drawn["edges"])
    assert all(node.color == _AMBER for node in drawn["nodes"])

    # --- the facts table for the selected step (the ego pose, by default) --------
    tables = [table.value for table in at.table]
    assert any(
        list(table["fact"]) == ["CAN speed", "Min longitudinal accel", "Hard braking"]
        and list(table["value"]) == ["36.0 km/h", "-7.50 m/s²", "yes"]
        for table in tables
    )
    captions = [str(caption.value) for caption in at.caption]
    assert any("EgoPose node" in caption for caption in captions)   # where CAN lives
    assert any("Amber" in caption and "hollow" in caption for caption in captions)

    # --- context nodes on: the full subgraph ------------------------------------
    at.toggle(key="scenario_graph_context").set_value(True).run(timeout=30)
    assert not at.exception
    with_context = calls[-1]
    assert len(with_context["nodes"]) == 8
    assert {"object:f2", "sample:v1", "location:boston-seaport"} <= {
        node.id for node in with_context["nodes"]
    }
    assert len(with_context["edges"]) == 8

    # --- step 1: only the Scene node is revealed --------------------------------
    at.select_slider(key="scenario_path_step").set_value("1 · Scene scene-X").run(timeout=30)
    assert not at.exception
    first_step = calls[-1]
    amber = [node.id for node in first_step["nodes"] if node.color == _AMBER]
    assert amber == ["scene:sceneX"]
    # The other four on-path nodes are still DRAWN (the force layout must not jump
    # between steps) -- just hollow: white fill, grey border.
    hollow = [
        node.id for node in first_step["nodes"]
        if isinstance(node.color, dict) and node.color.get("background") == "#FFFFFF"
    ]
    assert sorted(hollow) == [
        "category:human.pedestrian.adult", "egopose:s1", "object:f1", "sample:s1",
    ]
    tables = [table.value for table in at.table]
    assert any(list(table["fact"]) == ["Scene", "Location", "Lighting", "Rain"] for table in tables)


def test_scenario_graph_reveal_survives_switching_to_another_event(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Phase 9b Task 5) The step the viewer left the slider on belongs to the
    PREVIOUS event: "3 · Pedestrian · 5.0 m" is s1's third step and no step of
    v1's at all (its matched pedestrian is at 7 m). Selecting another event must
    fall back to that event's own last step -- the whole path revealed -- not
    raise, and not silently reveal a step number that means something else here.
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    calls = _capture_agraph(monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    at.select_slider(key="scenario_path_step").set_value("3 · Pedestrian · 5.0 m").run(timeout=30)
    assert not at.exception

    at.session_state["scenario_preset"] = "night_pedestrians"
    at.session_state["scenario_token"] = "v1"
    at.run(timeout=30)
    assert not at.exception

    slider = at.select_slider(key="scenario_path_step")
    assert list(slider.options) == [
        "1 · Scene scene-X", "2 · Keyframe", "3 · Pedestrian · 7.0 m", "4 · Ego pose",
    ]
    assert slider.value == "4 · Ego pose"
    assert all(node.color == _AMBER for node in calls[-1]["nodes"])
    # v1's OWN ego readings (can_speed_kmh 18.0, accel -0.5, not hard braking) --
    # the facts follow the selected event, not the one the slider was set on.
    tables = [table.value for table in at.table]
    assert any(
        list(table["value"]) == ["18.0 km/h", "-0.50 m/s²", "no"] for table in tables
    )


def test_scenario_graph_click_still_shows_the_clicked_nodes_meta(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Phase 6 behaviour, kept through Task 5's reveal) A clicked node's own
    ``meta`` table takes precedence over the selected step's facts table -- and
    the path narrative stays on screen either way."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)
    _capture_agraph(monkeypatch, clicked="object:f1")

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    tables = [table.value for table in at.table]
    # object:f1's own meta, key order as the staged JSON carries it (sorted).
    assert any(
        list(table["key"]) == [
            "category", "distance_to_ego_m", "ego_rel_x", "ego_rel_y", "group", "visibility",
        ]
        and list(table["value"]) == [
            "human.pedestrian.adult", "5.0", "3.0", "-1.0", "pedestrian", "4",
        ]
        for table in tables
    )
    # The step facts table is NOT drawn while a node is selected.
    assert not any("fact" in table.columns for table in tables)
    assert any(
        "pedestrian at 5.00 m" in str(caption.value) for caption in at.caption
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


# --- Phase 9b (Task 4): the one-screen event viewer + its CAN curve -------------
#
# The viewer's two curves are built by render.curve_charts from
# filters.filmstrip_steps, so the Scenario page and tour step 2 draw the SAME pair
# (spec docs/superpowers/specs/2026-08-22-demo-phase9b-design.md sec2). These tests
# read the charts back off the wire the way Streamlit sends them: st.altair_chart
# marshals the vega-lite spec as a JSON string on the element proto and hoists each
# layer's own data into a named, Arrow-encoded dataset beside it.

_CAN_SPEED_TITLE = "CAN speed (km/h)"
_EGO_SPEED_TITLE = "ego speed (km/h)"
# Short: the long "CAN longitudinal accel (m/s²)" clipped to "CAN longitudinal
# accel (n" at the viewer's chart width. The full words stay in the caption under
# the pair (asserted below), so only the axis got shorter.
_CAN_ACCEL_TITLE = "CAN accel (m/s²)"


def _chart_specs(at: Any) -> list[dict[str, Any]]:
    return [json.loads(chart.proto.spec) for chart in at.get("vega_lite_chart")]


def _y_titles(at: Any) -> list[str]:
    """One y-axis title per chart on the page -- the line layer's, which is the
    layer that carries the series (the rule layers encode x only)."""
    return [str(spec["layer"][0]["encoding"]["y"]["title"]) for spec in _chart_specs(at)]


def _all_y_titles(at: Any) -> list[str]:
    """Every y-axis title on the page, layered charts flattened -- ``bar_chart``
    returns a plain chart when ``zero_line=False`` and a two-layer one otherwise,
    and this reads both shapes."""
    titles: list[str] = []
    for spec in _chart_specs(at):
        for layer in spec.get("layer", [spec]):
            title = (layer.get("encoding", {}).get("y") or {}).get("title")
            if title is not None:
                titles.append(str(title))
    return titles


def _strategy_chart_frames(at: Any) -> list[Any]:
    """The data behind every chart on the page that carries a ``strategy`` column
    -- Streamlit hoists each altair chart's own frame into an Arrow-encoded dataset
    beside the spec, so this reads the bars back exactly as the page drew them."""
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

    frames = []
    for chart in at.get("vega_lite_chart"):
        for dataset in chart.proto.datasets:
            frame = convert_arrow_bytes_to_pandas_df(dataset.data.data)
            if "strategy" in frame.columns:
                frames.append(frame)
    return frames


def _rule_steps(chart: Any) -> list[str]:
    """The step label(s) the chart's selected-step rule sits on, decoded from the
    Arrow dataset Streamlit hoisted the rule's own one-row frame into."""
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

    spec = json.loads(chart.proto.spec)
    rule = spec["layer"][1]
    datasets = {
        dataset.name: convert_arrow_bytes_to_pandas_df(dataset.data.data)
        for dataset in chart.proto.datasets
    }
    return [str(value) for value in datasets[rule["data"]["name"]]["step"]]


@pytest.fixture()
def built_demo_data_without_can_speed(built_demo_data: Path) -> Path:
    """The same package as a pre-0.8 `demo build` would have written it: no
    per-step CAN speed on ``scenario_events.parquet``.

    Same edit-the-built-package idiom as ``built_demo_data_without_explain`` --
    the app's own fallback keys off the COLUMNS being absent
    (``filters.filmstrip_steps``), so dropping them from the built table is
    exactly the state an older package puts the page in.
    """
    path = built_demo_data / "scenario_events.parquet"
    events = pd.read_parquet(path)
    events = events.drop(columns=[c for c in events.columns if c.startswith("can_speed")])
    assert not [c for c in events.columns if c.startswith("can_speed")]
    events.to_parquet(path, index=False)
    return built_demo_data


def test_scenario_viewer_is_one_screen_with_can_curve(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The viewer is one screen: the frame beside the two CAN curves, then ONE
    metric row, then the filmstrip -- not the six stacked full-width panels the
    page drew before (spec sec2).

    The fixture's flagship event ("s1") has exactly two filmstrip steps (its only
    non-NA neighbour is "v1" at t+1), so the curves' x order is ["current", "t+1"]
    -- read straight off the chart spec, which is also what proves the axis is the
    nominal step sequence rather than a time axis.
    """
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    # (a) the two curves, titled as CAN readings (the package carries can_speed_*)
    titles = _y_titles(at)
    assert _CAN_SPEED_TITLE in titles
    assert _CAN_ACCEL_TITLE in titles
    specs = _chart_specs(at)
    for spec in specs:
        assert spec["layer"][0]["encoding"]["x"]["sort"] == ["current", "t+1"]
    captions = [str(c.value) for c in at.caption]
    assert any(
        "Keyframes are ~0.5 s apart; speed and longitudinal acceleration from the "
        "CAN bus" in c
        for c in captions
    )

    # (b) six cards -- the ego/context/model panels' figures merged -- laid out
    # THREE to a row (visual walk: at six per row st.metric truncated the speed,
    # accel and model values). Same cards, same order, two rows.
    assert len(at.metric) == 6
    labels = [str(m.label) for m in at.metric]
    assert labels == [
        "CAN speed", "Min long. accel", "Min ped dist", "Peds within 10m",
        "Lighting / rain", "Model result",
    ]
    values = {str(m.label): str(m.value) for m in at.metric}
    assert values["CAN speed"] == "36.0 km/h"
    assert values["Min long. accel"] == "-7.50 m/s²"
    assert values["Min ped dist"] == "5.0 m"
    assert values["Peds within 10m"] == "1"
    assert values["Lighting / rain"] == "day"
    # "s1" is not in the curated prediction set. The CARD says so in two words
    # (item M3, Phase 9b consolidated review: the 33-character sentence truncates as
    # an st.metric value, and this is the flagship preset's default state) and the
    # page's own pinned wording is _render_viewer's caption beside the frame --
    # EXACTLY ONCE (visual walk: the metric row said it a second time).
    assert values["Model result"] == "not curated"
    assert sum("not in the curated prediction set" in c for c in captions) == 1

    # (c) the three stacked panels' headings are gone
    markdowns = [str(m.value) for m in at.markdown]
    for heading in ("Ego dynamics", "Scene context", "Model context"):
        assert not any(heading in text for text in markdowns), heading

    # (d) the viewer's pinned trust chrome survives the relayout
    assert any("1 event found · SQL 1 / Graph 1 ✓" in c for c in captions)
    assert any("recorded experiment output" in c for c in captions)

    # (e) ... and a model-family preset still gets the graph-holds-GT-only note
    at.session_state["scenario_preset"] = "fn_pedestrians_night"
    at.run(timeout=30)
    assert not at.exception
    assert any(
        "graph holds GT only — model verdict comes from the prediction set" in str(c.value)
        for c in at.caption
    )


def test_scenario_curve_rule_follows_the_filmstrip_slider(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filmstrip slider is the curves' cursor: moving it moves the vertical
    rule on BOTH charts, so the strip and the curve always point at the same
    keyframe."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception
    charts = at.get("vega_lite_chart")
    assert len(charts) == 2
    assert all(_rule_steps(chart) == ["current"] for chart in charts)

    at.select_slider(key="scenario_filmstrip").set_value("t+1").run(timeout=30)
    assert not at.exception
    charts = at.get("vega_lite_chart")
    assert len(charts) == 2
    assert all(_rule_steps(chart) == ["t+1"] for chart in charts)


def test_scenario_viewer_titles_ego_speed_on_an_old_package(
    built_demo_data_without_can_speed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The honesty rule "a CAN curve must be CAN": a package built before v0.8 has
    no per-step CAN speed, so the speed series falls back to the GT ego pose and
    the chart says "ego speed" -- it never labels an ego-pose reading as CAN. The
    acceleration series IS CAN on every package, so its title is unchanged."""
    pytest.importorskip("streamlit")
    at = _scenarios_apptest(built_demo_data_without_can_speed, monkeypatch)

    at.session_state["scenario_preset"] = "hard_braking_near_pedestrians"
    at.session_state["scenario_token"] = "s1"
    at.run(timeout=30)
    assert not at.exception

    titles = _y_titles(at)
    assert _EGO_SPEED_TITLE in titles
    assert _CAN_SPEED_TITLE not in titles
    assert _CAN_ACCEL_TITLE in titles
    captions = [str(c.value) for c in at.caption]
    assert any(
        "Keyframes are ~0.5 s apart; ego speed from the GT ego pose, longitudinal "
        "acceleration from the CAN bus" in c
        for c in captions
    )
    assert not any("speed and longitudinal acceleration from the CAN bus" in c for c in captions)
    # ... and the metric card names the same source as the chart it sits under.
    assert "Ego speed" in [str(m.label) for m in at.metric]
    assert "CAN speed" not in [str(m.label) for m in at.metric]


def test_tour_step_2_shows_the_can_curve(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tour step 2 draws the SAME pair of curves under its filmstrip (no slider --
    a tour screen is one visual, not a control panel), so the condensed step and
    the deep page tell the identical story about the ego's motion."""
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 2)
    assert not at.exception

    titles = _y_titles(at)
    assert titles == [_CAN_SPEED_TITLE, _CAN_ACCEL_TITLE]
    charts = at.get("vega_lite_chart")
    assert all(_rule_steps(chart) == ["current"] for chart in charts)
    assert any(
        "Keyframes are ~0.5 s apart; speed and longitudinal acceleration from the "
        "CAN bus" in str(c.value)
        for c in at.caption
    )


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
    # The package's own model labels (filters.model_label): the two detection
    # models by their raw names, the Phase-9b weak pair by the labels that say what
    # each was trained on.
    assert at.radio(key="al_model").options == [
        "baseline",
        "graph_rate_night",
        "weak_graph_rate_night (pseudo labels, yolov8n)",
        "weak_graph_rate_night_gt (GT-labelled twin, yolov8n)",
    ]

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


def test_active_learning_page_compares_the_acquisition_strategies(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec sec4: the acquisition comparison sits ABOVE the all-arms chart and
    answers "why graph-aware mining rather than similarity mining?" from the arms'
    own mined-set composition -- scenes covered and night share, two charts over
    the same budget.

    This fixture ran two of the three strategies (it has no ``mined`` arm), so both
    the heading's count word and the callout are COMPUTED from what is in the
    package: the strategies are listed side by side with no ranking claim, and the
    "found near-duplicates" lesson is not written at all.
    """
    pytest.importorskip("streamlit")
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    subheaders = [str(s.value) for s in at.subheader]
    # 20 = this arm's own 120 training images minus the baseline's 100, never a
    # literal; "Two" because two of the three strategies have a row here.
    assert "Two ways to pick 20 frames" in subheaders
    assert subheaders.index("Two ways to pick 20 frames") < subheaders.index(
        "Every arm, one chart"
    )

    titles = _all_y_titles(at)
    assert "scenes covered" in titles
    assert "night share" in titles
    drawn = _strategy_chart_frames(at)
    assert len(drawn) == 2
    for frame in drawn:
        assert list(frame["strategy"]) == ["Random sample", "Graph-aware mining"]

    captions = [str(caption.value) for caption in at.caption]
    assert any(
        caption.startswith("**Graph-aware mining** (`graph_rate_night`) —")
        for caption in captions
    )

    learned = [str(m.value) for m in at.markdown if "What we learned" in str(m.value)]
    assert any(
        "Random sample: 1 scene · 0% night · +0.0100" in text
        and "Graph-aware mining: 1 scene · 100% night · +0.0300" in text
        for text in learned
    )
    assert not any("near-duplicates" in text for text in learned)


def _add_mined_arm(built_demo_data: Path) -> None:
    """Give the built package the third acquisition strategy, with the real
    package's own composition for the two arms the lesson compares (`mined` 219
    scenes / 0% night / +0.0072; `graph_rate_night` 368 scenes / 30.9% night).

    Deliberately absent from the base fixture -- which is what makes the neutral
    branch of the strategy callout testable at all (see the test above) -- and
    added here the same way ``_add_weak_night_twin_arm`` adds the weak twin: only
    the columns the section reads are set, the rest come back NaN through
    ``pd.concat``'s own column align.
    """
    path = built_demo_data / "active_learning_results.parquet"
    arms = pd.read_parquet(path)
    graph = arms["arm"] == "graph_rate_night"
    arms.loc[graph, "n_scenes"] = 368
    arms.loc[graph, "night_share"] = 0.309
    extra = pd.DataFrame({
        "arm": ["mined"],
        "family": ["mined"],
        "round_order": [int(arms["round_order"].max()) + 1],
        "n_train_images": [118],
        "overall_map5095": [0.215],
        "night_map5095": [0.1072],
        "delta_overall": [0.015],
        "delta_night": [0.0072],
        "n_scenes": pd.array([219], dtype="Int64"),
        "night_share": [0.0],
    })
    pd.concat([arms, extra], ignore_index=True).to_parquet(path, index=False)


def test_active_learning_strategy_lesson_is_computed_not_asserted(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With all three strategies in the package the callout may make its comparison
    -- and only because this package's own numbers support every clause of it: the
    graph arm really does cover more scenes AND a higher night share than the
    similarity arm, and really does hold the table's best night gain (spec's
    "computed, never asserted, superlatives")."""
    pytest.importorskip("streamlit")
    _add_mined_arm(built_demo_data)
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    assert "Three ways to pick 20 frames" in [str(s.value) for s in at.subheader]
    learned = [str(m.value) for m in at.markdown if "What we learned" in str(m.value)]
    assert any(
        "Similarity mining found near-duplicates: 219 scenes, 0% night, +0.0072 "
        "night mAP50-95; graph-aware mining spread the same budget over 368 scenes "
        "at 31% night and took the best night gain (+0.0300)." in text
        for text in learned
    )


def test_active_learning_why_selected_panel_leads_with_reason_chips(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) Spec sec4: the per-frame panel opens with the frame's selection facts as
    chips, above the factor lines that spell the same facts out. "wA" carries one
    visible pedestrian GT box and no routed failure mass, so the pedestrian chip is
    written and the routed-failures chip is not -- and no chip anywhere says "rate"
    (there is no per-frame failure rate in the package; the zero-pedestrian branch
    is pinned in tests/test_demo_filters.py::test_reason_chips_count_and_pluralise_
    visible_pedestrian_gt).
    """
    pytest.importorskip("streamlit")
    at = _active_learning_apptest(built_demo_data, monkeypatch)
    at.session_state["al_frame_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception

    markdowns = [str(m.value) for m in at.markdown]
    chips = next(text for text in markdowns if ":blue-badge[night]" in text)
    assert chips == (
        ":blue-badge[night] :blue-badge[1 pedestrian GT box] "
        ":blue-badge[community #0 · 10 frames] "
        ":blue-badge[mass rank 1 of 2] :blue-badge[quota 1] "
        ":blue-badge[night-pass pick (floor 1)] :blue-badge[degree rank 2 of 10]"
    )
    assert "rate" not in chips
    factors = next(text for text in markdowns if text.startswith("**Night frame:**"))
    assert markdowns.index(chips) < markdowns.index(factors)


def test_active_learning_page_writes_no_reason_chips_without_the_explain_group(
    built_demo_data_without_explain: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No staged selection facts, no chips: the panel says exactly what is missing
    rather than tagging the frame with reasons this package cannot support."""
    pytest.importorskip("streamlit")
    at = _active_learning_apptest(built_demo_data_without_explain, monkeypatch)
    at.session_state["al_frame_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception

    assert not any(":blue-badge[" in str(m.value) for m in at.markdown)
    assert any(
        "per-frame community and routed mass are not included in this package" in str(i.value)
        for i in at.info
    )


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
    assert [str(title.value) for title in at.title] == [
        "From model failure to better training data"
    ]
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


# --- Phase 9b (Task 8): Weak Supervision's one frame, three views -------------------
#
# Spec docs/superpowers/specs/2026-08-22-demo-phase9b-design.md sec6: the page stops
# stating the weak-supervision result only as a retention number. Row 1 shows what the
# VLM's counts bought on ONE accepted train-pool frame (GT | pseudo labels | the
# verdict and the count vote), row 2 what the two checkpoints trained on those frames
# then did on a held-out val frame neither of them saw -- the weak-labelled arm beside
# its GT-labelled twin, with the per-box difference underneath. The count-bucket chart
# below the loss split is the same story from the verifier's side. Every pinned string
# and key of the earlier phases survives verbatim (asserted in the tests above).

# filters._FIXED_BOX_COLUMNS, copied rather than imported (this module never imports
# the app package at collection time; the page tests reach it through AppTest only).
_FIXED_BOX_COLUMNS = [
    "annotation_token", "category_group", "distance_to_ego_m", "baseline_claim", "arm_claim",
]
_WEAK_ARM_LABEL = "weak_graph_rate_night (pseudo labels, yolov8n)"
_GT_ARM_LABEL = "weak_graph_rate_night_gt (GT-labelled twin, yolov8n)"
_RESULT_ABSENT_NOTE = "held-out weak-arm predictions need demo_data >= 0.8 — rerun demo build"
_BUCKETS_ABSENT_NOTE = "count-bucket chart needs demo_data >= 0.8 — rerun demo build"


def _image_captions(at: Any) -> list[str]:
    """Every caption the page attached to an ``st.image``. The six "One frame, three
    views" images are the only captioned ones on this page (the galleries caption
    their thumbs with a separate ``st.caption``), so this is also how the section's
    two rows of three are counted."""
    return [caption for image in at.image for caption in image.captions if caption]


def _bucket_chart(at: Any) -> tuple[dict[str, Any], Any] | None:
    """The count-bucket chart's spec and the frame Streamlit hoisted it into --
    identified by the ``bucket`` column, the way ``_strategy_chart_frames`` finds the
    strategy chart."""
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

    for chart in at.get("vega_lite_chart"):
        for dataset in chart.proto.datasets:
            frame = convert_arrow_bytes_to_pandas_df(dataset.data.data)
            if "bucket" in frame.columns:
                return json.loads(chart.proto.spec), frame
    return None


def test_weak_supervision_one_frame_three_views(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The section's two rows: the train-pool showcase frame (wA -- accepted, one
    pseudo box, one visible pedestrian GT box) and the held-out val frame (v0 --
    the GT-labelled twin catches the pedestrian a2 the weak arm misses), three
    images each, with the pair's provenance stated for what it is."""
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    assert "One frame, three views" in [str(h.value) for h in at.subheader]
    captions = _image_captions(at)
    assert len(captions) == 6
    # row 1: the same train-pool frame under three layers
    assert any(caption.startswith("ground truth") for caption in captions)
    assert any(caption.startswith("pseudo labels") for caption in captions)
    assert any(caption.startswith("both layers") for caption in captions)
    # row 2: the two checkpoints, each named for what it was trained on
    assert _WEAK_ARM_LABEL in captions
    assert _GT_ARM_LABEL in captions

    page_captions = [str(c.value) for c in at.caption]
    assert any(
        "held-out val frame — neither detector saw it in training" in caption
        for caption in page_captions
    )
    # the inference is RECORDED output of an offline run, labelled as such
    assert any(
        "demo infer, CPU, checkpoint of the training run" in caption
        for caption in page_captions
    )
    assert any(
        "pseudo boxes from weak_labels.parquet" in caption for caption in page_captions
    )
    # ... and the row-2 OVERLAYS are drawn here from the package tables (item M1,
    # Phase 9b consolidated review): the row stated where the predictions came from
    # and left the drawing itself unattributed, unlike row 1 just above it.
    assert any(
        "overlay drawn from gt_boxes.parquet and predictions.parquet" in caption
        for caption in page_captions
    )
    # row 1's count vote is a static table, so the galleries' own count dataframes
    # stay the only two of that shape on the page (pinned above).
    vote = [t.value for t in at.table if list(getattr(t.value, "columns", [])) == [
        "vlm_count", "gt_count"
    ]]
    assert len(vote) == 1


def test_weak_supervision_result_table_follows_the_detector_radio(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The radio picks which of the two detectors the per-box table credits. It
    opens on the GT-labelled twin -- the frame was chosen because the twin fixes
    something -- whose one upgraded box is v0's pedestrian a2; the weak-labelled
    arm upgrades nothing over the twin, and the page says so rather than showing an
    empty grid."""
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    radio = at.radio(key="ws_result_model")
    assert radio.options == [_WEAK_ARM_LABEL, _GT_ARM_LABEL]
    assert radio.value == "weak_graph_rate_night_gt"
    boxes = [
        d.value for d in at.dataframe
        if list(getattr(d.value, "columns", [])) == _FIXED_BOX_COLUMNS
    ]
    assert len(boxes) == 1
    assert list(boxes[0]["annotation_token"]) == ["a2"]

    at.radio(key="ws_result_model").set_value("weak_graph_rate_night").run(timeout=30)
    assert not at.exception
    assert not [
        d for d in at.dataframe
        if list(getattr(d.value, "columns", [])) == _FIXED_BOX_COLUMNS
    ]
    assert any("no such box on this frame" in str(c.value) for c in at.caption)


def test_weak_supervision_count_bucket_chart_and_callout(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crowding buckets, drawn from vlm_count_buckets.parquet: one bar per
    bucket in count order, ``n`` named as frame-class PAIRS (the exporter pools the
    ten count fields, so a bucket counts pairs, not frames), and a callout whose
    numbers are the table's own MAEs (this fixture: 1/17 wrong on the 17 empty
    pairs, then 1, 1 and 2).
    """
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    assert not at.exception

    found = _bucket_chart(at)
    assert found is not None
    spec, frame = found
    assert list(frame["bucket"]) == ["0", "1-3", "4-9", "10+"]
    assert spec["encoding"]["x"]["sort"] == ["0", "1-3", "4-9", "10+"]
    assert spec["encoding"]["y"]["title"] == "count MAE"

    captions = [str(c.value) for c in at.caption]
    assert any(
        "n = frame-class pairs" in caption and "0 → 17" in caption and "10+ → 1" in caption
        for caption in captions
    )
    # Item I1, Phase 9b consolidated review: the pooling is over the VLM's TEN count
    # fields (49,860 = 4,986 x 10 in the real package), not the five detector classes
    # the count-vote table above shows -- the caption said "five" and invited the
    # reader to equate the two sets.
    assert any(
        "ten count fields" in caption and "five detector classes" in caption
        for caption in captions
    )
    assert not any("The five classes are pooled" in caption for caption in captions)
    assert any("vlm_count_buckets.parquet" in caption for caption in captions)

    learned = [str(m.value) for m in at.markdown if "What we learned" in str(m.value)]
    assert any(
        "Why crowded frames defeat the VLM" in text
        and "0.06" in text and "1.00" in text and "2.00" in text
        for text in learned
    )


@pytest.fixture()
def built_demo_data_with_non_monotone_buckets(built_demo_data: Path) -> Path:
    """The same package with a bucket table whose MAEs FALL between two buckets --
    the shape the callout's neutral branch exists for."""
    path = built_demo_data / "vlm_count_buckets.parquet"
    buckets = pd.read_parquet(path)
    falls = {"0": 0.50, "1-3": 2.00, "4-9": 0.25, "10+": 1.00}
    buckets["mae"] = [falls[str(bucket)] for bucket in buckets["bucket"]]
    buckets.to_parquet(path, index=False)
    return built_demo_data


def test_weak_supervision_count_bucket_callout_is_neutral_when_the_maes_fall(
    built_demo_data_with_non_monotone_buckets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item M9, Phase 9b consolidated review: "rises with the crowd" is a claim about
    THIS package's table, so it is only written when the table's own MAEs never fall.
    Nothing covered the other branch -- on a table that falls between buckets the
    callout must say the neutral thing and still quote every bucket's own figure.
    """
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data_with_non_monotone_buckets, monkeypatch)
    assert not at.exception

    learned = [str(m.value) for m in at.markdown if "What we learned" in str(m.value)]
    assert any("moves with the crowd" in text for text in learned)
    assert not any("rises with the crowd" in text for text in learned)
    assert any(
        "0.50 on empty frame-class pairs" in text
        and "2.00 at 1-3 objects" in text
        and "0.25 at 4-9 objects" in text
        and "1.00 at 10+ objects" in text
        for text in learned
    )


def test_weak_supervision_without_the_count_buckets(
    built_demo_data_without_buckets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package built before 0.8 carries no bucket table: the note names the
    version and the rebuild, and no chart or callout is drawn from nothing."""
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data_without_buckets, monkeypatch)
    assert not at.exception

    assert any(_BUCKETS_ABSENT_NOTE in str(i.value) for i in at.info)
    assert _bucket_chart(at) is None
    assert not any(
        "Why crowded frames defeat the VLM" in str(m.value) for m in at.markdown
    )


def test_weak_supervision_without_the_weak_arm_predictions(
    built_demo_data_without_weak_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the two checkpoints' held-out predictions there is no second row to
    draw: the page says which package version carries them instead of comparing a
    model this package never evaluated. Row 1 is unaffected -- it needs no
    predictions at all."""
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data_without_weak_models, monkeypatch)
    assert not at.exception

    assert any(_RESULT_ABSENT_NOTE in str(i.value) for i in at.info)
    assert not [r for r in at.radio if r.key == "ws_result_model"]
    assert len(_image_captions(at)) == 3
    assert "Weak-arm detections on this frame" not in [
        str(e.label) for e in at.get("expander")
    ]


def test_weak_supervision_folds_the_weak_arms_out_of_the_gallery_panel(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gallery's downstream panel keeps its arm-level lines in the open and
    folds the weak checkpoints' own story away: the curated frames are train-pool
    frames, so neither checkpoint has a detection on them, and saying so belongs
    behind an expander rather than beside the result."""
    pytest.importorskip("streamlit")
    at = _weak_supervision_apptest(built_demo_data, monkeypatch)
    at.session_state["ws_accepted_token"] = "wA"
    at.run(timeout=30)
    assert not at.exception

    assert "Weak-arm detections on this frame" in [str(e.label) for e in at.get("expander")]
    folded = [
        str(m.value) for m in at.markdown
        if "`weak_graph_rate_night`" in str(m.value) and "held-out val frames" in str(m.value)
    ]
    assert len(folded) == 1


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
    ("views/tour.py", "From model failure to better training data"),
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

    assert [str(title.value) for title in at.title] == [
        "From model failure to better training data"
    ]
    captions = [str(caption.value) for caption in at.caption]
    assert any(text.startswith("Step 1 of 7 · We found a blind spot") for text in captions)
    # the page's subtitle, written directly under the title
    assert any(
        text.startswith("See how the data engine finds a perception weakness")
        for text in captions
    )
    # (step 0) the headline the three cards are the evidence for
    assert [str(head.value) for head in at.subheader] == [
        "Aggregate accuracy was hiding a night-driving blind spot."
    ]

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
    # (step 0) the single derived sentence: all three numbers off the same baseline
    # row the cards are read from (0.2000 / 0.1000 / ped 0.0500).
    assert any(
        "`baseline` scores 0.2000 mAP50-95 overall but 0.1000 at night — and 0.0500 "
        "on night pedestrians, the case that matters most." in text
        for text in markdowns
    )
    # ... closed by the takeaway callout
    assert any(
        ":material/school:" in text
        and "Aggregate metrics hide important failure slices." in text
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
    assert any(
        text.startswith("Step 2 of 7 · What the failure looks like") for text in captions
    )
    # Back is live in the SAME run the click landed in: the buttons move tour_step in
    # an on_click callback, which streamlit runs BEFORE the script redraws the nav row
    # (mutating it inline after the row is drawn would leave Back greyed out on the
    # step the viewer just walked into).
    assert at.button(key="tour_back").disabled is False

    # the step's own headline, and the toggle's two arms under their story names
    # (the widget KEY is untouched -- only how the options read).
    assert [str(head.value) for head in at.subheader] == [
        "Here the baseline misses a pedestrian at night."
    ]
    assert at.radio(key="tour_hero_model").options == [
        "Baseline (no mined data)",
        "Graph + night targeting",
    ]
    assert len(at.image) == 1
    assert any("defeats all five models" in text for text in captions)
    # v0 is a val row in the fixture's frame_manifest, so the held-out line is written
    assert "Held-out validation frame — never in any training set" in captions

    markdowns = [str(block.value) for block in at.markdown]
    # the compact badge legend, under the overlay
    assert any(":green-badge[green — ground truth]" in text for text in markdowns)
    # the derived fact line, each model under its story label: baseline never claims
    # v0's pedestrian (a2); graph_rate_night claims it at conf 0.80 (a plain tp here).
    assert any(
        "Pedestrian" in text and "Baseline (no mined data): no claim" in text
        and "Graph + night targeting: 0.800" in text
        for text in markdowns
    )
    # the bridge out of one frame and into the mining steps
    assert any(
        "Finding one failure is easy — the hard part is finding the rest of the "
        "dataset where the same thing happens." in text
        for text in markdowns
    )
    # and the counted (never assumed) miss -> hit claim
    assert any("only night frame" in text for text in markdowns)
    assert any("recomputed in this app" in text for text in captions)

    # --- Back -> step 0 ------------------------------------------------------------
    at.button(key="tour_back").click().run(timeout=30)
    assert not at.exception
    assert any(
        str(caption.value).startswith("Step 1 of 7 · We found a blind spot")
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

    # --- step 2: "Where else does this happen?" ------------------------------------
    _walk_to_step(at, 2)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        text.startswith("Step 3 of 7 · Where else does this happen?") for text in captions
    )
    # the headline the event, its filmstrip and its curves are the evidence for
    assert [str(head.value) for head in at.subheader] == [
        "Is this one bad photo, or a recurring driving scenario?"
    ]
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
    # the one mechanism sentence: what the search matched on to find those events
    assert any(
        "The system searches driving context — night, braking, pedestrians near the "
        "ego — not just similar-looking images." in text
        for text in markdowns
    )
    # ... closed by the step's takeaway callout
    assert any(
        ":material/school:" in text
        and "Perception failures must be analyzed in driving context." in text
        for text in markdowns
    )
    assert "Scenario Search" in str(at.button(key="tour_open_event").label)

    # --- step 3: "What data should we add?" ----------------------------------------
    at.button(key="tour_next").click().run(timeout=30)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        text.startswith("Step 4 of 7 · What data should we add?") for text in captions
    )
    assert [str(head.value) for head in at.subheader] == [
        "Thousands of candidate training frames — which are worth labelling?"
    ]
    # the one-line selection path, above the fold that details it
    assert any(
        "**failed val frame → similarity community → representative train-pool "
        "frames → selected for retraining**" in text
        for text in markdowns
    )
    # selection_factors, rendered exactly as the Active Learning page renders them
    # (the fixture's first candidate frame was taken by the night pass, floor 1) --
    # folded away now: mechanism detail, not the step's headline claim.
    assert any(text.startswith("**Night frame:** yes ✓") for text in markdowns)
    assert any("**Picked in:** night pass (night floor 1)" in text for text in markdowns)
    fold = next(block for block in at.expander if block.label == "How selection works")
    folded = [str(block.value) for block in fold.markdown]
    assert any(text.startswith("**Night frame:** yes ✓") for text in folded)
    assert any("Nothing about this frame on its own selected it" in text for text in folded)
    # the selection is REPRODUCED (demo al-explain), not recomputed or recorded
    assert any("reproduced selection" in text for text in captions)
    # the whole system path, in one grey line under the step
    assert any(
        text.startswith(
            "System path: nuScenes → validated Parquet → SQL / Neo4j / CAN → failure "
            "analysis → scenario search → active learning → YOLO retraining → evaluation"
        )
        for text in captions
    )
    assert "Active Learning" in str(at.button(key="tour_open_al_frame").label)

    # --- step 4: "We changed the training data" ------------------------------------
    at.button(key="tour_next").click().run(timeout=30)
    assert not at.exception
    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert any(
        text.startswith("Step 5 of 7 · We changed the training data") for text in captions
    )
    # the intervention headline -- a claim the composition line under the cards
    # then supports with this package's own night shares
    assert [str(head.value) for head in at.subheader] == [
        "We did not just add data — we changed what the model trains on."
    ]
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
    # WHAT changed, from the arms table's own night shares (arm 1.0, random 0.0) --
    # and no similarity clause, because this package never ran `mined`.
    assert (
        "Targeted mining: **100% night** · random control: **0% night**" in markdowns
    )
    assert not any("similarity mining:" in text for text in markdowns)
    # ONE chart, and it is the five-strategy story chart (story labels on the axis),
    # not the deep page's 13-arm chart: the fixture carries three of the strategies.
    assert len(at.get("vega_lite_chart")) == 1
    drawn = _strategy_chart_frames(at)
    assert len(drawn) == 1
    assert list(drawn[0]["strategy"]) == [
        "Baseline (no mined data)",
        "Random sample — control",
        "Graph + night targeting",
    ]
    assert "Δ night mAP50-95" in _all_y_titles(at)
    # the raw ids stay on screen, in small text under the chart
    assert any(
        "Baseline (no mined data) (`baseline`) · Random sample — control (`random`) · "
        "Graph + night targeting (`graph_rate_night`)" in text
        for text in captions
    )
    # the fairness statement. The fixture's charted arms do NOT share a budget
    # (random 115 vs graph_rate_night 120 train images), so no frame count is
    # claimed -- test_tour_retrain_step_charts_similarity_mining_when_that_arm_ran
    # pins the other branch.
    assert (
        "Same detector · same budget · same training configuration · scored on the "
        "same held-out split — Random sample is the control." in markdowns
    )
    # the winner is computed over the WHOLE arm table, not asserted
    assert (
        "Best intervention for the diagnosed night weakness: **Graph + night "
        "targeting** (`graph_rate_night`, +0.0300 night)." in markdowns
    )
    # the mined set's spread, with the noun agreeing: one scene here, so the
    # "not near-duplicates" clause (true only of a many-scene set) is not written
    assert "The 20 frames came from 1 scene." in markdowns
    assert not any("near-duplicates" in text for text in markdowns)
    # ... and no similarity explanation: there is no `mined` arm to say it about
    assert not any("Visual similarity alone" in text for text in captions)
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


def _add_mined_arm_at_the_shared_budget(built_demo_data: Path) -> None:
    """Give the built package the similarity-mining arm, at the SAME training-set
    size as every other charted arm.

    Two things the base fixture cannot show on the retrain step, in one package:
    the real package's `mined` composition (219 scenes, 0 % night, +0.0072 night --
    the numbers the §37 explanation is about), and a set of charted arms that all
    spent the same budget, which is what lets the fairness statement name a frame
    count at all. Injected the way ``_add_mined_arm`` / ``_add_weak_night_twin_arm``
    inject theirs: only the columns the step reads are set, the rest come back NaN
    through ``pd.concat``'s own column align.
    """
    path = built_demo_data / "active_learning_results.parquet"
    arms = pd.read_parquet(path)
    arms.loc[arms["arm"] == "random", "n_train_images"] = 120
    extra = pd.DataFrame({
        "arm": ["mined"],
        "family": ["mined"],
        "round_order": [int(arms["round_order"].max()) + 1],
        "n_train_images": [120],
        "overall_map5095": [0.215],
        "night_map5095": [0.1072],
        "delta_overall": [0.015],
        "delta_night": [0.0072],
        "n_scenes": pd.array([219], dtype="Int64"),
        "night_share": [0.0],
    })
    pd.concat([arms, extra], ignore_index=True).to_parquet(path, index=False)


def test_tour_retrain_step_charts_similarity_mining_when_that_arm_ran(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a `mined` row in the package the retrain step gains the fourth bar, its
    night-share clause, the §37 explanation of what visual similarity alone bought
    -- and, because every charted arm now shares a training-set size, the fairness
    statement names the budget instead of just claiming one.

    ``test_tour_walks_steps_2_to_5`` pins the other side of all four branches on the
    base fixture (no `mined` row, unequal budgets).
    """
    pytest.importorskip("streamlit")
    _add_mined_arm_at_the_shared_budget(built_demo_data)
    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 4)
    assert not at.exception

    captions = [str(caption.value) for caption in at.caption]
    markdowns = [str(block.value) for block in at.markdown]
    assert (
        "Targeted mining: **100% night** · random control: **0% night** · "
        "similarity mining: **0% night**" in markdowns
    )
    drawn = _strategy_chart_frames(at)
    assert len(drawn) == 1
    assert list(drawn[0]["strategy"]) == [
        "Baseline (no mined data)",
        "Random sample — control",
        "Similarity mining",
        "Graph + night targeting",
    ]
    assert any("Similarity mining (`mined`)" in text for text in captions)
    # §37: the similarity arm's 0 % night, explained where it is charted
    assert any(
        "Visual similarity alone concentrated on daytime appearance (0% night); the "
        "graph + night-floor arm explicitly preserved night coverage." in text
        for text in captions
    )
    # every charted non-baseline arm is at 120 training images, so the budget is named
    assert (
        "Same detector · same 120-frame budget · same training configuration · scored "
        "on the same held-out split — Random sample is the control." in markdowns
    )
    # the winner is still computed: `mined` (+0.0072) does not beat the tour's arm
    assert (
        "Best intervention for the diagnosed night weakness: **Graph + night "
        "targeting** (`graph_rate_night`, +0.0300 night)." in markdowns
    )


def test_tour_step_3_leads_with_the_same_reason_chips(
    built_demo_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec sec4: step 3 opens with the same chip row the Active Learning page's
    per-frame panel does, between the frame and the factor lines -- built by the
    same ``filters.reason_chips`` from the same explain row, so the tour cannot
    tell a different story about why a frame was picked. Its frame ("v0") does
    carry one visible pedestrian GT box, which the chips count."""
    pytest.importorskip("streamlit")
    at = _tour_apptest(built_demo_data, monkeypatch)
    _walk_to_step(at, 3)
    assert not at.exception

    markdowns = [str(block.value) for block in at.markdown]
    chips = next(text for text in markdowns if ":blue-badge[night]" in text)
    assert chips == (
        ":blue-badge[night] :blue-badge[1 pedestrian GT box] "
        ":blue-badge[community #0 · 10 frames] :blue-badge[mass rank 1 of 2] "
        ":blue-badge[quota 1] :blue-badge[night-pass pick (floor 1)] "
        ":blue-badge[degree rank 1 of 10]"
    )
    assert "rate" not in chips
    factors = next(text for text in markdowns if text.startswith("**Night frame:**"))
    assert markdowns.index(chips) < markdowns.index(factors)


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
        # singular nouns: the fixture's arm covers exactly one scene, and the
        # comparator one too -- the page must not say "1 scenes"
        "mined 20 frames across 1 scene, 100% at night" in text
        and "random covered 1 scene at 0% night" in text
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
        str(caption.value).startswith("Step 1 of 7 · We found a blind spot")
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
        "Graph + night targeting: 0.135 (low-confidence — below the 0.40 hit floor; "
        "the matching rule still counts it as a hit)" in text
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
    weak_supervision = importlib.import_module("views.weak_supervision")
    filters = importlib.import_module("filters")
    try:
        # (a) the strings the pages DO name -- compared attribute to attribute
        assert tour._NOT_CURATED_CAPTION == scenarios._NOT_CURATED_CAPTION
        # The overlay legend is now written on THREE pages (the Weak Supervision
        # page's held-out result row copies it too) -- item M5, Phase 9b
        # consolidated review: two of the three were compared, so a reword of the
        # weak page's copy could have drifted away unnoticed.
        assert tour._OVERLAY_LEGEND == weak_supervision._OVERLAY_LEGEND
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

