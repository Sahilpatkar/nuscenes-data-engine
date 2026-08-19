"""The demo app must stay deployable on Streamlit Cloud: no src/, no heavy deps."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

FORBIDDEN = ("nuscenes_data_engine", "requests", "torch", "lancedb", "neo4j", "duckdb")
DEMO_DIR = Path(__file__).resolve().parents[1] / "app" / "demo"


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
    lines = [
        line.split("==")[0].split(">=")[0].strip()
        for line in (DEMO_DIR / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert set(lines) <= {"streamlit", "pandas", "pyarrow", "pillow"}


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
    pd.DataFrame({"sample_data_token": ["s1", "s2", "s3"]}).to_parquet(
        processed / "samples.parquet"
    )
    pd.DataFrame({"sample_token": ["s1"] * 4}).to_parquet(processed / "annotations.parquet")
    pd.DataFrame(
        {
            # annotation_token: required since Task 1 (Phase 3) -- gt_boxes.parquet
            # (staged below) is joined onto this table by annotation_token to add
            # distance_to_ego_m; f1..f3 are otherwise unused by the flagship SQL.
            "annotation_token": ["f1", "f2", "f3"],
            "sample_token": ["s1", "s1", "s2"],
            "category_group": ["pedestrian", "pedestrian", "car"],
            "distance_to_ego_m": [5.0, 20.0, 3.0],
        }
    ).to_parquet(processed / "annotations_3d.parquet")
    pd.DataFrame(
        {
            "sample_token": ["s1", "s2"],
            "has_canbus": [True, True],
            "can_vel_mps": [10.0, 5.0],
            "can_speed_kmh": [36.0, 18.0],
            "is_hard_braking": [True, False],
        }
    ).to_parquet(processed / "canbus.parquet")
    pd.DataFrame({"sample_token": ["s1", "s2"], "speed_mps": [10.1, 4.9]}).to_parquet(
        processed / "ego_pose.parquet"
    )
    (al / "results.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "n_train_images": 100,
                    "overall": {"mAP50-95": 0.20},
                    "night": {"mAP50-95": 0.10},
                },
                "random": {
                    "n_train_images": 115,
                    "overall": {"mAP50-95": 0.21},
                    "night": {"mAP50-95": 0.11},
                },
                "weak_random": {
                    "n_train_images": 110,
                    "overall": {"mAP50-95": 0.2018},
                    "night": {"mAP50-95": 0.105},
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
            }
        )
    )
    # Phase 3: the hero is a hand-picked exemplar crop from the curated-frames group,
    # not an mlruns mosaic -- stage a minimal one-token curation group (val token
    # "v0") so run_build's now-mandatory hero-token resolution has a real crop to
    # copy. A genuine (if tiny) JPEG: st.image() in the AppTest run below actually
    # decodes it.
    staging = tmp_path / "curation_staging"
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        # Two buckets, not one: curation_buckets round-trips through parquet as a
        # numpy array (pyarrow's list dtype) -- a single-element array is falsy-
        # safe by accident (`bool()` of a length-1 array just returns that
        # element's truthiness), so this needs >= 2 entries to actually exercise
        # `array or []`-style bugs in the page (`ValueError: truth value of an
        # array with more than one element is ambiguous`).
        "curation_buckets": [["night_failure", "al_selected"]],
        # is_night/is_rain: Task 4's Failure Explorer sidebar builds its lighting/
        # rain filter options straight off these columns' actual values -- absent
        # here, the page would KeyError before an AppTest ever gets to render.
        "is_night": [True], "is_rain": [False],
        "n_preds_baseline": pd.array([1], dtype="Int64"),
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame({
        "sample_data_token": ["v0"], "model": ["baseline"], "category_group": ["car"],
        "x_min": [1.0], "y_min": [1.0], "x_max": [2.0], "y_max": [2.0],
        "conf": [0.9], "status": ["tp"], "matched_annotation_token": ["a1"],
    }).to_parquet(staging / "predictions.parquet")
    pd.DataFrame({
        "annotation_token": ["a1"], "sample_data_token": ["v0"],
        "category_group": ["car"], "x_min": [1.0], "y_min": [1.0],
        "x_max": [2.0], "y_max": [2.0], "matched_baseline": [True],
        # below_visibility_min: real gt_boxes always carries this column: the
        # Failure Explorer drops such rows everywhere (rendering + the box table),
        # so it must be present for the page to even read gt_boxes.parquet.
        "below_visibility_min": [False],
    }).to_parquet(staging / "gt_boxes.parquet")
    hero_bytes = io.BytesIO()
    Image.new("RGB", (2, 2), color=(120, 120, 120)).save(hero_bytes, format="JPEG")
    (staging / "crops" / "v0.jpg").write_bytes(hero_bytes.getvalue())

    out = tmp_path / "demo_data"
    config = {
        "paths": {
            "processed_dir": str(processed),
            "active_learning_dir": str(al),
            "mlruns_dir": str(tmp_path / "mlruns"),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(out),
        },
        "models": {"baseline": {"run": "runX", "imgsz": 640}},
        "hero": {"token": "v0"},
        "budgets": {"max_package_mb": 100},
        "flagship": {"expected_sql_count": 1, "cypher_count": 30, "cypher_source": "docs/GRAPH.md"},
        "curation": {"staging_dir": str(staging)},
    }
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))
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

    return out


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

    monkeypatch.setenv("DEMO_DATA_DIR", str(built_demo_data))
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    at = AppTest.from_file(str(DEMO_DIR / "main.py")).run()

    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values["Camera keyframes"] == "3"


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
