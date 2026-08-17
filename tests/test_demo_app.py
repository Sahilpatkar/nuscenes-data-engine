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
    mlruns = tmp_path / "mlruns"
    hero_dir = mlruns / "artifacts" / "runX" / "artifacts" / "ultralytics_run"
    hero_dir.mkdir(parents=True)
    # A genuine (if tiny) JPEG: st.image() in the AppTest run below actually decodes
    # it, unlike test_export_hero_copies_the_configured_mosaic's byte-copy check.
    hero_bytes = io.BytesIO()
    Image.new("RGB", (2, 2), color=(120, 120, 120)).save(hero_bytes, format="JPEG")
    (hero_dir / "val_batch0_pred.jpg").write_bytes(hero_bytes.getvalue())

    out = tmp_path / "demo_data"
    config = {
        "paths": {
            "processed_dir": str(processed),
            "active_learning_dir": str(al),
            "mlruns_dir": str(mlruns),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(out),
        },
        "models": {"baseline": "runX"},
        "hero": {"run": "baseline", "mosaic": "val_batch0_pred.jpg"},
        "budgets": {"max_package_mb": 100},
        "flagship": {"expected_sql_count": 1, "cypher_count": 30, "cypher_source": "docs/GRAPH.md"},
    }
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_build(config_path)
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
