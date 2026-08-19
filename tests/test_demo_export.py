"""Tests for the demo artifact builder (pure exporters; no torch, no network)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from nuscenes_data_engine.demo.exporters import export_overview


@pytest.fixture()
def tiny_inputs(tmp_path: Path) -> dict[str, Path]:
    """Miniature processed/ and active_learning/ trees with known numbers."""
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
            # annotation_token: Task 1 (Phase 3) needs a joinable token on every
            # annotations_3d row for the gt_boxes distance/size enrichment (proven
            # 100% joinable on real data) -- f1..f3 are the flagship rows' tokens,
            # otherwise unused by the flagship SQL itself.
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
            "can_speed_kmh": [36.0, 18.0],  # /3.6 == can_vel_mps here, by construction
            "is_hard_braking": [True, False],
        }
    ).to_parquet(processed / "canbus.parquet")
    pd.DataFrame(
        {"sample_token": ["s1", "s2"], "speed_mps": [10.1, 4.9]}
    ).to_parquet(processed / "ego_pose.parquet")
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
                    "night": {"mAP50-95": 0.099},
                },
                "weak_random": {
                    "n_train_images": 110,
                    "overall": {"mAP50-95": 0.2018},
                    "night": {"mAP50-95": 0.095},
                },
                "graph_rate_night": {
                    "n_train_images": 130,
                    "overall": {"mAP50-95": 0.22},
                    "night": {"mAP50-95": 0.1101},
                },
                "weak_graph_rate_night": {
                    "n_train_images": 122,
                    "overall": {"mAP50-95": 0.205},
                    "night": {"mAP50-95": 0.09},
                },
            }
        )
    )
    return {"processed": processed, "al": al}


def test_export_overview_derives_every_number(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    out = tmp_path / "demo_data"
    metrics = export_overview(
        processed_dir=tiny_inputs["processed"],
        al_dir=tiny_inputs["al"],
        out_dir=out,
        flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
    )
    on_disk = json.loads((out / "overview_metrics.json").read_text())
    assert on_disk == metrics
    assert metrics["scale"]["images"] == 3
    assert metrics["scale"]["boxes_2d"] == 4
    assert metrics["scale"]["objects_3d"] == 3
    assert metrics["scale"]["canbus_rows"] == 2
    # flagship: s1 is hard-braking AND has a pedestrian within 10 m -> exactly 1
    assert metrics["flagship"]["sql"] == 1
    assert metrics["flagship"]["cypher"] == 30
    assert metrics["flagship"]["cypher_source"] == "docs/GRAPH.md"
    # CAN validation: corr of (can_speed_kmh/3.6 == 10.0, 5.0) vs (10.1, 4.9) is
    # ~1.0, derived not hardcoded
    assert 0.99 <= metrics["can_speed_r"] <= 1.0
    # headline results derived from results.json
    assert metrics["results"]["best_night_arm"] == "graph_rate_night"
    assert metrics["results"]["best_night_delta"] == pytest.approx(0.0101, abs=1e-6)
    # weak-supervision retention: per-pair, with an attributed headline (the
    # documented Overview headline is the weak_random/random pair, not whichever
    # weak_ arm happens to be present)
    weak_retention = metrics["results"]["weak_retention"]
    assert weak_retention["headline_arm"] == "random"
    assert weak_retention["headline"] == pytest.approx(0.18, abs=1e-6)
    assert weak_retention["by_base_arm"]["graph_rate_night"] == pytest.approx(0.25, abs=1e-6)


def test_export_overview_excludes_negative_gt_gain_pairs(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    """A base arm that regressed vs baseline gets no weak-retention ratio.

    `if gt_gain:` (float-truthiness) only excluded an exactly-zero gain; a negative
    gt_gain would still produce a technically-computed but meaningless ratio (there is
    no such thing as "retention" of a regression). `if gt_gain > 0:` excludes it.
    """
    results_path = tiny_inputs["al"] / "results.json"
    results = json.loads(results_path.read_text())
    # graph_rate_night now UNDERPERFORMS baseline overall mAP (0.20) -> negative gt_gain.
    results["graph_rate_night"]["overall"]["mAP50-95"] = 0.15
    results_path.write_text(json.dumps(results))

    metrics = export_overview(
        processed_dir=tiny_inputs["processed"],
        al_dir=tiny_inputs["al"],
        out_dir=tmp_path / "demo_data",
        flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
    )
    weak_retention = metrics["results"]["weak_retention"]
    assert "graph_rate_night" not in weak_retention["by_base_arm"]
    # the headline (random) pair is a different base arm and is unaffected
    assert weak_retention["headline"] == pytest.approx(0.18, abs=1e-6)


def test_export_overview_missing_input_raises(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    (tiny_inputs["processed"] / "canbus.parquet").unlink()
    with pytest.raises(ValueError, match=r"canbus\.parquet"):
        export_overview(
            processed_dir=tiny_inputs["processed"],
            al_dir=tiny_inputs["al"],
            out_dir=tmp_path / "demo_data",
            flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
        )


def test_export_al_results_reshapes_and_sorts(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_results

    out = tmp_path / "demo_data"
    df = export_al_results(al_dir=tiny_inputs["al"], out_dir=out)
    on_disk = pd.read_parquet(out / "active_learning_results.parquet")
    pd.testing.assert_frame_equal(df, on_disk)
    assert list(df.columns) == [
        "arm", "n_train_images", "overall_map5095", "night_map5095",
        "delta_overall", "delta_night",
    ]
    assert list(df["arm"]) == sorted(df["arm"])          # deterministic order
    row = df.set_index("arm").loc["graph_rate_night"]
    assert row["delta_night"] == pytest.approx(0.0101, abs=1e-6)
    baseline = df.set_index("arm").loc["baseline"]
    assert baseline["delta_overall"] == 0.0


def test_export_al_results_missing_results_json_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_results

    al = tmp_path / "active_learning"
    al.mkdir()
    with pytest.raises(ValueError, match=r"results\.json"):
        export_al_results(al_dir=al, out_dir=tmp_path / "demo_data")


def test_export_weaksup_empty_dir_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    al = tmp_path / "active_learning"
    al.mkdir()
    with pytest.raises(ValueError, match=r"no \*_pseudo_summary\.json"):
        export_weaksup(al_dir=al, out_dir=tmp_path / "demo_data")


def test_export_weaksup_reads_summaries(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    al = tiny_inputs["al"]
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 1500, "n_accepted": 958, "retention": 0.639,
        "n_boxes": 1942, "mean_boxes_per_accepted_frame": 2.0,
        "mean_gt_boxes_per_accepted_frame": 3.9,
    }))
    out = tmp_path / "demo_data"
    df = export_weaksup(al_dir=al, out_dir=out)
    assert (out / "weak_supervision_results.parquet").is_file()
    row = df.set_index("arm").loc["random"]
    assert row["n_accepted"] == 958
    assert row["verifier_retention"] == pytest.approx(0.639)
    # arms without a summary file are simply absent, not an error
    assert "graph_rate_night" not in set(df["arm"])


def test_export_thumbs_writes_one_jpeg_per_token(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.exporters import export_thumbs

    db = lancedb.connect(str(tmp_path / "lancedb"))
    db.create_table(
        "frames",
        data=[
            {"sample_data_token": "t1", "thumbnail": b"\xff\xd8\xff\xe0one"},
            {"sample_data_token": "t2", "thumbnail": b"\xff\xd8\xff\xe0two"},
            {"sample_data_token": "t3", "thumbnail": b"\xff\xd8\xff\xe0three"},
        ],
    )
    out = tmp_path / "demo_data"
    written = export_thumbs(
        lancedb_path=tmp_path / "lancedb", table="frames",
        tokens=["t1", "t3"], out_dir=out,
    )
    assert sorted(p.name for p in written) == ["t1.jpg", "t3.jpg"]
    assert (out / "sample_frames" / "thumbs" / "t1.jpg").read_bytes().endswith(b"one")


def test_export_thumbs_unknown_token_raises(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.exporters import export_thumbs

    db = lancedb.connect(str(tmp_path / "lancedb"))
    db.create_table("frames", data=[{"sample_data_token": "t1", "thumbnail": b"x"}])
    with pytest.raises(ValueError, match="ghost"):
        export_thumbs(
            lancedb_path=tmp_path / "lancedb", table="frames",
            tokens=["t1", "ghost"], out_dir=tmp_path / "demo_data",
        )


def test_export_thumbs_empty_token_list_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_thumbs

    with pytest.raises(ValueError, match="empty token list"):
        export_thumbs(
            lancedb_path=tmp_path / "lancedb", table="frames",
            tokens=[], out_dir=tmp_path / "demo_data",
        )


def test_export_thumbs_malformed_token_raises_value_error(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_thumbs

    bad_token = "abc'); DROP TABLE frames;--"
    with pytest.raises(ValueError, match="malformed tokens") as exc_info:
        export_thumbs(
            lancedb_path=tmp_path / "lancedb", table="frames",
            tokens=["t1", bad_token], out_dir=tmp_path / "demo_data",
        )
    assert bad_token in str(exc_info.value)


def _write_demo_config(
    tmp_path: Path, tiny_inputs: dict[str, Path], *, models: dict[str, Any]
) -> Path:
    """Shared demo.yaml builder for ``build_config`` and the legacy-model-shape test
    below — factored out so the two differ only in ``models``, not in every other
    path/budget/flagship field.
    """
    al = tiny_inputs["al"]
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 10, "n_accepted": 6, "retention": 0.6,
        "n_boxes": 12, "mean_boxes_per_accepted_frame": 2.0,
        "mean_gt_boxes_per_accepted_frame": 3.0,
    }))
    config = {
        "paths": {
            "processed_dir": str(tiny_inputs["processed"]),
            "active_learning_dir": str(al),
            "mlruns_dir": str(tmp_path / "mlruns"),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(tmp_path / "demo_data"),
        },
        "models": models,
        # Phase 3: hero is a hand-picked exemplar crop token from the curated-frames
        # group, not an mlruns mosaic (see the dated amendment in
        # docs/superpowers/specs/2026-08-12-demo-phase1-design.md) -- absent by
        # default here, same as `curation:` below; tests that need a successful
        # build opt in via `_stage_and_pick_hero`.
        "hero": {"token": None},
        "budgets": {"max_package_mb": 100},
        "flagship": {"expected_sql_count": 1, "cypher_count": 30,
                     "cypher_source": "docs/GRAPH.md"},
        # Task 4: absent by default -- no `demo curate`/`demo infer` run has staged
        # anything under this dir in most fixtures here, so run_build must skip the
        # curation group loudly rather than erroring. The curation-specific tests
        # below stage real files under this same staging_dir.
        "curation": {"staging_dir": str(tmp_path / "curation_staging")},
    }
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _stage_minimal_curation(staging_dir: Path) -> None:
    """Write a minimal, fully-staged curation dir: one val token "v0" with
    n_preds_baseline coverage, gt_boxes a1 (20x20 -> small) / a2 (200x200 -> large)
    for v0, one predictions row for v0, and a single crop file v0.jpg.

    Factored out of the per-test staging blocks below (test_build_includes_
    curation_group_when_staged and its siblings) for the Task 1 gt-box-enrichment
    and hero-token tests, which all need the same minimal shape -- the older
    curation tests keep their own inline staging rather than being churned to
    adopt this.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "crops").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        "n_preds_baseline": pd.array([1], dtype="Int64"),
    }).to_parquet(staging_dir / "frame_manifest.parquet")
    pd.DataFrame({
        "sample_data_token": ["v0"], "model": ["baseline"], "category_group": ["car"],
        "x_min": [1.0], "y_min": [1.0], "x_max": [2.0], "y_max": [2.0],
        "conf": [0.9], "status": ["tp"], "matched_annotation_token": ["a1"],
    }).to_parquet(staging_dir / "predictions.parquet")
    pd.DataFrame({
        "annotation_token": ["a1", "a2"], "sample_data_token": ["v0", "v0"],
        "category_group": ["pedestrian", "car"],
        "x_min": [0.0, 0.0], "y_min": [0.0, 0.0],
        "x_max": [20.0, 200.0], "y_max": [20.0, 200.0],
        "matched_baseline": [True, True],
    }).to_parquet(staging_dir / "gt_boxes.parquet")
    (staging_dir / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")


def _stage_and_pick_hero(config_path: Path, *, token: str = "v0") -> dict[str, Any]:
    """Stage a minimal curated-frames group and point ``hero.token`` at it.

    Most ``build_config``-based tests below don't care about curation or the hero
    at all -- they need SOME successful build to exercise an unrelated code path
    (manifest determinism, budget/flagship validation, ...). Since Phase 3's hero
    flow always requires a real curated crop (see
    test_build_null_hero_token_fails_loudly /
    test_build_hero_token_without_curation_fails_loudly for the failure paths this
    guards), every such test opts in explicitly via this helper rather than
    ``build_config`` silently defaulting to "staged" -- which would make those two
    curation-absent-specific tests unable to express their own precondition.
    Returns the parsed config dict for further test-specific mutation.
    """
    config = yaml.safe_load(config_path.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": token}
    config_path.write_text(yaml.safe_dump(config))
    return config


@pytest.fixture()
def build_config(tmp_path: Path, tiny_inputs: dict[str, Path]) -> Path:
    # Production shape: a {run, imgsz} mapping (configs/demo.yaml since the Task 3
    # review round) -- the shape every other fixture in this module now exercises.
    # The flat run-id-string legacy shape gets its own dedicated coverage in
    # test_build_accepts_legacy_flat_model_shape below.
    return _write_demo_config(
        tmp_path, tiny_inputs, models={"baseline": {"run": "runX", "imgsz": 640}}
    )


def test_build_accepts_legacy_flat_model_shape(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    """Carried review item: ``models:`` entries may still be a flat run-id string
    (the shape every fixture used before the Task 3 review round introduced
    ``{run, imgsz}``) -- ``_include_curation``'s val-coverage check iterates
    ``config["models"]`` keys, which works identically for either value shape, and
    the Phase 3 hero flow no longer reads ``config["models"]`` at all -- so this
    just needs to confirm a build with the legacy shape still succeeds end to end.
    """
    from nuscenes_data_engine.demo.build import run_build

    config_path = _write_demo_config(tmp_path, tiny_inputs, models={"baseline": "runX"})
    config = _stage_and_pick_hero(config_path)
    manifest = run_build(config_path)
    out = Path(config["paths"]["out_dir"])
    assert (out / "sample_frames" / "hero.jpg").is_file()
    assert manifest["validation"]["flagship_sql_count"] == 1


def test_build_writes_validated_manifest(build_config: Path, tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    manifest = run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["git_sha"]
    assert manifest["outputs"]["overview_metrics.json"]["sha256"]
    assert manifest["outputs"]["active_learning_results.parquet"]["rows"] == 5
    assert manifest["validation"]["flagship_sql_count"] == 1
    assert manifest["validation"]["package_mb"] < 1


def test_build_is_deterministic(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    out = Path(config["paths"]["out_dir"])
    run_build(build_config)
    first = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    run_build(build_config)
    second = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert first == second                    # byte-stable outputs (manifest has built_at)


def test_build_fails_on_wrong_flagship_count(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    config["flagship"]["expected_sql_count"] = 30      # tiny fixture yields 1, not 30
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="flagship"):
        run_build(build_config)


def test_build_fails_over_size_budget(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    config["budgets"]["max_package_mb"] = 0
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="budget"):
        run_build(build_config)


def test_git_sha_anchored_to_package_repo_not_cwd(tmp_path: Path) -> None:
    """_git_sha() must anchor to this package's repo, not an unrelated repo at cwd."""
    from nuscenes_data_engine.demo.build import _git_sha

    repo_root = Path(__file__).resolve().parents[1]
    expected = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert _git_sha() == expected
    finally:
        os.chdir(cwd)


def test_build_wipes_stale_residue(build_config: Path) -> None:
    """A leftover file from a removed exporter or failed prior build must not survive."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # manifest.json makes this look like an existing demo package (not a typo'd
    # out_dir) so the wipe-guard (test_build_refuses_to_wipe_a_non_package_dir) lets
    # the rebuild proceed.
    (out_dir / "manifest.json").write_text("{}")
    (out_dir / "stale.json").write_text("{}")

    manifest = run_build(build_config)

    assert not (out_dir / "stale.json").exists()
    assert "stale.json" not in manifest["outputs"]


def test_build_refuses_to_wipe_a_non_package_dir(build_config: Path) -> None:
    """A typo'd out_dir pointing at a real, unrelated directory must not be wiped."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True)
    (out_dir / "not_a_demo_package.txt").write_text("please don't delete me")

    with pytest.raises(ValueError, match="refusing to wipe"):
        run_build(build_config)

    assert (out_dir / "not_a_demo_package.txt").is_file()


def test_build_recovers_from_a_failed_build_without_manual_cleanup(build_config: Path) -> None:
    """A failed build must not self-lock the retry.

    manifest.json is written last, so a failed build (here: the documented flagship-
    mismatch failure) leaves overview_metrics.json on disk but no manifest.json. The
    wipe guard must still recognise that tree as a demo package (via
    _PACKAGE_MARKERS) and let the corrected retry through, not blame a config typo
    that doesn't exist and demand a manual rm -rf.
    """
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    config["flagship"]["expected_sql_count"] = 30  # tiny fixture yields 1, not 30 -> fails
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="flagship"):
        run_build(build_config)

    out_dir = Path(config["paths"]["out_dir"])
    assert (out_dir / "overview_metrics.json").is_file()
    assert not (out_dir / "manifest.json").is_file()

    config["flagship"]["expected_sql_count"] = 1  # fix the config back to reality
    build_config.write_text(yaml.safe_dump(config))
    manifest = run_build(build_config)  # must not raise "refusing to wipe"

    assert (out_dir / "manifest.json").is_file()
    assert manifest["validation"]["flagship_sql_count"] == 1


def test_build_rewipes_an_existing_demo_package(build_config: Path) -> None:
    """A dir that already looks like a demo package (has manifest.json) is fine to wipe."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    run_build(build_config)  # first build creates out_dir/manifest.json
    manifest = run_build(build_config)  # second build must not raise the new guard

    out_dir = Path(config["paths"]["out_dir"])
    assert (out_dir / "manifest.json").is_file()
    assert manifest["validation"]["n_arms"] == 5


def test_build_proceeds_when_out_dir_absent(build_config: Path) -> None:
    """An out_dir that simply doesn't exist yet (the common case) is fine, not an error."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    out_dir = Path(config["paths"]["out_dir"])
    assert not out_dir.exists()

    manifest = run_build(build_config)

    assert (out_dir / "manifest.json").is_file()
    assert manifest["validation"]["n_arms"] == 5


def test_build_fails_when_weak_retention_headline_missing(build_config: Path) -> None:
    """A results.json missing the documented headline (random) pair must fail loudly,
    not silently publish `"headline": null` to the committed package."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    al_dir = Path(config["paths"]["active_learning_dir"])
    results_path = al_dir / "results.json"
    results = json.loads(results_path.read_text())
    del results["random"]
    del results["weak_random"]
    results_path.write_text(json.dumps(results))

    with pytest.raises(ValueError, match="weak-retention headline"):
        run_build(build_config)


def test_build_hashes_all_real_inputs(build_config: Path) -> None:
    """manifest['inputs'] must cover everything the build actually reads, not just results.json."""
    from nuscenes_data_engine.demo.build import run_build

    _stage_and_pick_hero(build_config)
    manifest = run_build(build_config)
    assert len(manifest["inputs"]) >= 7
    assert any(key.endswith("canbus.parquet") for key in manifest["inputs"])


def test_build_skips_absent_curation_with_manifest_note(build_config: Path) -> None:
    """No `demo curate`/`demo infer` staging (build_config's default) must not fail
    the build -- Task 2's TRINITY-unreachable fallback means Tasks 1-4 still merge
    on their own, and since data/demo_curation is gitignored, a fresh clone with a
    null hero token must still produce a working Phase-1-only package (spec-review
    fix: the null-token check must be gated on curation presence, not fire
    unconditionally). It must, however, say so loudly in the manifest, not silently
    proceed as if the curation group never existed.

    Whether hero.token is even required now depends on curation:
    test_build_hero_token_without_curation_fails_loudly covers "token configured
    but curation absent" (still an error -- the config references curated data that
    isn't there), and test_build_null_hero_token_fails_loudly covers "curation
    present but no token" (still an error -- pick one)."""
    from nuscenes_data_engine.demo.build import run_build

    manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "absent"
    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    assert not (out / "frame_manifest.parquet").exists()
    assert "sample_frames/hero.jpg" not in manifest["outputs"]
    assert not (out / "sample_frames" / "hero.jpg").exists()


def test_build_includes_curation_group_when_staged(build_config: Path, tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
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
    }).to_parquet(staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    config["hero"] = {"token": "v0"}
    build_config.write_text(yaml.safe_dump(config))
    manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "included"
    out = Path(config["paths"]["out_dir"])
    assert (out / "frame_manifest.parquet").is_file()
    assert (out / "sample_frames" / "crops" / "v0.jpg").is_file()
    assert "frame_manifest.parquet" in manifest["outputs"]
    # staging parquets are hashed as inputs (they are what the build actually read)
    assert any(
        Path(key).name == "frame_manifest.parquet" for key in manifest["inputs"]
    )


def test_build_raises_on_stale_crop_not_in_manifest(build_config: Path) -> None:
    """Licensing hazard: crops are nuScenes-derived imagery. A re-curation with a
    different token set must not silently ship a PRIOR run's crop alongside the
    current manifest -- stale imagery in staging means the operator's pipeline state
    is inconsistent (e.g. `demo curate` re-ran but `demo infer` hasn't cleaned up the
    old crops dir), not something to quietly carry forward into the package."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
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
    }).to_parquet(staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    # stale imagery: left over from a PRIOR curation run, not a token in this manifest.
    (staging / "crops" / "stale_token.jpg").write_bytes(b"\xff\xd8\xff\xe0stale")

    with pytest.raises(ValueError, match="stale_token"):
        run_build(build_config)


def test_build_fails_when_val_token_lacks_model_predictions(build_config: Path) -> None:
    """The never-ran case, column entirely absent: frame_manifest.parquet carries no
    n_preds_baseline column at all (demo infer never ran with this model) -- must
    fail loudly, not be conflated with a legitimate zero-prediction finding."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        # no n_preds_baseline column at all
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame(columns=["sample_data_token", "model", "category_group", "x_min",
                          "y_min", "x_max", "y_max", "conf", "status",
                          "matched_annotation_token"]).to_parquet(
        staging / "predictions.parquet")
    pd.DataFrame(columns=["annotation_token", "sample_data_token", "category_group",
                          "x_min", "y_min", "x_max", "y_max"]).to_parquet(
        staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    with pytest.raises(ValueError, match="predictions"):
        run_build(build_config)


def test_build_fails_when_val_token_has_na_n_preds(build_config: Path) -> None:
    """The never-ran case, column present but NA: distinct code path from the column
    being entirely absent (a model dropped from a later infer run, or leftover
    NA from a rerun with a different roster) -- must still fail loudly."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        "n_preds_baseline": pd.array([pd.NA], dtype="Int64"),
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame(columns=["sample_data_token", "model", "category_group", "x_min",
                          "y_min", "x_max", "y_max", "conf", "status",
                          "matched_annotation_token"]).to_parquet(
        staging / "predictions.parquet")
    pd.DataFrame(columns=["annotation_token", "sample_data_token", "category_group",
                          "x_min", "y_min", "x_max", "y_max"]).to_parquet(
        staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    with pytest.raises(ValueError, match="predictions"):
        run_build(build_config)


def test_build_passes_val_coverage_with_a_zero_prediction_count(build_config: Path) -> None:
    """Task-5 fix: 'ran and found nothing' (n_preds_<model> == 0) is legitimate
    coverage, not a validation failure -- real Task-5 case: baseline and
    graph_rate_night both found zero boxes on a genuine total-miss frame, and the
    old predictions.parquet-row-presence check couldn't tell that apart from
    'never ran', wrongly failing the build on a token the demo actually wants."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        "n_preds_baseline": pd.array([0], dtype="Int64"),
    }).to_parquet(staging / "frame_manifest.parquet")
    # baseline ran and found nothing on v0 -- predictions.parquet has zero rows for
    # it, same as a model that never ran would; n_preds is what disambiguates.
    pd.DataFrame(columns=["sample_data_token", "model", "category_group", "x_min",
                          "y_min", "x_max", "y_max", "conf", "status",
                          "matched_annotation_token"]).to_parquet(
        staging / "predictions.parquet")
    pd.DataFrame(columns=["annotation_token", "sample_data_token", "category_group",
                          "x_min", "y_min", "x_max", "y_max"]).to_parquet(
        staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    config["hero"] = {"token": "v0"}
    build_config.write_text(yaml.safe_dump(config))

    manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "included"


def test_build_raises_named_error_on_partial_curation_staging(build_config: Path) -> None:
    """Review round 2, item 1: a staging dir with ONLY frame_manifest.parquet (i.e.
    `demo curate` ran but `demo infer` has not yet) is an operator mid-flow, not a
    no-curation machine -- treating it as silently "absent" would hide their
    mistake. It must fail loudly and name `demo infer` as the next step, and it must
    not crash with a raw FileNotFoundError out of the copy2 loop first."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    staging.mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    # predictions.parquet / gt_boxes.parquet deliberately absent -- `demo infer` has
    # not run yet.
    with pytest.raises(ValueError, match="demo infer"):
        run_build(build_config)


def test_build_tolerates_a_corrupt_lancedb_store(
    build_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Review round 2, item 2: `lancedb_path.is_dir()` can be True while the store
    itself has no usable table inside (e.g. an existing-but-empty directory) --
    export_thumbs then raises from lancedb's own guts. Thumbnails are supplementary
    (crops already cover every token), so this must degrade to a warning and let the
    build finish, not crash it -- the crop-or-thumb validation still runs and would
    still fail loudly if coverage were actually lost."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
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
    }).to_parquet(staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    config["hero"] = {"token": "v0"}
    build_config.write_text(yaml.safe_dump(config))

    # Exists (is_dir() True) but has no valid lance table inside -- open_table raises.
    Path(config["paths"]["lancedb_path"]).mkdir(parents=True)

    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "included"
    assert "thumbnail export skipped" in caplog.text


def test_build_enriches_gt_boxes_with_distance_and_size(build_config: Path) -> None:
    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": "v0"}
    build_config.write_text(yaml.safe_dump(config))
    processed = Path(config["paths"]["processed_dir"])
    # a1/a2 join onto the annotation_token values _stage_minimal_curation already
    # staged for v0's gt_boxes. This APPENDS to (rather than overwrites) the fixture's
    # existing annotations_3d.parquet: the flagship rows (f1..f3, see tiny_inputs'
    # extended annotation_token column) must survive untouched so the flagship SQL
    # count doesn't move -- a1/a2 carry no sample_token, so the flagship join (which
    # requires a matching canbus.sample_token) simply never sees them.
    existing = pd.read_parquet(processed / "annotations_3d.parquet")
    extra = pd.DataFrame({
        "annotation_token": ["a1", "a2"],
        "distance_to_ego_m": [4.5, 31.0],
    })
    pd.concat([existing, extra], ignore_index=True).to_parquet(
        processed / "annotations_3d.parquet"
    )
    manifest = run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    gt = pd.read_parquet(out / "gt_boxes.parquet").set_index("annotation_token")
    assert gt.loc["a1", "distance_to_ego_m"] == 4.5
    assert gt.loc["a1", "size_bucket"] == "small"     # 20x20 = 400 < 1024
    assert gt.loc["a2", "size_bucket"] == "large"     # 200x200 = 40000 >= 9216
    assert manifest["validation"]["curation"] == "included"


def test_size_bucket_boundaries_exact() -> None:
    from nuscenes_data_engine.demo.build import _size_bucket

    assert _size_bucket(32.0 * 32.0 - 1) == "small"
    assert _size_bucket(32.0 * 32.0) == "medium"
    assert _size_bucket(96.0 * 96.0 - 1) == "medium"
    assert _size_bucket(96.0 * 96.0) == "large"


def test_build_hero_from_token_copies_crop_and_records_token(build_config: Path) -> None:
    import json as _json

    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": "v0"}          # v0 is a staged crop in the fixture
    build_config.write_text(yaml.safe_dump(config))
    run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    assert (out / "sample_frames" / "hero.jpg").read_bytes() == (
        out / "sample_frames" / "crops" / "v0.jpg"
    ).read_bytes()
    metrics = _json.loads((out / "overview_metrics.json").read_text())
    assert metrics["hero_token"] == "v0"


def test_build_null_hero_token_fails_loudly(build_config: Path) -> None:
    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": None}
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="hero token"):
        run_build(build_config)


def test_build_hero_token_without_curation_fails_loudly(build_config: Path) -> None:
    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())   # fixture has no staging by default
    config["hero"] = {"token": "v0"}
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="hero"):
        run_build(build_config)
