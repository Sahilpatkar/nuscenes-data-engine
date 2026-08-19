"""``demo build`` — run all exporters, write and validate the package manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.config import load_yaml
from nuscenes_data_engine.demo import exporters

logger = logging.getLogger("nuscenes_data_engine")

# src/nuscenes_data_engine/demo/build.py -> repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# A package is recognisable by any known output, not only the last-written manifest:
# a failed build leaves overview_metrics.json (written first) but no manifest, and
# recovery must not require a manual rm -rf.
_PACKAGE_MARKERS = ("manifest.json", "overview_metrics.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    """The HEAD sha of this package's own repo, not whatever repo happens to be cwd.

    A bare ``git rev-parse HEAD`` resolves against the process's current working
    directory, which silently returns an unrelated repo's sha (or "unknown") if
    ``demo build`` is ever invoked from inside a different git checkout.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # detached environments
        return "unknown"


def _include_curation(config: dict[str, Any], out_dir: Path) -> str:
    """Copy the demo-phase-2 curation group into the package, or note its absence.

    Absent staging (no ``demo curate`` + ``demo infer`` run yet — the documented
    TRINITY-unreachable fallback, see Task 2 of the phase-2 plan) is not a build
    failure: it logs a warning and the manifest records "absent" so the published
    package is honest about what it ships, instead of silently proceeding as if
    Phase 2 never happened. A PARTIAL staging dir (``frame_manifest.parquet`` present
    but ``predictions.parquet``/``gt_boxes.parquet`` missing — ``demo curate`` ran,
    ``demo infer`` has not) is a different case entirely: that's an operator
    mid-flow, not a no-curation machine, so it raises a named ``ValueError`` pointing
    at ``demo infer`` rather than silently shipping "absent" and hiding the mistake.
    When fully staged, the three curation parquets and every frame's crop are copied
    in, thumbnails are bulk-exported from LanceDB when that store exists (skipped
    with a warning both when the store is simply absent and when it exists but is
    unusable, e.g. no valid table inside — none of this module's test fixtures carry
    a real store, so full coverage of the export itself is the Task 5 operational
    run), and two invariants are validated before the copy is trusted: every ``val``
    token has RECORDED coverage (``n_preds_<model>`` not-NA — zero predictions counts
    as coverage, only "never ran" doesn't) from every configured model, and every
    curated token has at least a crop or a thumbnail to render.
    """
    curation_cfg = config.get("curation")
    if not curation_cfg:
        logger.warning(
            "demo build: no `curation:` config section — shipping without the "
            "curated-frames group (see docs/DEMO.md)"
        )
        return "absent"
    staging_dir = Path(curation_cfg["staging_dir"])
    manifest_path = staging_dir / "frame_manifest.parquet"
    predictions_path = staging_dir / "predictions.parquet"
    gt_path = staging_dir / "gt_boxes.parquet"
    if not manifest_path.is_file():
        logger.warning(
            "demo build: no curation staging at %s — shipping without the curated-"
            "frames group (run `demo curate` + `demo infer` first, see docs/DEMO.md)",
            staging_dir,
        )
        return "absent"
    # A manifest with no predictions/gt_boxes means `demo curate` ran but `demo
    # infer` has not -- an operator mid-flow, not a no-curation machine. Silently
    # treating this as "absent" would hide their mistake; a bare FileNotFoundError
    # out of the copy2 loop below would be equally unhelpful (names no next step).
    if not (predictions_path.is_file() and gt_path.is_file()):
        raise ValueError(
            f"demo build: curation staging at {staging_dir} is partial — "
            "frame_manifest present but predictions.parquet/gt_boxes.parquet "
            "missing; run `demo infer` first (see docs/DEMO.md)"
        )

    manifest = pd.read_parquet(manifest_path)
    tokens = list(manifest["sample_data_token"])
    token_set = set(tokens)

    for name in ("frame_manifest.parquet", "gt_boxes.parquet", "predictions.parquet"):
        shutil.copy2(staging_dir / name, out_dir / name)

    crops_dest = out_dir / "sample_frames" / "crops"
    crops_dest.mkdir(parents=True, exist_ok=True)
    for crop in sorted((staging_dir / "crops").glob("*")):
        # Crops are nuScenes-derived imagery (licensing-controlled) -- a re-curation
        # with a different token set must not silently ship a PRIOR run's crop
        # alongside the current manifest. This is an operator-pipeline-state
        # inconsistency, not something to skip past quietly.
        if crop.stem not in token_set:
            raise ValueError(
                f"demo build: stale crop {crop.name} not in the current manifest — "
                "re-run demo infer (crops are licensing-controlled)"
            )
        shutil.copy2(crop, crops_dest / crop.name)

    lancedb_path = Path(config["paths"]["lancedb_path"])
    thumb_tokens: set[str] = set()
    if lancedb_path.is_dir():
        # is_dir() only proves the path exists -- the store inside can still be
        # empty/corrupt (no valid table), which raises from lancedb's own guts, not
        # a clean, named error. Thumbnails are supplementary (crops already cover
        # every token); degrade to a warning rather than crash the whole build --
        # the crop-or-thumb validation below still runs and still fails loudly if
        # coverage were actually lost.
        try:
            exporters.export_thumbs(
                lancedb_path=lancedb_path,
                table=config["paths"]["lancedb_table"],
                tokens=tokens,
                out_dir=out_dir,
            )
            thumb_tokens = set(tokens)
        except Exception as exc:
            # Broad on purpose (store corruption, missing table, a bad token) --
            # don't narrow the message to "store unusable", which would misattribute
            # e.g. a missing-token ValueError from export_thumbs's own validation to
            # store corruption. The exception text speaks for itself.
            logger.warning("demo build: thumbnail export skipped (%s)", exc)
    else:
        logger.warning(
            "demo build: no LanceDB store at %s — curated frames ship without "
            "thumbnails (crops only)",
            lancedb_path,
        )

    # Coverage is read off frame_manifest's n_preds_<model> columns, not off
    # predictions.parquet row presence: a model that ran and found nothing writes
    # ZERO prediction rows, which is indistinguishable from "never ran" by row count
    # alone (Task 5's real run hit this: baseline and graph_rate_night both found
    # zero boxes on a genuine total-miss frame). n_preds_<model> disambiguates —
    # 0 is a recorded finding, NA (or the column being absent entirely, e.g. a model
    # dropped between infer runs) is "never ran".
    model_names = sorted(config["models"])
    val_rows = manifest.loc[manifest["split"] == "val"].set_index("sample_data_token")
    for token in val_rows.index:
        for model_name in model_names:
            col = f"n_preds_{model_name}"
            ran = col in val_rows.columns and pd.notna(val_rows.loc[token, col])
            if not ran:
                raise ValueError(
                    f"demo build: val token {token!r} has no predictions for model "
                    f"{model_name!r} — did `demo infer` run with every configured model?"
                )

    crop_tokens = {p.stem for p in crops_dest.glob("*")}
    for token in tokens:
        if token not in crop_tokens and token not in thumb_tokens:
            raise ValueError(
                f"demo build: token {token!r} has neither a crop nor a thumbnail"
            )

    return "included"


def run_build(config_path: Path) -> dict[str, Any]:
    """Run every registered exporter, then write demo_data/manifest.json.

    Validation fails loudly (ValueError) if a declared input is missing, the
    flagship SQL count differs from the configured expectation, or the package
    exceeds the size budget — a silently wrong or bloated package must never be
    committed.
    """
    config = load_yaml(config_path)
    paths = config["paths"]
    processed = Path(paths["processed_dir"])
    al_dir = Path(paths["active_learning_dir"])
    out_dir = Path(paths["out_dir"])
    mlruns_dir = Path(paths["mlruns_dir"])

    # The package is fully regenerated on every build; anything not produced by this
    # build must not survive — a partial tree from a failed build carries
    # overview_metrics.json and is safely regenerated. But out_dir comes straight
    # from config — a typo'd `out_dir: data/processed` would otherwise happily wipe
    # real pipeline artifacts, so refuse unless the directory already looks like a
    # demo package (or doesn't exist yet).
    resolved = out_dir.resolve()
    if resolved.exists() and not any((resolved / m).is_file() for m in _PACKAGE_MARKERS):
        raise ValueError(
            f"refusing to wipe {resolved}: it exists but has no manifest.json, so it "
            "does not look like a demo package (typo'd out_dir?)"
        )
    shutil.rmtree(out_dir, ignore_errors=True)

    metrics = exporters.export_overview(
        processed_dir=processed, al_dir=al_dir, out_dir=out_dir,
        flagship_cypher={
            "count": config["flagship"]["cypher_count"],
            "source": config["flagship"]["cypher_source"],
        },
    )
    al_df = exporters.export_al_results(al_dir=al_dir, out_dir=out_dir)
    weak_df = exporters.export_weaksup(al_dir=al_dir, out_dir=out_dir)
    # `models:` entries are either a plain run-id string (test fixtures, and demo.yaml
    # before the Task 3 review round) or a {run, imgsz} mapping (demo.yaml since —
    # imgsz varies per checkpoint, see configs/demo.yaml's `models:` comment). Accept
    # both rather than forcing every fixture/config onto one shape for a single field.
    hero_entry = config["models"][config["hero"]["run"]]
    hero_run = hero_entry["run"] if isinstance(hero_entry, dict) else hero_entry
    hero_mosaic = config["hero"]["mosaic"]
    exporters.export_hero(
        mlruns_dir=mlruns_dir, run_id=hero_run, mosaic=hero_mosaic, out_dir=out_dir,
    )
    curation_status = _include_curation(config, out_dir)

    expected = config["flagship"]["expected_sql_count"]
    if metrics["flagship"]["sql"] != expected:
        raise ValueError(
            f"flagship SQL count {metrics['flagship']['sql']} != expected {expected} — "
            "the source tables changed; re-verify before publishing"
        )
    headline = metrics["results"]["weak_retention"]["headline"]
    if headline is None:
        raise ValueError("weak-retention headline (random pair) missing from results.json")
    package_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    package_mb = package_bytes / 1e6
    if package_mb > config["budgets"]["max_package_mb"]:
        raise ValueError(
            f"demo_data is {package_mb:.1f} MB — over the {config['budgets']['max_package_mb']} MB budget"
        )

    outputs: dict[str, Any] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entry: dict[str, Any] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        if path.suffix == ".parquet":
            entry["rows"] = len(pd.read_parquet(path))
        outputs[str(path.relative_to(out_dir))] = entry

    # Hash everything the build actually read, not just results.json: the
    # processed parquets, every weak-sup summary, and the hero mosaic source —
    # an unnoticed change to any of these silently changes the published numbers.
    hero_src = (
        mlruns_dir / "artifacts" / hero_run / "artifacts" / "ultralytics_run" / hero_mosaic
    )
    inputs: dict[str, str] = {
        str(al_dir / "results.json"): _sha256(al_dir / "results.json"),
        str(hero_src): _sha256(hero_src),
    }
    for name in exporters.PROCESSED_INPUTS:
        path = processed / name
        inputs[str(path)] = _sha256(path)
    for path in sorted(al_dir.glob("*_pseudo_summary.json")):
        inputs[str(path)] = _sha256(path)
    if curation_status == "included":
        staging_dir = Path(config["curation"]["staging_dir"])
        for name in ("frame_manifest.parquet", "gt_boxes.parquet", "predictions.parquet"):
            path = staging_dir / name
            inputs[str(path)] = _sha256(path)

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "inputs": inputs,
        "outputs": outputs,
        "validation": {
            "flagship_sql_count": metrics["flagship"]["sql"],
            "package_mb": round(package_mb, 2),
            "n_arms": len(al_df),
            "n_weak_arms": len(weak_df),
            "curation": curation_status,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("demo build: %d outputs, %.2f MB -> %s", len(outputs), package_mb, out_dir)
    return manifest
