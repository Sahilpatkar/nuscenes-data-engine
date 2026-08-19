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
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("demo build: %d outputs, %.2f MB -> %s", len(outputs), package_mb, out_dir)
    return manifest
