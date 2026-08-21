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
from nuscenes_data_engine.demo.events import build_events
from nuscenes_data_engine.demo.exporters import write_json

logger = logging.getLogger("nuscenes_data_engine")

# src/nuscenes_data_engine/demo/build.py -> repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# A package is recognisable by any known output, not only the last-written manifest:
# a failed build leaves overview_metrics.json (written first) but no manifest, and
# recovery must not require a manual rm -rf.
_PACKAGE_MARKERS = ("manifest.json", "overview_metrics.json")

# Bumped by hand alongside docs/DEMO.md's package-layout table whenever a rebuild
# changes the package's shape (new/removed columns or files), not on every build --
# most rebuilds (a results.json update, a re-picked hero token) keep this the same.
# 0.4 (Phase 5): scenario_events.parquet (six presets + filmstrip neighbors) and
# semantic_search_results.parquet (recorded semsearch) join the package, plus their
# filmstrip/gallery thumbnails.
# 0.5 (Phase 6): graph_subgraphs/<preset>.json (six presets, from `demo subgraphs`)
# joins the package when staged, and overview_metrics.json's flagship.cypher becomes
# a COMPUTED (live Neo4j) value instead of a sourced one whenever it does.
# 0.6 (Phase 7, Task 1): active_learning_results.parquet gains round_order/family/
# night pedestrian+slice columns/n_boxes/mined-set composition; al_communities.
# parquet (always) and al_exemplars.json (validated, curation-gated) join the
# package; frame_manifest.parquet gains weak_verdict/al_selected_by when curation
# is included.
# 0.6 (Phase 7, Task 2, same version): weak_supervision_results.parquet gains
# n_rejected/tolerance/conf/n_unparsed/gt_boxes_per_candidate_frame/gt_boxes_per_
# rejected_frame; weak_loss_decomposition.parquet and weak_verifier_by_class.
# parquet join the package unconditionally; weak_labels.parquet and vlm_counts.
# parquet join it when curation is included.
_PACKAGE_VERSION = "0.6"

# Filmstrip neighbor columns (demo/events.py's t_minus2..t_plus2) -- NA at scene
# edges, so every token collection over these columns must drop the NA entries.
_FILMSTRIP_NEIGHBOR_COLUMNS = ("t_minus2", "t_minus1", "t_plus1", "t_plus2")

# The six scenario presets subgraph_export.py exports one JSON per (mirrors that
# module's private _ALL_PRESETS -- kept as its own tuple here rather than imported,
# same reasoning _SEMSEARCH_COLUMNS below is its own constant: this list is build.py's
# OWN contract about what a complete graph_subgraphs/ staging dir must contain, not a
# re-export of the exporter's internals).
_SUBGRAPH_PRESETS = (
    "hard_braking_near_pedestrians",
    "night_pedestrians",
    "fast_cyclists",
    "rain_vru",
    "fn_pedestrians_night",
    "low_conf_braking",
)


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


# COCO detection-eval size convention (https://cocodataset.org/#detection-eval):
# small < 32^2 px^2, medium < 96^2 px^2, else large. ``area`` is the 2D box area at
# native (1600x900) scale, not display-crop scale.
def _size_bucket(area: float) -> str:
    if pd.isna(area) or area < 0:
        raise ValueError(f"demo build: degenerate box area {area!r} — corrupt gt_boxes row?")
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


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

    for name in ("frame_manifest.parquet", "predictions.parquet"):
        shutil.copy2(staging_dir / name, out_dir / name)

    # Transform-on-copy (Phase 3): the staged gt_boxes.parquet itself stays
    # untouched on disk (it is still hashed as an input below) -- the PUBLISHED copy
    # gains distance_to_ego_m (left-joined from the processed annotations_3d table
    # on annotation_token, proven 100% joinable on real data) and size_bucket (COCO
    # area convention, see _size_bucket). A left join keeps any row that fails to
    # match as NA rather than silently dropping it -- dropped GT boxes would be a
    # much worse failure than an unenriched one.
    gt_boxes = pd.read_parquet(gt_path)
    staged_row_count = len(gt_boxes)
    # Column-pruned (not a full-table read then column-select): measured 2.4x faster
    # and roughly half the peak RSS on the real annotations_3d table, which carries
    # many unrelated columns this join never touches.
    annotations_3d = pd.read_parquet(
        Path(config["paths"]["processed_dir"]) / "annotations_3d.parquet",
        columns=["annotation_token", "distance_to_ego_m"],
    )
    gt_boxes = gt_boxes.merge(annotations_3d, on="annotation_token", how="left")
    if len(gt_boxes) != staged_row_count:
        raise ValueError(
            "demo build: annotation_token join fanout — annotations_3d has "
            f"duplicate tokens? ({staged_row_count} staged gt_boxes rows -> "
            f"{len(gt_boxes)} after the join)"
        )
    missing_distance = int(gt_boxes["distance_to_ego_m"].isna().sum())
    if missing_distance:
        logger.warning(
            "demo build: %d/%d gt_boxes rows have no distance_to_ego_m match in "
            "annotations_3d (annotation_token not found)",
            missing_distance, len(gt_boxes),
        )
    box_area = (gt_boxes["x_max"] - gt_boxes["x_min"]) * (gt_boxes["y_max"] - gt_boxes["y_min"])
    gt_boxes["size_bucket"] = box_area.map(_size_bucket)
    gt_boxes.to_parquet(out_dir / "gt_boxes.parquet", index=False)

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


def _export_thumbs_deduped(
    *, config: dict[str, Any], out_dir: Path, tokens: set[str], context: str
) -> None:
    """Export thumbs for ``tokens`` not already on disk (e.g. from a prior step).

    Mirrors ``_include_curation``'s degrade-to-warning treatment of a missing/
    unusable LanceDB store: thumbnails here are supplementary to the parquet data
    that already shipped, so a store problem must not fail the whole build.
    ``context`` names the caller in the warning (e.g. "event", "semsearch") so a
    skipped export is traceable to which group it affected.
    """
    thumb_dir = out_dir / "sample_frames" / "thumbs"
    existing = {p.stem for p in thumb_dir.glob("*.jpg")} if thumb_dir.is_dir() else set()
    to_export = sorted(tokens - existing)
    if not to_export:
        return
    lancedb_path = Path(config["paths"]["lancedb_path"])
    if not lancedb_path.is_dir():
        logger.warning(
            "demo build: no LanceDB store at %s — %s frames ship without thumbnails",
            lancedb_path, context,
        )
        return
    try:
        exporters.export_thumbs(
            lancedb_path=lancedb_path,
            table=config["paths"]["lancedb_table"],
            tokens=to_export,
            out_dir=out_dir,
        )
    except Exception as exc:
        # Broad on purpose, same rationale as _include_curation's own thumbnail
        # export: store corruption, a missing table, or a genuinely absent token
        # must all degrade to a warning, not crash a build that already has the
        # data these thumbnails merely illustrate.
        logger.warning("demo build: %s thumbnail export skipped (%s)", context, exc)


def _include_events(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Compute ``scenario_events.parquet`` live and export its thumbnails.

    Unlike curation, events are always attempted — they are derived entirely from
    the (always-required) processed tables, never from curation staging alone —
    but the two model-result presets (``fn_pedestrians_night``, ``low_conf_
    braking``) still need the curated-val staging parquets, so ``staging_dir`` is
    resolved the same way ``_include_curation`` checks readiness: only when all
    three staged files exist, else ``None`` (``build_events`` itself warns and
    skips those two presets in that case — see events.py::_load_staging).

    ``flagship_expected`` threads ``config["flagship"]["expected_sql_count"]``
    straight to ``build_events``, which raises if its own pre-cap
    ``hard_braking_near_pedestrians`` count disagrees — the same number the SQL
    check earlier in ``run_build`` already asserts, from an independent
    (pandas, not DuckDB) computation over the same tables.
    """
    curation_cfg = config.get("curation")
    staging_dir = None
    if curation_cfg:
        candidate = Path(curation_cfg["staging_dir"])
        if all(
            (candidate / name).is_file()
            for name in ("frame_manifest.parquet", "gt_boxes.parquet", "predictions.parquet")
        ):
            staging_dir = candidate

    events = build_events(
        processed_dir=Path(config["paths"]["processed_dir"]),
        staging_dir=staging_dir,
        presets_cfg=config["presets"],
        flagship_expected=config["flagship"]["expected_sql_count"],
    )
    events.to_parquet(out_dir / "scenario_events.parquet", index=False)

    tokens: set[str] = set(events["sample_data_token"])
    for col in _FILMSTRIP_NEIGHBOR_COLUMNS:
        tokens |= set(events[col].dropna())
    _export_thumbs_deduped(config=config, out_dir=out_dir, tokens=tokens, context="event")

    # The manifest's flagship_events is the PRE-cap ASSERTED count (item 2,
    # consolidated review), not a re-derivation from the (post-cap) returned
    # frame: build_events already raised above if its own pre-cap
    # hard_braking_near_pedestrians count disagreed with expected_sql_count, so by
    # this point the two are guaranteed equal -- config["flagship"]["expected_sql_
    # count"] IS that asserted number. Counting tags on the post-cap frame instead
    # would silently under-report if a future config ever set cap_per_preset below
    # expected_sql_count (today cap_per_preset=30 == expected_sql_count=30, so
    # nothing is actually capped away, but the manifest key's own meaning should
    # not depend on that coincidence holding forever).
    flagship_events = config["flagship"]["expected_sql_count"]
    # "frame" carries the events DataFrame itself (not a manifest scalar like the
    # other three keys): _include_subgraphs needs it to check the staged subgraphs
    # are keyed by THIS build's events, not a stale earlier run's (item M2, Phase 6
    # review). run_build reads the manifest keys by name, so it never leaks into
    # manifest.json.
    return {
        "events": "included",
        "flagship_events": flagship_events,
        "n_events": len(events),
        "frame": events,
    }


_SEMSEARCH_COLUMNS = ("query", "rank", "sample_data_token", "score", "k")


def _validate_semsearch(df: pd.DataFrame, path: Path) -> None:
    """Reject a malformed/stale staged semsearch parquet BEFORE it's copied into
    the package (item 7, consolidated review) -- a schema mismatch (e.g. from a
    pre-`k`-column ``demo semsearch`` run) must fail the build loudly rather than
    ship a file the gallery page can't fully render.
    """
    missing = [c for c in _SEMSEARCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"demo build: {path} missing columns {missing}")
    if df.empty:
        raise ValueError(f"demo build: {path} has zero rows — re-run `demo semsearch`")
    if df["sample_data_token"].isna().any():
        raise ValueError(f"demo build: {path} has null sample_data_token rows")
    # rank must be exactly 1..n per query, in order -- the gallery page sorts by
    # rank and trusts it starts at 1 with no gaps.
    bad_queries = [
        query
        for query, ranks in df.groupby("query")["rank"]
        if sorted(ranks) != list(range(1, len(ranks) + 1))
    ]
    if bad_queries:
        raise ValueError(
            f"demo build: {path} rank column isn't a clean 1..n sequence per "
            f"query for: {sorted(bad_queries)}"
        )


def _include_semsearch(config: dict[str, Any], out_dir: Path) -> str:
    """Copy the staged recorded-semsearch group into the package, or note its absence.

    Mirrors ``_include_curation``'s absent/present handling: no ``curation:``
    config section, or no staged ``semantic_search_results.parquet`` (``demo
    semsearch`` hasn't run yet), logs a warning and the manifest records "absent" —
    a package built without it is still honest and still usable, same rationale as
    curation's own TRINITY-unreachable fallback. When present, the staged parquet
    is validated (``_validate_semsearch``) BEFORE it's copied in.
    """
    curation_cfg = config.get("curation")
    if not curation_cfg:
        logger.warning(
            "demo build: no `curation:` config section — shipping without the "
            "recorded semantic search (see docs/DEMO.md)"
        )
        return "absent"
    staging_dir = Path(curation_cfg["staging_dir"])
    semsearch_path = staging_dir / "semantic_search_results.parquet"
    if not semsearch_path.is_file():
        logger.warning(
            "demo build: no semsearch staging at %s — shipping without the "
            "recorded semantic search (run `demo semsearch` first, see docs/DEMO.md)",
            semsearch_path,
        )
        return "absent"

    staged = pd.read_parquet(semsearch_path)
    _validate_semsearch(staged, semsearch_path)

    shutil.copy2(semsearch_path, out_dir / "semantic_search_results.parquet")
    tokens = set(staged["sample_data_token"])
    _export_thumbs_deduped(config=config, out_dir=out_dir, tokens=tokens, context="semsearch")
    return "included"


def _include_subgraphs(
    config: dict[str, Any], out_dir: Path, *, events: pd.DataFrame | None
) -> dict[str, Any]:
    """Copy the staged graph-subgraph export into the package, or note its absence.

    Neo4j is an OPERATIONAL dependency (spec §1) — same rationale as
    ``_include_semsearch``'s recorded-search staging: ``demo build`` never talks to
    the live graph itself, it only copies+validates what ``demo subgraphs`` already
    staged at ``curation.staging_dir/graph_subgraphs/``. No ``curation:`` config
    section, or no ``graph_subgraphs/`` staging dir at all (``demo subgraphs`` hasn't
    run yet), is "absent" — a warning, not a build failure, mirroring every other
    optional group's TRINITY/Neo4j-unreachable fallback.

    A staging dir that EXISTS but doesn't hold all six preset JSONs (``demo
    subgraphs`` interrupted mid-run, or a stale partial staging from before a preset
    existed) is a different case: an operator mid-flow, not a no-subgraphs machine,
    same rationale as ``_include_curation``'s partial-staging ``ValueError`` — it
    must not be silently swallowed into "absent" (hiding the mistake) nor silently
    shipped partial (a page some presets can't render for).

    A staging dir holding all six files but subgraphs for the WRONG EVENTS is a
    third case, and the one an operator actually hits (item M2, Phase 6 review):
    ``demo subgraphs`` exports one subgraph per event its own ``build_events`` run
    tagged, so any change to the processed tables or the preset thresholds between
    that run and this build leaves the staging keyed by events this package no
    longer ships (and missing the ones it does). Nothing downstream would fail —
    the page just shows "no subgraph staged for this event" on every card — so it
    is checked here, per preset, against ``events``: the tokens this build's own
    events frame ranks for that preset (a non-null ``preset_rank_<preset>``, which
    is exactly the ``preset in preset_tags`` rule ``export_subgraphs`` itself picks
    its event tokens by — both are written from the same post-cap kept-mask in
    events.py::build_events). ``events`` is ``None`` only for a caller that built no
    events frame at all, and then the check is skipped rather than guessed at.

    When fully staged, all six preset JSONs are copied verbatim (they are already
    the exact package-ready shape ``subgraph_export.export_subgraphs`` wrote) and the
    flagship preset's ``cypher_count`` is read back out — that's the one number
    ``run_build`` needs to upgrade ``overview_metrics.json``'s sourced flagship.cypher
    to a computed one.
    """
    curation_cfg = config.get("curation")
    if not curation_cfg:
        logger.warning(
            "demo build: no `curation:` config section — shipping without the "
            "graph subgraphs panel (see docs/DEMO.md)"
        )
        return {"status": "absent"}
    staging_dir = Path(curation_cfg["staging_dir"]) / "graph_subgraphs"
    if not staging_dir.is_dir():
        logger.warning(
            "demo build: no graph_subgraphs staging at %s — shipping without the "
            "interactive graph panel (run `demo subgraphs` first, see docs/GRAPH.md)",
            staging_dir,
        )
        return {"status": "absent"}

    missing = [
        preset for preset in _SUBGRAPH_PRESETS if not (staging_dir / f"{preset}.json").is_file()
    ]
    if missing:
        raise ValueError(
            f"demo build: graph_subgraphs staging at {staging_dir} is missing "
            f"preset(s) {missing} — re-run `demo subgraphs` (see docs/GRAPH.md)"
        )

    payloads = {
        preset: json.loads((staging_dir / f"{preset}.json").read_text())
        for preset in _SUBGRAPH_PRESETS
    }
    if events is not None:
        for preset in _SUBGRAPH_PRESETS:
            rank_column = f"preset_rank_{preset}"
            expected = set(events.loc[events[rank_column].notna(), "sample_data_token"])
            staged = set(payloads[preset].get("events", {}))
            if staged != expected:
                raise ValueError(
                    f"demo build: graph_subgraphs staging at {staging_dir} is stale for "
                    f"preset {preset!r} — it holds {len(staged)} event subgraph(s) but "
                    f"this build's events frame ranks {len(expected)} "
                    f"(missing {sorted(expected - staged)[:3]}, "
                    f"unexpected {sorted(staged - expected)[:3]}) — re-run "
                    "`demo subgraphs` (see docs/GRAPH.md)"
                )

    dest_dir = out_dir / "graph_subgraphs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for preset in _SUBGRAPH_PRESETS:
        shutil.copy2(staging_dir / f"{preset}.json", dest_dir / f"{preset}.json")

    return {
        "status": "included",
        "flagship_cypher": payloads["hard_braking_near_pedestrians"]["cypher_count"],
    }


def _add_al_selection_columns(
    manifest: pd.DataFrame, *, al_dir: Path, weak_arm: str, al_arm: str
) -> pd.DataFrame:
    """Add ``weak_verdict``/``al_selected_by`` to a copied ``frame_manifest``,
    derived from the local AL arm parquets at build time (never hand-tagged).

    ``weak_verdict`` is ``"accepted"`` for a token in ``<weak_arm>_accepted.
    parquet``, ``"rejected"`` for a token in the weak arm's candidate pool
    (``<weak_arm>.parquet``) but not accepted, else NA. ``al_selected_by`` is
    ``al_arm`` for a token in ``<al_arm>.parquet`` (the arm's selected set), else
    NA. Both are pandas nullable ``string`` columns (``pd.NA``, not None/NaN) so a
    parquet round-trip keeps them queryable as strings.
    """
    candidate = set(
        pd.read_parquet(exporters._require(al_dir / f"{weak_arm}.parquet"))["sample_data_token"]
    )
    accepted = set(
        pd.read_parquet(
            exporters._require(al_dir / f"{weak_arm}_accepted.parquet")
        )["sample_data_token"]
    )
    selected = set(
        pd.read_parquet(exporters._require(al_dir / f"{al_arm}.parquet"))["sample_data_token"]
    )

    def verdict(token: str) -> Any:
        if token in accepted:
            return "accepted"
        if token in candidate:
            return "rejected"
        return pd.NA

    manifest = manifest.copy()
    manifest["weak_verdict"] = manifest["sample_data_token"].map(verdict).astype("string")
    manifest["al_selected_by"] = (
        manifest["sample_data_token"]
        .map(lambda token: al_arm if token in selected else pd.NA)
        .astype("string")
    )
    return manifest


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
    al_df = exporters.export_al_results(al_dir=al_dir, out_dir=out_dir, processed_dir=processed)
    weak_df = exporters.export_weaksup(al_dir=al_dir, out_dir=out_dir, processed_dir=processed)
    curation_status = _include_curation(config, out_dir)

    # Phase 7 (Task 1): weak_verdict/al_selected_by are derived from the local AL
    # arm parquets, only meaningful once there's a real frame_manifest to tag — so
    # gated on curation_status, same rationale as every other curated-frames-
    # dependent step. al_communities.parquet is NOT gated: it comes entirely from
    # al_dir's persisted community diagnostics + the AL config's n_mine, so the
    # community chart works even on a fresh clone with curation absent (spec §2).
    # export_al_exemplars always runs too (it enforces its OWN absent/included rule
    # -- an empty token list only when curation_present is False, same as
    # hero.token below), reading the manifest/predictions this step just produced
    # when curation is included, or empty frames otherwise (never touched when the
    # configured token list is empty, which absent curation requires).
    al_cfg = config["al"]
    al_config_path = Path(paths["active_learning_config"])
    n_mine = load_yaml(al_config_path)["mining"]["n_mine"]

    manifest_df = pd.DataFrame()
    predictions_df = pd.DataFrame()
    if curation_status == "included":
        manifest_path = out_dir / "frame_manifest.parquet"
        manifest_df = _add_al_selection_columns(
            pd.read_parquet(manifest_path),
            al_dir=al_dir,
            weak_arm=config["curation"]["weak_arm"],
            al_arm=config["curation"]["al_arm"],
        )
        manifest_df.to_parquet(manifest_path, index=False)
        predictions_df = pd.read_parquet(out_dir / "predictions.parquet")

    communities_df = exporters.export_al_communities(
        al_dir=al_dir, out_dir=out_dir, arms=tuple(al_cfg["community_arms"]), n_mine=n_mine,
    )

    exemplar_tokens = exporters.export_al_exemplars(
        config=config, manifest=manifest_df, predictions=predictions_df, out_dir=out_dir,
        curation_present=(curation_status == "included"),
    )

    # Phase 7 (Task 2): the loss decomposition and by-class verifier tables are
    # curation-INDEPENDENT (pure results.json / *_pseudo_summary.json reads, same
    # as export_weaksup above) -- they run unconditionally. weak_labels/vlm_counts
    # need a real curated frame_manifest (their token lists come straight off it)
    # and are simply not written when curation is absent, same rationale as every
    # other curated-frames-dependent step above.
    weak_loss_df = exporters.export_weak_loss_decomposition(al_dir=al_dir, out_dir=out_dir)
    exporters.export_weak_verifier_by_class(al_dir=al_dir, out_dir=out_dir)

    # weak_loss_decomposition's retention is computed independently of (but from
    # the same results.json numbers as) export_overview's own by_base_arm ratio --
    # the two must never be allowed to silently drift apart. A base arm overview
    # excludes (non-positive gt_gain) has nothing to check against and is skipped,
    # not treated as a mismatch.
    weak_retention_by_arm = metrics["results"]["weak_retention"]["by_base_arm"]
    for _, row in weak_loss_df.iterrows():
        base_arm = row["base_arm"]
        overview_retention = weak_retention_by_arm.get(base_arm)
        if overview_retention is not None and abs(row["retention"] - overview_retention) > 1e-9:
            raise ValueError(
                "demo build: weak_loss_decomposition retention for base arm "
                f"{base_arm!r} ({row['retention']!r}) != overview_metrics.json's "
                f"results.weak_retention.by_base_arm[{base_arm!r}] "
                f"({overview_retention!r}) — the two recipes have drifted"
            )

    n_weak_labels = 0
    n_vlm_counts = 0
    if curation_status == "included":
        weak_arm = config["curation"]["weak_arm"]
        curated_tokens = list(manifest_df["sample_data_token"])
        weak_labels_df = exporters.export_weak_labels(
            al_dir=al_dir, arm=weak_arm, tokens=curated_tokens, out_dir=out_dir,
        )
        n_weak_labels = len(weak_labels_df)

        vlm_tokens = [
            token
            for token, buckets in zip(
                manifest_df["sample_data_token"], manifest_df["curation_buckets"], strict=True
            )
            if "weak_accepted" in buckets or "weak_rejected" in buckets
        ]
        vlm_counts_df = exporters.export_vlm_counts(
            al_dir=al_dir, processed_dir=processed, tokens=vlm_tokens, out_dir=out_dir,
        )
        n_vlm_counts = len(vlm_counts_df)

    # Phase 3: the hero is a hand-picked exemplar crop from the curated-frames group
    # (configs/demo.yaml `hero.token`), not an mlruns mosaic — see the dated
    # amendment in docs/superpowers/specs/2026-08-12-demo-phase1-design.md. It is
    # resolved AFTER curation because the crop it copies only exists once curation
    # has been included; export_overview ran BEFORE curation (Overview must stay
    # renderable even with curation absent), so overview_metrics.json's hero_token
    # is patched in here via the same write_json helper the exporters use, rather
    # than passed to export_overview at call time.
    #
    # Gated on curation_status FIRST (spec-review fix): the Phase-2 TRINITY-fallback
    # guarantee is that a build with curation absent (e.g. a fresh clone, where
    # data/demo_curation is gitignored) still succeeds as a Phase-1-only package.
    # A null/missing hero token is part of that same "nothing curated yet" state and
    # must not be an error on its own — it only becomes mandatory once there is
    # curated data to pick a crop from. A *non-null* token with curation absent is
    # still a real error: the config references curated data that isn't there.
    hero_token = (config.get("hero") or {}).get("token")
    if curation_status != "included":
        if hero_token:
            raise ValueError(
                f"demo build: hero token {hero_token!r} is configured but curation is "
                f"{curation_status!r} — the hero crop comes from the curated-frames "
                "group; run `demo curate` + `demo infer` first (see docs/DEMO.md)"
            )
        logger.warning(
            "demo build: no curation staging and no hero token — shipping a "
            "Phase-1-only package (no sample_frames/hero.jpg)"
        )
    else:
        if not hero_token:
            raise ValueError(
                "demo build: no hero token configured (hero.token in configs/demo.yaml) "
                "— pick a hero token (see docs/DEMO.md)"
            )
        hero_crop = out_dir / "sample_frames" / "crops" / f"{hero_token}.jpg"
        if not hero_crop.is_file():
            raise ValueError(
                f"demo build: hero crop not found for token {hero_token!r}: {hero_crop}"
            )
        (out_dir / "sample_frames" / "hero.jpg").write_bytes(hero_crop.read_bytes())
        overview_path = out_dir / "overview_metrics.json"
        overview_metrics = json.loads(overview_path.read_text())
        overview_metrics["hero_token"] = hero_token
        write_json(overview_path, overview_metrics)

    expected = config["flagship"]["expected_sql_count"]
    if metrics["flagship"]["sql"] != expected:
        raise ValueError(
            f"flagship SQL count {metrics['flagship']['sql']} != expected {expected} — "
            "the source tables changed; re-verify before publishing"
        )
    headline = metrics["results"]["weak_retention"]["headline"]
    if headline is None:
        raise ValueError("weak-retention headline (random pair) missing from results.json")

    # Phase 5: events + semsearch run after curation (their thumbs may dedup
    # against curation's already-exported ones) and before the size-budget check /
    # outputs hash sweep below, so their files count toward both.
    events_result = _include_events(config, out_dir)
    semsearch_status = _include_semsearch(config, out_dir)

    # Phase 6: the graph subgraphs group runs last among the optional groups, same
    # slot the plan calls for (after semsearch, before the size-budget check below,
    # so its files count toward both). When present, its flagship's COMPUTED Cypher
    # count both upgrades overview_metrics.json's sourced flagship.cypher (the
    # write_json-patch mechanism the hero_token patch above already established) and
    # is asserted equal to the flagship SQL count already validated a few lines up
    # (metrics["flagship"]["sql"]) — the plan's headline "the SQL and Cypher counts
    # agree" claim, now for real instead of sourced from docs.
    subgraphs_result = _include_subgraphs(config, out_dir, events=events_result["frame"])
    if subgraphs_result["status"] == "included":
        flagship_cypher = subgraphs_result["flagship_cypher"]
        flagship_sql = metrics["flagship"]["sql"]
        if flagship_sql != flagship_cypher:
            raise ValueError(
                "demo build: computed flagship Cypher count disagrees with the SQL "
                f"count — sql={flagship_sql} cypher={flagship_cypher} (the graph and "
                "processed tables have drifted; re-verify before publishing)"
            )
        overview_path = out_dir / "overview_metrics.json"
        overview_metrics = json.loads(overview_path.read_text())
        overview_metrics["flagship"]["cypher"] = flagship_cypher
        overview_metrics["flagship"]["cypher_source"] = "computed (neo4j, demo subgraphs)"
        write_json(overview_path, overview_metrics)

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

    # Hash everything the build actually read, not just results.json: the processed
    # parquets and every weak-sup summary — an unnoticed change to any of these
    # silently changes the published numbers. The hero source is now a curated crop
    # (out_dir/sample_frames/crops/<token>.jpg) rather than a separate mlruns
    # mosaic, so there is no extra hero-specific INPUT to hash here — it isn't
    # covered by the curation-staging hash loop below either (that loop only
    # covers the three staged parquets, not crop imagery). The crop (and the
    # hero.jpg copy of it) is still captured, just as an OUTPUT: the outputs loop
    # above hashes every file under out_dir, sample_frames/crops/ and hero.jpg
    # included.
    inputs: dict[str, str] = {
        str(al_dir / "results.json"): _sha256(al_dir / "results.json"),
    }
    for name in exporters.PROCESSED_INPUTS:
        path = processed / name
        inputs[str(path)] = _sha256(path)
    for path in sorted(al_dir.glob("*_pseudo_summary.json")):
        inputs[str(path)] = _sha256(path)
    # Phase 7 (Task 2): export_weaksup's own per-arm inputs -- every arm with a
    # *_pseudo_summary.json (weak_df's own "arm" column) needs its candidate/
    # accepted token-list parquets for the rejected-side recipe. Unconditional,
    # like export_weaksup itself (not gated on curation_status).
    for arm in weak_df["arm"]:
        for name in (f"{arm}.parquet", f"{arm}_accepted.parquet"):
            path = al_dir / name
            inputs[str(path)] = _sha256(path)
    # Phase 7 (Task 1): al_communities.parquet's own inputs -- always hashed, since
    # the export itself is unconditional (see the comment above its call site).
    inputs[str(al_config_path)] = _sha256(al_config_path)
    for arm in al_cfg["community_arms"]:
        path = al_dir / f"communities_{arm}.json"
        inputs[str(path)] = _sha256(path)
    if curation_status == "included":
        staging_dir = Path(config["curation"]["staging_dir"])
        for name in ("frame_manifest.parquet", "gt_boxes.parquet", "predictions.parquet"):
            path = staging_dir / name
            inputs[str(path)] = _sha256(path)
        # weak_verdict/al_selected_by's own arm-parquet inputs (weak_arm and al_arm
        # may be the same arm today, e.g. both "graph_rate_night" -- a dict key
        # naturally dedupes that case rather than hashing the same file twice under
        # two names).
        weak_arm = config["curation"]["weak_arm"]
        al_arm = config["curation"]["al_arm"]
        for name in (f"{weak_arm}.parquet", f"{weak_arm}_accepted.parquet", f"{al_arm}.parquet"):
            path = al_dir / name
            inputs[str(path)] = _sha256(path)
        # Phase 7 (Task 2): weak_labels/vlm_counts' own inputs -- only read (and
        # only hashed) once curation is included, same gating as the export calls
        # above.
        pseudo_labels_path = al_dir / f"{weak_arm}_pseudo_labels.parquet"
        inputs[str(pseudo_labels_path)] = _sha256(pseudo_labels_path)
        vlm_labels_path = al_dir / "autolabel_weak" / "labels.parquet"
        inputs[str(vlm_labels_path)] = _sha256(vlm_labels_path)
    # Events need no staging input of their own — build_events reads only the
    # already-hashed processed tables (+ the same three curation-staging parquets
    # hashed just above, when present). Semsearch's own staged parquet is a
    # distinct input the build actually read, hashed only when included.
    if semsearch_status == "included":
        semsearch_path = Path(config["curation"]["staging_dir"]) / "semantic_search_results.parquet"
        inputs[str(semsearch_path)] = _sha256(semsearch_path)
    if subgraphs_result["status"] == "included":
        subgraphs_staging = Path(config["curation"]["staging_dir"]) / "graph_subgraphs"
        for preset in _SUBGRAPH_PRESETS:
            path = subgraphs_staging / f"{preset}.json"
            inputs[str(path)] = _sha256(path)

    manifest: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "package_version": _PACKAGE_VERSION,
        "inputs": inputs,
        "outputs": outputs,
        "validation": {
            "flagship_sql_count": metrics["flagship"]["sql"],
            "package_mb": round(package_mb, 2),
            "n_arms": len(al_df),
            "n_weak_arms": len(weak_df),
            "n_communities": len(communities_df),
            "n_exemplars": len(exemplar_tokens),
            "n_weak_labels": n_weak_labels,
            "n_vlm_counts": n_vlm_counts,
            "curation": curation_status,
            "events": events_result["events"],
            "flagship_events": events_result["flagship_events"],
            "n_events": events_result["n_events"],
            "semsearch": semsearch_status,
            "subgraphs": subgraphs_result["status"],
        },
    }
    if subgraphs_result["status"] == "included":
        manifest["validation"]["flagship_cypher"] = subgraphs_result["flagship_cypher"]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("demo build: %d outputs, %.2f MB -> %s", len(outputs), package_mb, out_dir)
    return manifest
