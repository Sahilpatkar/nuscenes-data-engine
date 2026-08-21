"""Demo artifact exporters — every published number is derived at export time.

Pure functions over the repo's local artifacts (parquet, results.json, MLflow files,
LanceDB thumbnails). Nothing here talks to a network, a GPU, or the live stack; the
public demo app reads only the files these functions write (design:
docs/superpowers/specs/2026-08-12-demo-phase1-design.md).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from nuscenes_data_engine.demo.curate import front_camera_hits

logger = logging.getLogger("nuscenes_data_engine")

# Default over-request factor for semantic search: the LanceDB frame store spans
# all six camera channels, but every semsearch result must be CAM_FRONT (mirrors
# demo curate's semantic bucket, cli.py's `semantic_oversample`) -- 8x leaves
# enough headroom for `front_camera_hits` to still fill `k` after filtering.
# Overridable via configs/demo.yaml's `semsearch.oversample` (item 7, consolidated
# review) -- this is only the fallback when that key is absent.
_DEFAULT_SEMSEARCH_OVERSAMPLE = 8

# Verbatim from docs/GRAPH.md — the flagship SQL/Cypher parity query.
FLAGSHIP_SQL = """
    SELECT count(DISTINCT c.sample_token)
    FROM canbus c JOIN annotations_3d a USING (sample_token)
    WHERE c.is_hard_braking AND a.category_group = 'pedestrian'
      AND a.distance_to_ego_m < 10
"""

PROCESSED_INPUTS = (
    "samples.parquet",
    "annotations.parquet",
    "annotations_3d.parquet",
    "canbus.parquet",
    "ego_pose.parquet",
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required demo input missing: {path}")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as sorted, indented JSON, creating parent dirs as needed.

    Public since Phase 3: ``build.py`` reuses it to patch ``hero_token`` into
    overview_metrics.json after curation/hero resolution, which runs after this
    module's own export_overview call has already written the file once.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    """Run a single-row, single-column aggregate query and return its value.

    ``fetchone()`` is typed as returning ``tuple | None`` even though a bare
    aggregate query (``count(*)``, ``corr(...)``) always yields exactly one row;
    guard it explicitly rather than asserting past strict mypy.
    """
    row = con.execute(sql).fetchone()
    if row is None:
        raise ValueError(f"aggregate query returned no rows: {sql}")
    return row[0]


def export_overview(
    *,
    processed_dir: Path,
    al_dir: Path,
    out_dir: Path,
    flagship_cypher: dict[str, Any],
) -> dict[str, Any]:
    """Derive the Overview page's numbers from local artifacts and write the JSON.

    Nothing is hardcoded: scale = parquet row counts, r = live correlation of CAN
    speed vs GT-derived ego speed, flagship = the documented SQL re-run over the
    local tables. The Cypher twin written here is the *sourced* fallback
    (docs/GRAPH.md), labelled as such — ``run_build`` overwrites
    ``flagship.cypher``/``cypher_source`` with the *computed* value whenever
    ``demo subgraphs``' staging is present (build.py::_include_subgraphs).

    Phase 3's hero_token is deliberately NOT a parameter here: the hero token is
    only resolved after curation is included, which runs after this export
    (Overview must stay renderable even with curation absent) -- ``run_build``
    patches ``hero_token`` into the written JSON afterward via ``write_json``
    instead.
    """
    for name in PROCESSED_INPUTS:
        _require(processed_dir / name)
    results = json.loads(_require(al_dir / "results.json").read_text())

    con = duckdb.connect()
    for name in PROCESSED_INPUTS:
        table = name.removesuffix(".parquet")
        # duckdb.BinderException: CREATE VIEW (a DDL statement) cannot take a
        # prepared-statement parameter, so bind the path via the relation API
        # instead of `execute(..., [path])` (verified against duckdb 1.5.4).
        con.read_parquet(str(processed_dir / name)).create_view(table)
    scale = {
        "images": _scalar(con, "SELECT count(*) FROM samples"),
        "boxes_2d": _scalar(con, "SELECT count(*) FROM annotations"),
        "objects_3d": _scalar(con, "SELECT count(*) FROM annotations_3d"),
        "canbus_rows": _scalar(con, "SELECT count(*) FROM canbus"),
    }
    # docs/DATA.md's documented validation correlates the CAN vehicle_monitor
    # wheel-speed stream (an independent modality from GT pose) against
    # ego-pose-derived speed; `can_vel_mps` comes from the CAN pose/localization
    # stream instead and is a weaker independence claim, even though it happens to
    # round to the same 0.999 today. `WHERE c.has_canbus` is belt-and-braces here —
    # corr() already skips NULL pairs, and only has_canbus=False rows have a NULL
    # can_speed_kmh — but it documents the intent.
    can_speed_r = _scalar(
        con,
        "SELECT corr(c.can_speed_kmh / 3.6, e.speed_mps) FROM canbus c "
        "JOIN ego_pose e USING (sample_token) WHERE c.has_canbus",
    )
    if can_speed_r is None:
        raise ValueError("CAN speed correlation undefined — empty join?")
    flagship_sql = _scalar(con, FLAGSHIP_SQL)

    baseline = results["baseline"]["night"]["mAP50-95"]
    overall_baseline = results["baseline"]["overall"]["mAP50-95"]
    # weak_ arms are excluded here: they are pseudo-label training runs, not
    # independent night-targeting arms, so they can't win "best night gain". Note
    # the margin is narrow — in the real data weak_graph_rate_night_gt is +0.0099
    # against the winning graph_rate_night's +0.0101 — so a future results.json
    # change flipping the winner is a real result change, not a bug here.
    night_deltas = {
        arm: entry["night"]["mAP50-95"] - baseline
        for arm, entry in results.items()
        if arm != "baseline" and not arm.startswith("weak_")
    }
    best_night_arm = max(night_deltas, key=lambda arm: night_deltas[arm])

    # Per-pair weak-supervision retention (weak_<base> vs <base>, both vs baseline),
    # keyed by the GT base arm. `_gt` arms (e.g. weak_random_gt) are a different,
    # separately-documented comparison (GT training on the verifier-kept subset) and
    # are excluded here. The Overview headline is *attributed*, not just the largest
    # or most recent pair: docs/DEMO_PLAN.md and docs/ACTIVE_LEARNING.md's published
    # 18% figure is specifically the weak_random/random pair — other pairs (e.g.
    # weak_graph_rate_night/graph_rate_night, ~39% in the real data) are real but
    # different results, and must never be silently swapped in as "the" headline.
    weak_pairs: dict[str, float] = {}
    for arm, entry in results.items():
        if not arm.startswith("weak_") or arm.endswith("_gt"):
            continue
        base = arm.removeprefix("weak_")
        if base not in results:
            continue
        gt_gain = results[base]["overall"]["mAP50-95"] - overall_baseline
        # Strictly positive only: `if gt_gain:` let a *negative* GT gain through too
        # (float-truthiness only excludes exactly zero), which would publish a
        # technically-computed but meaningless ratio — "retention" of a regression
        # isn't a real quantity. A base arm that regressed vs baseline just doesn't
        # get a weak-retention entry.
        if gt_gain > 0:
            weak_gain = entry["overall"]["mAP50-95"] - overall_baseline
            weak_pairs[base] = round(weak_gain / gt_gain, 4)

    metrics: dict[str, Any] = {
        "scale": scale,
        "can_speed_r": round(float(can_speed_r), 3),
        "flagship": {
            "sql": flagship_sql,
            "cypher": flagship_cypher["count"],
            "cypher_source": flagship_cypher["source"],
        },
        "results": {
            "best_night_arm": best_night_arm,
            "best_night_delta": round(night_deltas[best_night_arm], 4),
            "weak_retention": {
                "headline_arm": "random",  # the documented 18% pair
                "headline": weak_pairs.get("random"),
                "by_base_arm": weak_pairs,
            },
        },
    }
    write_json(out_dir / "overview_metrics.json", metrics)
    return metrics


def _arm_family(arm: str) -> str:
    """``baseline`` -> "baseline"; any ``weak_*`` arm -> "weak"; everything else is
    its own family (mined/random/graph/rate/strat/rate_strat/graph_rate/
    graph_rate_night are each singletons, not grouped with one another)."""
    if arm == "baseline":
        return "baseline"
    if arm.startswith("weak_"):
        return "weak"
    return arm


def export_al_results(*, al_dir: Path, out_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Reshape results.json into the arm-comparison table the demo charts read.

    Phase 7 (Task 1): widens the table with ``round_order`` (results.json's own
    insertion order -- lost by the plain dict iteration below, since the returned
    rows are re-sorted by ``arm`` for a stable, deterministic file), ``family``, the
    night pedestrian/day/rain/clear slice numbers, the weak arms' ``n_boxes``, and
    mined-set composition (``n_scenes``/``night_share``/``rain_share``) via
    ``active_learning.report.arm_composition``. ``per_class``/``slices`` are read
    with ``.get`` throughout so an older results.json without them yields NA
    (pandas ``None``) rather than a ``KeyError`` -- every existing column and value
    is otherwise unchanged, and rows stay sorted by ``arm``.
    """
    from nuscenes_data_engine.active_learning.report import arm_composition

    results = json.loads(_require(al_dir / "results.json").read_text())
    base_overall = results["baseline"]["overall"]["mAP50-95"]
    base_night = results["baseline"]["night"]["mAP50-95"]
    composition = arm_composition(al_dir, processed_dir)
    round_order = {arm: i for i, arm in enumerate(results)}
    rows = []
    for arm, entry in results.items():
        overall = entry["overall"]
        night = entry["night"]
        slices = entry.get("slices", {})
        comp = composition.get(arm, {})
        rows.append(
            {
                "arm": arm,
                "n_train_images": entry.get("n_train_images"),
                "overall_map5095": overall["mAP50-95"],
                "night_map5095": night["mAP50-95"],
                "delta_overall": round(overall["mAP50-95"] - base_overall, 4),
                "delta_night": round(night["mAP50-95"] - base_night, 4),
                "round_order": round_order[arm],
                "family": _arm_family(arm),
                "overall_map50": overall.get("mAP50"),
                "night_map50": night.get("mAP50"),
                "night_precision": night.get("precision"),
                "night_recall": night.get("recall"),
                "night_ped_map5095": night.get("per_class", {}).get("pedestrian"),
                "day_map5095": slices.get("time_of_day/day", {}).get("mAP50-95"),
                "rain_map5095": slices.get("weather/rain", {}).get("mAP50-95"),
                "clear_map5095": slices.get("weather/clear", {}).get("mAP50-95"),
                "n_boxes": entry.get("n_boxes"),
                "n_scenes": comp.get("n_scenes"),
                "night_share": comp.get("night_share"),
                "rain_share": comp.get("rain_share"),
            }
        )
    df = pd.DataFrame(sorted(rows, key=lambda row: row["arm"])).reset_index(drop=True)
    df["n_boxes"] = df["n_boxes"].astype("Int64")
    df["n_scenes"] = df["n_scenes"].astype("Int64")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "active_learning_results.parquet", index=False)
    return df


def export_al_communities(
    *,
    al_dir: Path,
    out_dir: Path,
    arms: tuple[str, ...] = ("graph_rate", "graph_rate_night"),
    n_mine: int,
) -> pd.DataFrame:
    """Merge the per-arm community diagnostics (``communities_<arm>.json``) into one
    table: ``community, size, night_members, mass, quota_<arm>`` (one quota column
    per arm), sorted by ``community``.

    ``size``/``night_members``/``mass`` are per-COMMUNITY facts (the community
    structure and its routed failure mass are identical regardless of which arm's
    quota allocation is being looked at) -- the two (or more) files are expected to
    agree on them exactly (``mass`` to floating-point tolerance) and disagreement
    names the offending community rather than silently picking one file's value.
    Each ``quota_<arm>`` column must sum to ``n_mine`` (the arm's mining budget) --
    a mismatch means the diagnostics file doesn't describe the run that actually
    produced ``<arm>.parquet``, and is a build-time ``ValueError``, not a silently
    wrong chart. A ``community == -1`` backfill-sentinel row, if present in any
    file, is kept (not dropped as an outlier) and flagged via a boolean
    ``is_backfill`` column.
    """
    if not arms:
        raise ValueError("export_al_communities: empty arms")
    per_arm: dict[str, pd.DataFrame] = {}
    for arm in arms:
        path = _require(al_dir / f"communities_{arm}.json")
        records = json.loads(path.read_text())
        per_arm[arm] = pd.DataFrame.from_records(
            records, columns=["community", "size", "night_members", "mass", "quota"]
        ).set_index("community")

    base_arm = arms[0]
    merged = per_arm[base_arm][["size", "night_members", "mass"]].copy()
    for arm in arms[1:]:
        other = per_arm[arm]
        if set(merged.index) != set(other.index):
            symmetric_diff = sorted(set(merged.index) ^ set(other.index))
            raise ValueError(
                f"export_al_communities: communities_{base_arm}.json and "
                f"communities_{arm}.json disagree on which communities exist: "
                f"{symmetric_diff}"
            )
        for community in merged.index:
            for col in ("size", "night_members"):
                left = merged.loc[community, col]
                right = other.loc[community, col]
                if left != right:
                    raise ValueError(
                        f"export_al_communities: community {community} disagrees "
                        f"between communities_{base_arm}.json and communities_{arm}.json "
                        f"on {col!r}: {left!r} != {right!r}"
                    )
            left_mass = merged.loc[community, "mass"]
            right_mass = other.loc[community, "mass"]
            if abs(left_mass - right_mass) >= 1e-9:
                raise ValueError(
                    f"export_al_communities: community {community} disagrees between "
                    f"communities_{base_arm}.json and communities_{arm}.json on "
                    f"'mass': {left_mass!r} != {right_mass!r}"
                )

    for arm in arms:
        quota = per_arm[arm]["quota"]
        merged[f"quota_{arm}"] = quota
        total = int(quota.sum())
        if total != n_mine:
            raise ValueError(
                f"export_al_communities: quota_{arm} sums to {total}, expected "
                f"n_mine={n_mine} (communities_{arm}.json doesn't match the mining "
                "config it's supposed to describe)"
            )

    merged["is_backfill"] = merged.index == -1
    merged = merged.reset_index().sort_values("community").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "al_communities.parquet", index=False)
    return merged


def export_al_exemplars(
    *,
    config: dict[str, Any],
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    out_dir: Path,
    curation_present: bool,
) -> list[str]:
    """Validate ``configs/demo.yaml``'s hand-approved ``al.exemplar_tokens`` against
    the curated frame manifest and write ``al_exemplars.json``.

    Every token must be a ``split == "val"`` row of ``manifest`` with
    ``fixes_fn_vs_<baseline>_<arm> == True`` (the arm caught a box the baseline
    missed) and must have ``predictions`` rows for EVERY model in
    ``config["models"]`` (else the before/after page can't render one of its
    panels). A null/empty token list is only valid when ``curation_present`` is
    False (the same "nothing curated yet" rule ``run_build`` applies to
    ``hero.token``) -- an empty list while curation IS included is a build error,
    since the exemplars would then just silently never render.
    """
    al_cfg = config.get("al") or {}
    arm = al_cfg.get("arm")
    baseline = al_cfg.get("baseline")
    tokens = list(al_cfg.get("exemplar_tokens") or [])

    if not tokens:
        if curation_present:
            raise ValueError(
                "export_al_exemplars: al.exemplar_tokens is empty but curation is "
                "included — pick hand-approved exemplar tokens (see docs/DEMO.md)"
            )
        write_json(out_dir / "al_exemplars.json", {"arm": arm, "baseline": baseline, "tokens": []})
        return []

    if not curation_present:
        raise ValueError(
            f"export_al_exemplars: al.exemplar_tokens configured ({len(tokens)} "
            "tokens) but curation is absent — exemplars are validated against the "
            "curated frame manifest; run `demo curate` + `demo infer` first (see "
            "docs/DEMO.md)"
        )

    flag_col = f"fixes_fn_vs_{baseline}_{arm}"
    val_manifest = manifest.loc[manifest["split"] == "val"].set_index("sample_data_token")
    model_names = sorted(config["models"])
    pred_tokens_by_model = {
        model: set(predictions.loc[predictions["model"] == model, "sample_data_token"])
        for model in model_names
    }
    for token in tokens:
        if token not in val_manifest.index:
            raise ValueError(
                f"export_al_exemplars: token {token!r} is not a val row of "
                "frame_manifest.parquet"
            )
        flag = val_manifest.loc[token].get(flag_col) if flag_col in val_manifest.columns else None
        if not (pd.notna(flag) and bool(flag)):
            raise ValueError(
                f"export_al_exemplars: token {token!r} does not have {flag_col} == True"
            )
        for model in model_names:
            if token not in pred_tokens_by_model[model]:
                raise ValueError(
                    f"export_al_exemplars: token {token!r} has no predictions for "
                    f"model {model!r}"
                )

    write_json(out_dir / "al_exemplars.json", {"arm": arm, "baseline": baseline, "tokens": tokens})
    return tokens


def export_weaksup(*, al_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Collect every ``*_pseudo_summary.json`` into one weak-supervision table.

    Arms without a summary are absent rather than an error — the demo shows what
    was actually run.

    Phase-7 note: the documented 50%/32% loss decomposition (docs/ACTIVE_LEARNING.md)
    needs a 3-way results.json comparison per base arm (e.g. random vs weak_random_gt
    vs weak_random) that these per-arm summaries alone don't carry. The rejected-frame
    GT-box mean (documented 7.61) is also not in the summaries — recompute it from the
    arm's mined-token list minus its accepted tokens, joined to samples.parquet's
    n_boxes per frame.
    """
    rows = []
    for path in sorted(al_dir.glob("*_pseudo_summary.json")):
        summary = json.loads(path.read_text())
        rows.append(
            {
                "arm": summary["arm"],
                "n_candidates": summary["n_candidates"],
                "n_accepted": summary["n_accepted"],
                "verifier_retention": summary["retention"],
                "n_pseudo_boxes": summary["n_boxes"],
                "boxes_per_accepted_frame": summary["mean_boxes_per_accepted_frame"],
                "gt_boxes_per_accepted_frame": summary["mean_gt_boxes_per_accepted_frame"],
            }
        )
    if not rows:
        raise ValueError(f"no *_pseudo_summary.json found under {al_dir}")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "weak_supervision_results.parquet", index=False)
    return df


def export_semsearch(
    *,
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    queries: list[str],
    k: int,
    staging_dir: Path,
    oversample: int = _DEFAULT_SEMSEARCH_OVERSAMPLE,
) -> pd.DataFrame:
    """Run the recorded semantic-search queries, write ``semantic_search_results.parquet``.

    ``search_fn(query, k)`` is an injected callable (production: a lazily-built
    ``SearchEngine.search_text``, so this module itself never imports torch/lancedb)
    returning an ORDERED (nearest-first) list of result dicts with at least
    ``sample_data_token``, ``channel``, ``score``. Each query is over-requested by
    ``oversample`` (``configs/demo.yaml``'s ``semsearch.oversample``, default 8 when
    absent) and filtered to CAM_FRONT via ``curate.py::front_camera_hits`` — the
    same helper (and the same reason: the LanceDB store spans all six camera
    channels) ``demo curate``'s semantic bucket reuses, so the filtering logic
    lives in exactly one place.

    Writes ``(query, rank, sample_data_token, score, k)`` rows in rank order (rank
    restarts at 1 per query; ``k`` is the CONFIGURED target repeated on every row of
    that query, not how many actually came back — the demo page's gallery caption
    ("N of k front-camera hits") needs both numbers) to
    ``staging_dir/semantic_search_results.parquet``, creating ``staging_dir`` if
    needed, and returns the same DataFrame.
    """
    if not queries:
        raise ValueError("export_semsearch: empty queries")

    rows: list[dict[str, Any]] = []
    for query in queries:
        raw = search_fn(query, k * oversample)
        front = front_camera_hits(raw, k)
        for rank, hit in enumerate(front, start=1):
            rows.append(
                {
                    "query": query,
                    "rank": rank,
                    "sample_data_token": hit["sample_data_token"],
                    "score": hit["score"],
                    "k": k,
                }
            )

    df = pd.DataFrame(rows, columns=["query", "rank", "sample_data_token", "score", "k"])
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(staging_dir / "semantic_search_results.parquet", index=False)
    return df


def export_thumbs(
    *, lancedb_path: Path, table: str, tokens: list[str], out_dir: Path
) -> list[Path]:
    """Export LanceDB-embedded thumbnails as ``thumbs/<token>.jpg`` files.

    Reads the store directly (no SearchEngine, no encoder, no torch). Every
    requested token must exist — a silent miss would leave a demo page with a
    broken image, so missing tokens raise naming themselves.

    Tokens are validated up front, before touching the store: they are
    interpolated into a SQL ``IN (...)`` clause via ``repr()``, which breaks on
    quotes (raising a bare ``RuntimeError`` from the query engine, not a
    meaningful error) and is a syntax error for an empty list.
    """
    if not tokens:
        raise ValueError("export_thumbs: empty token list")
    bad = [t for t in tokens if not t.replace("-", "").isalnum()]
    if bad:
        raise ValueError(f"export_thumbs: malformed tokens (expected hex-like): {bad[:5]}")

    import lancedb  # lazy: optional dependency, never needed by the demo app itself

    frames = (
        lancedb.connect(str(lancedb_path))
        .open_table(table)
        .search()
        .where(f"sample_data_token IN ({', '.join(repr(t) for t in tokens)})")
        .select(["sample_data_token", "thumbnail"])
        .limit(len(tokens))
        .to_list()
    )
    by_token = {row["sample_data_token"]: row["thumbnail"] for row in frames}
    missing = [token for token in tokens if token not in by_token]
    if missing:
        raise ValueError(f"tokens missing from the LanceDB store: {missing}")
    thumb_dir = out_dir / "sample_frames" / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for token in sorted(tokens):
        path = thumb_dir / f"{token}.jpg"
        path.write_bytes(by_token[token])
        written.append(path)
    return written
