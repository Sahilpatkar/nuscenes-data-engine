"""``demo/al_explain.py`` — re-derive the ``graph_rate_night`` selection, validated.

The round-3 active-learning run persisted only its OUTPUT (the 1500-token
``graph_rate_night.parquet``) and per-COMMUNITY diagnostics
(``communities_graph_rate_night.json``). The per-FRAME selection facts the Active
Learning page wants to show — which community a frame belongs to, how much routed
failure mass landed on it, which of ``select_by_mass``'s passes picked it — were
never written down. This module re-derives them at export time by re-running the
experiment's own procedure (``active_learning/graph_mining.py::run_graph_mining``
for the ``rate_mass`` arms) against the local Neo4j + LanceDB, using the SAME
functions and the SAME config keys — nothing is re-typed here.

Because a re-derivation is only worth shipping if it IS the experiment, the export
is gated on exact reproduction (spec
docs/superpowers/specs/2026-08-21-demo-phase7-design.md §1): the re-derived token
set must equal the persisted parquet exactly, and the re-derived diagnostics must
equal the persisted JSON row-for-row (``community``/``size``/``night_members``/
``quota`` exactly, ``mass`` to the 4 dp ``select_by_mass`` itself rounds to). Any
mismatch raises and stages NOTHING — there is no "approximately matches" mode.

Neo4j/LanceDB are OPERATIONAL dependencies, handled like ``demo subgraphs``: the
``demo al-explain`` CLI command stages ``al_explain/{al_selection_explain.parquet,
al_explain_validation.json}`` under ``data/demo_curation/``, and ``demo build``
copies them in as a declared-optional group (absent / partial-error / included).

Four public pure functions plus the runner:

- ``route_failure_mass_per_frame`` makes the same neighbour queries
  ``graph_mining.route_failure_mass`` makes, but also keeps per-frame accumulators.
  The runner calls BOTH and asserts their community totals agree to 1e-9 — the
  original stays the oracle, this copy can never silently drift.
- ``attribute_pick_pass`` re-applies ``select_by_mass``'s two passes (with the real
  ``allocate_by_mass``) to label each selected token ``night``/``main``/
  ``backfill``; the labels must partition the selected list.
- ``validate_reproduction`` is the gate described above.
- ``build_explain_rows`` assembles the staged table.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nuscenes_data_engine.active_learning.graph_mining import (
    DEFAULT_GRAPH_ARMS,
    _louvain_communities,
    allocate_by_mass,
    route_failure_mass,
    select_by_mass,
)
from nuscenes_data_engine.active_learning.mining import (
    mine_candidates,
    read_vectors,
    score_failures,
)
from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.data_engine.graph import connection
from nuscenes_data_engine.demo.exporters import write_json

logger = logging.getLogger("nuscenes_data_engine")

# Only the mass-weighted arms have a routed-failure-mass story to explain; the
# round-1 `size` weighting selects by community size and never calls
# route_failure_mass at all, so this module refuses it rather than pretending.
_REQUIRED_WEIGHTING = "rate_mass"

# Frames a seeded night backfill drew from outside every community get this
# sentinel community, matching select_by_mass's own -1 diagnostics row.
_NO_COMMUNITY = -1

EXPLAIN_COLUMNS = (
    "sample_data_token",
    "arm",
    "is_night",
    "scene_name",
    "community",
    "community_size",
    "community_night_members",
    "community_mass",
    "community_mass_rank",
    "community_quota",
    "degree",
    "degree_rank_in_community",
    "pick_pass",
    "n_failures_routed",
    "mass_routed",
)

EXPLAIN_PARQUET = "al_selection_explain.parquet"
VALIDATION_JSON = "al_explain_validation.json"


def route_failure_mass_per_frame(
    tbl: Any,
    failures: pd.DataFrame,
    communities: dict[str, int],
    pool_scenes: list[str],
    channel: str,
    top_k: int,
    route_k: int,
) -> tuple[dict[int, float], dict[str, tuple[int, float]]]:
    """``graph_mining.route_failure_mass``, plus a per-frame accumulator.

    Step for step the original: rank failures by the smoothed rate, take the top
    ``top_k`` (ties broken by token), read their stored vectors, and spread each
    one's score over the communities of its ``route_k`` nearest pool frames.
    Neighbours outside every community are skipped (they carry no community mass and
    get no per-frame row); failures without a stored embedding are skipped with a
    warning, exactly as the original does.

    Returns ``(community masses, {token: (n_failures_routed, mass_routed)})``. Only
    frames that actually received mass appear in the second dict — callers default
    the rest to ``(0, 0.0)``. The community masses are accumulated in the same order
    as the original, so they are bit-identical to it, not merely close.
    """
    scored = failures.assign(score=score_failures(failures, "smoothed_rate"))
    top = scored.sort_values(["score", "sample_data_token"], ascending=[False, True]).head(top_k)
    vectors = read_vectors(tbl, list(top["sample_data_token"]))
    vector_by_token = dict(zip(vectors["sample_data_token"], vectors["vector"], strict=True))
    missing = [t for t in top["sample_data_token"] if t not in vector_by_token]
    if missing:
        logger.warning(
            "route_failure_mass_per_frame: %d/%d top failures have no stored embedding "
            "and contribute no mass (first: %s)",
            len(missing), len(top), missing[0],
        )
    masses: dict[int, float] = defaultdict(float)
    routed_n: dict[str, int] = defaultdict(int)
    routed_mass: dict[str, float] = defaultdict(float)
    for token, score in zip(top["sample_data_token"], top["score"], strict=True):
        vector = vector_by_token.get(token)
        if vector is None:
            continue
        neighbors = mine_candidates(
            tbl, np.asarray(vector, dtype=np.float32), pool_scenes, channel, route_k
        )
        for neighbor in neighbors["sample_data_token"]:
            community = communities.get(neighbor)
            if community is not None:
                masses[community] += float(score)
                routed_n[neighbor] += 1
                routed_mass[neighbor] += float(score)
    per_frame = {token: (routed_n[token], routed_mass[token]) for token in routed_n}
    return dict(masses), per_frame


def _ranked_members(
    communities: dict[str, int], degrees: dict[str, float]
) -> dict[int, list[str]]:
    """Each community's members, most-connected first (ties by token) — the exact
    ordering ``select_by_mass`` ranks and takes its quotas from."""
    members: dict[int, list[str]] = defaultdict(list)
    for token, community in communities.items():
        members[community].append(token)
    return {
        c: sorted(tokens, key=lambda t: (-degrees.get(t, 0.0), t))
        for c, tokens in members.items()
    }


def attribute_pick_pass(
    communities: dict[str, int],
    degrees: dict[str, float],
    masses: dict[int, float],
    selected: list[str],
    *,
    night_tokens: set[str],
    night_floor: int,
    n_mine: int,
) -> dict[str, str]:
    """Label each selected token with the ``select_by_mass`` pass that picked it.

    Re-applies the same two passes with the same ``allocate_by_mass``: the night
    pass takes the top-``night_quota[c]`` NIGHT members of each community by degree;
    a shortfall is filled by a seeded draw from the night pool (the ``backfill``
    tokens, which this function does not need the RNG to identify — they are exactly
    the tokens ``selected`` holds between the night take and the main pass, the
    order ``select_by_mass`` appends in); the main pass then takes each community's
    top unchosen frames up to its mass-weighted quota.

    Raises ``ValueError`` unless the three labels partition ``selected`` exactly —
    an attribution that doesn't cover the set is a re-derivation bug, not something
    to paper over with a default label.
    """
    ranked = _ranked_members(communities, degrees)
    passes: dict[str, str] = {}

    night_take: list[str] = []
    picked: dict[int, int] = defaultdict(int)
    if night_floor > 0:
        night_ranked = {c: [t for t in ranked[c] if t in night_tokens] for c in ranked}
        night_caps = {c: len(tokens) for c, tokens in night_ranked.items()}
        night_quotas = allocate_by_mass(masses, night_caps, night_floor, floors={})
        for c in sorted(night_quotas):
            take = night_ranked[c][: night_quotas[c]]
            night_take.extend(take)
            picked[c] += len(take)
    for token in night_take:
        passes[token] = "night"

    # The seeded backfill (if any) immediately follows the night take in `selected`
    # and brings it up to exactly `night_floor` frames.
    chosen = set(night_take)
    shortfall = night_floor - len(night_take)
    if shortfall > 0:
        backfilled = [t for t in selected if t not in chosen][:shortfall]
        for token in backfilled:
            passes[token] = "backfill"
            chosen.add(token)
            if token in communities:
                picked[communities[token]] += 1

    capacities = {c: len([t for t in ranked[c] if t not in chosen]) for c in ranked}
    floors = {c: 0 if picked[c] > 0 or capacities[c] == 0 else 1 for c in ranked}
    quotas = allocate_by_mass(masses, capacities, n_mine - len(chosen), floors)
    for c in sorted(quotas):
        for token in [t for t in ranked[c] if t not in chosen][: quotas[c]]:
            passes[token] = "main"

    if set(passes) != set(selected) or len(passes) != len(selected):
        only_selected = sorted(set(selected) - set(passes))
        only_attributed = sorted(set(passes) - set(selected))
        raise ValueError(
            "attribute_pick_pass: the night/main/backfill passes do not partition the "
            f"selected set — {len(only_selected)} selected token(s) unattributed "
            f"(first: {only_selected[0] if only_selected else None}), "
            f"{len(only_attributed)} attributed token(s) not selected "
            f"(first: {only_attributed[0] if only_attributed else None})"
        )
    return passes


def validate_reproduction(
    selected: list[str],
    diagnostics: list[dict[str, Any]],
    *,
    persisted_tokens: list[str],
    persisted_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare a re-derivation against what the AL run actually persisted.

    ``selected_match``: same number of tokens AND the same set (the arm parquet is a
    set, not a ranking — ``run_graph_mining`` writes it in pass order and treats it
    as a set everywhere downstream). ``communities_match``: the diagnostics agree
    row-for-row on ``community``/``size``/``night_members``/``quota`` exactly and on
    ``mass`` to 4 dp (the precision ``select_by_mass`` rounds to before persisting —
    comparing tighter than the file's own precision could only ever fail).

    ``first_mismatch`` is a human-readable description of the first difference found
    (tokens before communities), or ``None`` when the re-derivation is exact.
    """
    first_mismatch: str | None = None

    rederived = set(selected)
    persisted = set(persisted_tokens)
    selected_match = True
    if len(selected) != len(persisted_tokens):
        selected_match = False
        first_mismatch = (
            f"selected count differs: re-derived {len(selected)} vs run "
            f"{len(persisted_tokens)}"
        )
    elif rederived != persisted:
        selected_match = False
        only_run = sorted(persisted - rederived)
        only_rederived = sorted(rederived - persisted)
        first_mismatch = (
            f"selected set differs: {len(only_run)} token(s) in the run but not "
            f"re-derived (first: {only_run[0] if only_run else None}), "
            f"{len(only_rederived)} re-derived but not in the run "
            f"(first: {only_rederived[0] if only_rederived else None})"
        )

    communities_match = True
    if len(diagnostics) != len(persisted_diagnostics):
        communities_match = False
        if first_mismatch is None:
            first_mismatch = (
                f"community count differs: re-derived {len(diagnostics)} rows vs run "
                f"{len(persisted_diagnostics)} rows"
            )
    else:
        for got, want in zip(diagnostics, persisted_diagnostics, strict=True):
            for field in ("community", "size", "night_members", "quota"):
                if int(got[field]) != int(want[field]):
                    communities_match = False
                    if first_mismatch is None:
                        first_mismatch = (
                            f"community {want['community']}: {field} re-derived "
                            f"{int(got[field])} != run {int(want[field])}"
                        )
                    break
            if round(float(got["mass"]), 4) != round(float(want["mass"]), 4):
                communities_match = False
                if first_mismatch is None:
                    first_mismatch = (
                        f"community {want['community']}: mass re-derived "
                        f"{round(float(got['mass']), 4)} != run "
                        f"{round(float(want['mass']), 4)} (compared to 4 dp)"
                    )
            if not communities_match:
                break

    return {
        "selected_match": selected_match,
        "communities_match": communities_match,
        "n_selected": len(selected),
        "n_communities": len(diagnostics),
        "first_mismatch": first_mismatch,
    }


def build_explain_rows(
    selected: list[str],
    communities: dict[str, int],
    degrees: dict[str, float],
    masses: dict[int, float],
    per_frame: dict[str, tuple[int, float]],
    passes: dict[str, str],
    diagnostics: list[dict[str, Any]],
    meta: pd.DataFrame,
    *,
    arm: str,
) -> pd.DataFrame:
    """One row per selected frame, sorted by token (deterministic bytes).

    ``meta`` is the pool's ``samples.parquet`` slice indexed by
    ``sample_data_token`` (``is_night``, ``scene_name``). Community-level facts come
    from ``diagnostics`` (the same records the run persisted), except
    ``community_mass_rank``, which is computed from the FULL-precision ``masses`` so
    two communities only tie when they really do — 1 is the heaviest, ties share the
    smaller rank ("min" method). ``community_mass`` is the 4-dp value the
    diagnostics carry.

    A frame the night backfill drew from outside every community gets community -1
    and the sentinel row's counts; its mass rank and degree rank are NA, since it is
    not ranked within any community (``degree`` is NA too — it was never in the
    Louvain projection).
    """
    by_community = {int(row["community"]): row for row in diagnostics}
    ranked = _ranked_members(communities, degrees)
    degree_rank = {
        token: index + 1
        for tokens in ranked.values()
        for index, token in enumerate(tokens)
    }
    real_masses = {
        c: float(masses.get(c, 0.0)) for c in by_community if c != _NO_COMMUNITY
    }
    mass_rank = (
        pd.Series(real_masses).rank(method="min", ascending=False).astype(int).to_dict()
        if real_masses
        else {}
    )
    is_night = meta["is_night"].to_dict()
    scene_name = meta["scene_name"].to_dict()

    records: list[dict[str, Any]] = []
    for token in sorted(selected):
        community = communities.get(token, _NO_COMMUNITY)
        row = by_community.get(community)
        if row is None:
            raise ValueError(
                f"build_explain_rows: no diagnostics row for community {community} "
                f"(token {token!r}) — the diagnostics and the selection disagree"
            )
        n_routed, mass_routed = per_frame.get(token, (0, 0.0))
        records.append(
            {
                "sample_data_token": token,
                "arm": arm,
                "is_night": bool(is_night[token]),
                "scene_name": str(scene_name[token]),
                "community": int(community),
                "community_size": int(row["size"]),
                "community_night_members": int(row["night_members"]),
                "community_mass": float(row["mass"]),
                "community_mass_rank": mass_rank.get(community, pd.NA),
                "community_quota": int(row["quota"]),
                "degree": degrees.get(token, float("nan")),
                "degree_rank_in_community": degree_rank.get(token, pd.NA),
                "pick_pass": passes[token],
                "n_failures_routed": int(n_routed),
                "mass_routed": float(mass_routed),
            }
        )

    frame = pd.DataFrame.from_records(records, columns=list(EXPLAIN_COLUMNS))
    frame["community_mass_rank"] = frame["community_mass_rank"].astype("Int64")
    frame["degree_rank_in_community"] = frame["degree_rank_in_community"].astype("Int64")
    frame["degree"] = frame["degree"].astype("float64")
    return frame


def _open_frames_table(cfg: dict[str, Any]) -> Any:
    """The LanceDB frames table the AL config points at (same call as run_mining)."""
    from nuscenes_data_engine.data_engine import store

    engine_cfg = load_yaml(Path(cfg.get("engine_config", "configs/engine.yaml"))).get("lancedb", {})
    return store.open_frames_table(
        Path(engine_cfg.get("path", "data/lancedb")), engine_cfg.get("table", "frames"), dim=0
    )


def _gds_version(driver: Any, database: str) -> str:
    """The live GDS version — the build the re-derivation actually ran against."""
    rows = connection.read_query(driver, "RETURN gds.version() AS v", database=database)
    return str(rows[0]["v"])


def run_al_explain(
    *,
    al_config_path: Path,
    processed_dir: Path,
    al_dir: Path,
    out_dir: Path,
    arm: str = "graph_rate_night",
) -> dict[str, Any]:
    """Re-derive ``arm``'s selection and stage the per-frame facts, if it reproduces.

    Mirrors ``graph_mining.run_graph_mining`` step for step for a ``rate_mass`` arm,
    reusing its own functions and reading every parameter from
    ``configs/active_learning.yaml`` (nothing re-typed): Louvain over the pool's
    ``SIMILAR_TO`` projection, failure-mass routing through LanceDB, then
    ``select_by_mass`` with the arm's night quota.

    Nothing is written until ``validate_reproduction`` passes — on any mismatch this
    raises ``ValueError`` naming the first difference and ``out_dir`` is left
    untouched (it is not even created).
    """
    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    cfg = load_yaml(al_config_path)
    settings = get_settings()
    channel = cfg.get("split", {}).get("channel", "CAM_FRONT")
    graph_cfg = cfg.get("graph_mining", {})
    arms_cfg = {**DEFAULT_GRAPH_ARMS, **graph_cfg.get("arms", {})}
    if arm not in arms_cfg:
        raise ValueError(f"Unknown graph-mining arm {arm!r} (expected one of {sorted(arms_cfg)})")
    arm_cfg = arms_cfg[arm] or {}
    weighting = arm_cfg.get("weighting")
    if weighting != _REQUIRED_WEIGHTING:
        raise ValueError(
            f"demo al-explain: arm {arm!r} has weighting {weighting!r}, but only "
            f"{_REQUIRED_WEIGHTING!r} arms route failure mass — there is nothing to explain"
        )
    quotas_cfg = {flag: int(v) for flag, v in (arm_cfg.get("quotas") or {}).items()}
    night_floor = quotas_cfg.get("is_night", 0)
    n_mine = int(graph_cfg.get("n_mine", cfg.get("mining", {}).get("n_mine", 1500)))
    route_k = int(graph_cfg.get("route_k", 10))
    seed = int(graph_cfg.get("seed", cfg.get("mining", {}).get("seed", 64)))
    top_k = int(cfg.get("sweep", {}).get("top_k_failures", 1000))
    database = settings.neo4j_database

    split = pd.read_parquet(al_dir / "split.parquet")
    pool_scenes = sorted(split[split["role"] == "pool"]["scene_name"])
    pool_frames = frames_for_scenes(processed_dir, set(pool_scenes), channel)
    samples = pd.read_parquet(
        processed_dir / "samples.parquet",
        columns=["sample_data_token", "scene_name", "is_night"],
    ).set_index("sample_data_token")
    pool_meta = samples.loc[sorted(pool_frames)]

    driver = connection.get_driver(settings)
    try:
        communities, degrees = _louvain_communities(
            driver, sorted(pool_frames), database, seed=seed
        )
        gds_version = _gds_version(driver, database)
    finally:
        connection.close(driver)
    logger.info(
        "demo al-explain: Louvain over %d pool frames -> %d connected in %d communities "
        "(GDS %s)",
        len(pool_frames), len(communities), len(set(communities.values())), gds_version,
    )

    tbl = _open_frames_table(cfg)
    failures = pd.read_parquet(al_dir / "failures.parquet")
    masses, per_frame = route_failure_mass_per_frame(
        tbl, failures, communities, pool_scenes, channel, top_k, route_k
    )
    # The original function is the ORACLE: the per-frame copy is only trustworthy if
    # its community totals are the same numbers the experiment allocated budget by.
    oracle = route_failure_mass(
        tbl, failures, communities, pool_scenes, channel, top_k, route_k
    )
    if set(masses) != set(oracle) or any(
        abs(masses[c] - oracle[c]) > 1e-9 for c in oracle
    ):
        raise ValueError(
            "demo al-explain: route_failure_mass_per_frame disagrees with "
            "graph_mining.route_failure_mass — the per-frame routing has drifted from "
            "the experiment's own function"
        )

    night_tokens = set(pool_meta.index[pool_meta["is_night"]])
    selected, diagnostics = select_by_mass(
        communities,
        degrees,
        masses,
        n_mine,
        night_tokens=night_tokens,
        night_floor=night_floor,
        night_pool=sorted(night_tokens),
        seed=seed,
    )
    passes = attribute_pick_pass(
        communities, degrees, masses, selected,
        night_tokens=night_tokens, night_floor=night_floor, n_mine=n_mine,
    )

    persisted_tokens = list(
        pd.read_parquet(al_dir / f"{arm}.parquet")["sample_data_token"]
    )
    persisted_diagnostics = json.loads((al_dir / f"communities_{arm}.json").read_text())
    validation = validate_reproduction(
        selected, diagnostics,
        persisted_tokens=persisted_tokens, persisted_diagnostics=persisted_diagnostics,
    )
    if not (validation["selected_match"] and validation["communities_match"]):
        raise ValueError(
            f"demo al-explain: arm {arm!r} did NOT reproduce the persisted run — "
            f"{validation['first_mismatch']}; nothing staged (the per-frame facts are "
            "only shippable if the re-derivation IS the experiment)"
        )

    rows = build_explain_rows(
        selected, communities, degrees, masses, per_frame, passes, diagnostics,
        pool_meta, arm=arm,
    )
    mass_total = round(float(sum(masses.values())), 4)
    payload: dict[str, Any] = {
        **validation,
        "arm": arm,
        "mass_total": mass_total,
        "gds_version": gds_version,
        "config": {
            "n_mine": n_mine,
            "route_k": route_k,
            "top_k": top_k,
            "night_floor": night_floor,
            "seed": seed,
            "channel": channel,
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_dir / EXPLAIN_PARQUET, index=False)
    write_json(out_dir / VALIDATION_JSON, payload)

    pass_counts = {
        name: int((rows["pick_pass"] == name).sum())
        for name in ("night", "main", "backfill")
    }
    summary: dict[str, Any] = {
        **validation,
        "arm": arm,
        "n_rows": len(rows),
        "n_pool_connected": len(communities),
        "mass_total": mass_total,
        "mass_routed_total": float(sum(mass for _n, mass in per_frame.values())),
        "gds_version": gds_version,
        "pick_pass": pass_counts,
        "out_dir": str(out_dir),
    }
    logger.info(
        "demo al-explain: arm %s reproduced (selected_match=%s communities_match=%s) — "
        "%d rows, %d communities, mass_total %.4f, passes %s -> %s",
        arm, validation["selected_match"], validation["communities_match"], len(rows),
        validation["n_communities"], mass_total, pass_counts, out_dir,
    )
    return summary
