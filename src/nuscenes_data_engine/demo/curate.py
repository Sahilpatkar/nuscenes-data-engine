"""``demo curate`` — deterministic bucket selection into a frame manifest + rsync filelist.

Buckets sample deterministically (stable sort, explicit tie-breaks — score desc then
token asc; ascending for ``clean_success``; token asc alone where no score exists) from
the AL failure ledger (``failures.parquet``), hard-braking canbus events, the AL/weak
arms, and an injected semantic-search callable. No RNG anywhere — "seeded" curation is
satisfied by determinism. Buckets are merged into one manifest (deduping tokens that
land in more than one bucket) plus the rsync filelist that ``demo infer`` reads.

Every curated token must be a CAM_FRONT frame — the LanceDB store the semantic bucket
draws from spans all six camera channels, and a non-CAM_FRONT token silently reaching
``demo infer``/rsync would be a wrong-shaped image for the box matcher and crops. The
check is applied to every bucket's output, not just the semantic one, so it also
guards any bucket added later.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

_KNOWN_BUCKETS = frozenset(
    {
        "night_failure",
        "day_failure",
        "hard_braking",
        "al_selected",
        "weak_accepted",
        "weak_rejected",
        "clean_success",
        "semantic",
    }
)

# Failure-ledger stat columns copied onto the manifest for tokens that have them (i.e.
# the curated token also appears in failures.parquet); train_pool-only tokens get NA.
# n_gt/n_matched/n_fn/n_low_conf use pandas' nullable Int64, failure_score Float64 —
# plain numpy dtypes would force these to float64/object on the NA rows, which is a
# smaller version of the same "NaN poisons a boolean-shaped column" bug C1 fixes for
# canbus (see the scene_name/is_night/is_rain handling below, sourced from
# samples.parquet instead for exactly that reason).
_INT_STAT_COLUMNS = ("n_gt", "n_matched", "n_fn", "n_low_conf")
_FLOAT_STAT_COLUMNS = ("failure_score",)

_MANIFEST_COLUMNS = (
    "sample_data_token",
    "split",
    "curation_buckets",
    "filename",
    "scene_name",
    "is_night",
    "is_rain",
    *_INT_STAT_COLUMNS,
    *_FLOAT_STAT_COLUMNS,
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required curation input missing: {path}")
    return path


def _top_tokens(
    df: pd.DataFrame, *, score_col: str, quota: int, ascending: bool = False
) -> list[str]:
    """Deterministic top-``quota`` tokens: sort by ``score_col`` (desc unless
    ``ascending``), tie-broken by ``sample_data_token`` ascending."""
    if quota <= 0 or df.empty:
        return []
    ordered = df.sort_values(
        [score_col, "sample_data_token"], ascending=[ascending, True], kind="stable"
    )
    return list(ordered["sample_data_token"].head(quota))


def _warn_if_underfilled(bucket_name: str, selected: list[str], quota: int) -> None:
    if quota > 0 and len(selected) < quota:
        logger.warning(
            "demo curate: bucket %r under-filled (%d/%d tokens available)",
            bucket_name,
            len(selected),
            quota,
        )


def run_curate(
    *,
    processed_dir: Path,
    al_dir: Path,
    staging_dir: Path,
    quotas: dict[str, int],
    al_arm: str,
    weak_arm: str,
    semantic_hits: Callable[[list[str]], list[tuple[str, list[str]]]],
    semantic_queries: list[str],
) -> pd.DataFrame:
    """Build the curated frame manifest and write it (+ the rsync filelist) to disk.

    Every returned token resolves to a ``filename`` via ``samples.parquet`` and must be
    channel CAM_FRONT — either failing raises ``ValueError`` naming the token, since a
    manifest row that ``demo infer``/rsync cannot locate or that is the wrong camera is
    worse than not curating it.

    ``semantic_hits`` returns an ORDERED ``list[(token, bucket_labels)]`` — hits are
    expected to already be ranked by relevance, so curation truncates to quota in the
    given order rather than re-sorting (token order is not a proxy for relevance).
    """
    unknown_quotas = set(quotas) - _KNOWN_BUCKETS
    if unknown_quotas:
        raise ValueError(f"curation: unknown quota keys: {sorted(unknown_quotas)}")

    processed_dir = Path(processed_dir)
    al_dir = Path(al_dir)
    staging_dir = Path(staging_dir)

    failures = pd.read_parquet(_require(al_dir / "failures.parquet"))
    samples = pd.read_parquet(_require(processed_dir / "samples.parquet"))
    canbus = pd.read_parquet(_require(processed_dir / "canbus.parquet"))
    al_arm_tokens = pd.read_parquet(_require(al_dir / f"{al_arm}.parquet"))
    weak_arm_tokens = pd.read_parquet(_require(al_dir / f"{weak_arm}.parquet"))
    accepted = pd.read_parquet(_require(al_dir / f"{weak_arm}_accepted.parquet"))

    # is_hard_braking is `bool | null` (docs/DATA.md:166): null means "no CAN window
    # to evaluate", not True/False — ingestion writes None, not False, when
    # has_canbus is False. A plain boolean mask over a column containing None raises
    # ("Cannot mask with non-boolean array containing NA"); null is not-hard-braking
    # for curation purposes, so it's coerced explicitly rather than silently NaN-ing.
    # (`.where(notna, False).astype(bool)` instead of `.fillna(False).astype(bool)` —
    # the latter is semantically identical but pandas 2.x emits a FutureWarning about
    # its implicit object->bool downcast on a fillna result.)
    _raw_hard_braking = canbus["is_hard_braking"]
    hard_braking_mask = _raw_hard_braking.where(_raw_hard_braking.notna(), False).astype(bool)

    simple_buckets: dict[str, list[str]] = {}

    simple_buckets["night_failure"] = _top_tokens(
        failures[failures["is_night"]],
        score_col="failure_score",
        quota=quotas.get("night_failure", 0),
    )
    simple_buckets["day_failure"] = _top_tokens(
        failures[~failures["is_night"]],
        score_col="failure_score",
        quota=quotas.get("day_failure", 0),
    )
    _warn_if_underfilled(
        "night_failure", simple_buckets["night_failure"], quotas.get("night_failure", 0)
    )
    _warn_if_underfilled(
        "day_failure", simple_buckets["day_failure"], quotas.get("day_failure", 0)
    )

    # Hard braking: canbus events (keyed by sample_token) joined to their CAM_FRONT
    # sample_data_token — no inherent "score", so tokens sort ascending alone.
    hard_quota = quotas.get("hard_braking", 0)
    if hard_quota > 0:
        cam_front = samples.loc[
            samples["channel"] == "CAM_FRONT", ["sample_token", "sample_data_token"]
        ]
        hard_braking_tokens = (
            canbus[hard_braking_mask]
            .merge(cam_front, on="sample_token", how="inner")["sample_data_token"]
            .drop_duplicates()
            .sort_values(kind="stable")
        )
        simple_buckets["hard_braking"] = list(hard_braking_tokens.head(hard_quota))
    else:
        simple_buckets["hard_braking"] = []
    _warn_if_underfilled("hard_braking", simple_buckets["hard_braking"], hard_quota)

    # Weak supervision: accepted/rejected split of the weak arm's mined pool — computed
    # before al_selected so al_selected can exclude tokens already claimed here. With
    # al_arm == weak_arm (today's config), all three would otherwise slice the same
    # token-ascending prefix of the same pool, wasting quota on redundant tokens.
    accepted_tokens = set(accepted["sample_data_token"])
    weak_pool = sorted(weak_arm_tokens["sample_data_token"])
    weak_accepted_quota = quotas.get("weak_accepted", 0)
    weak_rejected_quota = quotas.get("weak_rejected", 0)
    accepted_sorted = [t for t in weak_pool if t in accepted_tokens]
    rejected_sorted = [t for t in weak_pool if t not in accepted_tokens]
    simple_buckets["weak_accepted"] = (
        accepted_sorted[:weak_accepted_quota] if weak_accepted_quota > 0 else []
    )
    simple_buckets["weak_rejected"] = (
        rejected_sorted[:weak_rejected_quota] if weak_rejected_quota > 0 else []
    )
    _warn_if_underfilled(
        "weak_accepted", simple_buckets["weak_accepted"], weak_accepted_quota
    )
    _warn_if_underfilled(
        "weak_rejected", simple_buckets["weak_rejected"], weak_rejected_quota
    )

    weak_claimed = set(simple_buckets["weak_accepted"]) | set(simple_buckets["weak_rejected"])
    al_quota = quotas.get("al_selected", 0)
    al_pool = [t for t in sorted(al_arm_tokens["sample_data_token"]) if t not in weak_claimed]
    simple_buckets["al_selected"] = al_pool[:al_quota] if al_quota > 0 else []
    _warn_if_underfilled("al_selected", simple_buckets["al_selected"], al_quota)

    # Clean success: no false negatives, no low-conf hits, a non-trivial GT count —
    # lowest failure_score first (the "cleanest" frames), not highest.
    clean_mask = (
        (failures["n_fn"] == 0) & (failures["n_low_conf"] == 0) & (failures["n_gt"] >= 3)
    )
    clean_quota = quotas.get("clean_success", 0)
    simple_buckets["clean_success"] = _top_tokens(
        failures[clean_mask], score_col="failure_score", quota=clean_quota, ascending=True
    )
    _warn_if_underfilled("clean_success", simple_buckets["clean_success"], clean_quota)

    token_buckets: dict[str, set[str]] = {}
    for bucket_name, tokens in simple_buckets.items():
        for token in tokens:
            token_buckets.setdefault(token, set()).add(bucket_name)

    # Semantic: an injected callable returning an ORDERED [(token, bucket labels)] —
    # truncated in the given (assumed relevance-ranked) order, not re-sorted.
    semantic_quota = quotas.get("semantic", 0)
    semantic_hit_list = semantic_hits(semantic_queries) if semantic_quota > 0 else []
    semantic_selected = semantic_hit_list[:semantic_quota]
    for token, bucket_labels in semantic_selected:
        token_buckets.setdefault(token, set()).update(bucket_labels)
    _warn_if_underfilled(
        "semantic", [token for token, _ in semantic_selected], semantic_quota
    )

    val_tokens = set(failures["sample_data_token"])
    failure_lookup = failures.set_index("sample_data_token")
    samples_lookup = samples.set_index("sample_data_token")

    rows: list[dict[str, Any]] = []
    for token in sorted(token_buckets):
        if token not in samples_lookup.index:
            raise ValueError(f"curation: token {token!r} has no filename in samples.parquet")
        channel = samples_lookup.loc[token, "channel"]
        if channel != "CAM_FRONT":
            raise ValueError(
                f"curation: token {token!r} is channel {channel!r}, not CAM_FRONT"
            )
        row: dict[str, Any] = {
            "sample_data_token": token,
            "split": "val" if token in val_tokens else "train_pool",
            "curation_buckets": sorted(token_buckets[token]),
            "filename": samples_lookup.loc[token, "filename"],
            # Sourced from samples.parquet for every row (val and train_pool alike) —
            # non-null for every token, unlike the failure-ledger stats below which
            # only exist for val tokens. failures.parquet's own scene_name/is_night
            # columns are not used here for exactly that reason (they would be
            # missing/NaN for train_pool rows, turning these columns object-dtype
            # with None holes — the same class of bug C1 fixes for canbus, just for
            # the manifest's own output instead of an input; the reviewer verified
            # samples.parquet's scene_name agrees with failures.parquet's on every
            # val token, so this is a like-for-like source swap).
            "scene_name": samples_lookup.loc[token, "scene_name"],
            "is_night": bool(samples_lookup.loc[token, "is_night"]),
            "is_rain": bool(samples_lookup.loc[token, "is_rain"]),
        }
        if token in failure_lookup.index:
            for col in (*_INT_STAT_COLUMNS, *_FLOAT_STAT_COLUMNS):
                row[col] = failure_lookup.loc[token, col]
        rows.append(row)

    manifest = pd.DataFrame(rows, columns=list(_MANIFEST_COLUMNS))
    manifest["is_night"] = manifest["is_night"].astype(bool)
    manifest["is_rain"] = manifest["is_rain"].astype(bool)
    for col in _INT_STAT_COLUMNS:
        manifest[col] = manifest[col].astype("Int64")
    for col in _FLOAT_STAT_COLUMNS:
        manifest[col] = manifest[col].astype("Float64")

    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(staging_dir / "frame_manifest.parquet")
    filelist_text = "".join(f"{name}\n" for name in manifest["filename"])
    (staging_dir / "rsync_filelist.txt").write_text(filelist_text)

    logger.info(
        "demo curate: %d tokens (%d val, %d train_pool) -> %s",
        len(manifest),
        int((manifest["split"] == "val").sum()) if len(manifest) else 0,
        int((manifest["split"] == "train_pool").sum()) if len(manifest) else 0,
        staging_dir,
    )
    return manifest
