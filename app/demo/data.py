"""Cached loaders over the committed demo_data/ package — the app's only data source."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# DEMO_DATA_DIR lets tests point the app at a tmp-built package (Streamlit Cloud and
# the real app never set it, so the committed demo_data/ stays the default).
_DEFAULT_DEMO_DATA = Path(__file__).resolve().parents[2] / "demo_data"
DEMO_DATA = Path(os.environ.get("DEMO_DATA_DIR", str(_DEFAULT_DEMO_DATA)))


# The one note every Phase-7 section shows when the committed package predates the
# tables it reads (consolidated review M6 -- it was a private constant duplicated
# verbatim in both Phase-7 views).
STALE_PACKAGE_NOTE = "needs demo_data >= 0.6 (the Phase-7 tables) — rerun `demo build`"


def package_missing() -> bool:
    return not (DEMO_DATA / "manifest.json").is_file()


@st.cache_data
def load_manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DEMO_DATA / "manifest.json").read_text())
    return data


@st.cache_data
def load_overview() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DEMO_DATA / "overview_metrics.json").read_text())
    return data


@st.cache_data
def load_al_results() -> pd.DataFrame:
    return pd.read_parquet(DEMO_DATA / "active_learning_results.parquet")


@st.cache_data
def load_weaksup() -> pd.DataFrame:
    return pd.read_parquet(DEMO_DATA / "weak_supervision_results.parquet")


@st.cache_data
def load_frame_manifest() -> pd.DataFrame:
    return pd.read_parquet(DEMO_DATA / "frame_manifest.parquet")


@st.cache_data
def load_gt_boxes() -> pd.DataFrame:
    return pd.read_parquet(DEMO_DATA / "gt_boxes.parquet")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(DEMO_DATA / "predictions.parquet")


def _events_path() -> Path:
    return DEMO_DATA / "scenario_events.parquet"


def events_available() -> bool:
    """``scenario_events.parquet`` shipped starting package_version 0.4 (Phase 5) --
    a package built by an older ``demo build`` (or a stale committed demo_data/)
    won't have it. The Scenario Search page checks this BEFORE calling
    ``load_events()`` and shows a directive ``st.error`` instead of letting a bare
    ``FileNotFoundError`` (or a downstream ``rank_events`` ``ValueError`` on a
    frame missing every ``preset_rank_*`` column) surface as a traceback
    (item 4, consolidated review).
    """
    return _events_path().is_file()


@st.cache_data
def load_events() -> pd.DataFrame:
    return pd.read_parquet(_events_path())


# semsearch's schema (build.py's _include_semsearch / exporters.export_semsearch):
# query, rank, sample_data_token, score, k.
_EMPTY_SEMSEARCH_COLUMNS = ["query", "rank", "sample_data_token", "score", "k"]


@st.cache_data
def load_semsearch() -> pd.DataFrame:
    """The recorded semantic-search gallery, or an empty frame when absent.

    `demo build` ships without `semantic_search_results.parquet` when `demo
    semsearch` hasn't been run (manifest.json's validation.semsearch == "absent" --
    mirrors curation's own honest-absence handling, build.py::_include_semsearch) --
    a fresh clone or a torch-free machine must still render this page, just with an
    empty gallery section rather than a crash.
    """
    path = DEMO_DATA / "semantic_search_results.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_SEMSEARCH_COLUMNS)
    return pd.read_parquet(path)


def _subgraphs_dir() -> Path:
    return DEMO_DATA / "graph_subgraphs"


def subgraphs_available() -> bool:
    """Whether `demo subgraphs` was staged for this package (build.py's `demo
    build` copies `graph_subgraphs/<preset>.json` for all six presets together, or
    none at all -- a partial staging fails the build rather than shipping, see
    tests/test_demo_export.py) -- mirrors `events_available`'s file-existence
    check so the Scenario page can show an honest absent note instead of a bare
    FileNotFoundError.
    """
    return _subgraphs_dir().is_dir()


@st.cache_data
def load_subgraphs(preset: str) -> dict[str, Any] | None:
    """One preset's `graph_subgraphs/<preset>.json` (subgraph_export.export_
    subgraphs' `{preset, count_cypher, sql_count, cypher_count, parity, note?,
    events}` shape), or `None` when the file is absent -- either the whole package
    has no subgraphs (`subgraphs_available()` is False) or (defensively) this one
    preset's file specifically is missing.
    """
    path = _subgraphs_dir() / f"{preset}.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# --- Phase 7: the Active Learning tables ------------------------------------------
#
# al_communities.parquet and al_exemplars.json are written by every `demo build`
# from package_version 0.6 on; al_selection_explain.parquet /
# al_explain_validation.json only when `demo al-explain` (Neo4j + GDS + LanceDB)
# staged them, which a fresh clone never has (manifest.json's validation.al_explain
# == "absent"). All four load through load_semsearch's graceful-absence pattern --
# an older/stale committed package must render the page, just with the sections
# that have no data saying so, rather than a bare FileNotFoundError.

_EMPTY_COMMUNITY_COLUMNS = [
    "community", "size", "night_members", "mass",
    "quota_graph_rate", "quota_graph_rate_night", "is_backfill",
]

_EMPTY_EXPLAIN_COLUMNS = [
    "sample_data_token", "arm", "is_night", "scene_name", "community",
    "community_size", "community_night_members", "community_mass",
    "community_mass_rank", "community_quota", "degree", "degree_rank_in_community",
    "pick_pass", "n_failures_routed", "mass_routed",
]


@st.cache_data
def load_al_communities() -> pd.DataFrame:
    """The 97 Louvain communities with each arm's quota, or an empty frame."""
    path = DEMO_DATA / "al_communities.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_COMMUNITY_COLUMNS)
    return pd.read_parquet(path)


@st.cache_data
def load_al_exemplars() -> dict[str, Any]:
    """`al_exemplars.json`'s {arm, baseline, tokens}, or empty defaults."""
    path = DEMO_DATA / "al_exemplars.json"
    if not path.is_file():
        return {"arm": None, "baseline": None, "tokens": []}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _al_explain_path() -> Path:
    return DEMO_DATA / "al_selection_explain.parquet"


def al_explain_available() -> bool:
    """Whether `demo al-explain`'s per-frame selection facts shipped in this
    package -- checked BEFORE the "why was this frame selected?" panel claims
    anything, so an absent group draws the honest note instead (spec §3d)."""
    return _al_explain_path().is_file()


@st.cache_data
def load_al_explain() -> pd.DataFrame:
    path = _al_explain_path()
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_EXPLAIN_COLUMNS)
    return pd.read_parquet(path)


@st.cache_data
def load_al_explain_validation() -> dict[str, Any]:
    """The re-derivation's own validation record (night floor, pass sizes, GDS
    version), or an empty dict when the group is absent."""
    path = DEMO_DATA / "al_explain_validation.json"
    if not path.is_file():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# --- Phase 7: the Weak Supervision tables -----------------------------------------
#
# weak_loss_decomposition.parquet and weak_verifier_by_class.parquet ship with every
# `demo build` from package_version 0.6 on; weak_labels.parquet and vlm_counts.parquet
# only when the curation group was staged (build.py's `_include_curation` branch --
# they are the curated weak frames' pseudo boxes and VLM count votes). All four load
# through load_semsearch's graceful-absence pattern, so an older/stale committed
# package renders the page with the sections that have no data saying so, rather than
# a bare FileNotFoundError.

_EMPTY_WEAK_LOSS_COLUMNS = [
    "base_arm", "gt_gain", "weak_gt_gain", "weak_gain", "retention",
    "dropped_frame_cost", "dropped_frame_share", "label_cost", "label_share", "headline",
]

_EMPTY_WEAK_BY_CLASS_COLUMNS = [
    "arm", "category_group", "n_rejected_disagreements",
    "n_accepted_mutual_zero", "mutual_zero_share",
]

_EMPTY_WEAK_LABELS_COLUMNS = [
    "sample_data_token", "category_group", "x_min", "y_min", "x_max", "y_max", "score",
]

_EMPTY_VLM_COUNTS_COLUMNS = [
    "sample_data_token", "parse_status", "label_confidence", "vlm_time_of_day",
    "vlm_weather", "vlm_car", "vlm_truck", "vlm_bus", "vlm_pedestrian", "vlm_bicycle",
    "gt_car", "gt_truck", "gt_bus", "gt_pedestrian", "gt_bicycle",
]

_EMPTY_VLM_COUNT_BUCKETS_COLUMNS = ["model", "bucket", "n", "mae"]


@st.cache_data
def load_weak_loss() -> pd.DataFrame:
    """Each weak/GT pair's gain split (retained / dropped-frame / label cost)."""
    path = DEMO_DATA / "weak_loss_decomposition.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_WEAK_LOSS_COLUMNS)
    return pd.read_parquet(path)


@st.cache_data
def load_weak_by_class() -> pd.DataFrame:
    """The verifier's per-class rejected disagreements and mutual-zero shares."""
    path = DEMO_DATA / "weak_verifier_by_class.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_WEAK_BY_CLASS_COLUMNS)
    return pd.read_parquet(path)


@st.cache_data
def load_weak_labels() -> pd.DataFrame:
    """The curated accepted frames' verified pseudo boxes (native 1600x900 coords).

    An accepted frame with no rows here is the mutual-zero case (the verifier
    accepted zero VLM-counted objects against zero detector boxes -- the VLM emits
    per-class counts, never boxes), not missing data --
    ``filters.weak_frame_summary`` is what turns that absence into a label.
    """
    path = DEMO_DATA / "weak_labels.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_WEAK_LABELS_COLUMNS)
    return pd.read_parquet(path)


@st.cache_data
def load_vlm_counts() -> pd.DataFrame:
    """One row per curated frame carrying a weak verdict: the VLM's per-class counts
    next to GT's.

    Scoped to ``frame_manifest.weak_verdict`` (78 frames in the shipped package),
    which is exactly what the page's accepted/rejected tabs render -- consolidated
    review I1; it used to be scoped to the two curation buckets (40), leaving 38
    rendered frames without a row.
    """
    path = DEMO_DATA / "vlm_counts.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_VLM_COUNTS_COLUMNS)
    return pd.read_parquet(path)


@st.cache_data
def load_vlm_count_buckets() -> pd.DataFrame:
    """The VLM's count MAE by GT-count bucket, per model (package >= 0.8).

    ``n`` counts frame x CLASS pairs, not frames -- the builder pools all ten count
    fields into one bucket table (``eval_count_buckets``), so any caption over this
    table has to say "frame-class pairs".

    Absent from every package built before 0.8 (and from any built on a machine
    without ``data/autolabel/labels.parquet``), so it follows the same graceful-
    absence shape as the other optional tables: the empty frame with the four
    columns, and the page draws its own note.
    """
    path = DEMO_DATA / "vlm_count_buckets.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=_EMPTY_VLM_COUNT_BUCKETS_COLUMNS)
    return pd.read_parquet(path)


# --- the recorded chat session ----------------------------------------------------
#
# chat_replays.json + chat_replay_summary.json ship only when `demo chat-record`
# staged a recording (a paid, one-off run against a live provider) and `demo build`
# copied it (manifest.json's validation.chat_replay == "included"). A fresh clone --
# or any package built by an older `demo build` -- carries neither file, so both
# load through load_semsearch's graceful-absence pattern and the "Ask the Dataset"
# page draws its honest absent note instead of a bare FileNotFoundError.

_CHAT_REPLAY_FILES = ("chat_replays.json", "chat_replay_summary.json")


def chat_replay_available() -> bool:
    """Whether this package carries a recorded chat session.

    BOTH files, checked on disk: the page reads the recording's model and date out
    of the summary beside the replays themselves, and `demo build` only ever copies
    the pair (a half-staged group fails the build). The check is on the FILES, not
    on manifest.json's ``validation.chat_replay`` -- a package built before that key
    existed has no key to read, and the files are what the page actually opens.
    """
    return all((DEMO_DATA / name).is_file() for name in _CHAT_REPLAY_FILES)


@st.cache_data
def load_chat_replays() -> list[dict[str, Any]]:
    """The recorded replays (showcase first, then graded), or an empty list."""
    path = DEMO_DATA / "chat_replays.json"
    if not path.is_file():
        return []
    data: list[dict[str, Any]] = json.loads(path.read_text())
    return data


@st.cache_data
def load_chat_replay_summary() -> dict[str, Any]:
    """The recording's own summary (model, provider, recorded_at, counts, whether
    the search engine and the graph were available), or an empty dict."""
    path = DEMO_DATA / "chat_replay_summary.json"
    if not path.is_file():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def hero_path() -> Path:
    return DEMO_DATA / "sample_frames" / "hero.jpg"


def crop_path(token: str) -> Path:
    return DEMO_DATA / "sample_frames" / "crops" / f"{token}.jpg"


def frame_image_path(token: str) -> Path | None:
    """The best available gallery image for a token -- thumb first (galleries are
    grids of small images), crop as the fallback, ``None`` when the package carries
    neither.

    Shared by the Active Learning and Weak Supervision galleries (consolidated
    review M6 -- it was duplicated verbatim in both views).
    """
    thumb = thumb_path(token)
    if thumb.is_file():
        return thumb
    crop = crop_path(token)
    return crop if crop.is_file() else None


def thumb_path(token: str) -> Path:
    return DEMO_DATA / "sample_frames" / "thumbs" / f"{token}.jpg"
