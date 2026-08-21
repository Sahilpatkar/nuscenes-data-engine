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


def hero_path() -> Path:
    return DEMO_DATA / "sample_frames" / "hero.jpg"


def crop_path(token: str) -> Path:
    return DEMO_DATA / "sample_frames" / "crops" / f"{token}.jpg"


def thumb_path(token: str) -> Path:
    return DEMO_DATA / "sample_frames" / "thumbs" / f"{token}.jpg"
