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


@st.cache_data
def load_events() -> pd.DataFrame:
    return pd.read_parquet(DEMO_DATA / "scenario_events.parquet")


# semsearch's schema (build.py's _include_semsearch / exporters.export_semsearch):
# query, rank, sample_data_token, score.
_EMPTY_SEMSEARCH_COLUMNS = ["query", "rank", "sample_data_token", "score"]


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


def hero_path() -> Path:
    return DEMO_DATA / "sample_frames" / "hero.jpg"


def crop_path(token: str) -> Path:
    return DEMO_DATA / "sample_frames" / "crops" / f"{token}.jpg"


def thumb_path(token: str) -> Path:
    return DEMO_DATA / "sample_frames" / "thumbs" / f"{token}.jpg"
