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


def hero_path() -> Path:
    return DEMO_DATA / "sample_frames" / "hero.jpg"
