"""Shared rendering primitives for the demo pages."""

from __future__ import annotations

import streamlit as st


def metric_cards(items: list[tuple[str, str]], per_row: int = 4) -> None:
    """A row of st.metric cards: [(label, value), ...]."""
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        for col, (label, value) in zip(st.columns(len(chunk)), chunk, strict=True):
            col.metric(label, value)


def story_arrows(steps: list[tuple[str, str]]) -> None:
    """The vertical Problem -> ... -> Result narrative (DEMO_PLAN.md §9)."""
    for index, (title, body) in enumerate(steps):
        st.markdown(f"**{title}**  \n{body}")
        if index < len(steps) - 1:
            st.markdown("<div style='text-align:center'>&#8595;</div>", unsafe_allow_html=True)


def recorded_banner(text: str) -> None:
    st.caption(f":material/history: {text}")
