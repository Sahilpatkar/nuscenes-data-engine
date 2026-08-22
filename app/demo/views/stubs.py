"""Placeholder page(s) that ship the six-page navigation before their phase lands.

Phase 7 replaced the Weak Supervision stub with the real page (views/
weak_supervision.py); only the Phase-8 chat replay is still a stub.
"""

from __future__ import annotations

import streamlit as st


def _stub(title: str, purpose: str, phase: int) -> None:
    st.title(title)
    st.info(f"{purpose}  \n\n_Ships in Phase {phase} of the demo build._")


def chat_replay() -> None:
    _stub(
        "Ask the Dataset (recorded)",
        "Recorded question-and-answer sessions with the dataset chat agent, with SQL "
        "steps and retrieved frames.",
        8,
    )
