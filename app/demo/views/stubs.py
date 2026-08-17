"""Placeholder pages that ship the five-page navigation before their phase lands."""

from __future__ import annotations

import streamlit as st


def _stub(title: str, purpose: str, phase: int) -> None:
    st.title(title)
    st.info(f"{purpose}  \n\n_Ships in Phase {phase} of the demo build._")


def failures() -> None:
    _stub(
        "Failure Explorer",
        "See where and why the detector fails — GT vs prediction overlays, filterable "
        "by condition, class, and failure type.",
        3,
    )


def scenarios() -> None:
    _stub(
        "Scenario Search",
        "Graph + CAN-bus driving-scenario queries ('hard braking near pedestrians') "
        "with a synchronized event viewer.",
        5,
    )


def active_learning() -> None:
    _stub(
        "Active Learning",
        "How the system chooses training data — 13 arms, and why graph_rate_night "
        "won the night.",
        7,
    )


def weak_supervision() -> None:
    _stub(
        "Weak Supervision",
        "What VLM auto-labels kept — and lost — versus ground truth.",
        7,
    )


def chat_replay() -> None:
    _stub(
        "Ask the Dataset (recorded)",
        "Recorded question-and-answer sessions with the dataset chat agent, with SQL "
        "steps and retrieved frames.",
        8,
    )
