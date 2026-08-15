"""Overview — the page that explains the project in seconds (DEMO_PLAN.md §1)."""

from __future__ import annotations

import streamlit as st
from render import metric_cards

from data import hero_path, load_overview


def render() -> None:
    st.title("nuScenes Perception Data Engine")
    st.markdown(
        "> A system for training, evaluating, diagnosing, and improving "
        "autonomous-driving perception models using active learning, semantic "
        "search, knowledge graphs, CAN-bus context, and weak supervision."
    )
    metrics = load_overview()
    scale = metrics["scale"]
    st.subheader("Scale")
    metric_cards(
        [
            ("Camera keyframes", f"{scale['images']:,}"),
            ("2D boxes", f"{scale['boxes_2d']:,}"),
            ("3D object observations", f"{scale['objects_3d']:,}"),
            ("CAN-bus rows", f"{scale['canbus_rows']:,}"),
        ]
    )
    st.subheader("Headline results")
    results = metrics["results"]
    flagship = metrics["flagship"]
    weak = results["weak_retention"]
    retention = weak["headline"]
    # "Retention" is ambiguous: the weak-sup summaries also publish a *verifier*
    # retention (fraction of candidate pseudo-labels the verifier kept — a different
    # quantity, ~64% for the random arm). This card is the training-outcome quantity:
    # how much of the GT-trained arm's mAP gain a weak-labeled arm captures.
    retention_text = "n/a" if retention is None else f"{retention:.0%}"
    metric_cards(
        [
            ("Best night gain", f"+{results['best_night_delta']:.4f}"),
            ("CAN speed vs ego-motion", f"r = {metrics['can_speed_r']:.3f}"),
            ("Weak-sup share of GT gain", retention_text),
            ("Graph = SQL flagship", f"{flagship['sql']} = {flagship['cypher']}"),
        ]
    )
    other_pairs = ", ".join(
        f"`{arm}` ({weak['by_base_arm'][arm]:.0%})"
        for arm in weak["by_base_arm"]
        if arm != weak["headline_arm"]
    )
    other_pairs_note = (
        f"the {other_pairs} pair is carried in `demo_data/overview_metrics.json`; "
        "the Weak Supervision page presents it in Phase 7."
        if other_pairs
        else "no other weak/GT pairs are in this build."
    )
    st.caption(
        f"Best night arm: `{results['best_night_arm']}` · flagship Cypher twin "
        f"sourced from {flagship['cypher_source']} until the live-graph export lands. "
        f"The card above is {retention_text} of the ground-truth mAP gain, for the "
        f"`{weak['headline_arm']}` pair (the documented headline); {other_pairs_note}"
    )
    if hero_path().is_file():
        st.subheader("The model at work")
        st.image(str(hero_path()), caption="Validation-batch predictions (baseline yolov8n)")
    st.divider()
    st.caption(
        "Every number above is derived from the pipeline's real artifacts at "
        "export time. Interactive pages label recorded outputs as recorded; "
        "the live stack (training, search, chat) runs locally via docker compose."
    )
