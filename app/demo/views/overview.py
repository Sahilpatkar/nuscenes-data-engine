"""Overview — the page that explains the project in seconds (DEMO_PLAN.md §1)."""

from __future__ import annotations

import streamlit as st
from PIL import Image
from render import draw_overlay, metric_cards

from data import crop_path, hero_path, load_gt_boxes, load_overview, load_predictions

# Phase 3: docs/superpowers/specs/2026-08-18-demo-phase3-design.md §4 -- fixed
# hero caption, always the baseline model (the hero is a baseline-vs-GT story,
# not a per-viewer model choice). This is a description of THIS SPECIFIC
# hero_token's content (configs/demo.yaml hero.token), not a generic label --
# picking a different token means rewriting this string to match (see
# docs/DEMO.md's "Picking the hero token" section). Ground truth for the token
# picked in the Task-5 operational run (final-review correction): baseline
# misses a shadowed car AND a pedestrian; graph_rate_night recovers ONLY the
# pedestrian (a low-confidence match, still a hit per the matching rules) -- the
# far car defeats all three models. The original caption ("the orange dashed
# box is a miss ... catches", singular) was wrong on both counts: it claimed a
# single miss where two dashed boxes render, and implied the visually-dominant
# one (the far car) is the one that gets caught, when it's the one nothing
# catches.
_HERO_CAPTION = (
    "baseline misses a shadowed car and a pedestrian; the night-targeted "
    "retrain recovers the pedestrian (a low-confidence hit) — the far car "
    "defeats all three models. Explore more in the Failure Explorer."
)


def _hero_overlay(hero_token: str) -> Image.Image | None:
    """The hero crop rendered live through draw_overlay, or None if the token's
    boxes are absent (a curated-but-not-fully-enriched package, or a hero token
    picked before Task 1's gt_boxes enrichment shipped) -- callers fall back to
    the plain crop in that case rather than showing a boxless "overlay"."""
    crop = crop_path(hero_token)
    if not crop.is_file():
        return None
    gt = load_gt_boxes()
    gt_token = gt.loc[gt["sample_data_token"] == hero_token]
    if "below_visibility_min" in gt_token.columns:
        gt_token = gt_token.loc[~gt_token["below_visibility_min"]]
    if "matched_baseline" not in gt_token.columns:
        return None
    gt_token = gt_token.rename(columns={"matched_baseline": "matched"})
    preds = load_predictions()
    preds_token = preds.loc[
        (preds["sample_data_token"] == hero_token) & (preds["model"] == "baseline")
    ]
    if gt_token.empty and preds_token.empty:
        return None
    return draw_overlay(Image.open(crop), gt_token, preds_token, mode="overlay", scale=0.6)


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
    # One decimal place here (39.4%, not 39%) to match the figure as documented in
    # docs/DEMO.md — the two must agree on the same rounding for the same number.
    other_pairs = ", ".join(
        f"`{arm}` ({weak['by_base_arm'][arm]:.1%})"
        for arm in weak["by_base_arm"]
        if arm != weak["headline_arm"]
    )
    other_pairs_note = (
        f"The {other_pairs} pair is carried in `demo_data/overview_metrics.json`. "
        "The Weak Supervision page presents it in Phase 7."
        if other_pairs
        else "No other weak/GT pairs are in this build."
    )
    st.caption(
        f"Best night arm: `{results['best_night_arm']}` · flagship Cypher twin "
        f"sourced from {flagship['cypher_source']} until the live-graph export lands."
    )
    st.caption(
        f"The card above is {retention_text} of the ground-truth mAP gain for the "
        f"`{weak['headline_arm']}` pair (the documented headline). {other_pairs_note}"
    )
    if hero_path().is_file():
        st.subheader("The model at work")
        hero_token = metrics.get("hero_token")
        overlay = _hero_overlay(hero_token) if hero_token else None
        if overlay is not None:
            st.image(overlay, caption=_HERO_CAPTION)
        else:
            st.image(str(hero_path()), caption="Validation-batch predictions (baseline yolov8n)")
    st.divider()
    st.caption(
        "Every number above is derived from the pipeline's real artifacts at "
        "export time. Interactive pages label recorded outputs as recorded; "
        "the live stack (training, search, chat) runs locally via docker compose."
    )
