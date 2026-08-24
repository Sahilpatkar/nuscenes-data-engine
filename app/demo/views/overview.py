"""Overview — the page that explains the project in seconds (DEMO_PLAN.md §1).

Phase 9a (spec §4): outcome-first. The page now leads with a CTA into the guided
tour and the loop strip, keeps only four flagship metrics in the headline row, and
folds everything else into three collapsed expanders — the headline cards'
provenance ("Where these numbers come from", directly under them), the dataset-scale
cards and CAN check, and the architecture restatement. Nothing on the page
disappears, it just stops competing with the loop for the first screenful.
"""

from __future__ import annotations

from typing import Any

import nav
import pandas as pd
import streamlit as st
from filters import visible_gt
from PIL import Image
from render import draw_overlay, legend, loop_breadcrumb, metric_cards, provenance

from data import (
    crop_path,
    hero_path,
    load_al_exemplars,
    load_al_results,
    load_gt_boxes,
    load_overview,
    load_predictions,
)

# Phase 3: docs/superpowers/specs/2026-08-18-demo-phase3-design.md §4 -- fixed
# hero caption, always the baseline model (the hero is a baseline-vs-GT story,
# not a per-viewer model choice). This is a description of THIS SPECIFIC
# hero_token's content (configs/demo.yaml hero.token), not a generic label --
# picking a different token means rewriting this string to match (see
# docs/DEMO.md's "Picking the hero token" section). Ground truth for the token
# picked in the Task-5 operational run (final-review correction): baseline
# misses a shadowed car AND a pedestrian; graph_rate_night recovers ONLY the
# pedestrian (a low-confidence match, still a hit per the matching rules) -- the
# far car defeats every model in the package. The original caption ("the orange
# dashed box is a miss ... catches", singular) was wrong on both counts: it
# claimed a single miss where two dashed boxes render, and implied the
# visually-dominant one (the far car) is the one that gets caught, when it's the
# one nothing catches.
#
# "five", not "three" (item I2, Phase 9b consolidated review): the package this
# phase ships carries FIVE models, and the claim was re-verified against them --
# all five miss the car, and graph_rate_night alone claims the pedestrian, at
# conf 0.135 (a low-confidence hit). A sentence that counts models has to be
# re-counted whenever `demo infer` changes how many there are.
HERO_CAPTION = (
    "baseline misses a shadowed car and a pedestrian; the night-targeted "
    "retrain recovers the pedestrian (a low-confidence hit) — the far car "
    "defeats all five models. Explore more in the Failure Explorer."
)


def _baseline_arm() -> str:
    """The arm table's own name for the un-mined checkpoint, read from
    ``al_exemplars.json`` (which ``demo build`` writes both names into) exactly as
    ``views/tour.py::_load`` reads it -- Phase 9a review M5, where this page instead
    hardcoded "baseline" and would have silently compared against nothing on a
    package that named its baseline anything else. "baseline" remains the fallback
    for a package with no ``al_exemplars.json`` at all (the loader's own default is
    ``None``), which is what every package built so far actually calls it."""
    return str(load_al_exemplars().get("baseline") or "baseline")


# docs/PROJECT.md §2 "Architecture", restated (and only restated -- no new
# claims): the two-machine topology table, the component map as a compact bullet
# flow, and the CI sentence, verbatim in substance. This deployment (the public
# Streamlit app) runs none of it -- it only reads the committed demo_data/
# package -- which is why the drawer closes on that line rather than on the CI
# sentence.
_ARCHITECTURE_MARKDOWN = (
    "| | GPU server (\"TRINITY\", multi-node) | Infra machine (Mac, this repo's "
    "demo host) |\n"
    "|---|---|---|\n"
    "| Role | Compute: ingest, train, evaluate, embed, label, mine | Ops: "
    "registry UI, serving, monitoring, demo, chat |\n"
    "| Hardware | RTX 3080 Ti nodes (11.9 GB each) | Apple M4 Pro, 48 GB |\n"
    "| Writes | Plain files (Parquet, weights, `mlruns/`, LanceDB) | Docker "
    "stack, synced artifacts |\n"
    "| Sync | `scripts/gpu-run.sh` pushes the current branch, runs any CLI "
    "command remotely; results rsync back | `docker compose up` |\n"
    "\n"
    "- Ingestion → processed Parquet tables → training / evaluation / registry "
    "(MLflow) → serving (FastAPI) → Streamlit demo, with monitoring (Evidently "
    "drift) alongside\n"
    "- Data engine: 6a embed (SigLIP2 → LanceDB) · 6b VLM auto-labelling · 6c "
    "chat agent · 6d active learning\n"
    "- Analytics: DuckDB over the same Parquet tables\n"
    "\n"
    "CI (GitHub Actions) runs two jobs on every PR: **quality** (ruff + mypy "
    "strict + the torch-free test suite) and **smoke-train** (a 1-epoch CPU "
    "training run on a tiny fixture dataset).\n"
    "\n"
    "This deployment runs none of it — it reads `demo_data/` only."
)


def _hero_overlay(hero_token: str) -> Image.Image | None:
    """The hero crop rendered live through draw_overlay, or None if the token's
    boxes are absent (a curated-but-not-fully-enriched package, or a hero token
    picked before Task 1's gt_boxes enrichment shipped) -- callers fall back to
    the plain crop in that case rather than showing a boxless "overlay"."""
    crop = crop_path(hero_token)
    if not crop.is_file():
        return None
    gt_token = visible_gt(load_gt_boxes(), hero_token)
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


def _night_pedestrian_card(
    arms: pd.DataFrame, arm_name: str | None
) -> tuple[str, str, str] | None:
    """("Night pedestrian mAP50-95", "0.117", "+0.0345 vs baseline"), or None when
    either row is absent from this package's arm table or carries no
    night-pedestrian slice (an older results.json never wrote
    ``night.per_class``) -- the headline row omits the card entirely rather than
    showing "nan".

    A single figure plus a delta, not the "0.083 → 0.117" arrow: the arrow is
    13 characters and truncates in a 1-of-4 st.metric card at <=1200px (Phase 9a
    final review)."""
    if arm_name is None:
        return None
    base_rows = arms.loc[arms["arm"] == _baseline_arm()]
    arm_rows = arms.loc[arms["arm"] == arm_name]
    if base_rows.empty or arm_rows.empty:
        return None
    base_ped = base_rows.iloc[0].get("night_ped_map5095")
    arm_ped = arm_rows.iloc[0].get("night_ped_map5095")
    if base_ped is None or arm_ped is None or pd.isna(base_ped) or pd.isna(arm_ped):
        return None
    base_ped, arm_ped = float(base_ped), float(arm_ped)
    return (
        "Night pedestrian mAP50-95",
        f"{arm_ped:.3f}",
        f"{arm_ped - base_ped:+.4f} vs baseline",
    )


def _loop_beats(results: dict[str, Any], arms: pd.DataFrame) -> list[str | None]:
    """The loop strip's four one-number beats, one per ``LOOP_STAGES`` entry, in
    ``st.columns(4)`` order -- ``None`` for a beat whose underlying value is NA
    rather than a card reading "nan" (an arm table without the mined-set
    composition columns, or missing a row for the best-night arm)."""
    beats: list[str | None] = [None, None, None, None]
    base_rows = arms.loc[arms["arm"] == _baseline_arm()]
    arm_name = results.get("best_night_arm")
    arm_rows = arms.loc[arms["arm"] == arm_name] if arm_name is not None else arms.iloc[:0]

    if not base_rows.empty:
        base_row = base_rows.iloc[0]
        night, overall = base_row.get("night_map5095"), base_row.get("overall_map5095")
        if pd.notna(night) and pd.notna(overall):
            beats[0] = (
                f"**Detect weakness** — night mAP50-95 {float(night):.4f} vs "
                f"overall {float(overall):.4f}"
            )

    if not base_rows.empty and not arm_rows.empty:
        base_n, arm_n = base_rows.iloc[0].get("n_train_images"), arm_rows.iloc[0].get(
            "n_train_images"
        )
        n_scenes = arm_rows.iloc[0].get("n_scenes")
        if bool(pd.notna(base_n)) and bool(pd.notna(arm_n)):
            if bool(pd.notna(n_scenes)):
                n_mined = int(arm_n) - int(base_n)
                scenes = int(n_scenes)
                # Number/noun agreement, as in tour._result_data_added and
                # active_learning._strategy_triple.
                beats[1] = (
                    f"**Find useful data** — {n_mined:,} "
                    f"frame{'' if n_mined == 1 else 's'} across "
                    f"{scenes} scene{'' if scenes == 1 else 's'}"
                )
            beats[2] = f"**Retrain** — {int(base_n):,} → {int(arm_n):,} training images"

    best_night_delta = results.get("best_night_delta")
    if best_night_delta is not None and pd.notna(best_night_delta):
        beats[3] = f"**Measure impact** — night {float(best_night_delta):+.4f}"

    return beats


def render() -> None:
    st.title("nuScenes Perception Data Engine")
    st.markdown(
        "> A system for training, evaluating, diagnosing, and improving "
        "autonomous-driving perception models using active learning, semantic "
        "search, knowledge graphs, CAN-bus context, and weak supervision."
    )

    metrics = load_overview()
    results = metrics["results"]
    flagship = metrics["flagship"]
    weak = results["weak_retention"]
    retention = weak["headline"]
    # "Retention" is ambiguous: the weak-sup summaries also publish a *verifier*
    # retention (fraction of candidate pseudo-labels the verifier kept — a different
    # quantity, ~64% for the random arm). This card is the training-outcome quantity:
    # how much of the GT-trained arm's mAP gain a weak-labeled arm captures.
    retention_text = "n/a" if retention is None else f"{retention:.0%}"
    arms = load_al_results()

    # --- CTA: the default path into the guided tour -----------------------------
    if st.button("Explore a model failure →", key="overview_start_tour", type="primary"):
        st.session_state[nav.TOUR_STEP_KEY] = 0
        st.switch_page(nav.page("tour"))
    st.caption(
        "A 2-3 minute guided walk: night weakness → missed pedestrian → scenario "
        "mining → selected frames → retraining → improvement."
    )

    # --- loop strip ---------------------------------------------------------------
    loop_breadcrumb(
        None,
        caption="System loop — Detect weakness → Find useful data → Retrain → "
        "Measure impact",
    )
    beat_columns = st.columns(4)
    for column, beat in zip(beat_columns, _loop_beats(results, arms), strict=True):
        if beat:
            column.markdown(beat)

    # --- headline: exactly four flagship metrics -----------------------------------
    st.subheader("Headline results")
    can_card = ("CAN speed vs ego-motion", f"r = {metrics['can_speed_r']:.3f}")
    ped_card = _night_pedestrian_card(arms, results.get("best_night_arm"))
    metric_cards(
        [
            ("Best night gain", f"+{results['best_night_delta']:.4f}"),
            ped_card if ped_card is not None else can_card,
            ("Weak-sup share of GT gain", retention_text),
            ("Graph = SQL flagship", f"{flagship['sql']} = {flagship['cypher']}"),
        ]
    )
    provenance("recorded", "overview_metrics.json + active_learning_results.parquet")

    # Provenance, not architecture (Phase 9a review M1): these two captions say
    # where the four cards just above come from and which pair the retention figure
    # is about, so they belong under the cards -- collapsed, because a viewer who
    # takes the numbers at face value should not have to read past them, and one
    # who doesn't should not have to open a drawer labelled "Architecture" to find
    # the answer.
    with st.expander("Where these numbers come from"):
        # One decimal place here (39.4%, not 39%) to match the figure as documented in
        # docs/DEMO.md — the two must agree on the same rounding for the same number.
        other_pairs = ", ".join(
            f"`{arm}` ({weak['by_base_arm'][arm]:.1%})"
            for arm in weak["by_base_arm"]
            if arm != weak["headline_arm"]
        )
        other_pairs_note = (
            f"The {other_pairs} pair is carried in `demo_data/overview_metrics.json`. "
            "The Weak Supervision page presents it."
            if other_pairs
            else "No other weak/GT pairs are in this build."
        )
        # The flagship Cypher twin's provenance, stated as it actually is (item I5,
        # Phase 6 review). Since `demo subgraphs` + `demo build` compute that count
        # against the live graph, cypher_source reads "computed (neo4j, demo
        # subgraphs)" -- and the old fixed sentence then claimed the number was
        # "sourced from computed (neo4j, demo subgraphs) until the live-graph export
        # lands", which is both ungrammatical and false about an export that already
        # landed. A package built without that staging still ships the sourced value
        # (build.py::_include_subgraphs' absent path), and keeps the original wording.
        if str(flagship["cypher_source"]).startswith("computed"):
            flagship_note = (
                "flagship Cypher twin computed live against Neo4j at build time "
                f"(SQL {flagship['sql']} / Cypher {flagship['cypher']})."
            )
        else:
            flagship_note = (
                f"flagship Cypher twin sourced from {flagship['cypher_source']} until "
                "the live-graph export lands."
            )
        st.caption(f"Best night arm: `{results['best_night_arm']}` · {flagship_note}")
        # Named, not "the card above" (Phase 9a review M1): the caption no longer
        # sits directly under that card, and a positional reference to a metric row
        # is fragile even when it does.
        st.caption(
            f"The Weak-sup share of GT gain card is {retention_text} of the "
            f"ground-truth mAP gain for the `{weak['headline_arm']}` pair (the "
            f"documented headline). {other_pairs_note}"
        )

    if hero_path().is_file():
        st.subheader("The model at work")
        hero_token = metrics.get("hero_token")
        overlay = _hero_overlay(hero_token) if hero_token else None
        if overlay is not None:
            st.image(overlay, caption=HERO_CAPTION)
            # Only under the LIVE overlay (Phase 10 spec sec3, review amendment):
            # the legend describes the colours draw_overlay just drew. The fallback
            # below is YOLO's own val-batch mosaic, drawn in YOLO's colours, and
            # this legend would misdescribe it.
            legend()
        else:
            st.image(str(hero_path()), caption="Validation-batch predictions (baseline yolov8n)")
        provenance("recomputed", "overlay drawn from gt_boxes.parquet and predictions.parquet")

    # "& data checks" (Phase 9a review M2): the CAN-speed correlation card below is
    # not a scale figure, it is the CAN-bus sanity check the demo's Q5 answer points
    # at -- the label has to cover both or the card is unfindable.
    with st.expander("Dataset scale & data checks"):
        scale = metrics["scale"]
        scale_cards = [
            ("Camera keyframes", f"{scale['images']:,}"),
            ("2D boxes", f"{scale['boxes_2d']:,}"),
            ("3D object observations", f"{scale['objects_3d']:,}"),
            ("CAN-bus rows", f"{scale['canbus_rows']:,}"),
        ]
        # The CAN card lives here unless it's already the headline row's
        # pedestrian-card fallback (an older package with no night-pedestrian
        # slice) -- never shown twice.
        if ped_card is not None:
            scale_cards.append(can_card)
        metric_cards(scale_cards)

    with st.expander("Architecture (for technical reviewers)"):
        st.markdown(_ARCHITECTURE_MARKDOWN)

    st.divider()
    st.caption(
        "Every number above is derived from the pipeline's real artifacts at "
        "export time. Interactive pages label recorded outputs as recorded; "
        "the live stack (training, search, chat) runs locally via docker compose."
    )
    # docs/DEMO_PLAN.md:697's credibility statement, verbatim and in its own caption:
    # it answers the first question a visitor has about a hosted demo -- whether the
    # numbers were produced here (they were not; this deployment only serves them).
    # Pinned by tests/test_demo_app.py::test_overview_footer_carries_the_credibility_
    # statement, so a reword here has to be a deliberate one.
    st.caption(
        "Results shown here were generated by the full offline pipeline. The public "
        "application serves curated experiment outputs for reproducibility and "
        "demonstration."
    )
