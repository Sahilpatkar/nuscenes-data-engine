"""Guided tour — the demo's default 2-3 minute path through the loop (Phase 9a,
docs/superpowers/specs/2026-08-22-demo-phase9a-design.md §1), retold as *problem →
intervention → impact* by Phase 10
(docs/superpowers/specs/2026-08-23-demo-phase10-design.md §2): the page itself
renders as "From model failure to better training data" and its steps carry the
story titles below, while the sidebar entry main.py registers stays "Guided tour".

Seven steps, one screen each: the weakness, the frame it shows up on, mining for
more like it, why one mined frame was picked, the retrain, the before/after pair on
one hand-approved frame, and the result screen. The step index is the page's only
state (``st.session_state["tour_step"]``); Back/Next move it.

This module writes prose, never figures: every number on a step is derived here
from the same package tables the deep pages read, each step says where its numbers
came from (``render.provenance``), and each step lights its stage(s) in the loop
breadcrumb -- one stage for an ordinary step, all four for the result screen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import nav
import pandas as pd
import streamlit as st
from filters import (
    FILMSTRIP_STEPS,
    arm_story_label,
    failure_flags,
    filmstrip_steps,
    fixed_boxes,
    frame_quota_before,
    gain_text,
    gt_for_render,
    parity_short,
    rank_events,
    reason_chips,
    relative_gain,
    selection_factors,
    severity_caption,
    strategy_coverage,
    tour_frame_candidates,
    tour_strategies,
    upgrade_callout,
    visible_gt,
    visible_gt_boxes,
)
from nav import TOUR_STEP_KEY
from PIL import Image
from render import (
    LOOP_STAGES,
    bar_chart,
    chip_row,
    chip_row_text,
    curve_caption,
    curve_charts,
    draw_overlay,
    learned,
    legend,
    loop_breadcrumb,
    metric_cards,
    provenance,
)

from data import (
    STALE_PACKAGE_NOTE,
    al_explain_available,
    crop_path,
    events_available,
    load_al_communities,
    load_al_exemplars,
    load_al_explain,
    load_al_explain_validation,
    load_al_results,
    load_events,
    load_frame_manifest,
    load_gt_boxes,
    load_overview,
    load_predictions,
    load_subgraphs,
    load_weak_loss,
    load_weaksup,
    subgraphs_available,
    thumb_path,
)
from views.overview import HERO_CAPTION

# The matching rule's confidence floor: a prediction that claims a GT box with
# conf below it is a "low_conf" hit -- still a hit (the box's matched_<model> is
# True), just an unsure one. The package ships no copy of the value (no manifest
# key, no column), so it is restated here from its source: `sweep.conf_hit: 0.40`
# in configs/active_learning.yaml, the default that
# src/nuscenes_data_engine/active_learning/matching.py applies and
# src/nuscenes_data_engine/demo/infer.py runs the demo's own matching with.
_CONF_HIT_FLOOR = 0.40

_NO_HERO_CROP_NOTE = "hero crop not in this package"

# The tour's mining step queries the flagship dynamics preset: it is the one the
# Scenario page opens on, the one `demo subgraphs` records a SQL/Cypher parity for,
# and the one whose events are hard braking near pedestrians -- the population the
# night weakness of steps 0-1 lives in.
_TOUR_PRESET = "hard_braking_near_pedestrians"

# Text copied verbatim (not imported) from the two pages these steps condense --
# every one of them is a module-private constant or an inline string over there, and
# a tour step must say exactly what the deep page behind it says:
#   views/scenarios.py -- _NOT_CURATED_CAPTION, and `render`'s events-absent error;
#   views/active_learning.py -- _EXPLAIN_ABSENT_NOTE, _EXPLAIN_FRAME_ABSENT_NOTE,
#   _TRAIN_POOL_NOTE, and _render_before_after's legend / empty-exemplar note.
_NOT_CURATED_CAPTION = "not in the curated prediction set"
_EVENTS_ABSENT_NOTE = "needs demo_data >= 0.4 — rerun demo build"
_EXPLAIN_ABSENT_NOTE = (
    "per-frame community and routed mass are not included in this package "
    "(demo al-explain)"
)
_EXPLAIN_FRAME_ABSENT_NOTE = (
    "this frame is not in the staged selection facts — re-run `demo al-explain`"
)
_TRAIN_POOL_NOTE = "train-pool frame — no predictions (models never saw it as a test image)"
# Step 0: the three-beat opening the story site opens on too (web-frontend-v1 spec
# §1.1: plain-language problem → the numbers → what the system is FOR). Both
# sentences are copied from that spec rather than imported: `web/build_data.py`
# ships them as bundle fields for the site, and the two front-ends must say exactly
# the same thing -- the same copy-contract the tour keeps with the deep pages.
_PROBLEM_SENTENCE = (
    "The detector looked reasonable overall — but performance dropped sharply at "
    "night, especially for pedestrians."
)
_PURPOSE_SENTENCE = (
    "The goal of the system: automatically find failures like this and turn them "
    "into better training data."
)
# Step 1: what the frame IS (written only where the manifest says so), and the
# question it hands to the mining steps.
_HELD_OUT_CAPTION = "Held-out validation frame — never in any training set"
_BRIDGE_TO_MINING = (
    "Finding one failure is easy — the hard part is finding the rest of the dataset "
    "where the same thing happens."
)
# Step 2: the one sentence that says what the scenario search matches ON. The
# event, its filmstrip and its CAN curves above it are the evidence that the
# failure recurs; this is the mechanism that found them.
_MINING_MECHANISM = (
    "The system searches driving context — night, braking, pedestrians near the ego — "
    "not just similar-looking images."
)
# Step 3: the centerpiece (web-frontend-v1 spec §1.2) -- the lede that says what
# the system does INSTEAD of randomly adding images, and the chain it walks to do
# it, in the words a viewer already has. The implementation-oriented chain
# (`_SELECTION_PATH`) says the same thing in the package's own vocabulary, so it is
# what the "How selection works" fold opens with, above the seven recorded factors.
_LEDE_SENTENCE = (
    "Instead of randomly adding more images, the system searches the training pool "
    "for examples related to the diagnosed failure."
)
_SELECTION_CHAIN: tuple[str, ...] = (
    "Failed validation frame",
    "relevant driving scenario",
    "candidate training frames",
    "targeted retraining set",
)
_PLAIN_SELECTION_PATH = "**" + " → ".join(_SELECTION_CHAIN) + "**"
_SELECTION_PATH = (
    "**failed val frame → similarity community → representative train-pool frames → "
    "selected for retraining**"
)
_ARCHITECTURE_STRIP = (
    "System path: nuScenes → validated Parquet → SQL / Neo4j / CAN → failure analysis "
    "→ scenario search → active learning → YOLO retraining → evaluation"
)
# Step 4: the intervention screen (Phase 10 spec sec2 row 4). The headline is a
# claim about COMPOSITION -- the line under the cards has to support it out of this
# package's own night shares, or the headline would be the one sentence on the step
# nothing computed.
_RETRAIN_HEADLINE = "We did not just add data — we changed what the model trains on."
_STRATEGY_CHART_TITLE = "Night mAP50-95 vs baseline, by acquisition strategy"
# What the fairness statement promises, minus the budget clause (which is written
# only when the charted arms really did share a training-set size) and minus the
# control clause (only when the control arm is one of them).
_FAIRNESS_TAIL = "same training configuration · scored on the same held-out split"
# The fairness statement is what makes the chart a comparison rather than a
# leaderboard, so (web-frontend-v1 spec §1.3) it is not prose that happens to sit
# near the chart: it is a bordered strip directly above it, its clauses as chips,
# with the control clause -- the arm every other bar is measured against -- the one
# chip in accent. Written only when the control arm is actually charted.
_FAIRNESS_LEAD = "Fair comparison"
_FAIRNESS_CONTROL = "Random sample is the control"
# A mined set whose night share rounds to 0 % at the shares' own precision. The
# similarity explanation (brief sec37) is written only below this, so a set with any
# real night coverage is never described as a daytime one.
_NIGHT_ABSENT_SHARE = 0.005

# Step 5: the question the whole tour has been walking towards, asked above the two
# pictures that answer it -- and the fold the per-box evidence sits in, so the screen
# reads as an answer rather than as a table.
_AFTER_HEADLINE = "Did the targeted retraining fix the kind of failure we started with?"
_PER_BOX_FOLD = "Technical details — per-box claims"
# The two evidence tiers (web-frontend-v1 spec §1.4): the frame is labelled as ONE
# example before it is shown, and the arm-level numbers under it are labelled as
# the result -- the structure itself answers "did you just pick a flattering
# image?", rather than a caveat sentence somewhere below having to.
_ONE_EXAMPLE_TIER = "One example — one hand-approved frame, illustrative, not the metric"
_AGGREGATE_TIER = "**The aggregate result:**"

_NO_EXEMPLAR_NOTE = "no exemplar frames in this package"
_NO_UPGRADED_BOXES_NOTE = "no upgraded boxes on this frame"

_NO_TOUR_EVENT_NOTE = f"no `{_TOUR_PRESET}` events in this package"
_NO_TOUR_FRAME_NOTE = "no curated frame selected by the arm is in this package"

# Step 5's honesty line, drawn only when the package's own numbers make it true --
# see _hero_recovery_is_low_conf.
_HERO_HONESTY_LINE = (
    "The hero frame's pedestrian recovery (step 2) is a low-confidence claim and does "
    "not pass this table's confident-detection rule; the hand-approved exemplars do."
)

# Step 6: the result screen's headline and its closing thesis. The headline is
# itself a CLAIM about this package's numbers -- "measurable improvement" is true
# only where the arm's recorded night delta really is a gain -- so which of the two
# is written is computed in _result_headline, never decided here. The thesis under
# the four answers is a claim about the METHOD, which those answers are the
# evidence for; it states no figure, so it needs no guard.
_CLOSED_LOOP_IMPROVED = "Closed the loop: weakness → targeted data → measurable improvement"
_CLOSED_LOOP_MEASURED = "Closed the loop: weakness → targeted data → measured result"
_CLOSING_THESIS = (
    "**Instead of blindly retraining the model, the system diagnoses where it fails, "
    "finds the data that can address the weakness, and measures whether the "
    "intervention actually works.**"
)

# scenario_events' t-2..t+2 neighbour columns in strip order, with the event itself
# (column None) in the middle. Phase 9b: filters.FILMSTRIP_STEPS is the single
# definition (this step order is shared with views/scenarios.py and with
# filters.filmstrip_steps, which this page's strip reads below); the name stays
# because the copy-contract test pins the tour's strip order to the Scenario
# page's, and that is what it names here.
_FILMSTRIP_STEPS: tuple[tuple[str | None, str], ...] = FILMSTRIP_STEPS


@dataclass
class _TourData:
    """Every table a step can read, loaded once per run (the loaders are cached,
    but a step should never have to know that).

    ``events`` is ``None`` when the package predates ``scenario_events.parquet``;
    ``explain``/``validation``/``communities``/``loss`` come back empty when their
    optional groups were never staged; ``arm`` is ``None`` on a package with no
    ``al_exemplars.json``. Steps degrade on those, they never assume them.
    """

    overview: dict[str, Any]
    arms: pd.DataFrame
    manifest: pd.DataFrame
    gt: pd.DataFrame
    preds: pd.DataFrame
    exemplars: dict[str, Any]
    explain: pd.DataFrame
    validation: dict[str, Any]
    loss: pd.DataFrame
    weaksup: pd.DataFrame
    communities: pd.DataFrame
    events: pd.DataFrame | None
    baseline: str
    arm: str | None


@dataclass(frozen=True)
class _Step:
    """One screen of the tour. ``stage`` is the loop stage(s) it lights: a single
    stage name for an ordinary step, a tuple to light more than one (the result
    screen lights all four -- the whole loop just ran), or ``None`` to light
    nothing; ``links`` are the "Go deeper" deep links, as ``(url_path, label)``
    pairs resolved through ``nav.page``."""

    key: str
    title: str
    stage: str | tuple[str, ...] | None
    render: Callable[[_TourData], None]
    links: tuple[tuple[str, str], ...] = ()


def _load() -> _TourData:
    exemplars = load_al_exemplars()
    arm = exemplars.get("arm")
    return _TourData(
        overview=load_overview(),
        arms=load_al_results(),
        manifest=load_frame_manifest(),
        gt=load_gt_boxes(),
        preds=load_predictions(),
        exemplars=exemplars,
        explain=load_al_explain(),
        validation=load_al_explain_validation(),
        loss=load_weak_loss(),
        weaksup=load_weaksup(),
        communities=load_al_communities(),
        # The one loader that opens a file older packages don't carry at all.
        events=load_events() if events_available() else None,
        # `demo build` writes both names into al_exemplars.json; "baseline" is the
        # arm table's own name for the un-mined checkpoint, and the fallback keeps
        # a package without the file rendering the recorded cards.
        baseline=str(exemplars.get("baseline") or "baseline"),
        arm=str(arm) if arm else None,
    )


def _night_miss_counts(data: _TourData) -> tuple[int, int, int] | None:
    """``(night val frames, of those with >= 1 baseline miss, val frames)``, or
    ``None`` when this package carries no ``matched_<baseline>`` column to count
    misses from."""
    if f"matched_{data.baseline}" not in data.gt.columns:
        return None
    flags = failure_flags(
        data.manifest, visible_gt_boxes(data.gt), data.preds, model=data.baseline
    )
    is_night = flags["is_night"].fillna(False).astype(bool)
    return int(is_night.sum()), int((is_night & flags["has_fn"]).sum()), len(flags)


def _render_weakness(data: _TourData) -> None:
    """Step 0 — the sliced evaluation that makes the weakness visible, in the three
    beats the story site opens on (web-frontend-v1 spec §1.1): the plain-language
    problem as the headline, the numbers that establish it (cards, the counted miss
    line, the derived sentence), and the purpose line that says what the system does
    about a diagnosis like this. Every figure is still derived here; the two fixed
    sentences are the site's own, word for word."""
    rows = data.arms.loc[data.arms["arm"] == data.baseline]
    if rows.empty:
        st.info(STALE_PACKAGE_NOTE)
        return
    row = rows.iloc[0]
    st.subheader(_PROBLEM_SENTENCE)

    cards = [
        ("Overall mAP50-95", f"{row['overall_map5095']:.4f}"),
        ("Night mAP50-95", f"{row['night_map5095']:.4f}"),
    ]
    # night_ped_map5095 is the night PEDESTRIAN class -- the mechanism the night
    # arms move. It is absent (or NA) on a package whose eval never wrote per-class
    # night metrics, and an absent card is better than a card reading "nan".
    night_ped = row.get("night_ped_map5095")
    has_night_ped = night_ped is not None and bool(pd.notna(night_ped))
    if has_night_ped:
        cards.append(("Night pedestrian mAP50-95", f"{float(night_ped):.4f}"))
    metric_cards(cards)

    # The counted line, recomputed here over the same val slice the Failure
    # Explorer shows. A package with no per-model matched columns still gets the
    # recorded cards above -- it just has nothing to count misses from, and says so.
    counts = _night_miss_counts(data)
    recomputed_detail = ""
    if counts is None:
        st.info(STALE_PACKAGE_NOTE)
    else:
        n_night, n_night_fn, n_val = counts
        st.markdown(
            f"**{n_night_fn} of {n_night} night validation frames carry at least one "
            f"baseline miss.**"
        )
        recomputed_detail = (
            f"frame counts from frame_manifest/gt_boxes/predictions ({n_val} curated "
            "val frames)"
        )
    # One sentence off the same baseline row the cards are read from -- and the
    # night-pedestrian clause is dropped on a package that has no such number,
    # exactly as its card is (a sentence must not out-claim the cards above it).
    night_ped_clause = (
        f" — and {float(night_ped):.4f} on night pedestrians, the case that matters most"
        if has_night_ped
        else ""
    )
    st.markdown(
        f"`{data.baseline}` scores {row['overall_map5095']:.4f} mAP50-95 overall but "
        f"{row['night_map5095']:.4f} at night{night_ped_clause}."
    )
    # Beat (c), and the last thing on the screen before the takeaway: the numbers
    # above are a diagnosis, and this is what the rest of the tour does with one.
    st.markdown(_PURPOSE_SENTENCE)
    learned("Aggregate metrics hide important failure slices.")
    provenance("recorded", "mAP from active_learning_results.parquet")
    if recomputed_detail:
        provenance("recomputed", recomputed_detail)


def _claim(
    preds: pd.DataFrame, *, token: str, model: str, annotation: str
) -> tuple[float, str] | None:
    """The highest-confidence row ``model`` filed against ``annotation`` on
    ``token``: ``(conf, status)``, or ``None`` when it never claimed the box at
    all. The raw pair ``_claim_text`` formats into its "no claim"/"conf (status)"
    prose -- and the result screen's hero-honesty line (Task 4) reads it back
    directly, rather than re-deriving the claim a second way."""
    rows = preds.loc[
        (preds["sample_data_token"] == token)
        & (preds["model"] == model)
        & (preds["matched_annotation_token"] == annotation)
    ]
    if rows.empty:
        return None
    row = rows.sort_values("conf", ascending=False).iloc[0]
    return float(row["conf"]), str(row["status"])


def _claim_text(preds: pd.DataFrame, *, token: str, model: str, annotation: str) -> str:
    """What ``model`` claimed about one GT box: its confidence and what kind of claim
    it is, or "no claim" when it never matched the box at all (an honest absence --
    a model that said nothing is not a model that said something weak).

    Both branches read as English (consolidated review M3): the low-confidence one
    always spelled itself out, while the confident one printed the table's raw
    ``status`` code beside it, so the same line said "0.135 (low-confidence — ...)"
    and "0.800 (tp)". A status this app has no word for falls back to its own id
    with the underscores spaced, rather than being handed a meaning nobody wrote.
    """
    claim = _claim(preds, token=token, model=model, annotation=annotation)
    if claim is None:
        return "no claim"
    conf, status = claim
    if status == "low_conf":
        return (
            f"{conf:.3f} (low-confidence — below the {_CONF_HIT_FLOOR:.2f} hit floor; "
            "the matching rule still counts it as a hit)"
        )
    if status == "tp":
        return f"{conf:.3f} (confident detection)"
    return f"{conf:.3f} ({status.replace('_', ' ')})"


def _pedestrian_facts(
    data: _TourData, *, token: str, gt_rows: pd.DataFrame, models: Sequence[str]
) -> list[str]:
    """One line per visible pedestrian GT box on the frame: how far away it is, and
    what each model claimed about it."""
    lines = []
    peds = gt_rows.loc[gt_rows["category_group"] == "pedestrian"]
    for row in peds.itertuples(index=False):
        # A box whose annotation never joined onto annotations_3d has no distance;
        # the claim is still the point of the line, so it renders without one.
        distance = getattr(row, "distance_to_ego_m", None)
        where = ""
        if distance is not None and bool(pd.notna(distance)):
            where = f" at {float(distance):.1f} m"
        claims = " · ".join(
            f"{arm_story_label(model)}: "
            + _claim_text(
                data.preds, token=token, model=model, annotation=str(row.annotation_token)
            )
            for model in models
        )
        lines.append(f"**Pedestrian{where}** — {claims}")
    return lines


def _pedestrian_miss_to_hit_frames(data: _TourData) -> set[str] | None:
    """Night val frames carrying a pedestrian the baseline missed and the arm hit,
    or ``None`` when this package has no column for one of the two models."""
    arm = data.arm
    baseline_column, arm_column = f"matched_{data.baseline}", f"matched_{arm}"
    if arm is None or not {baseline_column, arm_column} <= set(data.gt.columns):
        return None
    val = data.manifest.loc[data.manifest["split"] == "val"]
    night = set(
        val.loc[val["is_night"].fillna(False).astype(bool), "sample_data_token"].astype(str)
    )
    boxes = visible_gt_boxes(data.gt)
    upgraded = boxes.loc[
        (boxes["category_group"] == "pedestrian")
        & boxes["sample_data_token"].astype(str).isin(night)
        # .eq(False)/.eq(True) on the nullable-boolean column: NA is "not
        # evaluated", which is neither a miss nor a hit.
        & boxes[baseline_column].eq(False).fillna(False)
        & boxes[arm_column].eq(True).fillna(False)
    ]
    return {str(token) for token in upgraded["sample_data_token"]}


def _miss_to_hit_sentence(data: _TourData, hero: str) -> str | None:
    """The counted claim about how rare this frame is -- counted, never assumed:
    when the hero is not the single such frame, the sentence says how many there
    are instead of calling it the only one."""
    frames = _pedestrian_miss_to_hit_frames(data)
    if frames is None:
        return None
    if len(frames) == 1 and hero in frames:
        # Phrased without an article before the arm name -- the name is read from
        # the package, so "a/an <arm>" cannot be got right for every package.
        return (
            f"Across the whole curated val split this is the only night frame where "
            f"`{data.arm}` turns a `{data.baseline}` pedestrian miss into a hit."
        )
    return (
        f"{len(frames)} night val frames carry a pedestrian that `{data.baseline}` misses "
        f"and `{data.arm}` claims."
    )


def _held_out_caption(data: _TourData, token: str) -> None:
    """Write "never in any training set" for ``token`` -- but only where the manifest
    says so.

    Both the hero (step 1) and the hand-approved exemplar (step 5) are curated val
    frames on the shipped package, and the claim is the whole point of showing them.
    A package that staged either as a train-pool frame must not be told it was held
    out, so the split is read per frame rather than assumed, and the two steps share
    one gate rather than two that could drift.
    """
    rows = data.manifest.loc[data.manifest["sample_data_token"].astype(str) == token]
    if not rows.empty and str(rows.iloc[0].get("split")) == "val":
        st.caption(_HELD_OUT_CAPTION)


def _render_missed_pedestrian(data: _TourData) -> None:
    """Step 1 — the weakness on one frame: what each model claimed about the
    pedestrian the baseline missed."""
    hero = str(data.overview.get("hero_token") or "")
    crop = crop_path(hero) if hero else None
    if crop is None or not crop.is_file():
        st.info(_NO_HERO_CROP_NOTE)
        return

    st.subheader("Here the baseline misses a pedestrian at night.")
    models = [name for name in (data.baseline, data.arm) if name is not None]
    model = (
        st.radio(
            "Model",
            models,
            key="tour_hero_model",
            horizontal=True,
            format_func=arm_story_label,
        )
        or models[0]
    )
    gt_rows = visible_gt(data.gt, hero)
    frame_preds = data.preds.loc[
        (data.preds["sample_data_token"] == hero) & (data.preds["model"] == model)
    ]
    st.image(
        draw_overlay(
            Image.open(crop),
            gt_for_render(gt_rows, model),
            frame_preds,
            mode="overlay",
            scale=0.6,
        ),
        width="stretch",
    )
    legend()
    st.caption(HERO_CAPTION)
    _held_out_caption(data, hero)

    for line in _pedestrian_facts(data, token=hero, gt_rows=gt_rows, models=models):
        st.markdown(line)
    sentence = _miss_to_hit_sentence(data, hero)
    if sentence:
        st.markdown(sentence)
    st.markdown(f":gray[{_BRIDGE_TO_MINING}]")
    provenance("recomputed", "boxes, claims and counts from gt_boxes/predictions")


def _open(page_url: str, **state: str) -> None:
    """Assign a target page's own session-state keys, then switch to it.

    ``st.page_link`` carries no state, so the three "open this one over there"
    buttons write the keys the target page already reads -- ``scenario_preset``/
    ``scenario_token``, ``al_frame_token``, ``al_exemplar`` (that last one is the
    Active Learning page's exemplar SELECTBOX key, which is valid to set before the
    widget is instantiated) -- and switch afterwards. Called from a button's inline
    branch, never from an ``on_click`` callback.
    """
    for key, value in state.items():
        st.session_state[key] = value
    st.switch_page(nav.page(page_url))


def _render_event_frame(data: _TourData, row: pd.Series) -> None:
    """The mined event's frame, GT boxes only: crop first, thumb second -- the two
    branches ``views/scenarios.py::_render_viewer`` uses for an event outside the
    curated prediction set, and a caption saying WHY there are no predictions on it.

    Three cases, because "no predictions here" has two different reasons (Phase 9a
    review 5c) and a curated frame has neither:

    * ``in_curated_set`` -- no caption at all; predictions exist for this frame, and
      claiming otherwise would be a false absence claim.
    * not curated, but the package's ``frame_manifest`` does carry the token as a
      ``train_pool`` row -- ``_TRAIN_POOL_NOTE``: the frame IS in the package, it
      just was never a test image, which is the more informative half of the story
      (and the note the Active Learning page shows for the same kind of frame).
    * not curated and not in the manifest at all -- ``_NOT_CURATED_CAPTION``: the
      package carries this event's row and thumbs but not the frame itself.
    """
    token = str(row["sample_data_token"])
    gt_rows = gt_for_render(visible_gt(data.gt, token))
    crop, thumb = crop_path(token), thumb_path(token)
    if crop.is_file():
        st.image(draw_overlay(Image.open(crop), gt_rows, pd.DataFrame(), mode="gt", scale=0.6))
    elif thumb.is_file():
        st.image(draw_overlay(Image.open(thumb), gt_rows, pd.DataFrame(), mode="gt", scale=0.16))
    if bool(row.get("in_curated_set")):
        return
    manifest_rows = data.manifest.loc[data.manifest["sample_data_token"].astype(str) == token]
    train_pool = (
        not manifest_rows.empty and str(manifest_rows.iloc[0].get("split")) == "train_pool"
    )
    st.caption(_TRAIN_POOL_NOTE if train_pool else _NOT_CURATED_CAPTION)


def _render_event_filmstrip(row: pd.Series) -> None:
    """The event's t-2..t+2 thumbs with their step labels -- the Scenario page's
    filmstrip without its step slider (a tour screen is one visual, not a control
    panel). A scene-edge neighbour is NA and is left out of the strip entirely,
    exactly as that page leaves it out -- filters.filmstrip_steps is where that
    rule (and the strip order) now lives, for both pages."""
    steps = filmstrip_steps(row).steps
    if not steps:
        return
    for strip_column, step in zip(st.columns(len(steps)), steps, strict=True):
        with strip_column:
            thumb = thumb_path(step.token)
            if thumb.is_file():
                st.image(str(thumb))
            st.caption(step.label)


def _render_event_curves(row: pd.Series) -> None:
    """The event's CAN speed and CAN longitudinal acceleration over the same steps
    the strip above shows -- the motion cue a single still frame cannot give.

    Built by ``render.curve_charts`` from ``filters.filmstrip_steps``, exactly as
    the Scenario page builds its pair, so this step and the deep page behind it
    draw the identical curves (the page adds only its slider; here the rule is
    pinned to the event's own frame, since a tour screen is one visual rather than
    a control panel). The caption is the builder's own: it says what the two series
    ARE, and on a package older than v0.8 it says "ego speed" instead of CAN --
    never the other way round.
    """
    curve = filmstrip_steps(row)
    if not curve.steps:
        return
    current = next((step.label for step in curve.steps if step.is_current), None)
    speed, accel = curve_charts(curve, selected=current)
    speed_column, accel_column = st.columns(2)
    with speed_column:
        st.altair_chart(speed, width="stretch")
    with accel_column:
        st.altair_chart(accel, width="stretch")
    st.caption(curve_caption(curve))


def _render_mine(data: _TourData) -> None:
    """Step 2 — the weakness as a population: the flagship scenario query's top
    event, its own facts, and how many of the matching events are at night.

    Phase 10 (spec sec2 row 2) only re-frames what was already here: the headline
    asks the question the event, its filmstrip and its CAN curves answer, one
    sentence says what the search matched on, and the takeaway closes the step.
    """
    if data.events is None:
        st.info(_EVENTS_ABSENT_NOTE)
        return
    ranked = rank_events(data.events, _TOUR_PRESET)
    if ranked.empty:
        st.info(_NO_TOUR_EVENT_NOTE)
        return
    st.subheader("Is this one bad photo, or a recurring driving scenario?")
    row = ranked.iloc[0]
    token = str(row["sample_data_token"])

    left, right = st.columns([3, 2])
    with left:
        _render_event_frame(data, row)
    with right:
        st.markdown(f"**{row['scene_name']} · {severity_caption(_TOUR_PRESET, row.to_dict())}**")
        n_peds = int(row["n_peds_within_10m"]) if pd.notna(row["n_peds_within_10m"]) else 0
        nearest = row["min_dist_pedestrian_m"]
        facts = f"within 10 m: {n_peds} pedestrian{'' if n_peds == 1 else 's'}"
        if pd.notna(nearest):
            facts += f" · nearest at {float(nearest):.1f} m"
        st.markdown(facts)
        chip_row([
            "night" if bool(row["is_night"]) else "day",
            *(["rain"] if bool(row["is_rain"]) else []),
        ])
    _render_event_filmstrip(row)
    _render_event_curves(row)

    # The parity line is a trust indicator, not a claim about this event: a package
    # with no subgraphs staged simply omits it (the Scenario page's own no-op).
    payload = load_subgraphs(_TOUR_PRESET) if subgraphs_available() else None
    if payload is not None:
        st.caption(
            parity_short(int(payload["sql_count"]), payload["cypher_count"], payload["parity"])
        )

    n_night = int(ranked["is_night"].fillna(False).astype(bool).sum())
    st.markdown(f"{n_night} of {len(ranked)} matching events are at night.")
    st.markdown(_MINING_MECHANISM)
    learned("Perception failures must be analyzed in driving context.")
    provenance("recorded", "event counts computed at build against SQL and the Neo4j graph")
    provenance("recomputed", "overlay from gt_boxes.parquet")
    if st.button("Open this event in Scenario Search →", key="tour_open_event"):
        _open("scenarios", scenario_preset=_TOUR_PRESET, scenario_token=token)


def _tour_frame(data: _TourData, arm: str) -> tuple[str, float] | None:
    """``(token, image scale)`` for the "why selected" step: the first ranked
    candidate that has a crop, else the first that has a thumb (a partial package),
    else ``None`` -- the same crop-then-thumb fallback (and the same 0.6/0.16
    scales) the Active Learning page's own selected-frame panel uses."""
    candidates = tour_frame_candidates(data.manifest, data.explain, data.gt, arm=arm)
    for token in candidates:
        if crop_path(token).is_file():
            return token, 0.6
    for token in candidates:
        if thumb_path(token).is_file():
            return token, 0.16
    return None


def _flagship_rank(data: _TourData, token: str) -> int | None:
    """This frame's rank in the tour's own scenario query, or ``None`` when it is
    not one of that query's events at all -- the "the two mechanisms agree on this
    frame" sentence is only written when they actually do."""
    if data.events is None:
        return None
    ranked = rank_events(data.events, _TOUR_PRESET)
    tokens = [str(value) for value in ranked["sample_data_token"]]
    return tokens.index(token) + 1 if token in tokens else None


def _render_why_selected(data: _TourData) -> None:
    """Step 3 — one mined frame and the recorded reason it was picked, reproduced
    by ``demo al-explain`` rather than reasoned about here.

    Phase 10 (spec sec2 row 3) sets the question first, answers it in one line and
    folds the seven recorded factors and the causal closing sentence into "How
    selection works" -- the mechanism is one click away, not the first thing on the
    screen. web-frontend-v1 spec §1.2 makes that answer readable without the
    package's vocabulary: the lede says what the system does instead of adding
    images at random, the visible chain is the plain-language one, and
    ``_SELECTION_PATH`` -- the same chain in the package's own terms -- opens the
    fold, above the factors. The chip row stays where it was, between the frame and
    the fold: it is what the Active Learning page leads its own per-frame panel
    with, and this step must not summarise that panel into a different set of
    facts.
    """
    if not al_explain_available():
        st.info(_EXPLAIN_ABSENT_NOTE)
        return
    arm = data.arm
    if arm is None:
        st.info(STALE_PACKAGE_NOTE)
        return
    chosen = _tour_frame(data, arm)
    if chosen is None:
        st.info(_NO_TOUR_FRAME_NOTE)
        return
    token, scale = chosen

    st.subheader("Thousands of candidate training frames — which are worth labelling?")
    st.markdown(_LEDE_SENTENCE)
    frame_row = data.manifest.loc[data.manifest["sample_data_token"] == token].iloc[0]
    gt_rows = visible_gt(data.gt, token)
    image_path = crop_path(token) if scale == 0.6 else thumb_path(token)
    st.image(
        draw_overlay(
            Image.open(image_path),
            # matched_<arm> is NA on a train-pool frame (no model ever saw it as a
            # test image), which draw_overlay renders as plain GT -- never as a miss.
            gt_for_render(gt_rows, arm),
            pd.DataFrame(),
            mode="gt",
            scale=scale,
        )
    )
    if frame_row.get("split") == "train_pool":
        st.caption(_TRAIN_POOL_NOTE)

    explain_rows = data.explain.loc[data.explain["sample_data_token"] == token]
    if explain_rows.empty:
        # Unreachable for a candidate (tour_frame_candidates only returns tokens
        # with an explain row), kept so this panel can never invent a reason.
        st.caption(_EXPLAIN_FRAME_ABSENT_NOTE)
        return
    night_floor_value = (data.validation.get("config") or {}).get("night_floor")
    night_floor = int(night_floor_value) if night_floor_value is not None else None
    n_communities = int(data.validation.get("n_communities", len(data.communities)))
    explain_row = explain_rows.iloc[0].to_dict()
    # Phase 9b (Task 6): the Active Learning page's own chip row, from the same
    # helper over the same explain row -- this step condenses that panel, so it
    # must not summarise it into a different set of facts.
    chip_row(
        reason_chips(
            explain_row,
            gt_rows,
            n_communities=n_communities,
            night_floor=night_floor,
            quota_before=frame_quota_before(data.communities, explain_row, arm=arm),
        )
    )
    st.markdown(_PLAIN_SELECTION_PATH)
    with st.expander("How selection works"):
        # The same chain in the package's own vocabulary, above the factors that
        # detail it: the plain one above is the story, this one is the mechanism.
        st.markdown(_SELECTION_PATH)
        for label, value, flag in selection_factors(
            explain_row, n_communities=n_communities, night_floor=night_floor
        ):
            mark = "" if flag is None else (" ✓" if flag else " ✗")
            st.markdown(f"**{label}:** {value}{mark}")
        floor_text = (
            "a night floor takes a minimum number of night frames first"
            if night_floor is None
            else f"a night floor takes {night_floor} night frames first"
        )
        st.markdown(
            f"Nothing about this frame on its own selected it: its community carried "
            f"failure mass, that mass bought the community a quota, the frame ranked "
            f"high enough inside it by similarity — and {floor_text}."
        )

    rank = _flagship_rank(data, token)
    if rank is not None:
        st.markdown(
            f"This frame is itself flagship event #{rank} — the scenario query and the "
            "selection agree on it."
        )
    if str(frame_row.get("weak_verdict")) == "rejected":
        st.markdown(
            "Later, the weak-supervision verifier rejected this frame as too crowded to "
            "label automatically — step 7 shows why that matters."
        )
    n_selected = int(data.validation.get("n_selected", len(data.explain)))
    provenance(
        "reproduced",
        f"demo al-explain reproduced the run: {n_selected:,} frames, "
        f"{n_communities} communities",
    )
    st.caption(_ARCHITECTURE_STRIP)
    if st.button("Open this frame in Active Learning →", key="tour_open_al_frame"):
        _open("active_learning", al_frame_token=token)


def _night_share_line(data: _TourData, *, arm: str, arm_row: pd.Series) -> str | None:
    """What the mining CHANGED, in one bold line: the arm's own night share against
    the comparator arms this package actually ran.

    Nothing is guessed and nothing is filled in: a comparator with no row (or a row
    whose share was never written) loses its clause, and an arm with no share of its
    own has no line at all -- the sentence exists to compare shares, and a
    comparison missing its own subject is not one. A tour pointed AT a comparator
    (an ``arm`` of "mined", say) drops that clause too, rather than comparing an arm
    with itself.

    Every arm is named by its STORY LABEL -- the same name the chart's axis, the
    cards and step 5's captions use (consolidated review M6). This line used to
    invent a third name for each of them ("Targeted mining", "random control"), so
    one screen called the same arm three things and the viewer had to work out that
    they were one arm.
    """
    share = arm_row.get("night_share")
    if not bool(pd.notna(share)):
        return None
    clauses = [f"{arm_story_label(arm)}: **{float(share):.0%} night**"]
    for name in ("random", "mined"):
        if name == arm:
            continue
        rows = data.arms.loc[data.arms["arm"] == name]
        if rows.empty:
            continue
        value = rows.iloc[0].get("night_share")
        if not bool(pd.notna(value)):
            continue
        clauses.append(f"{arm_story_label(name)}: **{float(value):.0%} night**")
    return " · ".join(clauses)


def _charted_ids_caption(coverage: pd.DataFrame) -> str:
    """The charted strategies' raw arm ids, in small text under the chart: the tour
    fronts each arm with a story label, and this is where the identifier it stands
    for stays visible (spec's honesty rules -- readable names are tour-scope, they
    never replace the id)."""
    return " · ".join(
        f"{record.strategy} (`{record.arm}`)" for record in coverage.itertuples(index=False)
    )


def _fairness_parts(coverage: pd.DataFrame, *, baseline: str) -> tuple[list[str], str | None]:
    """What was held constant across the charted arms, clause by clause -- the
    promise that makes the chart a comparison rather than a leaderboard -- and the
    control clause, or ``None`` when the control arm is not one of the charted ones.

    The budget clause names a frame count only when every charted non-baseline arm
    really was trained on the same number of images (computed from the arm table,
    NA counting as "not known to be equal"); otherwise the clause still promises the
    same budget, without claiming a figure this package cannot show.

    Clauses rather than one sentence (web-frontend-v1 spec §1.3): the strip renders
    them as chips and accents the control one, and the derivation must live in one
    place whichever shape the screen draws them in.
    """
    others = coverage.loc[coverage["arm"] != baseline]
    sizes = others["n_train_images"]
    known = sizes.dropna()
    shared = {int(value) for value in known}
    budget = (
        f"same {shared.pop():,}-frame budget"
        if not others.empty and len(known) == len(sizes) and len(shared) == 1
        else "same budget"
    )
    clauses = ["Same detector", budget, *_FAIRNESS_TAIL.split(" · ")]
    # "Random sample", not the chart's own "Random sample — control" label: the
    # clause would otherwise read "... — control is the control".
    control = (
        _FAIRNESS_CONTROL if "random" in {str(name) for name in coverage["arm"]} else None
    )
    return clauses, control


def _render_fairness_strip(coverage: pd.DataFrame, *, baseline: str) -> None:
    """The fairness promise as the panel the chart is read through: an eyebrow, the
    held-constant clauses as chips, and the control clause as the one chip in accent
    -- the same grammar the story site's own strip uses, so the two front-ends bind
    the same promise to the same chart."""
    clauses, control = _fairness_parts(coverage, baseline=baseline)
    with st.container(border=True):
        st.markdown(f"**{_FAIRNESS_LEAD}**")
        chips = chip_row_text(clauses)
        if control is not None:
            chips += " " + chip_row_text([control], color="orange")
        st.markdown(chips)


def _best_night_arm_sentence(arms: pd.DataFrame, *, baseline: str) -> str | None:
    """Which INTERVENTION actually answered the diagnosed weakness best -- computed
    over the whole arm table (not just the charted five, and not assumed to be the
    arm the tour follows), or ``None`` when this package ran no intervention with a
    night delta to rank.

    The baseline is dropped before the ranking (consolidated review M2): its
    ``delta_night`` is +0.0000 by construction, since it is the comparator every
    other delta is measured against, so on a package where every arm regressed it
    held the highest night delta and was crowned "best intervention" -- a run where
    nothing worked reading as a run where doing nothing worked.

    And a ranking whose top arm still lost night mAP is not a win, so it is not
    written as one: the sentence then says no intervention beat the baseline and
    names the closest, which is the same fact without the promotion.
    """
    if "delta_night" not in arms.columns:
        return None
    ranked = arms.dropna(subset=["delta_night"])
    ranked = ranked.loc[ranked["arm"] != baseline]
    if ranked.empty:
        return None
    best = ranked.loc[ranked["delta_night"].idxmax()]
    name, delta = str(best["arm"]), float(best["delta_night"])
    if delta <= 0:
        return (
            f"No intervention beat the baseline on the diagnosed night weakness — "
            f"the closest was **{arm_story_label(name)}** (`{name}`, {delta:+.4f} night)."
        )
    return (
        f"Best intervention for the diagnosed night weakness: "
        f"**{arm_story_label(name)}** (`{name}`, {delta:+.4f} night)."
    )


def _similarity_sentence(arms: pd.DataFrame) -> str | None:
    """Why the similarity arm's night result is where it is (brief sec37) -- written
    only when this package ran that arm AND its mined set really is a daytime one
    (``_NIGHT_ABSENT_SHARE``).

    Takes any frame carrying ``arm``/``night_share``, because the two screens that
    say this say it about the same row and must say it the same way: step 4 asks it
    of the charted strategies (so the caption appears only under a chart that
    actually drew that bar) and step 6's second answer asks it of the whole arm
    table, next to the comparator clause about the very same set.
    """
    rows = arms.loc[arms["arm"] == "mined"]
    if rows.empty:
        return None
    share = rows.iloc[0]["night_share"]
    if not bool(pd.notna(share)) or float(share) >= _NIGHT_ABSENT_SHARE:
        return None
    return (
        f"Visual similarity alone concentrated on daytime appearance "
        f"({float(share):.0%} night); the graph + night-floor arm explicitly preserved "
        "night coverage."
    )


def _scene_diversity_sentence(base_row: pd.Series, arm_row: pd.Series) -> str | None:
    """How widely the mined frames are spread -- ``None`` when this package never
    recorded the arm's scene count.

    The "not near-duplicates" clause is a claim about the spread, so it is written
    only when the set really does span more than one scene; both nouns agree with
    their counts, the way ``active_learning._strategy_triple`` makes them.
    """
    n_scenes = arm_row.get("n_scenes")
    if not bool(pd.notna(n_scenes)):
        return None
    scenes = int(n_scenes)
    base_n, arm_n = base_row.get("n_train_images"), arm_row.get("n_train_images")
    if bool(pd.notna(base_n)) and bool(pd.notna(arm_n)):
        mined = int(arm_n) - int(base_n)
        subject = f"The {mined:,} frame{'' if mined == 1 else 's'}"
    else:
        subject = "The mined frames"
    sentence = f"{subject} came from {scenes:,} scene{'' if scenes == 1 else 's'}"
    if scenes > 1:
        sentence += " — targeted, but not near-duplicates of one scene"
    return sentence + "."


def _render_retrain(data: _TourData) -> None:
    """Step 4 — the intervention: what the mining changed about the training set,
    and how that choice compares with the other ways of spending the same budget.

    Phase 10 (spec sec2 row 4) narrows the deep page's 13-arm chart to the tour's
    five-strategy story (``tour_strategies`` + ``strategy_coverage``, the same pair
    the Active Learning page's own strategy section charts) and surrounds it with
    the three sentences that make it readable as an experiment: what changed, what
    was held constant, and which arm actually won. The four cards are untouched.

    web-frontend-v1 spec §1.3 promotes the middle one out of prose: what was held
    constant is a bordered chip strip directly ABOVE the chart, because a viewer who
    reads the bars first has already read them as a leaderboard.
    """
    arm = data.arm
    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm] if arm is not None else data.arms.iloc[:0]
    if arm is None or base_rows.empty or arm_rows.empty:
        st.info(STALE_PACKAGE_NOTE)
        return
    base_row, arm_row = base_rows.iloc[0], arm_rows.iloc[0]
    st.subheader(_RETRAIN_HEADLINE)

    # A card whose value is NA is omitted, never rendered as "nan": n_scenes/
    # night_share come from the arm's own mined-set composition, which an older
    # results.json (or an arm with no extra-frames file) never wrote.
    cards: list[tuple[str, str] | tuple[str, str, str]] = []
    base_n, arm_n = base_row.get("n_train_images"), arm_row.get("n_train_images")
    sizes_known = bool(pd.notna(base_n)) and bool(pd.notna(arm_n))
    if sizes_known:
        cards.append(("Frames mined", f"{int(arm_n) - int(base_n):,}"))
    n_scenes = arm_row.get("n_scenes")
    if bool(pd.notna(n_scenes)):
        cards.append(("Scenes covered", f"{int(n_scenes)}"))
    night_share = arm_row.get("night_share")
    if bool(pd.notna(night_share)):
        cards.append(("Night share", f"{float(night_share):.0%}"))
    if sizes_known:
        # "Training images", not "Training set" + a trailing " images": st.metric
        # renders its VALUE in a narrow column and the 22-character
        # "7,035 → 8,535 images" wrapped mid-arrow (Phase 9a review 5a). The
        # 13-character "7,035 → 8,535" arrow value on its own still truncates in
        # a 1-of-4 st.metric card at <=1200px (Phase 9a final review), so show
        # the after-count alone with the mined total as the delta.
        cards.append((
            "Training images",
            f"{int(arm_n):,}",
            f"+{int(arm_n) - int(base_n):,} mined frames",
        ))
    metric_cards(cards)

    line = _night_share_line(data, arm=arm, arm_row=arm_row)
    if line is not None:
        st.markdown(line)

    # The tour's five bars, not the deep page's thirteen: where we started, the
    # control, the obvious idea, the best score-based arm and this arm -- each read
    # off the same table, in the order the story tells them (the full 13-arm chart
    # stays on the Active Learning page).
    coverage = strategy_coverage(
        data.arms,
        strategies=tour_strategies(data.arms, baseline=data.baseline, arm=arm),
    )
    # Above the chart, not below it (see the docstring).
    _render_fairness_strip(coverage, baseline=data.baseline)
    st.altair_chart(
        bar_chart(
            coverage,
            x="strategy",
            y="delta_night",
            highlight=arm_story_label(arm),
            sort=[str(label) for label in coverage["strategy"]],
            zero_line=True,
            title=_STRATEGY_CHART_TITLE,
            y_title="Δ night mAP50-95",
            # The same tilt the deep page's arm charts use (consolidated review
            # M5): five story labels ("Random sample — control" is 23 characters)
            # crowd a chart this narrow, and `bar_chart` never drops or truncates a
            # label -- so drawn flat they would overlap instead.
            label_angle=-45,
        ),
        width="stretch",
    )
    st.caption(_charted_ids_caption(coverage))
    winner = _best_night_arm_sentence(data.arms, baseline=data.baseline)
    if winner is not None:
        st.markdown(winner)
    similarity = _similarity_sentence(coverage)
    if similarity is not None:
        st.caption(similarity)
    spread = _scene_diversity_sentence(base_row, arm_row)
    if spread is not None:
        st.markdown(spread)
    provenance("recorded", "active_learning_results.parquet")


def _night_map_sentence(data: _TourData, arm: str) -> str | None:
    """"Night mAP50-95 A → B (+D) on the held-out split.", or ``None`` when this
    package's arm table carries no row for one of the two models."""
    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm]
    if base_rows.empty or arm_rows.empty:
        return None
    base_night = float(base_rows.iloc[0]["night_map5095"])
    arm_night = float(arm_rows.iloc[0]["night_map5095"])
    # The table's own recorded delta where it has one (it is the number every other
    # page quotes); the difference of the two figures on screen otherwise.
    recorded = arm_rows.iloc[0].get("delta_night")
    delta = float(recorded) if bool(pd.notna(recorded)) else arm_night - base_night
    return (
        f"Night mAP50-95 {base_night:.4f} → {arm_night:.4f} ({delta:+.4f}) on the "
        "held-out split."
    )


def _hero_recovery_is_low_conf(data: _TourData) -> bool:
    """Whether the hero frame's pedestrian recovery really is the low-confidence
    claim step 5's honesty line calls it: the arm turns a baseline miss into a hit
    on that frame (``_pedestrian_miss_to_hit_frames``), yet ``fixed_boxes`` -- the
    confident-detection rule this step's table applies -- lists nothing for it.

    Derived rather than asserted: the shipped package's hero is exactly that case
    (conf 0.135, below the 0.40 hit floor), but a package whose hero was recovered
    confidently must not carry a sentence calling the recovery unsure."""
    hero = str(data.overview.get("hero_token") or "")
    arm = data.arm
    if not hero or arm is None:
        return False
    frames = _pedestrian_miss_to_hit_frames(data)
    if frames is None or hero not in frames:
        return False
    upgraded = fixed_boxes(
        visible_gt(data.gt, hero),
        data.preds.loc[data.preds["sample_data_token"] == hero],
        baseline=data.baseline,
        arm=arm,
    )
    return bool(upgraded.empty)


def _night_ped_sentence(data: _TourData, arm: str) -> str | None:
    """The DIAGNOSED slice's own before/after -- night pedestrian mAP50-95, stated
    absolute and relative -- or ``None`` when this package's arm table carries no
    night-pedestrian figure for one of the two models.

    The same two numbers the result screen's "Did the model improve?" answer reads,
    from the same two rows; ``gain_text`` is what turns them into a sentence (and
    drops the relative clause when the base leaves no honest ratio), so neither the
    percentage nor its sign is ever written here by hand.
    """
    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm]
    if base_rows.empty or arm_rows.empty:
        return None
    before = base_rows.iloc[0].get("night_ped_map5095")
    after = arm_rows.iloc[0].get("night_ped_map5095")
    if not (bool(pd.notna(before)) and bool(pd.notna(after))):
        return None
    return gain_text("Night pedestrian mAP50-95", float(before), float(after))


def _render_after(data: _TourData) -> None:
    """Step 5 — did the retraining fix the KIND of failure step 1 opened on?

    A before/after is two pictures (Phase 10 spec sec2 row 5), so the hand-approved
    exemplar is drawn twice, side by side, under the one box that actually changed
    hands — the model radio this step used to carry retires, and full inspection is
    the Active Learning deep link's job. The frame stays an illustration: the two
    sentences under it are arm-level results on the held-out split, which is what
    "did it work?" is answered with.

    web-frontend-v1 spec §1.4 makes that hierarchy explicit rather than implied: the
    picture is labelled "One example" before it is shown and the numbers under it are
    labelled as the aggregate result, so the screen's own structure answers "did you
    just pick a flattering image?".
    """
    arm = data.arm
    tokens = [str(token) for token in (data.exemplars.get("tokens") or [])]
    if not tokens or arm is None:
        st.info(_NO_EXEMPLAR_NOTE)
        return
    token = tokens[0]

    st.subheader(_AFTER_HEADLINE)
    # Tier 1, labelled before the picture rather than caveated after it.
    st.caption(_ONE_EXAMPLE_TIER)

    gt_rows = visible_gt(data.gt, token)
    frame_preds = data.preds.loc[data.preds["sample_data_token"] == token]
    upgraded = fixed_boxes(gt_rows, frame_preds, baseline=data.baseline, arm=arm)

    # The callout names ONE box, in the claim strings the table itself writes -- a
    # second rendering of the same claim here would be a second thing to keep true.
    callout = upgrade_callout(upgraded)
    if callout is not None:
        category, before, after = callout
        metric_cards(
            [
                (f"Before — {arm_story_label(data.baseline)}", f"{category}: {before}"),
                (f"After — {arm_story_label(arm)}", f"{category}: {after}"),
            ],
            per_row=2,
        )

    crop = crop_path(token)
    if crop.is_file():
        image = Image.open(crop)
        for column, model in zip(st.columns(2), (data.baseline, arm), strict=True):
            with column:
                st.image(
                    draw_overlay(
                        image,
                        gt_for_render(gt_rows, model),
                        frame_preds.loc[frame_preds["model"] == model],
                        mode="overlay",
                        scale=0.6,
                    ),
                    width="stretch",
                )
                # Story label AND raw id (the spec's honesty rule, consolidated
                # review I1): this was the one screen naming an arm readably with
                # the identifier it stands for nowhere on it.
                st.caption(f"{arm_story_label(model)} (`{model}`)")
        # Under the pair, not above it: the legend describes colours this step just
        # drew, so it is written only where an overlay was actually rendered.
        legend()

    # Tier 2: the arm-level numbers on the held-out split -- written under their own
    # lead-in, and the lead-in only where this package has such a number to state.
    ped_sentence = _night_ped_sentence(data, arm)
    sentence = _night_map_sentence(data, arm)
    if ped_sentence or sentence:
        st.markdown(_AGGREGATE_TIER)
    if ped_sentence:
        st.markdown(ped_sentence)
    if sentence:
        st.markdown(sentence)
    if _hero_recovery_is_low_conf(data):
        st.markdown(_HERO_HONESTY_LINE)
    _held_out_caption(data, token)

    if upgraded.empty:
        st.caption(_NO_UPGRADED_BOXES_NOTE)
    else:
        with st.expander(_PER_BOX_FOLD):
            st.dataframe(upgraded, hide_index=True)
    provenance("recomputed", "per-box claims from predictions.parquet")
    if st.button("Open this exemplar in Active Learning →", key="tour_open_exemplar"):
        _open("active_learning", al_exemplar=token)


def _hero_recovery_conf(data: _TourData) -> float | None:
    """The arm's raw confidence on the hero frame's low-confidence pedestrian
    recovery -- the same claim ``_hero_recovery_is_low_conf`` establishes is a
    low_conf hit, read back out through ``_claim`` (the pedestrian-claim helper
    step 1 already uses) rather than re-deriving it. Only meaningful when the
    caller has already checked ``_hero_recovery_is_low_conf``; ``None`` otherwise
    (defensive -- never hit on the shipped package)."""
    hero = str(data.overview.get("hero_token") or "")
    arm = data.arm
    if not hero or arm is None:
        return None
    gt_rows = visible_gt(data.gt, hero)
    baseline_column, arm_column = f"matched_{data.baseline}", f"matched_{arm}"
    if not {baseline_column, arm_column} <= set(gt_rows.columns):
        return None
    peds = gt_rows.loc[
        (gt_rows["category_group"] == "pedestrian")
        & gt_rows[baseline_column].eq(False).fillna(False)
        & gt_rows[arm_column].eq(True).fillna(False)
    ]
    if peds.empty:
        return None
    annotation = str(peds.iloc[0]["annotation_token"])
    claim = _claim(data.preds, token=hero, model=arm, annotation=annotation)
    return claim[0] if claim else None


def _result_weakness(data: _TourData) -> None:
    """Answer 1 -- "What weakness did we find?": the same recorded/recomputed
    numbers step 0 opens on, restated as the tour's own finding."""
    st.markdown("**What weakness did we find?**")
    rows = data.arms.loc[data.arms["arm"] == data.baseline]
    if rows.empty:
        st.info(STALE_PACKAGE_NOTE)
        return
    row = rows.iloc[0]
    st.markdown(
        f"Night mAP50-95 {float(row['night_map5095']):.4f} vs overall "
        f"{float(row['overall_map5095']):.4f}."
    )
    night_ped = row.get("night_ped_map5095")
    if night_ped is not None and bool(pd.notna(night_ped)):
        st.markdown(f"Night pedestrian mAP50-95 {float(night_ped):.4f}.")

    counts = _night_miss_counts(data)
    if counts is None:
        st.info(STALE_PACKAGE_NOTE)
    else:
        n_night, n_night_fn, _n_val = counts
        st.markdown(
            f"{n_night_fn} of {n_night} night validation frames carry at least one "
            "baseline miss."
        )
    provenance("recorded", "mAP from active_learning_results.parquet")
    if counts is not None:
        provenance("recomputed", "frame counts from frame_manifest/gt_boxes/predictions")


def _mined_comparator_clause(data: _TourData, *, name: str, label: str) -> str | None:
    """"{label} covered {n} scene(s) at {p:.0%} night" for the comparator arm
    ``name``, or ``None`` when this package carries no row for it (an older
    package, or one that never ran that arm) -- omitted rather than guessed.

    The noun agrees with the count, the way ``active_learning._strategy_triple``
    does: a one-scene comparator used to read "1 scenes".
    """
    rows = data.arms.loc[data.arms["arm"] == name]
    if rows.empty:
        return None
    row = rows.iloc[0]
    n_scenes, night_share = row.get("n_scenes"), row.get("night_share")
    if not (bool(pd.notna(n_scenes)) and bool(pd.notna(night_share))):
        return None
    scenes = int(n_scenes)
    return (
        f"{label} covered {scenes} scene{'' if scenes == 1 else 's'} at "
        f"{float(night_share):.0%} night"
    )


def _result_data_added(data: _TourData) -> None:
    """Answer 2 -- "What data did we add?": the arm's own mined-set composition,
    against the ``random``/``mined`` comparators when this package carries them."""
    st.markdown("**What data did we add?**")
    arm = data.arm
    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm] if arm is not None else data.arms.iloc[:0]
    if arm is None or base_rows.empty or arm_rows.empty:
        st.info(STALE_PACKAGE_NOTE)
        return
    base_row, arm_row = base_rows.iloc[0], arm_rows.iloc[0]
    base_n, arm_n = base_row.get("n_train_images"), arm_row.get("n_train_images")
    n_scenes, night_share = arm_row.get("n_scenes"), arm_row.get("night_share")
    if not (
        bool(pd.notna(base_n))
        and bool(pd.notna(arm_n))
        and bool(pd.notna(n_scenes))
        and bool(pd.notna(night_share))
    ):
        st.info(STALE_PACKAGE_NOTE)
        return

    n_mined = int(arm_n) - int(base_n)
    scenes = int(n_scenes)
    # Both nouns agree with their counts (the _strategy_triple rule): a one-frame,
    # one-scene arm used to read "1 frames across 1 scenes".
    sentence = (
        f"`{arm}` mined {n_mined:,} frame{'' if n_mined == 1 else 's'} across "
        f"{scenes} scene{'' if scenes == 1 else 's'}, "
        f"{float(night_share):.0%} at night"
    )
    clauses = [
        clause
        for clause in (
            _mined_comparator_clause(data, name="random", label="random"),
            _mined_comparator_clause(data, name="mined", label="similarity mining"),
        )
        if clause is not None
    ]
    if clauses:
        sentence += " — " + "; ".join(clauses)
    st.markdown(sentence + ".")
    # What the comparator clause above leaves unexplained: WHY the similarity arm's
    # set looks like that. Written from the same helper step 4 captions, so the two
    # screens cannot drift apart on it, and only when that arm's set really is a
    # daytime one.
    similarity = _similarity_sentence(data.arms)
    if similarity is not None:
        st.markdown(similarity)
    provenance(
        "recorded",
        "n_train_images, n_scenes, night_share from active_learning_results.parquet",
    )


def _result_improved(data: _TourData) -> None:
    """Answer 3 -- "Did the model improve?": the night result, the night
    pedestrian slice, the arm's own overall delta, and (only when a different arm
    actually won it) which arm took the best overall gain."""
    st.markdown("**Did the model improve?**")
    arm = data.arm
    if arm is None:
        st.info(STALE_PACKAGE_NOTE)
        return

    sentence = _night_map_sentence(data, arm)
    if sentence:
        st.markdown(sentence)

    # The diagnosed slice, in the phrasing the tour states it in everywhere else
    # (step 5 writes this same sentence from this same helper): absolute AND
    # relative, both computed by ``gain_text`` rather than written here.
    ped_sentence = _night_ped_sentence(data, arm)
    if ped_sentence:
        st.markdown(ped_sentence)

    arm_rows = data.arms.loc[data.arms["arm"] == arm]
    if not arm_rows.empty:
        delta_overall = arm_rows.iloc[0].get("delta_overall")
        if bool(pd.notna(delta_overall)):
            st.markdown(f"Overall mAP50-95 {float(delta_overall):+.4f} against baseline.")

    ranked = data.arms.dropna(subset=["delta_overall"])
    if not ranked.empty:
        best = ranked.loc[ranked["delta_overall"].idxmax()]
        if str(best["arm"]) != arm:
            st.markdown(
                f"The best overall arm is `{best['arm']}` ({float(best['delta_overall']):+.4f}); "
                "the night arm trades overall gain for night gain."
            )
    provenance("recorded", "mAP from active_learning_results.parquet")


def _result_failures(data: _TourData) -> None:
    """Answer 4 -- "What failed along the way?": the weak-supervision loss split,
    the headline pair's sparse-frame bias, the worst night arm (only when that arm
    really is a weak-supervision result), and (only when true) the hero recovery's
    own low-confidence honesty line."""
    st.markdown("**What failed along the way?**")
    wrote_recorded = False

    headline_rows = data.loss.loc[data.loss["headline"]]
    headline_base_arm: str | None = None
    if not headline_rows.empty:
        row = headline_rows.iloc[0]
        headline_base_arm = str(row["base_arm"])
        st.markdown(
            f"Weak (VLM-verified) labels kept {float(row['retention']):.1%} of the "
            f"ground-truth gain for the `{row['base_arm']}` pair: "
            f"{float(row['dropped_frame_share']):.1%} was lost with the frames the "
            f"verifier dropped, {float(row['label_share']):.1%} to label noise on "
            "the ones it kept."
        )
        wrote_recorded = True

    # ONE sparse-frames sentence, about the SAME pair the loss split above is about
    # (Phase 9a review I7). The real package carries a weak_supervision_results row
    # per weak arm, and looping them wrote the same sentence three times over on a
    # screen whose whole job is to answer one question in a few lines; the headline
    # pair is the one the rest of this answer already talks about.
    if headline_base_arm is not None and {
        "gt_boxes_per_accepted_frame",
        "gt_boxes_per_rejected_frame",
    } <= set(data.weaksup.columns):
        headline_weak = data.weaksup.loc[data.weaksup["arm"] == headline_base_arm]
        if not headline_weak.empty:
            weak_row = headline_weak.iloc[0]
            st.markdown(
                "The verifier kept sparse frames: "
                f"{float(weak_row['gt_boxes_per_accepted_frame']):.2f} vs "
                f"{float(weak_row['gt_boxes_per_rejected_frame']):.2f} GT boxes per accepted "
                f"vs rejected frame ({weak_row['arm']})."
            )
            wrote_recorded = True

    # The superlative is computed over EVERY arm, because that is what "of the N
    # arms" claims (Phase 9a review I2): the worst night arm in the whole table is
    # read first, and the sentence is written only when that arm really is a
    # weak-supervision RESULT -- a "weak_<base>" arm trained on pseudo labels,
    # never the GT-trained "weak_<base>_gt" twin export_weak_loss_decomposition
    # uses to compute the loss split above, and never a non-weak arm. When some
    # other arm is the worst there is no honest weak-supervision superlative to
    # write, so this screen writes none.
    night_ranked = data.arms.dropna(subset=["delta_night"])
    if not night_ranked.empty:
        worst = night_ranked.loc[night_ranked["delta_night"].idxmin()]
        worst_arm = str(worst["arm"])
        if worst_arm.startswith("weak_") and not worst_arm.endswith("_gt"):
            st.markdown(
                f"`{worst_arm}` is the worst night result of the {len(night_ranked)} arms "
                f"({float(worst['delta_night']):+.4f})."
            )
            wrote_recorded = True

    if wrote_recorded:
        provenance(
            "recorded",
            "weak_loss_decomposition.parquet, weak_supervision_results.parquet, "
            "active_learning_results.parquet",
        )

    hero_conf = _hero_recovery_conf(data) if _hero_recovery_is_low_conf(data) else None
    if hero_conf is not None:
        st.markdown(
            f"The hero frame's pedestrian recovery is a low-confidence claim "
            f"(conf {hero_conf:.3f}) — counted as a hit by the matching rule, not a "
            "confident detection."
        )
        provenance("recomputed", "claim confidence from predictions.parquet")

    if not wrote_recorded and hero_conf is None:
        st.info(STALE_PACKAGE_NOTE)


def _result_headline(arm_row: pd.Series) -> str | None:
    """Which of the two closing headlines this package has earned, or ``None`` when
    it recorded no night delta at all.

    The headline is a claim like any other sentence on the screen: "measurable
    improvement" is written only where the arm's own recorded ``delta_night`` is a
    gain, a regression (or a dead-flat result) gets "measured result" instead, and
    a package that never recorded the delta gets no headline rather than one its
    tables cannot support. The loop closed either way -- that is what the four
    answers below show -- so the honest branch still says so.
    """
    delta = arm_row.get("delta_night")
    if not bool(pd.notna(delta)):
        return None
    return _CLOSED_LOOP_IMPROVED if float(delta) > 0 else _CLOSED_LOOP_MEASURED


def _result_hero_cards(
    data: _TourData, *, base_row: pd.Series, arm_row: pd.Series
) -> list[tuple[str, str] | tuple[str, str, str]]:
    """The three numbers the whole tour was for: how much data was added, what made
    it targeted, and what that bought on the diagnosed slice.

    A card whose source is NA is omitted rather than rendered as "nan" (the step-4
    rule); the night-share delta clause is written only where this package really
    ran the random control; and the night-pedestrian card states its absolute gain
    alone when ``relative_gain`` finds no honest base to divide by. Values stay
    short -- st.metric renders them in a narrow column and truncates a long one
    mid-word (Phase 9a review 5a) -- so the fuller figures ride in the deltas.

    The night-pedestrian delta is the absolute gain and nothing else (consolidated
    review M5): it used to carry the "0.0826 → 0.1171" arrow as well, which is a
    second copy of the arrow answer 3's own ``gain_text`` sentence writes a few
    lines below, in a delta line narrower than the sentence.
    """
    cards: list[tuple[str, str] | tuple[str, str, str]] = []
    base_n, arm_n = base_row.get("n_train_images"), arm_row.get("n_train_images")
    if bool(pd.notna(base_n)) and bool(pd.notna(arm_n)):
        cards.append(("Targeted frames added", f"{int(arm_n) - int(base_n):,}"))

    night_share = arm_row.get("night_share")
    if bool(pd.notna(night_share)):
        share = f"{float(night_share):.0%}"
        control_rows = data.arms.loc[data.arms["arm"] == "random"]
        control = control_rows.iloc[0].get("night_share") if not control_rows.empty else None
        if control is not None and bool(pd.notna(control)):
            cards.append((
                "Night share of mined data",
                share,
                f"vs {float(control):.0%} random control",
            ))
        else:
            cards.append(("Night share of mined data", share))

    before, after = base_row.get("night_ped_map5095"), arm_row.get("night_ped_map5095")
    if bool(pd.notna(before)) and bool(pd.notna(after)):
        delta = float(after) - float(before)
        relative = relative_gain(float(before), float(after))
        if relative is None:
            cards.append(("Night-pedestrian mAP50-95", f"{delta:+.4f} absolute"))
        else:
            cards.append((
                "Night-pedestrian mAP50-95",
                f"{relative:+.1%} relative",
                f"{delta:+.4f} absolute",
            ))
    return cards


def _render_result(data: _TourData) -> None:
    """Step 6 — the result screen (spec §2 row 6): the computed headline and the
    three hero numbers, then the four bordered answers, then the one sentence they
    are the evidence for. Every number is read from the same package tables every
    earlier step reads. The loop breadcrumb lights all four stages here (see
    ``_STEPS`` below) -- by this screen the tour has walked the whole loop, not one
    stage of it."""
    arm = data.arm
    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm] if arm is not None else data.arms.iloc[:0]
    if not base_rows.empty and not arm_rows.empty:
        base_row, arm_row = base_rows.iloc[0], arm_rows.iloc[0]
        headline = _result_headline(arm_row)
        if headline is not None:
            st.subheader(headline)
        cards = _result_hero_cards(data, base_row=base_row, arm_row=arm_row)
        if cards:
            metric_cards(cards)

    with st.container(border=True):
        _result_weakness(data)
    with st.container(border=True):
        _result_data_added(data)
    with st.container(border=True):
        _result_improved(data)
    with st.container(border=True):
        _result_failures(data)
    st.markdown(_CLOSING_THESIS)


# Step numbering: 0-based in this module (the ``tour_step`` session key, ``_STEPS``
# indices, and the docstrings' "step N" references) and 1-based in everything the
# viewer reads ("Step 4 of 7", and the prose cross-references inside the steps
# themselves). "Step 4" in a rendered sentence therefore means ``_STEPS[3]``; in a
# docstring it means ``_STEPS[4]``.
_STEPS: tuple[_Step, ...] = (
    _Step(
        key="weakness",
        title="We found a blind spot",
        stage="Diagnose",
        render=_render_weakness,
        links=(("failures", "Failure Explorer — see every night frame the baseline missed"),),
    ),
    _Step(
        key="missed_pedestrian",
        title="What the failure looks like",
        stage="Diagnose",
        render=_render_missed_pedestrian,
        links=(("failures", "Failure Explorer — inspect this failure on every val frame"),),
    ),
    _Step(
        key="mine",
        title="Where else does this happen?",
        stage="Mine",
        render=_render_mine,
        links=(("scenarios", "Scenario Search — find every braking-near-pedestrian event"),),
    ),
    _Step(
        key="why_selected",
        title="What data should we add?",
        stage="Mine",
        render=_render_why_selected,
        links=(
            ("active_learning", "Active Learning — inspect the full mined set and its communities"),
        ),
    ),
    _Step(
        key="retrain",
        title="We changed the training data",
        stage="Train",
        render=_render_retrain,
        links=(("active_learning", "Active Learning — every arm, its quotas and the full table"),),
    ),
    _Step(
        key="after",
        title="Did it fix the failure?",
        stage="Evaluate",
        render=_render_after,
        links=(("active_learning", "Active Learning — every hand-approved before/after frame"),),
    ),
    _Step(
        key="result",
        title="Closed the loop",
        # All four stages, lit together: this screen doesn't add a stage of its
        # own, it closes the loop the first six screens walked one stage at a time.
        stage=LOOP_STAGES,
        render=_render_result,
        # Story sentences, not page descriptions -- and none of them names a
        # count: a static label is the one string on the screen nothing recomputes,
        # so a number in one would be the only unchecked figure in the tour.
        links=(
            ("overview", "Overview — the headline results in one screen"),
            ("failures", "Failure Explorer — every miss, filterable by condition"),
            ("scenarios", "Scenario Search — every preset, every matching event"),
            ("active_learning", "Active Learning — the full experiment, all arms"),
            ("weak_supervision", "Weak Supervision — where the rest of the gain went"),
            ("chat_replay", "Ask the Dataset — recorded chat replays"),
        ),
    ),
)


def _set_step(step: int) -> None:
    st.session_state[TOUR_STEP_KEY] = max(0, min(step, len(_STEPS) - 1))


def _current_step() -> int:
    raw = st.session_state.get(TOUR_STEP_KEY, 0)
    try:
        step = int(raw)
    except (TypeError, ValueError):
        step = 0
    return max(0, min(step, len(_STEPS) - 1))


# At most three "Go deeper" links per row: the result screen carries six, and six
# st.columns across the content width left each page_link ~150 px for a label like
# "Active Learning — inspect the full mined set and its communities", which wrapped
# to four lines of two words (Phase 9a review 5a). Rows of three give every label
# the width of two of the old columns; a step with one or two links still gets a
# single row.
_GO_DEEPER_PER_ROW = 3


def _render_go_deeper(step: _Step) -> None:
    if not step.links:
        return
    st.markdown("**Go deeper**")
    for start in range(0, len(step.links), _GO_DEEPER_PER_ROW):
        row = step.links[start : start + _GO_DEEPER_PER_ROW]
        # A short final row is laid out over the FULL row width (st.columns(len(row)))
        # rather than padded to three, so two links never render as two thirds of a
        # row with a hole where the third would be.
        for column, (url_path, label) in zip(st.columns(len(row)), row, strict=True):
            column.page_link(nav.page(url_path), label=label)


def render() -> None:
    st.title("From model failure to better training data")
    st.caption(
        "See how the data engine finds a perception weakness, mines targeted AV "
        "scenarios, and measures whether retraining fixes it."
    )
    data = _load()
    step = _current_step()
    st.session_state[TOUR_STEP_KEY] = step
    last = len(_STEPS) - 1

    # The nav row is drawn FIRST and moves the step in an on_click callback, which
    # streamlit runs BEFORE the script redraws -- so `step` here is already the
    # step the click asked for, and the row's own disabled flags describe it. (The
    # inline alternative -- read the button's return value and mutate afterwards --
    # renders one screen where the content has moved on but Back/Next are still
    # disabled for the step the viewer just left, and `st.rerun()` to fix that is
    # not available to AppTest's step-by-step run model.) The target index is
    # computed here rather than in the callback for the same reason: it is the step
    # the viewer was actually looking at when they clicked.
    columns = st.columns([1, 1, 6])
    columns[0].button(
        "← Back", key="tour_back", disabled=step == 0, on_click=_set_step, args=(step - 1,)
    )
    columns[1].button(
        "Next →",
        key="tour_next",
        type="primary",
        disabled=step == last,
        on_click=_set_step,
        args=(step + 1,),
    )

    current = _STEPS[step]
    # ``stage`` is a single name for an ordinary step, a tuple for the result
    # screen (all four, lit together), or None -- loop_breadcrumb already takes a
    # Sequence[str] | None, so a bare string is the only shape that needs wrapping.
    lit: Sequence[str] | None
    if current.stage is None:
        lit = None
    elif isinstance(current.stage, str):
        lit = [current.stage]
    else:
        lit = current.stage
    loop_breadcrumb(lit)
    st.caption(f"Step {step + 1} of {len(_STEPS)} · {current.title}")
    with st.container(border=True):
        current.render(data)
    _render_go_deeper(current)
    if step == last:
        st.button("Restart the tour", key="tour_restart", on_click=_set_step, args=(0,))
