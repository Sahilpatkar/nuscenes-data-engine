"""Guided tour — the demo's default 2-3 minute path through the loop (Phase 9a,
docs/superpowers/specs/2026-08-22-demo-phase9a-design.md §1).

Seven steps, one screen each: the weakness, the frame it shows up on, mining for
more like it, why one mined frame was picked, the retrain, the same kind of frame
after, and the result screen. The step index is the page's only state
(``st.session_state["tour_step"]``); Back/Next move it.

This module writes prose, never figures: every number on a step is derived here
from the same package tables the deep pages read, each step says where its numbers
came from (``render.provenance``), and each step lights its own stage in the loop
breadcrumb.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import nav
import pandas as pd
import streamlit as st
from filters import failure_flags, gt_for_render, model_label, visible_gt
from PIL import Image
from render import draw_overlay, loop_breadcrumb, metric_cards, provenance

from data import (
    STALE_PACKAGE_NOTE,
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
    load_weak_loss,
    load_weaksup,
)
from views.overview import HERO_CAPTION

TOUR_STEP_KEY = "tour_step"

# The matching rule's confidence floor: a prediction that claims a GT box with
# conf below it is a "low_conf" hit -- still a hit (the box's matched_<model> is
# True), just an unsure one. The package ships no copy of the value (no manifest
# key, no column), so it is restated here from its source: `sweep.conf_hit: 0.40`
# in configs/active_learning.yaml, the default that
# src/nuscenes_data_engine/active_learning/matching.py applies and
# src/nuscenes_data_engine/demo/infer.py runs the demo's own matching with.
_CONF_HIT_FLOOR = 0.40

_NO_HERO_CROP_NOTE = "hero crop not in this package"

# Steps 2-6 land in the next tasks (plan Tasks 3 and 4) -- an honest placeholder
# line, never a half-built claim.
_PLACEHOLDER_NOTE = "Step lands in the next task"


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
    """One screen of the tour. ``stage`` is the loop stage it lights (``None``
    lights nothing); ``links`` are the "Go deeper" deep links, as
    ``(url_path, label)`` pairs resolved through ``nav.page``."""

    key: str
    title: str
    stage: str | None
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


def _visible(gt: pd.DataFrame) -> pd.DataFrame:
    """``gt_boxes`` with the visibility-floor rows dropped -- the same slice the
    Failure Explorer counts and draws, since a row under the floor was never scored
    against and so was never missed by anyone."""
    if "below_visibility_min" not in gt.columns:
        return gt
    return gt.loc[~gt["below_visibility_min"].fillna(False)]


def _night_miss_counts(data: _TourData) -> tuple[int, int, int] | None:
    """``(night val frames, of those with >= 1 baseline miss, val frames)``, or
    ``None`` when this package carries no ``matched_<baseline>`` column to count
    misses from."""
    if f"matched_{data.baseline}" not in data.gt.columns:
        return None
    flags = failure_flags(data.manifest, _visible(data.gt), data.preds, model=data.baseline)
    is_night = flags["is_night"].fillna(False).astype(bool)
    return int(is_night.sum()), int((is_night & flags["has_fn"]).sum()), len(flags)


def _render_weakness(data: _TourData) -> None:
    """Step 0 — the sliced evaluation that makes the weakness visible."""
    rows = data.arms.loc[data.arms["arm"] == data.baseline]
    if rows.empty:
        st.info(STALE_PACKAGE_NOTE)
        return
    row = rows.iloc[0]

    cards = [
        ("Overall mAP50-95", f"{row['overall_map5095']:.4f}"),
        ("Night mAP50-95", f"{row['night_map5095']:.4f}"),
    ]
    # night_ped_map5095 is the night PEDESTRIAN class -- the mechanism the night
    # arms move. It is absent (or NA) on a package whose eval never wrote per-class
    # night metrics, and an absent card is better than a card reading "nan".
    night_ped = row.get("night_ped_map5095")
    if night_ped is not None and bool(pd.notna(night_ped)):
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
    st.markdown(
        f"A single overall number hides where a detector fails; the sliced evaluation "
        f"shows it — `{data.baseline}` scores {row['overall_map5095']:.4f} mAP50-95 "
        f"overall and {row['night_map5095']:.4f} at night. Night is the weakness this "
        f"tour follows: it is what the next steps mine for, retrain on, and measure."
    )
    provenance("recorded", "mAP from active_learning_results.parquet")
    if recomputed_detail:
        provenance("recomputed", recomputed_detail)


def _claim_text(preds: pd.DataFrame, *, token: str, model: str, annotation: str) -> str:
    """What ``model`` claimed about one GT box: its confidence and status, or "no
    claim" when it never matched the box at all (an honest absence -- a model that
    said nothing is not a model that said something weak)."""
    rows = preds.loc[
        (preds["sample_data_token"] == token)
        & (preds["model"] == model)
        & (preds["matched_annotation_token"] == annotation)
    ]
    if rows.empty:
        return "no claim"
    row = rows.sort_values("conf", ascending=False).iloc[0]
    conf, status = float(row["conf"]), str(row["status"])
    if status == "low_conf":
        return (
            f"{conf:.3f} (low_conf — below the {_CONF_HIT_FLOOR:.2f} hit floor; the "
            "matching rule still counts it as a hit)"
        )
    return f"{conf:.3f} ({status})"


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
            f"{model_label(model)}: "
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
    boxes = _visible(data.gt)
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


def _render_missed_pedestrian(data: _TourData) -> None:
    """Step 1 — the weakness on one frame: what each model claimed about the
    pedestrian the baseline missed."""
    hero = str(data.overview.get("hero_token") or "")
    crop = crop_path(hero) if hero else None
    if crop is None or not crop.is_file():
        st.info(_NO_HERO_CROP_NOTE)
        return

    models = [name for name in (data.baseline, data.arm) if name is not None]
    model = (
        st.radio(
            "Model", models, key="tour_hero_model", horizontal=True, format_func=model_label
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
    st.caption(HERO_CAPTION)

    for line in _pedestrian_facts(data, token=hero, gt_rows=gt_rows, models=models):
        st.markdown(line)
    sentence = _miss_to_hit_sentence(data, hero)
    if sentence:
        st.markdown(sentence)
    provenance("recomputed", "boxes, claims and counts from gt_boxes/predictions")


def _render_placeholder(data: _TourData) -> None:
    st.caption(_PLACEHOLDER_NOTE)


_STEPS: tuple[_Step, ...] = (
    _Step(
        key="weakness",
        title="The weakness: night",
        stage="Diagnose",
        render=_render_weakness,
        links=(("failures", "Failure Explorer — the whole val split, filterable"),),
    ),
    _Step(
        key="missed_pedestrian",
        title="One missed pedestrian",
        stage="Diagnose",
        render=_render_missed_pedestrian,
        links=(("failures", "Failure Explorer — the same overlay on every val frame"),),
    ),
    _Step(key="mine", title="Find more like it", stage="Mine", render=_render_placeholder),
    _Step(
        key="why_selected",
        title="Why this frame was picked",
        stage="Mine",
        render=_render_placeholder,
    ),
    _Step(
        key="retrain",
        title="Retrain on what was found",
        stage="Train",
        render=_render_placeholder,
    ),
    _Step(
        key="after",
        title="Same kind of frame, after",
        stage="Evaluate",
        render=_render_placeholder,
    ),
    _Step(
        key="result",
        title="What we found, added, gained — and what failed",
        stage="Evaluate",
        render=_render_placeholder,
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


def _render_go_deeper(step: _Step) -> None:
    if not step.links:
        return
    st.markdown("**Go deeper**")
    for column, (url_path, label) in zip(st.columns(len(step.links)), step.links, strict=True):
        column.page_link(nav.page(url_path), label=label)


def render() -> None:
    st.title("Guided tour")
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
    loop_breadcrumb([current.stage] if current.stage else None)
    st.caption(f"Step {step + 1} of {len(_STEPS)} · {current.title}")
    with st.container(border=True):
        current.render(data)
    _render_go_deeper(current)
    if step == last:
        st.button("Restart", key="tour_restart", on_click=_set_step, args=(0,))
