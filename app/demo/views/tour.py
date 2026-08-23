"""Guided tour — the demo's default 2-3 minute path through the loop (Phase 9a,
docs/superpowers/specs/2026-08-22-demo-phase9a-design.md §1).

Seven steps, one screen each: the weakness, the frame it shows up on, mining for
more like it, why one mined frame was picked, the retrain, the same kind of frame
after, and the result screen. The step index is the page's only state
(``st.session_state["tour_step"]``); Back/Next move it.

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
    failure_flags,
    fixed_boxes,
    gt_for_render,
    model_label,
    parity_short,
    rank_events,
    selection_factors,
    severity_caption,
    tour_frame_candidates,
    visible_gt,
    visible_gt_boxes,
)
from nav import TOUR_STEP_KEY
from PIL import Image
from render import (
    LOOP_STAGES,
    bar_chart,
    chip_row,
    draw_overlay,
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
_OVERLAY_LEGEND = (
    "Green = ground truth, orange dashed = a GT box this model missed, white = its "
    "true positives, yellow dotted = a claim below the confidence floor, red = a "
    "false positive."
)
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

# scenario_events' t-2..t+2 neighbour columns in strip order, with the event itself
# (column None) in the middle. Plain ASCII hyphens in the labels, matching
# views/scenarios.py's own _BEFORE_STEPS/_AFTER_STEPS -- ruff RUF001 flags the design
# doc's typographic U+2212 minus as an ambiguous character.
_FILMSTRIP_STEPS: tuple[tuple[str | None, str], ...] = (
    ("t_minus2", "t-2"),
    ("t_minus1", "t-1"),
    (None, "current"),
    ("t_plus1", "t+1"),
    ("t_plus2", "t+2"),
)


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
    """What ``model`` claimed about one GT box: its confidence and status, or "no
    claim" when it never matched the box at all (an honest absence -- a model that
    said nothing is not a model that said something weak)."""
    claim = _claim(preds, token=token, model=model, annotation=annotation)
    if claim is None:
        return "no claim"
    conf, status = claim
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
    exactly as that page leaves it out."""
    steps = []
    for column, label in _FILMSTRIP_STEPS:
        token = row["sample_data_token"] if column is None else row.get(column)
        if pd.notna(token):
            steps.append((label, str(token)))
    if not steps:
        return
    for strip_column, (label, token) in zip(st.columns(len(steps)), steps, strict=True):
        with strip_column:
            thumb = thumb_path(token)
            if thumb.is_file():
                st.image(str(thumb))
            st.caption(label)


def _render_mine(data: _TourData) -> None:
    """Step 2 — the weakness as a population: the flagship scenario query's top
    event, its own facts, and how many of the matching events are at night."""
    if data.events is None:
        st.info(_EVENTS_ABSENT_NOTE)
        return
    ranked = rank_events(data.events, _TOUR_PRESET)
    if ranked.empty:
        st.info(_NO_TOUR_EVENT_NOTE)
        return
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

    # The parity line is a trust indicator, not a claim about this event: a package
    # with no subgraphs staged simply omits it (the Scenario page's own no-op).
    payload = load_subgraphs(_TOUR_PRESET) if subgraphs_available() else None
    if payload is not None:
        st.caption(
            parity_short(int(payload["sql_count"]), payload["cypher_count"], payload["parity"])
        )

    n_night = int(ranked["is_night"].fillna(False).astype(bool).sum())
    st.markdown(f"{n_night} of {len(ranked)} matching events are at night.")
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
    by ``demo al-explain`` rather than reasoned about here."""
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
    for label, value, flag in selection_factors(
        explain_rows.iloc[0].to_dict(), n_communities=n_communities, night_floor=night_floor
    ):
        mark = "" if flag is None else (" ✓" if flag else " ✗")
        st.markdown(f"**{label}:** {value}{mark}")

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
    floor_text = (
        "a night floor takes a minimum number of night frames first"
        if night_floor is None
        else f"a night floor takes {night_floor} night frames first"
    )
    st.markdown(
        f"Nothing about this frame on its own selected it: its community carried failure "
        f"mass, that mass bought the community a quota, the frame ranked high enough "
        f"inside it by similarity — and {floor_text}."
    )
    n_selected = int(data.validation.get("n_selected", len(data.explain)))
    provenance(
        "reproduced",
        f"demo al-explain reproduced the run: {n_selected:,} frames, "
        f"{n_communities} communities",
    )
    if st.button("Open this frame in Active Learning →", key="tour_open_al_frame"):
        _open("active_learning", al_frame_token=token)


def _render_retrain(data: _TourData) -> None:
    """Step 4 — what the mining bought the training set, and where this arm lands
    among every arm the experiment ran."""
    arm = data.arm
    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm] if arm is not None else data.arms.iloc[:0]
    if base_rows.empty or arm_rows.empty:
        st.info(STALE_PACKAGE_NOTE)
        return
    base_row, arm_row = base_rows.iloc[0], arm_rows.iloc[0]

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

    ordered = (
        data.arms.sort_values("round_order").reset_index(drop=True)
        if "round_order" in data.arms.columns
        else data.arms
    )
    st.altair_chart(
        bar_chart(
            ordered,
            x="arm",
            y="delta_night",
            highlight=arm,
            # round_order's own order: the arms read as the experiment ran them.
            sort=[str(name) for name in ordered["arm"]],
            label_angle=-45,
            title="Night mAP50-95 vs baseline, all arms",
            y_title="Δ night mAP50-95",
        ),
        width="stretch",
    )
    st.markdown(
        "Every arm retrains the same detector with the same budget and is scored on the "
        "same held-out split; `random` is the control."
    )
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


def _render_after(data: _TourData) -> None:
    """Step 5 — a hand-approved before/after frame, and the arm-level number that
    is the actual result."""
    arm = data.arm
    tokens = [str(token) for token in (data.exemplars.get("tokens") or [])]
    if not tokens or arm is None:
        st.info(_NO_EXEMPLAR_NOTE)
        return
    token = tokens[0]
    models = [data.baseline, arm]
    model = (
        st.radio(
            "Model", models, key="tour_exemplar_model", horizontal=True, format_func=model_label
        )
        or models[0]
    )

    gt_rows = visible_gt(data.gt, token)
    frame_preds = data.preds.loc[data.preds["sample_data_token"] == token]
    crop = crop_path(token)
    if crop.is_file():
        st.image(
            draw_overlay(
                Image.open(crop),
                gt_for_render(gt_rows, model),
                frame_preds.loc[frame_preds["model"] == model],
                mode="overlay",
                scale=0.6,
            ),
            width="stretch",
        )
    st.caption(_OVERLAY_LEGEND)

    upgraded = fixed_boxes(gt_rows, frame_preds, baseline=data.baseline, arm=arm)
    if upgraded.empty:
        st.caption(_NO_UPGRADED_BOXES_NOTE)
    else:
        st.dataframe(upgraded, hide_index=True)

    sentence = _night_map_sentence(data, arm)
    if sentence:
        st.markdown(sentence)
    if _hero_recovery_is_low_conf(data):
        st.markdown(_HERO_HONESTY_LINE)
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
    """"{label} covered {n} scenes at {p:.0%} night" for the comparator arm
    ``name``, or ``None`` when this package carries no row for it (an older
    package, or one that never ran that arm) -- omitted rather than guessed."""
    rows = data.arms.loc[data.arms["arm"] == name]
    if rows.empty:
        return None
    row = rows.iloc[0]
    n_scenes, night_share = row.get("n_scenes"), row.get("night_share")
    if not (bool(pd.notna(n_scenes)) and bool(pd.notna(night_share))):
        return None
    return f"{label} covered {int(n_scenes)} scenes at {float(night_share):.0%} night"


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
    sentence = (
        f"`{arm}` mined {n_mined:,} frames across {int(n_scenes)} scenes, "
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

    base_rows = data.arms.loc[data.arms["arm"] == data.baseline]
    arm_rows = data.arms.loc[data.arms["arm"] == arm]
    if not base_rows.empty and not arm_rows.empty:
        base_ped = base_rows.iloc[0].get("night_ped_map5095")
        arm_ped = arm_rows.iloc[0].get("night_ped_map5095")
        if bool(pd.notna(base_ped)) and bool(pd.notna(arm_ped)):
            st.markdown(
                f"Night pedestrian mAP50-95 {float(base_ped):.4f} → {float(arm_ped):.4f}."
            )
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


def _render_result(data: _TourData) -> None:
    """Step 6 — the result screen (spec §2): four bordered answers, every number
    read from the same package tables every earlier step reads. The loop
    breadcrumb lights all four stages here (see ``_STEPS`` below) -- by this
    screen the tour has walked the whole loop, not one stage of it."""
    with st.container(border=True):
        _result_weakness(data)
    with st.container(border=True):
        _result_data_added(data)
    with st.container(border=True):
        _result_improved(data)
    with st.container(border=True):
        _result_failures(data)


# Step numbering: 0-based in this module (the ``tour_step`` session key, ``_STEPS``
# indices, ``_walk_to_step``) and 1-based in everything the viewer reads ("Step 4 of
# 7", and the prose cross-references inside the steps themselves). "Step 4" in a
# docstring or a rendered sentence therefore means ``_STEPS[3]``.
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
    _Step(
        key="mine",
        title="Find more like it",
        stage="Mine",
        render=_render_mine,
        links=(("scenarios", "Scenario Search — every preset, every matching event"),),
    ),
    _Step(
        key="why_selected",
        title="Why this frame was picked",
        stage="Mine",
        render=_render_why_selected,
        links=(("active_learning", "Active Learning — the whole mined set and its communities"),),
    ),
    _Step(
        key="retrain",
        title="Retrain on what was found",
        stage="Train",
        render=_render_retrain,
        links=(("active_learning", "Active Learning — every arm, its quotas and its table"),),
    ),
    _Step(
        key="after",
        title="Same kind of frame, after",
        stage="Evaluate",
        render=_render_after,
        links=(("active_learning", "Active Learning — every hand-approved before/after frame"),),
    ),
    _Step(
        key="result",
        title="What we found, added, gained — and what failed",
        # All four stages, lit together: this screen doesn't add a stage of its
        # own, it closes the loop the first six screens walked one stage at a time.
        stage=LOOP_STAGES,
        render=_render_result,
        links=(
            ("overview", "Overview — the whole loop and the flagship numbers"),
            ("failures", "Failure Explorer — the whole val split, filterable"),
            ("scenarios", "Scenario Search — every preset, every matching event"),
            ("active_learning", "Active Learning — the whole mined set and its communities"),
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
# "Active Learning — the whole mined set and its communities", which wrapped to four
# lines of two words (Phase 9a review 5a). Rows of three give every label the width
# of two of the old columns; a step with one or two links still gets a single row.
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
