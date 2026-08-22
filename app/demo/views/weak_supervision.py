"""Weak Supervision — what VLM-verified pseudo labels actually bought, and where the
rest of the ground-truth gain went (design: docs/superpowers/specs/2026-08-21-demo-
phase7-design.md §4).

Every number on this page is read from the committed package: the per-pair loss
split (`weak_loss_decomposition.parquet`), the verifier's own run summaries
(`weak_supervision_results.parquet`, `weak_verifier_by_class.parquet`), the trained
arms (`active_learning_results.parquet`), and the curated frames' pseudo boxes and
VLM count votes (`weak_labels.parquet`, `vlm_counts.parquet`). Only the section
prose is written here.

Three honesty rules this page is built around:

- The documented headline is the `random` pair (`headline == True` in the
  decomposition table); the other pair is carried beside it, never instead of it.
- "Retention" is two different quantities. The VERIFIER's retention is the share of
  candidate frames it accepted; the pair's retention is the share of the GT-trained
  arm's mAP gain the weak-trained arm captured. Both are on this page, each labelled
  for what it is.
- The curated weak frames are TRAIN-POOL frames: no model ever evaluated them, so
  they carry GT and pseudo boxes but no predictions, and the downstream result is
  stated at arm level rather than being read off a frame.
- The pseudo boxes are the BASELINE DETECTOR's, not the VLM's: the VLM emits
  scene-level per-class counts and never draws a box (active_learning/
  pseudo_label.py). Its role is the verifier's -- a frame is kept only when its
  counts agree with the detector's counts, class by class, within a tolerance. The
  page says that wherever a pseudo box or a score is on screen.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from filters import crowding_long, gt_for_render, loss_long, visible_gt, weak_frame_summary
from PIL import Image
from render import (
    bar_chart,
    draw_overlay,
    learned,
    loop_breadcrumb,
    metric_cards,
    provenance,
    story_arrows,
)

from data import (
    STALE_PACKAGE_NOTE as _STALE_PACKAGE_NOTE,
)
from data import (
    crop_path,
    frame_image_path,
    load_al_results,
    load_frame_manifest,
    load_gt_boxes,
    load_vlm_counts,
    load_weak_by_class,
    load_weak_labels,
    load_weak_loss,
    load_weaksup,
    thumb_path,
)

# How many gallery thumbs render before the "show all" checkbox -- 47 accepted +
# 31 rejected frames in one grid is a wall of images (consolidated review M8).
_GALLERY_PAGE = 24

# The exact claim the spec pins for a rejected frame (§4d): the comparison that
# rejected it was against the DETECTOR's counts, which this package does not carry
# -- so "no boxes" here is a fact about the pipeline, not missing data.
_REJECTED_NOTE = (
    "the verifier compared the VLM's counts to the detector's, which are not in the "
    "package — no pseudo boxes exist for rejected frames by construction"
)
_MUTUAL_ZERO_NOTE = (
    "mutual zero: the detector proposed no boxes here and the VLM's counts agreed, so "
    "the frame was accepted carrying no pseudo boxes at all — agreement that trains "
    "on an empty image"
)
# The columns Phase 7 (Task 2) added to weak_supervision_results.parquet. A package
# built before that ships the same file with the narrower schema (the committed
# demo_data/ is package_version 0.5 today), so the crowding section checks for them
# before reading any of them -- an AttributeError on a column that did not exist yet
# is not an honest absent note.
_CROWDING_COLUMNS = frozenset({
    "gt_boxes_per_accepted_frame", "gt_boxes_per_rejected_frame",
    "gt_boxes_per_candidate_frame", "tolerance", "conf",
})

_LEGEND = (
    "Green = ground truth. Blue = a pseudo-label box: a BASELINE-DETECTOR proposal "
    "(conf ≥ 0.5) kept because the VLM's per-class counts agreed within ±1 — the VLM "
    "emits counts, never boxes, so the number on the label is the detector's "
    "confidence. Train-pool frames carry no model predictions."
)


def _ordered_pairs(loss: pd.DataFrame) -> pd.DataFrame:
    """``weak_loss_decomposition`` with the documented headline row first (a stable
    sort, so the remaining pairs keep the exporter's own arm order)."""
    if loss.empty:
        return loss
    return loss.sort_values("headline", ascending=False, kind="stable").reset_index(drop=True)


def _text(value: object) -> str:
    """A package value as display text, "n/a" when it is missing.

    export_vlm_counts gives EVERY curated weak frame a row, with NA parse_status /
    confidence / scene fields when the VLM never labelled it (or its answer never
    parsed) -- printing that NA as "nan" would read as a value.
    """
    return "n/a" if value is None or pd.isna(value) else str(value)


def _map_value(arms: pd.DataFrame, arm: str, column: str) -> float | None:
    rows = arms.loc[arms["arm"] == arm]
    if rows.empty or pd.isna(rows.iloc[0][column]):
        return None
    return float(rows.iloc[0][column])


def _night_pedestrian_pair(loss: pd.DataFrame, arms: pd.DataFrame) -> tuple[str, str] | None:
    """The (base arm, weak arm) pair whose weak arm did WORST at night — the pair the
    night-pedestrian card follows, since that is where pseudo labels cost the most.

    Pairs whose trained arm isn't in this package are skipped rather than guessed at
    (the card then simply follows another pair, or isn't drawn at all)."""
    candidates = []
    for base in loss["base_arm"]:
        weak = f"weak_{base}"
        delta = _map_value(arms, weak, "delta_night")
        if delta is not None:
            candidates.append((delta, str(base), weak))
    if not candidates:
        return None
    _, base_arm, weak_arm = min(candidates)
    return base_arm, weak_arm


def _night_rank_caption(arms: pd.DataFrame, weak_arm: str) -> str:
    """Whether the weak arm really is the worst night arm in THIS package's table --
    computed, never asserted, so a package with different arms says the true rank."""
    ordered = arms.sort_values("delta_night", kind="stable").reset_index(drop=True)
    position = int(ordered.index[ordered["arm"] == weak_arm][0])
    delta = float(ordered.iloc[position]["delta_night"])
    total = len(ordered)
    if position == 0:
        return (
            f"`{weak_arm}` is the worst night result of the {total} arms "
            f"({delta:+.4f} night mAP50-95 against the baseline) — the pseudo labels "
            "did not merely fail to help at night, they cost night performance."
        )
    worst = ordered.iloc[0]
    return (
        f"`{weak_arm}` ranks {position + 1} of {total} arms by night delta "
        f"({delta:+.4f}); the worst in this package is `{worst['arm']}` "
        f"({float(worst['delta_night']):+.4f})."
    )


def _render_cards(loss: pd.DataFrame, weaksup: pd.DataFrame, arms: pd.DataFrame) -> None:
    """(a) Both retention quantities and the night pedestrian slice."""
    cards: list[tuple[str, str]] = []
    for row in _ordered_pairs(loss).itertuples(index=False):
        # The headline attribution comes from the table's own `headline` column, not
        # from an arm name written into this file.
        # Short labels, and no backticks: st.metric renders its label in a narrow
        # column and breaks mid-TOKEN, so "`graph_rate_night`" came out as
        # "graph_ra te_night" (consolidated review, real-browser finding 4). The arm
        # pair the night card follows moves to a caption below for the same reason.
        suffix = " (headline)" if bool(row.headline) else ""
        cards.append((
            f"GT gain retained — {row.base_arm}{suffix}", f"{float(row.retention):.1%}",
        ))
    for row in weaksup.itertuples(index=False):
        cards.append((
            f"Verifier retention — {row.arm}", f"{float(row.verifier_retention):.0%}"
        ))

    pair = _night_pedestrian_pair(loss, arms)
    night_pedestrian_caption = None
    if pair is None:
        cards.append(("Night pedestrian mAP50-95", "n/a"))
    else:
        base_arm, weak_arm = pair
        base_value = _map_value(arms, base_arm, "night_ped_map5095")
        weak_value = _map_value(arms, weak_arm, "night_ped_map5095")
        cards.append((
            "Night pedestrian mAP50-95",
            "n/a"
            if base_value is None or weak_value is None
            else f"{base_value:.3f} → {weak_value:.3f}",
        ))
        night_pedestrian_caption = (
            f"Night pedestrian mAP50-95 follows the `{base_arm}` → `{weak_arm}` pair — "
            "the weak arm with the worst night result in this package."
        )
    metric_cards(cards)
    provenance(
        "recorded", "weak_loss_decomposition.parquet + weak_supervision_results.parquet"
    )
    st.caption(
        "(headline) marks the pair docs/ACTIVE_LEARNING.md publishes as the "
        "weak-supervision result — the lower of the two shares, not the better one."
    )

    if night_pedestrian_caption is not None:
        st.caption(night_pedestrian_caption)
    st.caption(
        "Two different quantities called retention: the VERIFIER's retention is the "
        "share of candidate frames it accepted; the pair's retention is the share of "
        "the GT-labelled arm's mAP50-95 gain that the weak-labelled arm captured."
    )
    if pair is not None:
        st.caption(_night_rank_caption(arms, pair[1]))


def _render_loss_learned_callout(loss: pd.DataFrame, arms: pd.DataFrame) -> None:
    """Phase 9a (Task 6): the loss-split "what we learned" sentence, from the
    documented headline pair (``headline == True``) -- skipped when this package
    carries no headline row. A second sentence, about the OTHER pair, is added
    only when this package carries a second pair AND its weak arm is genuinely
    (computed here, never assumed) the worst night arm in the whole table."""
    headline_rows = loss.loc[loss["headline"]]
    if headline_rows.empty:
        return
    headline_row = headline_rows.iloc[0]
    base_arm = str(headline_row["base_arm"])
    retention = float(headline_row["retention"])
    dropped_frame_share = float(headline_row["dropped_frame_share"])
    label_share = float(headline_row["label_share"])
    text = (
        f"Free labels kept {retention:.1%} of the ground-truth gain for the "
        f"`{base_arm}` pair: {dropped_frame_share:.1%} was lost with the frames "
        f"the verifier dropped and {label_share:.1%} to label noise on the ones "
        "it kept."
    )

    other_rows = loss.loc[loss["base_arm"] != base_arm]
    if not other_rows.empty and not arms.empty:
        other_row = other_rows.iloc[0]
        other = str(other_row["base_arm"])
        other_retention = float(other_row["retention"])
        other_weak_arm = f"weak_{other}"
        worst_row = arms.loc[arms["delta_night"].idxmin()]
        if str(worst_row["arm"]) == other_weak_arm:
            worst_delta = float(worst_row["delta_night"])
            text += (
                f" The `{other}` pair retained {other_retention:.1%}, but its "
                f"weak arm posted the worst night result of the {len(arms)} arms "
                f"({worst_delta:+.4f})."
            )

    learned(text)


def _render_decomposition(loss: pd.DataFrame, arms: pd.DataFrame) -> None:
    """(b) Every pair's gain split into retained / dropped-frame cost / label cost."""
    st.subheader("Where the rest of the gain went")
    if loss.empty:
        st.info(_STALE_PACKAGE_NOTE)
        return

    ordered = _ordered_pairs(loss)
    long = pd.concat([loss_long(row) for _, row in ordered.iterrows()], ignore_index=True)
    st.altair_chart(
        bar_chart(
            long, x="base_arm", y="value", color_field="component", zero_line=False,
            sort=[str(arm) for arm in ordered["base_arm"]],
            y_title="mAP50-95 gain over baseline",
            title="Overall mAP50-95 gain over baseline, split by what weak supervision "
            "kept and how it lost the rest",
        ),
        width="stretch",
    )
    for _, row in ordered.iterrows():
        components = loss_long(row).set_index("component")["share"]
        st.caption(
            f"`{row['base_arm']}`: the weak-labelled arm keeps "
            f"**{components['retained']:.1%}** of the {float(row['gt_gain']):.4f} "
            f"mAP50-95 the GT-labelled arm gained. "
            f"{components['dropped-frame cost']:.1%} is lost to candidate frames the "
            f"verifier dropped (they were never trained on at all) and "
            f"{components['label cost']:.1%} to label noise on the frames it kept."
        )
    _render_loss_learned_callout(loss, arms)


def _render_crowding(weaksup: pd.DataFrame, by_class: pd.DataFrame) -> None:
    """(c) The bias the verifier's agreement rule introduces, per arm and per class."""
    st.subheader("What the verifier's rule selects for")
    if weaksup.empty or not set(weaksup.columns) >= _CROWDING_COLUMNS:
        st.info(_STALE_PACKAGE_NOTE)
        return
    pairs = crowding_long(weaksup)

    # x is the arm+side pair (not arm with side as the colour) so the two bars stand
    # SIDE BY SIDE: an accepted/rejected pair stacked on one bar would read as a
    # total, and the two means are not summable.
    pairs["candidate_set"] = pairs["arm"] + " · " + pairs["side"]
    st.altair_chart(
        bar_chart(
            pairs, x="candidate_set", y="gt_boxes_per_frame", color_field="side",
            zero_line=False, sort=[str(name) for name in pairs["candidate_set"]],
            title="GT boxes per frame — the candidates the verifier accepted vs the "
            "ones it rejected",
        ),
        width="stretch",
    )
    provenance(
        "recorded", "weak_supervision_results.parquet + weak_verifier_by_class.parquet"
    )
    for row in weaksup.itertuples(index=False):
        st.caption(
            f"`{row.arm}`: accepted frames average "
            f"{float(row.gt_boxes_per_accepted_frame):.2f} GT boxes/frame, rejected "
            f"ones {float(row.gt_boxes_per_rejected_frame):.2f} (all "
            f"{int(row.n_candidates):,} candidates: "
            f"{float(row.gt_boxes_per_candidate_frame):.2f}). The verifier accepts a "
            f"frame only when the VLM's per-class counts match the detector's within "
            f"{int(row.tolerance)} at conf >= {float(row.conf):g} — counting is easy "
            "on a sparse frame and hard on a crowded one, so the rule keeps the easy "
            "frames and drops exactly the busy ones a detector most needs."
        )

    if by_class.empty:
        st.info(_STALE_PACKAGE_NOTE)
        return
    table = by_class.copy()
    table["mutual_zero_share"] = table["mutual_zero_share"].map(
        lambda value: "n/a" if pd.isna(value) else f"{float(value):.1%}"
    )
    st.dataframe(table, hide_index=True)
    st.caption(
        "n_rejected_disagreements: candidates rejected because the VLM's count for "
        "that class disagreed with the detector's. n_accepted_mutual_zero / "
        "mutual_zero_share: accepted frames where BOTH counted zero of the class — "
        "agreement, but agreement that carries no supervision."
    )


def _render_frame(
    token: str, frames: pd.DataFrame, *, gt: pd.DataFrame, labels: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    """(d) One curated weak frame: its crop with GT + pseudo boxes (the DETECTOR's
    proposals the VLM's counts corroborated), its verdict badge, and the VLM-vs-GT
    counts the verifier actually compared."""
    row = frames.loc[frames["sample_data_token"] == token].iloc[0]
    vlm_rows = labels.loc[labels["sample_data_token"] == token]
    counts_rows = counts.loc[counts["sample_data_token"] == token]
    counts_row = None if counts_rows.empty else counts_rows.iloc[0]
    summary = weak_frame_summary(row.get("weak_verdict"), vlm_rows, counts_row)

    image_path, scale = crop_path(token), 0.6
    if not image_path.is_file():
        image_path, scale = thumb_path(token), 0.16
    if image_path.is_file():
        st.image(
            draw_overlay(
                Image.open(image_path), gt_for_render(visible_gt(gt, token)), pd.DataFrame(),
                mode="gt", scale=scale, pseudo_boxes=vlm_rows,
            )
        )
        st.caption(_LEGEND)
        provenance(
            "recomputed", "GT from gt_boxes.parquet; pseudo boxes from weak_labels.parquet"
        )

    st.markdown(f"**{summary['verdict_label']}**")
    if summary["mutual_zero"]:
        st.caption(_MUTUAL_ZERO_NOTE)
    if str(row.get("weak_verdict")) == "rejected":
        st.caption(_REJECTED_NOTE)

    st.markdown("**The VLM's counts vs ground truth**")
    st.dataframe(summary["table"], hide_index=True)
    st.caption(
        "This is the VLM's counting judged against GT — the verifier itself never saw "
        "GT: it compared these counts to the DETECTOR's, which this package does not "
        "carry."
    )
    if counts_row is None:
        st.caption(
            "this frame has no `vlm_counts` row in the package, so its counts read "
            "n/a rather than zero"
        )
    else:
        # label_confidence is a STRING enum the VLM emits ("high"/"low"), not a
        # number: formatting it as a float raised ValueError on every frame the VLM
        # actually labelled (consolidated review C1).
        st.caption(
            f"VLM parse: {_text(counts_row['parse_status'])} · label confidence "
            f"{_text(counts_row['label_confidence'])} · "
            f"it called the scene {_text(counts_row['vlm_time_of_day'])} / "
            f"{_text(counts_row['vlm_weather'])}."
        )


def _render_gallery(
    frames: pd.DataFrame, *, verdict: str, gt: pd.DataFrame, labels: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    """(d) One tab's thumb gallery, and the frame a viewer picks out of it."""
    if frames.empty:
        st.info(f"no {verdict} weak-supervision frames are curated into this package")
        return

    shown = frames
    if len(frames) > _GALLERY_PAGE:
        if st.checkbox(f"Show all {len(frames)} frames", key=f"ws_{verdict}_show_all"):
            st.caption(
                f"All {len(frames)} curated frames the verifier {verdict}. These are "
                "TRAIN-POOL frames: they were never test images, and no model ever ran "
                "on them."
            )
        else:
            shown = frames.head(_GALLERY_PAGE)
            st.caption(
                f"The first {_GALLERY_PAGE} of {len(frames)} curated frames the "
                f"verifier {verdict}, in package order. These are TRAIN-POOL frames: "
                "they were never test images, and no model ever ran on them."
            )
    else:
        st.caption(
            f"{len(frames)} curated frames the verifier {verdict}. These are TRAIN-POOL "
            "frames: they were never test images, and no model ever ran on them."
        )

    columns = st.columns(4)
    for position, row in enumerate(shown.itertuples()):
        token = str(row.sample_data_token)
        with columns[position % 4]:
            image_path = frame_image_path(token)
            if image_path is not None:
                st.image(str(image_path))
            st.caption(f"{row.scene_name} · {'night' if row.is_night else 'day'}")
            if st.button("View", key=f"ws_{verdict}_select_{token}"):
                st.session_state[f"ws_{verdict}_token"] = token

    chosen = st.session_state.get(f"ws_{verdict}_token")
    if not chosen or chosen not in set(frames["sample_data_token"]):
        return
    _render_frame(str(chosen), frames, gt=gt, labels=labels, counts=counts)


def _render_downstream(loss: pd.DataFrame, arms: pd.DataFrame) -> None:
    """(d) The result, stated where it actually exists: at arm level."""
    st.markdown("**What training on those frames did**")
    for row in _ordered_pairs(loss).itertuples(index=False):
        weak_arm = f"weak_{row.base_arm}"
        arm_rows = arms.loc[arms["arm"] == weak_arm]
        if arm_rows.empty:
            st.markdown(
                f"trained arm `{weak_arm}`: not in this package — no retrained "
                "checkpoint was exported for this pair."
            )
            continue
        arm_row = arm_rows.iloc[0]
        st.markdown(
            f"trained arm `{weak_arm}`: Δnight {float(arm_row['delta_night']):+.4f}, "
            f"Δoverall {float(arm_row['delta_overall']):+.4f} (train-pool frames carry "
            "no predictions — the result is arm-level)"
        )


def _render_story(loss: pd.DataFrame, weaksup: pd.DataFrame, arms: pd.DataFrame) -> None:
    """(e) The experiment end to end, every figure read from the package."""
    ordered = _ordered_pairs(loss)
    if ordered.empty:
        return
    pair = ordered.iloc[0]
    base_arm = str(pair["base_arm"])
    weak_arm = f"weak_{base_arm}"
    components = loss_long(pair).set_index("component")["share"]

    runs = (
        weaksup.loc[weaksup["arm"] == base_arm]
        if set(weaksup.columns) >= {"conf", "tolerance"}
        else weaksup.iloc[0:0]
    )
    if runs.empty:
        labelling = (
            "The baseline detector proposes boxes on every candidate frame, and a VLM "
            "is asked for per-class counts (car / truck / bus / pedestrian / bicycle) "
            "on the same frames — it never draws a box."
        )
        verification = (
            "A frame is kept only when the VLM's counts agree with the detector's, "
            "class by class; this package carries no run summary for this pair."
        )
    else:
        run = runs.iloc[0]
        labelling = (
            f"The baseline detector proposes boxes above conf "
            f"**{float(run['conf']):g}** on each of the "
            f"**{int(run['n_candidates']):,}** candidate frames, and a VLM is asked "
            "for per-class counts (car / truck / bus / pedestrian / bicycle) on the "
            "same frames — it never draws a box. No human annotation at all."
        )
        verification = (
            f"A frame's boxes are kept only when the VLM's count matches the "
            f"detector's within **{int(run['tolerance'])}** for every class: "
            f"**{float(run['verifier_retention']):.0%}** of the candidates survive "
            f"(**{int(run['n_accepted']):,}** frames, "
            f"**{int(run['n_pseudo_boxes']):,}** pseudo boxes)."
        )

    n_train = _map_value(arms, weak_arm, "n_train_images")
    training = (
        f"`{weak_arm}` is retrained on the surviving frames' pseudo boxes"
        + (f" — **{int(n_train):,}** training images." if n_train is not None else ".")
    )

    story_arrows([
        (
            "Hypothesis",
            "Labelling is the expensive part of the loop. The detector can propose "
            "boxes for free; if a VLM can be trusted to say which frames it got "
            "right, the kept frames are free training data — and a weak-labelled arm "
            "should capture most of the gain the ground-truth-labelled arm bought, "
            "for none of the annotation cost.",
        ),
        ("Labelling", labelling),
        ("Verification", verification),
        ("Training", training),
        (
            "Result",
            f"It captures **{float(pair['retention']):.1%}** of the `{base_arm}` pair's "
            f"GT gain. {components['dropped-frame cost']:.1%} of that gain was lost "
            f"with the frames the verifier dropped and "
            f"{components['label cost']:.1%} to label noise on the ones it kept — the "
            "verifier's own bias toward sparse frames is most of the story, not the "
            "box quality.",
        ),
    ])


def render() -> None:
    st.title("Weak Supervision")
    loop_breadcrumb(["Train", "Evaluate"])
    st.caption(
        "What VLM-verified pseudo labels kept — and lost — against ground truth. "
        "Every number here is read from the package."
    )

    loss = load_weak_loss()
    weaksup = load_weaksup()
    arms = load_al_results()
    if loss.empty and weaksup.empty:
        st.error(_STALE_PACKAGE_NOTE)
        return

    _render_cards(loss, weaksup, arms)

    st.divider()
    _render_decomposition(loss, arms)

    st.divider()
    _render_crowding(weaksup, load_weak_by_class())

    st.divider()
    st.subheader("What the VLM saw")
    manifest = load_frame_manifest()
    gt = load_gt_boxes()
    labels = load_weak_labels()
    counts = load_vlm_counts()
    if "weak_verdict" not in manifest.columns:
        st.info(_STALE_PACKAGE_NOTE)
    else:
        accepted_tab, rejected_tab = st.tabs(["accepted", "rejected"])
        for tab, verdict in ((accepted_tab, "accepted"), (rejected_tab, "rejected")):
            with tab:
                _render_gallery(
                    manifest.loc[manifest["weak_verdict"] == verdict].reset_index(drop=True),
                    verdict=verdict, gt=gt, labels=labels, counts=counts,
                )
        _render_downstream(loss, arms)

    st.divider()
    _render_story(loss, weaksup, arms)
