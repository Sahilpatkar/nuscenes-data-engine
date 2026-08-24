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

from itertools import pairwise

import pandas as pd
import streamlit as st
from filters import (
    crowding_long,
    fixed_boxes,
    gt_for_render,
    loss_long,
    model_label,
    visible_gt,
    weak_frame_summary,
    weak_result_token,
    weak_showcase_token,
)
from PIL import Image
from render import (
    bar_chart,
    draw_overlay,
    learned,
    legend,
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
    load_predictions,
    load_vlm_count_buckets,
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

# --- Phase 9b (Task 8): one frame, three views ------------------------------------
#
# Spec docs/superpowers/specs/2026-08-22-demo-phase9b-design.md sec6. Row 1 is the
# pipeline's INPUT side on one accepted train-pool frame (what a human labelled, what
# the detector proposed and the VLM's counts kept, and the verdict that kept it); row
# 2 its OUTPUT side on a val frame neither checkpoint ever trained on -- the arm
# trained on those pseudo labels beside the twin trained on the GT for the same
# frames. Both rows compare boxes ON THE FRAME SHOWN; what either arm is worth
# overall is a number, and stays in the cards and the loss split.

_SHOWCASE_ABSENT_NOTE = (
    "no accepted train-pool frame in this package carries both a pseudo box and a "
    "visible pedestrian GT box, so there is no frame to show all three layers on"
)
# The two held-out checkpoints are a package-0.8 group (`demo infer` over five
# models): an older package carries no matched_/prediction rows for either of them.
_RESULT_ABSENT_NOTE = (
    "held-out weak-arm predictions need demo_data >= 0.8 — rerun demo build"
)
_NO_RESULT_FRAME_NOTE = (
    "no held-out val frame in this package carries a GT box the GT-labelled twin "
    "detects and the weak-labelled arm misses — the two detectors agree on every "
    "box that was scored"
)
_NO_CROP_NOTE = "this frame's crop is not in the package, so its views cannot be drawn"
_NO_UPGRADED_BOX_NOTE = "no such box on this frame"

_BUCKETS_ABSENT_NOTE = "count-bucket chart needs demo_data >= 0.8 — rerun demo build"
# eval_count_buckets' own cut points (autolabel/evaluate.py), in count order -- the x
# axis is nominal, so the order has to be pinned or altair sorts "10+" between "0"
# and "1-3".
_BUCKET_ORDER = ("0", "1-3", "4-9", "10+")


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
    computed, never asserted, so a package with different arms says the true rank.

    Ranked over the arms with a known ``delta_night`` only (Phase 9a final review
    item 7): ``sort_values`` keeps NA rows (sorted last) rather than dropping
    them, so counting ``len(ordered)`` without first dropping them would count an
    arm this package never evaluated for night delta as one of "the N arms"."""
    ordered = (
        arms.dropna(subset=["delta_night"])
        .sort_values("delta_night", kind="stable")
        .reset_index(drop=True)
    )
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


def _frame_image(token: str) -> tuple[Image.Image, float] | None:
    """One curated frame's image and the scale its boxes need, or ``None`` when the
    package carries neither a crop nor a thumbnail for it.

    Boxes are stored in native 1600x900 coordinates, so the scale belongs to the
    image that was found: 0.6 for a crop, 0.16 for a thumb (draw_overlay's own
    convention). Shared by the frame panel and by both rows of "One frame, three
    views" -- three overlays of the same frame open the file once.
    """
    path, scale = crop_path(token), 0.6
    if not path.is_file():
        path, scale = thumb_path(token), 0.16
    if not path.is_file():
        return None
    return Image.open(path), scale


def _result_arms(gt: pd.DataFrame) -> tuple[str, str] | None:
    """The (weak-labelled arm, GT-labelled twin) pair this package was evaluated
    with, read off ``gt_boxes``' own ``matched_<model>`` columns.

    `demo infer` writes one ``matched_<model>`` column per configured model, and the
    control checkpoint's name is the arm's with a ``_gt`` suffix (configs/demo.yaml
    `models:`) -- so the pair is DISCOVERED here rather than written into this file,
    and a package whose infer run predates the two checkpoints simply has no pair.
    """
    models = {
        column.removeprefix("matched_")
        for column in gt.columns
        if column.startswith("matched_")
    }
    pairs = sorted(
        (twin.removesuffix("_gt"), twin)
        for twin in models
        if twin.endswith("_gt") and twin.removesuffix("_gt") in models
    )
    return pairs[0] if pairs else None


def _render_showcase_row(
    token: str, frames: pd.DataFrame, *, gt: pd.DataFrame, labels: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    """Row 1: one accepted train-pool frame as three layers -- the human labels, the
    pseudo labels, and the two together with the verdict that kept the frame."""
    row = frames.loc[frames["sample_data_token"] == token].iloc[0]
    vlm_rows = labels.loc[labels["sample_data_token"] == token]
    counts_rows = counts.loc[counts["sample_data_token"] == token]
    counts_row = None if counts_rows.empty else counts_rows.iloc[0]
    summary = weak_frame_summary(row.get("weak_verdict"), vlm_rows, counts_row)
    gt_rows = gt_for_render(visible_gt(gt, token))

    opened = _frame_image(token)
    views: list[tuple[str, Image.Image]] = []
    if opened is not None:
        image, scale = opened
        views = [
            (
                "ground truth — the boxes a human annotator drew on this train-pool frame",
                draw_overlay(image, gt_rows, pd.DataFrame(), mode="gt", scale=scale),
            ),
            (
                "pseudo labels — the baseline detector's proposals the VLM's counts kept",
                draw_overlay(
                    image, pd.DataFrame(), pd.DataFrame(), mode="gt", scale=scale,
                    pseudo_boxes=vlm_rows,
                ),
            ),
            (
                "both layers — what the verifier's vote was about",
                draw_overlay(
                    image, gt_rows, pd.DataFrame(), mode="gt", scale=scale,
                    pseudo_boxes=vlm_rows,
                ),
            ),
        ]

    columns = st.columns(3)
    for column, (caption, rendered) in zip(columns, views, strict=False):
        with column:
            st.image(rendered, caption=caption)
    with columns[2]:
        st.markdown(f"**{summary['verdict_label']}**")
        st.markdown("**The VLM's counts vs ground truth**")
        # st.table, not st.dataframe: five rows in a third-width column read better
        # as a static table than in a scrollable grid, and the galleries below keep
        # their own count tables as the page's only dataframes of this shape.
        st.table(summary["table"].set_index("class"))

    if not views:
        st.info(_NO_CROP_NOTE)
    else:
        st.caption(_LEGEND)
    st.caption(
        f"Train-pool frame `{token}` ({row.get('scene_name')}) — the verifier compared "
        "COUNTS, never boxes: it never saw the green layer, and the blue layer is "
        "what its vote let through."
    )
    provenance(
        "recomputed", "GT from gt_boxes.parquet; pseudo boxes from weak_labels.parquet"
    )


def _render_result_row(
    token: str, frames: pd.DataFrame, *, gt: pd.DataFrame, preds: pd.DataFrame,
    weak_arm: str, gt_arm: str,
) -> None:
    """Row 2: a held-out val frame with the same three-panel shape -- ground truth,
    then each checkpoint's own boxes over it -- and the per-box difference between
    the two, on this frame only."""
    row = frames.loc[frames["sample_data_token"] == token].iloc[0]
    gt_token = gt.loc[gt["sample_data_token"] == token]
    visible = visible_gt(gt, token)
    preds_token = preds.loc[preds["sample_data_token"] == token]

    opened = _frame_image(token)
    views: list[tuple[str, Image.Image]] = []
    if opened is not None:
        image, scale = opened
        views = [
            (
                "ground truth — every visible GT box on this held-out val frame",
                draw_overlay(
                    image, gt_for_render(visible), pd.DataFrame(), mode="gt", scale=scale
                ),
            ),
            *(
                (
                    model_label(model),
                    draw_overlay(
                        image, gt_for_render(visible, model),
                        preds_token.loc[preds_token["model"] == model],
                        mode="overlay", scale=scale,
                    ),
                )
                for model in (weak_arm, gt_arm)
            ),
        ]

    if not views:
        st.info(_NO_CROP_NOTE)
    else:
        for column, (caption, rendered) in zip(st.columns(3), views, strict=True):
            with column:
                st.image(rendered, caption=caption)
        legend()
    st.caption(
        f"`{token}` ({row.get('scene_name')}) is a held-out val frame — neither "
        "detector saw it in training."
    )
    # BOTH lines, the way row 1 states its own (item M1, Phase 9b consolidated
    # review): the boxes are a recorded offline inference run, and the three
    # overlays over them are drawn here, in this app, from the package's tables.
    # Stating only the first left the drawing unattributed.
    provenance("recorded", "demo infer, CPU, checkpoint of the training run")
    provenance("recomputed", "overlay drawn from gt_boxes.parquet and predictions.parquet")

    # The radio credits a detector, and the table under it lists what that detector
    # claims and the other one does not (fixed_boxes' upgrade rule). It opens on the
    # GT-labelled twin because that is the direction this frame was CHOSEN for
    # (weak_result_token) -- the reverse direction is one click away and is usually
    # empty, which is itself the finding.
    choice = st.radio(
        "Detector", [weak_arm, gt_arm], index=1, horizontal=True,
        key="ws_result_model", format_func=model_label,
    )
    other = gt_arm if choice == weak_arm else weak_arm
    upgraded = fixed_boxes(gt_token, preds_token, baseline=other, arm=choice)
    st.markdown(f"**GT boxes `{choice}` detects that `{other}` does not**")
    if upgraded.empty:
        st.caption(_NO_UPGRADED_BOX_NOTE)
    else:
        st.dataframe(upgraded, hide_index=True)
    st.caption(
        "Counted on this frame only — what each arm was worth overall is in the "
        "cards above and in the loss split below."
    )


def _render_three_views(
    *, frames: pd.DataFrame, gt: pd.DataFrame, preds: pd.DataFrame,
    labels: pd.DataFrame, counts: pd.DataFrame, arms: tuple[str, str] | None,
) -> None:
    """The section: the input row on a train-pool frame, the output row on a
    held-out val frame. Each row degrades to its own note -- they need different
    parts of the package, and one missing group must not take the other row down."""
    st.subheader("One frame, three views")
    st.caption(
        "What the VLM's counts bought on one curated frame — and what the two "
        "checkpoints trained on those frames then did on a frame neither of them saw."
    )

    showcase = weak_showcase_token(frames, labels, gt)
    if showcase is None:
        st.info(_SHOWCASE_ABSENT_NOTE)
    else:
        _render_showcase_row(showcase, frames, gt=gt, labels=labels, counts=counts)

    if arms is None:
        st.info(_RESULT_ABSENT_NOTE)
        return
    weak_arm, gt_arm = arms
    result = weak_result_token(frames, gt, preds, weak_arm=weak_arm, gt_arm=gt_arm)
    if result is None:
        st.info(_NO_RESULT_FRAME_NOTE)
        return
    _render_result_row(
        result, frames, gt=gt, preds=preds, weak_arm=weak_arm, gt_arm=gt_arm
    )


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
    # Ranked over arms with a known delta_night only (Phase 9a final review item
    # 7): an arm this package never evaluated for night delta (NaN) is not one of
    # "the N arms" the superlative names.
    night_ranked = arms.dropna(subset=["delta_night"])
    if not other_rows.empty and not night_ranked.empty:
        other_row = other_rows.iloc[0]
        other = str(other_row["base_arm"])
        other_retention = float(other_row["retention"])
        other_weak_arm = f"weak_{other}"
        worst_row = night_ranked.loc[night_ranked["delta_night"].idxmin()]
        if str(worst_row["arm"]) == other_weak_arm:
            worst_delta = float(worst_row["delta_night"])
            text += (
                f" The `{other}` pair retained {other_retention:.1%}, but its "
                f"weak arm posted the worst night result of the {len(night_ranked)} "
                f"arms ({worst_delta:+.4f})."
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


def _bucket_phrase(bucket: str, mae: float) -> str:
    """One bucket's MAE as prose. The "0" bucket is not "zero objects" in the
    ordinary sense: it is a frame-class pair the frame holds none of, which is
    exactly the pair the VLM finds easiest."""
    if bucket == "0":
        return f"{mae:.2f} on empty frame-class pairs"
    return f"{mae:.2f} at {bucket} objects"


def _render_count_buckets(buckets: pd.DataFrame) -> None:
    """The VLM's count error against how crowded the frame actually is -- the
    verifier's sparse-frame bias (the panel below) seen from the other side.

    ``n`` counts frame-class PAIRS: the builder pools the ten count fields into one
    bucket table, so a bucket holds one entry per (frame, class), not per frame, and
    the caption says so.
    """
    st.subheader("Where the VLM's counting breaks down")
    if buckets.empty:
        st.info(_BUCKETS_ABSENT_NOTE)
        return

    # One model per package today (the Phase-6b labelling run); the table carries
    # the name, so the caption states which VLM these errors belong to rather than
    # leaving the reader to assume.
    model = str(buckets["model"].iloc[0])
    rows = buckets.loc[buckets["model"] == model].assign(
        bucket=buckets.loc[buckets["model"] == model, "bucket"].astype(str)
    )
    order = [bucket for bucket in _BUCKET_ORDER if bucket in set(rows["bucket"])]
    ordered = rows.set_index("bucket").loc[order].reset_index()

    st.altair_chart(
        bar_chart(
            ordered, x="bucket", y="mae", sort=order, zero_line=False,
            y_title="count MAE",
            title="The VLM's mean count error by how many objects the frame actually holds",
        ),
        width="stretch",
    )
    pairs = ", ".join(
        f"{bucket} → {int(n):,}"
        for bucket, n in zip(ordered["bucket"], ordered["n"], strict=True)
    )
    # "ten", not "five" (item I1, Phase 9b consolidated review): eval_count_buckets
    # pools every one of autolabel.schema.COUNT_FIELDS -- cars, trucks, buses,
    # trailers, construction vehicles, motorcycles, bicycles, pedestrians, traffic
    # cones, barriers -- which is why the real package's n sums to 49,860 = 4,986
    # parsed frames x 10. The five-class count-vote table further up the page is a
    # DIFFERENT set (the detector's classes), and the caption said "five" in a way
    # that invited the reader to equate the two.
    st.caption(
        f"n = frame-class pairs: {pairs}. All ten count fields the VLM was asked "
        f"for are pooled here — more than the five detector classes in the "
        f"count-vote table above — so a bucket counts (frame, class) entries "
        f"rather than frames. VLM: {model}."
    )
    provenance(
        "recorded",
        "vlm_count_buckets.parquet — recomputed at build from the VLM's label table "
        "and the GT annotations",
    )

    maes = [float(value) for value in ordered["mae"]]
    # Computed, never asserted (the 9a rule): "rises" is a claim about THIS table,
    # so it is only written when the table's own MAEs never fall.
    trend = (
        "rises with the crowd"
        if all(later >= earlier for earlier, later in pairwise(maes))
        else "moves with the crowd"
    )
    phrases = ", ".join(
        _bucket_phrase(str(bucket), mae)
        for bucket, mae in zip(ordered["bucket"], maes, strict=True)
    )
    learned(
        f"Why crowded frames defeat the VLM — its mean count error {trend}: "
        f"{phrases}. The verifier's bias toward sparse frames, in the panel below, "
        "is this curve seen from the other side: the frames the VLM can count are "
        "the frames the rule keeps."
    )


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

    opened = _frame_image(token)
    if opened is not None:
        image, scale = opened
        st.image(
            draw_overlay(
                image, gt_for_render(visible_gt(gt, token)), pd.DataFrame(),
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


def _render_downstream(
    loss: pd.DataFrame, arms: pd.DataFrame, *, weak_pair: tuple[str, str] | None
) -> None:
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

    # Folded (Phase 9b, spec §6): the two checkpoints DID run per frame, just not on
    # these frames, and a viewer looking at a gallery frame reasonably wonders where
    # their boxes are. The answer is a null result, so it belongs behind a fold
    # rather than beside the arm-level one.
    if weak_pair is None:
        return
    weak_arm, gt_arm = weak_pair
    with st.expander("Weak-arm detections on this frame"):
        st.markdown(
            f"`{weak_arm}` and `{gt_arm}` ran over the held-out val frames only "
            "(`demo infer`, on CPU, from the training runs' own checkpoints). Every "
            "frame in these galleries is a TRAIN-POOL frame, so neither checkpoint "
            "has a detection on any of them — their frame-by-frame comparison is in "
            "*One frame, three views*, at the top of this page."
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

    # Loaded once, here: the three-views section and the galleries below read the
    # same four tables, and every loader is @st.cache_data anyway.
    manifest = load_frame_manifest()
    gt = load_gt_boxes()
    labels = load_weak_labels()
    counts = load_vlm_counts()
    preds = load_predictions()
    weak_pair = _result_arms(gt)

    st.divider()
    _render_three_views(
        frames=manifest, gt=gt, preds=preds, labels=labels, counts=counts, arms=weak_pair
    )

    st.divider()
    _render_decomposition(loss, arms)

    st.divider()
    _render_count_buckets(load_vlm_count_buckets())

    st.divider()
    _render_crowding(weaksup, load_weak_by_class())

    st.divider()
    st.subheader("What the VLM saw")
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
        _render_downstream(loss, arms, weak_pair=weak_pair)

    st.divider()
    _render_story(loss, weaksup, arms)
