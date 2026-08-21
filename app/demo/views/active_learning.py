"""Active Learning — how the system chose which frames to label, and whether
retraining on them helped (design: docs/superpowers/specs/2026-08-21-demo-phase7-
design.md §3).

Every number on this page is read from the committed package: the arm table
(`active_learning_results.parquet`), the Louvain communities and their per-arm
quotas (`al_communities.parquet`), the per-frame selection facts when `demo
al-explain` staged them (`al_selection_explain.parquet`), and the hand-approved
before/after tokens (`al_exemplars.json`, validated at build time). Only the
section prose is written here.

Two honesty rules this page is built around:

- A frame was selected because its COMMUNITY carried failure mass, which bought the
  community a quota, and the frame ranked high enough inside it by similarity
  degree. Failure mass routed to the frame ITSELF is sparse (1249 of the 1500
  selected frames have none) and is shown as context, never as the reason.
- "champion" is the yolov8m @960 checkpoint from the Phase-3 model comparison, not
  an AL arm; and the best OVERALL arm is `graph`, not the night-targeted arm this
  page follows. Both are said on screen (see `filters.model_label`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from filters import community_jump, fixed_boxes, model_label, selection_factors
from PIL import Image
from render import bar_chart, draw_overlay, metric_cards, story_arrows

from data import (
    al_explain_available,
    crop_path,
    load_al_communities,
    load_al_exemplars,
    load_al_explain,
    load_al_explain_validation,
    load_al_results,
    load_frame_manifest,
    load_gt_boxes,
    load_predictions,
    thumb_path,
)

# The exact absent-note the spec pins (§3d): what is missing, and which command
# produces it -- never a generic "high failure rate ✓" stand-in.
_EXPLAIN_ABSENT_NOTE = (
    "per-frame community and routed mass are not included in this package "
    "(demo al-explain)"
)
_EXPLAIN_FRAME_ABSENT_NOTE = (
    "this frame is not in the staged selection facts — re-run `demo al-explain`"
)
_TRAIN_POOL_NOTE = "train-pool frame — no predictions (models never saw it as a test image)"
_STALE_PACKAGE_NOTE = "needs demo_data >= 0.6 (the Phase-7 tables) — rerun `demo build`"

# The paired quota chart gets crowded well before all 97 communities fit; the
# heaviest ones are where the night floor's reallocation is visible anyway.
_TOP_COMMUNITIES = 12

_ARM_TABLE_COLUMNS = [
    "arm",
    "family",
    "round_order",
    "n_train_images",
    "delta_night",
    "delta_overall",
    "night_ped_map5095",
    "night_share",
    "n_scenes",
]


def _models_from_gt(gt: pd.DataFrame) -> list[str]:
    """The models this package recorded, from gt_boxes' matched_<model> columns --
    the same discovery the Failure Explorer does (never parsed out of a
    fixes_fn_vs_ column name, whose two model names can't be split unambiguously)."""
    return sorted(col.removeprefix("matched_") for col in gt.columns if col.startswith("matched_"))


def _frame_image_path(token: str) -> Path | None:
    thumb = thumb_path(token)
    if thumb.is_file():
        return thumb
    crop = crop_path(token)
    return crop if crop.is_file() else None


def _visible_gt(gt: pd.DataFrame, token: str) -> pd.DataFrame:
    """One frame's GT rows, visibility-floor rows dropped -- the Failure Explorer
    drops them everywhere for the same reason (they were never scored against)."""
    subset = gt.loc[gt["sample_data_token"] == token]
    return subset.loc[~subset["below_visibility_min"].fillna(False)]


def _gt_for_render(gt_rows: pd.DataFrame, model: str) -> pd.DataFrame:
    """``matched_<model>`` renamed to the ``matched`` column ``draw_overlay`` reads.

    A train-pool frame's ``matched_*`` values are all NA (no model ever evaluated
    it), which draw_overlay renders as plain GT rather than as misses -- "not
    evaluated" is not the same claim as "missed".
    """
    column = f"matched_{model}"
    if column in gt_rows.columns:
        return gt_rows.rename(columns={column: "matched"})
    return gt_rows.assign(matched=pd.Series(pd.NA, index=gt_rows.index, dtype="boolean"))


def _night_floor(validation: dict[str, Any]) -> int | None:
    floor = (validation.get("config") or {}).get("night_floor")
    return int(floor) if floor is not None else None


def _render_story(
    arm_row: pd.Series, base_row: pd.Series, *, arm: str, validation: dict[str, Any]
) -> None:
    """(a) The six beats, every figure read from the arm table."""
    night_floor = _night_floor(validation)
    if night_floor is None:
        # No al_explain group in this package: the mechanism is still true, the
        # exact floor just isn't in the package, so it is described in words rather
        # than asserted as a number.
        acquisition_floor = "a night floor — a minimum count of night frames taken first"
    else:
        acquisition_floor = f"a night floor of **{night_floor}** frames"

    mined = int(arm_row["n_train_images"]) - int(base_row["n_train_images"])
    # val_images is not a column of the shipped table today; if a future export adds
    # it, the beat names the split's size instead of just naming the split.
    val_images = arm_row.get("val_images")
    evaluation = "Both checkpoints are scored on the same held-out validation split"
    if pd.notna(val_images):
        evaluation += f" ({int(val_images):,} images)"

    story_arrows([
        (
            "Problem",
            f"The baseline detector scores **{base_row['night_map5095']:.4f}** night "
            f"mAP50-95 against **{base_row['overall_map5095']:.4f}** overall. Night is "
            "where it fails.",
        ),
        (
            "Hypothesis",
            "That gap is a data problem, not a model problem: the training pool "
            "under-represents the conditions the detector fails in. Mine the unlabelled "
            "pool for those frames, label them, retrain.",
        ),
        (
            "Acquisition",
            f"`{arm}` scores communities of visually similar pool frames by the failure "
            "mass routed into them, allocates the mining budget across communities in "
            f"proportion to that mass, and applies {acquisition_floor} so night frames "
            "cannot be out-voted by the day-heavy pool.",
        ),
        (
            "Training",
            f"The **{mined:,}** mined frames grow the training set from "
            f"**{int(base_row['n_train_images']):,}** to "
            f"**{int(arm_row['n_train_images']):,}** images, and the detector is "
            "retrained on it from the same starting checkpoint.",
        ),
        ("Evaluation", f"{evaluation} — the mined frames are train-pool frames, never test images."),
        (
            "Result",
            f"Night mAP50-95 **{base_row['night_map5095']:.4f} → "
            f"{arm_row['night_map5095']:.4f}** (**{arm_row['delta_night']:+.4f}**); "
            f"overall {arm_row['delta_overall']:+.4f}.",
        ),
    ])


def _render_arm_chart(arms: pd.DataFrame, *, arm: str) -> None:
    """(b) All arms in the order the experiment ran them, night first."""
    st.subheader("Every arm, one chart")
    ordered = arms.sort_values("round_order").reset_index(drop=True)
    order = [str(name) for name in ordered["arm"]]

    st.altair_chart(
        bar_chart(
            ordered, x="arm", y="delta_night", highlight=arm, sort=order,
            title="Night mAP50-95 vs baseline",
        ),
        width="stretch",
    )
    st.altair_chart(
        bar_chart(
            ordered, x="arm", y="delta_overall", highlight=arm, sort=order,
            title="Overall mAP50-95 vs baseline",
        ),
        width="stretch",
    )

    best_overall = ordered.loc[ordered["delta_overall"].idxmax()]
    arm_row = ordered.loc[ordered["arm"] == arm].iloc[0]
    st.caption(
        f"Best overall arm: `{best_overall['arm']}` ({best_overall['delta_overall']:+.4f}). "
        f"`{arm}` is the best NIGHT arm ({arm_row['delta_night']:+.4f} night, "
        f"{arm_row['delta_overall']:+.4f} overall) — it was built to buy night "
        "performance, and that is the trade it makes. Weak-supervision arms are greyed: "
        "they are pseudo-label training runs, shown on the Weak Supervision page."
    )
    with st.expander("Per-arm table"):
        columns = [column for column in _ARM_TABLE_COLUMNS if column in ordered.columns]
        st.dataframe(ordered[columns], hide_index=True)
        st.caption(
            "night_ped_map5095 is the night pedestrian class — the mechanism the "
            "night arms move; night_share/n_scenes describe each arm's own mined set."
        )


def _render_community_section(communities: pd.DataFrame, *, arm: str) -> None:
    """(c) Community mass -> quota, and what the night floor changed."""
    st.subheader(f"How `{arm}` chooses: community mass → quota")
    if communities.empty:
        st.info(_STALE_PACKAGE_NOTE)
        return

    after = f"quota_{arm}"
    quota_columns = sorted(c for c in communities.columns if c.startswith("quota_"))
    if after not in quota_columns:
        st.info(_STALE_PACKAGE_NOTE)
        return
    # The other arm's quota over the SAME communities -- the run's own control for
    # "what did the night floor change?", since the two allocations differ in
    # nothing else. None when this package only exported one arm's quota, in which
    # case the paired comparison below simply isn't drawn (there is nothing to
    # compare) rather than charting one column against itself.
    before = next((c for c in quota_columns if c != after), None)

    # The -1 backfill sentinel is a bucket for frames drawn from outside every
    # community, not a community: charting it as one would put a mass-0 outlier on
    # both axes.
    real = communities.loc[communities["community"] >= 0].copy()
    real["night_share"] = real["night_members"] / real["size"]

    st.caption(
        f"{len(real)} communities of visually similar pool frames (Louvain over the "
        f"frame-similarity graph). Each one's mined quota is proportional to the "
        f"failure mass routed into it — {int(real[after].sum()):,} frames in total."
    )
    scatter = (
        alt.Chart(real)
        .mark_circle(size=70, opacity=0.8)
        .encode(
            x=alt.X("mass:Q", title="failure mass routed to the community"),
            y=alt.Y(f"{after}:Q", title=f"frames mined ({arm})"),
            color=alt.Color(
                "night_share:Q", scale=alt.Scale(scheme="blues"), title="night share"
            ),
            tooltip=[
                column
                for column in ("community", "size", "night_members", "mass", before, after)
                if column is not None
            ],
        )
    )
    st.altair_chart(scatter, width="stretch")

    if before is None:
        return

    top = real.sort_values(["mass", "community"], ascending=[False, True]).head(_TOP_COMMUNITIES)
    paired = top.melt(
        id_vars=["community"], value_vars=[before, after],
        var_name="allocation", value_name="frames",
    )
    paired["community"] = "#" + paired["community"].astype(str)
    st.altair_chart(
        bar_chart(
            paired, x="community", y="frames", color_field="allocation", zero_line=False,
            sort=[f"#{community}" for community in top["community"]],
            title=f"Quota per community, {before.removeprefix('quota_')} vs "
            f"{after.removeprefix('quota_')} (top {len(top)} by mass)",
        ),
        width="stretch",
    )

    jump = community_jump(communities, before=before, after=after)
    if jump is None:
        return
    ratio = jump["ratio"]
    # Plain ASCII "x", not the typographic multiplication sign the design doc uses
    # (ruff RUF001 flags U+00D7 as an ambiguous character) -- same call scenarios.py
    # made for the minus sign.
    multiple = "" if ratio is None else f" — {ratio:.1f}x"
    st.caption(
        f"The all-night community #{jump['community']} ({jump['size']:,} frames) went "
        f"from {jump['quota_before']} to {jump['quota_after']} mined frames under the "
        f"night floor{multiple}. That reallocation is the whole difference between the "
        f"two arms."
    )


def _render_selected_frame(
    frame_row: pd.Series,
    gt: pd.DataFrame,
    *,
    arm: str,
    explain: pd.DataFrame,
    n_communities: int,
    night_floor: int | None,
) -> None:
    """(d) One selected frame: its crop with GT, its facts, and why it was picked."""
    token = str(frame_row["sample_data_token"])
    gt_rows = _visible_gt(gt, token)

    image_path = crop_path(token)
    scale = 0.6
    if not image_path.is_file():
        image_path = thumb_path(token)
        scale = 0.16
    if image_path.is_file():
        st.image(
            draw_overlay(
                Image.open(image_path), _gt_for_render(gt_rows, arm), pd.DataFrame(),
                mode="gt", scale=scale,
            )
        )
    if frame_row.get("split") == "train_pool":
        st.caption(_TRAIN_POOL_NOTE)

    facts = st.columns(3)
    facts[0].metric("Scene", str(frame_row.get("scene_name", "n/a")))
    facts[1].metric("Night", "Yes" if frame_row.get("is_night") else "No")
    facts[2].metric("GT boxes", str(len(gt_rows)))

    if not al_explain_available():
        return
    rows = explain.loc[explain["sample_data_token"] == token]
    if rows.empty:
        st.caption(_EXPLAIN_FRAME_ABSENT_NOTE)
        return
    st.markdown("**Why this frame**")
    factors = selection_factors(
        rows.iloc[0].to_dict(), n_communities=n_communities, night_floor=night_floor
    )
    for label, value, flag in factors:
        mark = "" if flag is None else (" ✓" if flag else " ✗")
        st.markdown(f"**{label}:** {value}{mark}")


def _render_gallery(
    manifest: pd.DataFrame,
    gt: pd.DataFrame,
    communities: pd.DataFrame,
    *,
    arm: str,
    validation: dict[str, Any],
) -> None:
    """(d) The curated slice of the arm's mined set, and the factor panel."""
    st.subheader("Why was this frame selected?")
    if "al_selected_by" not in manifest.columns:
        st.info(_STALE_PACKAGE_NOTE)
        return
    selected = manifest.loc[manifest["al_selected_by"] == arm].reset_index(drop=True)
    if selected.empty:
        st.info(f"no `{arm}`-selected frames are curated into this package")
        return

    st.caption(
        f"{len(selected)} of the arm's mined frames are curated into this package with "
        "crops and GT. Mining runs over the unlabelled TRAIN pool — every selection "
        "decision below was made before any of these frames was labelled."
    )
    if not al_explain_available():
        st.info(_EXPLAIN_ABSENT_NOTE)

    columns = st.columns(4)
    for position, row in enumerate(selected.itertuples()):
        token = str(row.sample_data_token)
        with columns[position % 4]:
            image_path = _frame_image_path(token)
            if image_path is not None:
                st.image(str(image_path))
            st.caption(f"{row.scene_name} · {'night' if row.is_night else 'day'}")
            if st.button("View", key=f"al_frame_select_{token}"):
                st.session_state["al_frame_token"] = token

    chosen = st.session_state.get("al_frame_token")
    if not chosen or chosen not in set(selected["sample_data_token"]):
        return
    frame_row = selected.loc[selected["sample_data_token"] == chosen].iloc[0]
    # n_communities from the community table the chart above draws (not the
    # validation JSON's own count), so the panel's "rank N of M" and that chart can
    # never disagree; the validation record is the fallback when the table is absent.
    n_communities = (
        int((communities["community"] >= 0).sum())
        if not communities.empty
        else int(validation.get("n_communities", 0))
    )
    _render_selected_frame(
        frame_row, gt,
        arm=arm, explain=load_al_explain(), n_communities=n_communities,
        night_floor=_night_floor(validation),
    )


def _exemplar_label(
    token: str, manifest: pd.DataFrame, gt: pd.DataFrame, preds: pd.DataFrame,
    *, arm: str, baseline: str,
) -> str:
    rows = manifest.loc[manifest["sample_data_token"] == token]
    scene = str(rows.iloc[0].get("scene_name", token)) if not rows.empty else token
    is_night = bool(rows.iloc[0].get("is_night")) if not rows.empty else False
    n_fixed = len(
        fixed_boxes(
            _visible_gt(gt, token),
            preds.loc[preds["sample_data_token"] == token],
            baseline=baseline,
            arm=arm,
        )
    )
    boxes = "box" if n_fixed == 1 else "boxes"
    return f"{scene} · {'night' if is_night else 'day'} · {n_fixed} upgraded {boxes}"


def _render_before_after(
    arms: pd.DataFrame,
    manifest: pd.DataFrame,
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    arm: str,
    baseline: str,
    tokens: list[str],
) -> None:
    """(e) The hand-approved before/after frames, and what changed on each box."""
    st.subheader("Before / after")
    if not tokens:
        st.info("no exemplar frames in this package")
        return

    labels = {
        token: _exemplar_label(token, manifest, gt, preds, arm=arm, baseline=baseline)
        for token in tokens
    }
    token = st.selectbox(
        "Exemplar frame", options=tokens, format_func=lambda t: labels[t], key="al_exemplar"
    )
    models = _models_from_gt(gt)
    if not models:
        st.info(_STALE_PACKAGE_NOTE)
        return
    model = st.radio(
        "Model", models, index=models.index(baseline) if baseline in models else 0,
        format_func=model_label, key="al_model", horizontal=True,
    )

    gt_rows = _visible_gt(gt, token)
    frame_preds = preds.loc[preds["sample_data_token"] == token]
    crop = crop_path(token)
    if crop.is_file():
        st.image(
            draw_overlay(
                Image.open(crop),
                _gt_for_render(gt_rows, model),
                frame_preds.loc[frame_preds["model"] == model],
                mode="overlay",
                scale=0.6,
            )
        )
    st.caption(
        "Green = ground truth, orange dashed = a GT box this model missed, white = its "
        "true positives, yellow dotted = a claim below the confidence floor, red = a "
        "false positive."
    )

    upgraded = fixed_boxes(gt_rows, frame_preds, baseline=baseline, arm=arm)
    st.markdown(f"**Boxes `{arm}` upgraded over `{baseline}`**")
    if upgraded.empty:
        st.caption("no upgraded boxes on this frame")
    else:
        st.dataframe(upgraded, hide_index=True)

    arm_row = arms.loc[arms["arm"] == arm].iloc[0]
    base_row = arms.loc[arms["arm"] == baseline].iloc[0]
    # The champion sentence only when this package actually recorded that model --
    # naming a checkpoint the viewer can't select would be noise.
    champion_note = (
        f" `{model_label('champion')}` is a different, model-size checkpoint, not an "
        "AL arm — it is here as a second opinion on the same frame."
        if "champion" in models
        else ""
    )
    st.caption(
        f"One frame is an illustration, not the result. The result is the arm: night "
        f"mAP50-95 {base_row['night_map5095']:.4f} → {arm_row['night_map5095']:.4f} on "
        f"the held-out val split.{champion_note}"
    )

    story_arrows([
        ("Failure", f"`{baseline}` misses (or barely claims) a box on a val frame like this one."),
        (
            "Selected for acquisition",
            f"`{arm}` mines visually similar TRAIN-pool frames from the communities that "
            "failure mass routes into, with the night floor applied.",
        ),
        ("Added to training data", "Those frames are labelled and added to the training set."),
        ("Retrained model", "The detector is retrained on the enlarged set."),
        (
            "Improved prediction",
            f"On the same held-out frame, `{arm}` now detects the box — and across the "
            f"whole val split it gains {arm_row['delta_night']:+.4f} night mAP50-95.",
        ),
    ])


def render() -> None:
    st.title("Active Learning")
    st.caption(
        "Which frames the system asks to have labelled next, why, and what "
        "retraining on them bought. Every number here is read from the package."
    )

    arms = load_al_results()
    package = load_al_exemplars()
    arm = package.get("arm")
    baseline = package.get("baseline")
    known_arms = set(arms["arm"]) if not arms.empty else set()
    if not arm or not baseline or not {arm, baseline} <= known_arms:
        st.error(_STALE_PACKAGE_NOTE)
        return

    arm_row = arms.loc[arms["arm"] == arm].iloc[0]
    base_row = arms.loc[arms["arm"] == baseline].iloc[0]
    validation = load_al_explain_validation()

    metric_cards([
        (f"Night mAP50-95 (`{baseline}`)", f"{base_row['night_map5095']:.4f}"),
        (f"Night mAP50-95 (`{arm}`)", f"{arm_row['night_map5095']:.4f}"),
        ("Night delta", f"{arm_row['delta_night']:+.4f}"),
        (
            "Frames mined",
            f"{int(arm_row['n_train_images']) - int(base_row['n_train_images']):,}",
        ),
    ])

    _render_story(arm_row, base_row, arm=str(arm), validation=validation)

    st.divider()
    _render_arm_chart(arms, arm=str(arm))

    st.divider()
    communities = load_al_communities()
    _render_community_section(communities, arm=str(arm))

    st.divider()
    manifest = load_frame_manifest()
    gt = load_gt_boxes()
    preds = load_predictions()
    _render_gallery(manifest, gt, communities, arm=str(arm), validation=validation)

    st.divider()
    _render_before_after(
        arms, manifest, gt, preds,
        arm=str(arm), baseline=str(baseline),
        tokens=[str(token) for token in package.get("tokens") or []],
    )
