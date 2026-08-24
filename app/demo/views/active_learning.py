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

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from filters import (
    community_jump,
    fixed_boxes,
    frame_quota_before,
    gt_for_render,
    model_label,
    quota_column_pair,
    reason_chips,
    selection_factors,
    strategy_coverage,
    visible_gt,
)
from PIL import Image
from render import (
    bar_chart,
    chip_row,
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
    al_explain_available,
    crop_path,
    frame_image_path,
    load_al_communities,
    load_al_exemplars,
    load_al_explain,
    load_al_explain_validation,
    load_al_results,
    load_frame_manifest,
    load_gt_boxes,
    load_overview,
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

# The paired quota chart gets crowded well before all 97 communities fit; the
# heaviest ones are where the night floor's reallocation is visible anyway.
_TOP_COMMUNITIES = 12

# How many gallery thumbs render before the "show all" checkbox -- all 78 curated
# selected frames in one grid is 78 images and 78 buttons (consolidated review M8).
_GALLERY_PAGE = 24

# Phase 9b (Task 6): the acquisition strategies the experiment ran over the SAME
# mining budget -- the comparison that answers "why the graph arm rather than
# similarity mining?" before the all-arms chart shows the results. Keys are arm
# names in active_learning_results.parquet, values the on-screen labels, so a
# viewer reads a strategy rather than an identifier off the chart axis.
_STRATEGIES = {
    "random": "Random sample",
    "mined": "Similarity mining",
    "graph_rate_night": "Graph-aware mining",
}

# How each strategy WORKS -- fixed prose, because the mechanism is the same in
# every package; every NUMBER beside it is read from the arm table.
_STRATEGY_NOTES = {
    "random": "draws frames uniformly at random from the unlabelled pool.",
    "mined": (
        "takes the unlabelled frames nearest the failure clusters in embedding space."
    ),
    "graph_rate_night": (
        "scores communities of visually similar pool frames by the failure mass "
        "routed into them, turns that mass into a per-community quota, and applies "
        "a night floor before the quotas are filled."
    ),
}

# The section heading counts the strategies THIS package carries rather than
# claiming three when a run skipped one -- the same "computed, never asserted"
# rule the superlatives on this page follow.
_STRATEGY_COUNT_WORDS = {2: "Two", 3: "Three"}

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
    # val_images ships from package_version 0.6 (6019 on every arm); .get keeps an
    # older package rendering, with the beat naming the split but not its size.
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


def _render_night_inversion_callout(arms: pd.DataFrame) -> None:
    """Phase 9a (Task 6): the "what we learned" sentence for the arm chart -- the
    best NIGHT arm's own weak-supervised twin is (in the real package) the WORST
    night arm in the table, which is the point: the night gain came from the
    mined FRAMES, not from training on cheaper (weak) labels.

    ``best_night_arm`` is read from overview_metrics.json (the same night-only,
    weak-excluded winner the Overview headline uses), never recomputed here --
    the two must always name the same arm. Skipped whenever that arm or its
    ``weak_<arm>`` twin isn't a row in THIS package's arm table, rather than
    guessed at.

    Both superlatives are CHECKED against this package's own table before they are
    written (Phase 9a review I1, the ``_night_rank_caption`` convention on the Weak
    Supervision page): "best ... worst" is claimed only when the two arms really do
    hold the max and the min of ``delta_night``. On any other package the same two
    numbers are contrasted with no superlative in the sentence at all -- the point
    (targeting night bought night; cheaper labels did not) survives without a rank
    claim the table does not support.
    """
    night_arm = (load_overview().get("results") or {}).get("best_night_arm")
    if night_arm is None:
        return
    weak_arm = f"weak_{night_arm}"
    night_rows = arms.loc[arms["arm"] == night_arm]
    weak_rows = arms.loc[arms["arm"] == weak_arm]
    if night_rows.empty or weak_rows.empty:
        return
    delta_night = float(night_rows.iloc[0]["delta_night"])
    weak_delta_night = float(weak_rows.iloc[0]["delta_night"])

    ranked = arms.dropna(subset=["delta_night"])
    best = ranked.loc[ranked["delta_night"].idxmax()]
    worst = ranked.loc[ranked["delta_night"].idxmin()]
    if str(best["arm"]) == night_arm and str(worst["arm"]) == weak_arm:
        learned(
            f"Targeting night bought night: `{night_arm}` ({delta_night:+.4f} night "
            f"mAP50-95) is the best night arm of {len(ranked)}, while its "
            f"weak-supervised twin `{weak_arm}` ({weak_delta_night:+.4f}) is the "
            "worst — the night gain came from the frames, not from cheaper labels."
        )
        return
    learned(
        f"Targeting night bought night: `{night_arm}` posts {delta_night:+.4f} night "
        f"mAP50-95 against the baseline, while its weak-supervised twin "
        f"`{weak_arm}` posts {weak_delta_night:+.4f} — the night gain came from the "
        "frames, not from cheaper labels."
    )


def _mined_count(
    arm_row: pd.Series, base_row: pd.Series, validation: dict[str, Any]
) -> int | None:
    """How many frames the arm mined: the growth of the training set, or the
    mining budget `demo al-explain` recorded, or ``None`` when this package states
    neither (the section then says "the mined frames" instead of a count it cannot
    derive)."""
    arm_images, base_images = arm_row.get("n_train_images"), base_row.get("n_train_images")
    if pd.notna(arm_images) and pd.notna(base_images):
        return int(arm_images) - int(base_images)
    n_mine = (validation.get("config") or {}).get("n_mine")
    return int(n_mine) if n_mine is not None else None


def _coverage_row(coverage: pd.DataFrame, arm: str) -> pd.Series | None:
    """One strategy's coverage row, or ``None`` when this package has no row for
    that arm -- or has one but is missing any of the three numbers a comparison
    between strategies would have to state."""
    rows = coverage.loc[coverage["arm"] == arm]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if any(pd.isna(row[column]) for column in ("n_scenes", "night_share", "delta_night")):
        return None
    return row


def _best_night_arm(arms: pd.DataFrame) -> str | None:
    """The arm holding this table's best night gain, or ``None`` when no arm has a
    night delta at all -- the same ``idxmax`` over the non-NA deltas the night
    inversion callout above ranks with."""
    if "delta_night" not in arms.columns:
        return None
    ranked = arms.dropna(subset=["delta_night"])
    if ranked.empty:
        return None
    return str(ranked.loc[ranked["delta_night"].idxmax()]["arm"])


def _strategy_triple(row: pd.Series) -> str:
    """One strategy's (scenes, night share, night delta), each rendered as "n/a"
    when this package doesn't carry it.

    The noun agrees with the count (item M8, Phase 9b consolidated review): a
    one-scene arm read "1 scenes". "n/a scenes" keeps the plural -- there is no
    count to agree with.
    """
    n_scenes = None if pd.isna(row["n_scenes"]) else int(row["n_scenes"])
    scenes = (
        "n/a scenes"
        if n_scenes is None
        else f"{n_scenes:,} scene{'' if n_scenes == 1 else 's'}"
    )
    night = "n/a" if pd.isna(row["night_share"]) else f"{float(row['night_share']):.0%}"
    delta = "n/a" if pd.isna(row["delta_night"]) else f"{float(row['delta_night']):+.4f}"
    return f"{row['strategy']}: {scenes} · {night} night · {delta}"


def _strategy_lesson(coverage: pd.DataFrame, arms: pd.DataFrame) -> str:
    """The strategy section's "what we learned" sentence, COMPUTED from this
    package's own table (Phase 9a's rule, restated in the 9b spec: superlatives are
    computed, never asserted).

    The comparison -- similarity mining found near-duplicates while the graph arm
    spread the same budget wider and took the best night gain -- is written only
    when all three of its clauses are true HERE: the graph arm covers strictly more
    scenes than the similarity arm, at a strictly higher night share, and holds the
    best night delta of every arm in the table. Any other package (including one
    that never ran ``mined``) gets the same numbers with no ranking claim at all:
    each strategy's scenes/night share/night delta, side by side.
    """
    mined = _coverage_row(coverage, "mined")
    graph = _coverage_row(coverage, "graph_rate_night")
    if (
        mined is not None
        and graph is not None
        and int(graph["n_scenes"]) > int(mined["n_scenes"])
        and float(graph["night_share"]) > float(mined["night_share"])
        and _best_night_arm(arms) == "graph_rate_night"
    ):
        # Same count/noun agreement as _strategy_triple (Phase 9b review M8): a
        # one-scene arm must not read "1 scenes" here either.
        mined_scenes = int(mined["n_scenes"])
        graph_scenes = int(graph["n_scenes"])
        return (
            f"Similarity mining found near-duplicates: {mined_scenes:,} "
            f"scene{'' if mined_scenes == 1 else 's'}, "
            f"{float(mined['night_share']):.0%} night, {float(mined['delta_night']):+.4f} "
            "night mAP50-95; graph-aware mining spread the same budget over "
            f"{graph_scenes:,} scene{'' if graph_scenes == 1 else 's'} at "
            f"{float(graph['night_share']):.0%} night "
            f"and took the best night gain ({float(graph['delta_night']):+.4f})."
        )
    triples = "; ".join(_strategy_triple(row) for _, row in coverage.iterrows())
    return (
        "Same budget, spent differently — scenes covered, night share of the mined "
        f"set, and night mAP50-95 against the baseline: {triples}."
    )


def _render_strategies(arms: pd.DataFrame, *, n_mined: int | None) -> bool:
    """(b, ahead of the arm chart) The acquisitions side by side: what each
    strategy's own mined set covers, and what it bought at night.

    Returns whether anything was drawn -- a package carrying fewer than two of the
    strategies has no comparison to make, and the section is skipped entirely
    rather than charting one bar against itself.
    """
    coverage = strategy_coverage(arms, strategies=_STRATEGIES)
    if len(coverage) < 2:
        return False

    count = _STRATEGY_COUNT_WORDS.get(len(coverage), str(len(coverage)))
    st.subheader(
        f"{count} ways to pick the mined frames"
        if n_mined is None
        else f"{count} ways to pick {n_mined:,} frames"
    )
    st.caption(
        "Every arm below spent the same mining budget on the same unlabelled pool "
        "— these are the sets they came back with."
    )

    order = [str(label) for label in coverage["strategy"]]
    highlight = _STRATEGIES["graph_rate_night"]
    scenes_column, night_column = st.columns(2)
    with scenes_column:
        st.altair_chart(
            bar_chart(
                coverage, x="strategy", y="n_scenes", highlight=highlight, sort=order,
                zero_line=False, y_title="scenes covered",
                title="Scenes the mined frames came from",
            ),
            width="stretch",
        )
    with night_column:
        st.altair_chart(
            bar_chart(
                coverage, x="strategy", y="night_share", highlight=highlight, sort=order,
                zero_line=False, y_title="night share",
                title="Night share of the mined set",
            ),
            width="stretch",
        )
    provenance("recorded", "active_learning_results.parquet")
    for record in coverage.itertuples(index=False):
        note = _STRATEGY_NOTES.get(str(record.arm))
        if note is not None:
            st.caption(f"**{record.strategy}** (`{record.arm}`) — {note}")
    learned(_strategy_lesson(coverage, arms))
    return True


def _render_arm_chart(arms: pd.DataFrame, *, arm: str) -> None:
    """(b) All arms in the order the experiment ran them, night first."""
    st.subheader("Every arm, one chart")
    ordered = arms.sort_values("round_order").reset_index(drop=True)
    order = [str(name) for name in ordered["arm"]]

    # labelAngle=-45 (bar_chart never truncates a label): 13 arms across ~1100 px
    # clipped the weak arms' names to "weak_graph_rate..." horizontally, and an arm
    # name is an identifier -- a clipped one names nothing (consolidated review,
    # real-browser finding 2).
    st.altair_chart(
        bar_chart(
            ordered, x="arm", y="delta_night", highlight=arm, sort=order,
            label_angle=-45, title="Night mAP50-95 vs baseline",
        ),
        width="stretch",
    )
    st.altair_chart(
        bar_chart(
            ordered, x="arm", y="delta_overall", highlight=arm, sort=order,
            label_angle=-45, title="Overall mAP50-95 vs baseline",
        ),
        width="stretch",
    )
    provenance("recorded", "active_learning_results.parquet")

    best_overall = ordered.loc[ordered["delta_overall"].idxmax()]
    arm_row = ordered.loc[ordered["arm"] == arm].iloc[0]
    st.caption(
        f"Best overall arm: `{best_overall['arm']}` ({best_overall['delta_overall']:+.4f}). "
        f"`{arm}` is the best NIGHT arm ({arm_row['delta_night']:+.4f} night, "
        f"{arm_row['delta_overall']:+.4f} overall) — it was built to buy night "
        "performance, and that is the trade it makes. Weak-supervision arms are greyed: "
        "they are pseudo-label training runs, shown on the Weak Supervision page."
    )
    _render_night_inversion_callout(arms)
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
    # ONE derivation of the (after, before) quota pair, shared with
    # filters.frame_quota_before -- which puts this same jump on a reason chip, and
    # derived the pair itself until item M4 of the Phase 9b consolidated review. The
    # helper's own docstring carries the rules (empty table / no column for this arm
    # -> None, i.e. a stale package for this section; a `before` of None when the
    # package exported only one arm's quota, in which case the paired comparison
    # below simply isn't drawn rather than charting a column against itself).
    pair = quota_column_pair(communities, arm=arm)
    if pair is None:
        st.info(_STALE_PACKAGE_NOTE)
        return
    after, before = pair

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
    # grouped=True: the two quotas are ALTERNATIVES (the same budget allocated two
    # ways), not parts of a total -- stacked, community #10301's 83 and 323 read as
    # 406 mined frames, a number that never existed (consolidated review,
    # real-browser finding 1).
    st.altair_chart(
        bar_chart(
            paired, x="community", y="frames", color_field="allocation", zero_line=False,
            grouped=True, sort=[f"#{community}" for community in top["community"]],
            title=f"Quota per community, {before.removeprefix('quota_')} vs "
            f"{after.removeprefix('quota_')} (top {len(top)} by mass, side by side — "
            "the two arms are alternatives, not parts of a total)",
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
    communities: pd.DataFrame,
    *,
    arm: str,
    explain: pd.DataFrame,
    n_communities: int,
    night_floor: int | None,
    validation: dict[str, Any],
) -> None:
    """(d) One selected frame: its crop with GT, its facts, and why it was picked."""
    token = str(frame_row["sample_data_token"])
    gt_rows = visible_gt(gt, token)

    image_path = crop_path(token)
    scale = 0.6
    if not image_path.is_file():
        image_path = thumb_path(token)
        scale = 0.16
    if image_path.is_file():
        st.image(
            draw_overlay(
                Image.open(image_path), gt_for_render(gt_rows, arm), pd.DataFrame(),
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
    explain_row = rows.iloc[0].to_dict()
    st.markdown("**Why this frame**")
    # Phase 9b (Task 6): the same facts the lines below spell out, as a chip row
    # that can be read at a glance -- built by filters.reason_chips from THIS row,
    # so a chip can never claim something the panel under it doesn't.
    chip_row(
        reason_chips(
            explain_row,
            gt_rows,
            n_communities=n_communities,
            night_floor=night_floor,
            quota_before=frame_quota_before(communities, explain_row, arm=arm),
        )
    )
    factors = selection_factors(
        explain_row, n_communities=n_communities, night_floor=night_floor
    )
    for label, value, flag in factors:
        mark = "" if flag is None else (" ✓" if flag else " ✗")
        st.markdown(f"**{label}:** {value}{mark}")

    # Phase 9a (Task 6): the re-derivation's own validation record, not the
    # n_communities/night_floor this function was already handed above (those come
    # from the community table, with the validation JSON only as ITS fallback) --
    # this line is specifically about what `demo al-explain` itself validated.
    n_selected = validation.get("n_selected")
    n_communities_validated = validation.get("n_communities")
    if n_selected is not None and n_communities_validated is not None:
        provenance(
            "reproduced",
            f"demo al-explain reproduced the run: {int(n_selected):,} frames, "
            f"{n_communities_validated} communities",
        )


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

    # Night frames first: this arm is night-TARGETED, so the frames that show what
    # it was built to find must not sit below the fold behind 40 day frames
    # (consolidated review M8).
    ordered = selected.sort_values(
        ["is_night", "scene_name", "sample_data_token"], ascending=[False, True, True]
    ).reset_index(drop=True)
    shown = ordered
    if len(ordered) > _GALLERY_PAGE:
        if st.checkbox(f"Show all {len(ordered)} frames", key="al_gallery_show_all"):
            st.caption(
                f"All {len(ordered)} of the arm's mined frames curated into this "
                "package with crops and GT, night frames first."
            )
        else:
            shown = ordered.head(_GALLERY_PAGE)
            st.caption(
                f"The first {_GALLERY_PAGE} of {len(ordered)} of the arm's mined frames "
                "curated into this package with crops and GT, night frames first."
            )
    else:
        st.caption(
            f"{len(ordered)} of the arm's mined frames are curated into this package "
            "with crops and GT, night frames first."
        )
    st.caption(
        "Mining runs over the unlabelled TRAIN pool — every selection decision below "
        "was made before any of these frames was labelled."
    )
    if not al_explain_available():
        st.info(_EXPLAIN_ABSENT_NOTE)

    columns = st.columns(4)
    for position, row in enumerate(shown.itertuples()):
        token = str(row.sample_data_token)
        with columns[position % 4]:
            image_path = frame_image_path(token)
            if image_path is not None:
                st.image(str(image_path))
            st.caption(f"{row.scene_name} · {'night' if row.is_night else 'day'}")
            if st.button("View", key=f"al_frame_select_{token}"):
                st.session_state["al_frame_token"] = token

    chosen = st.session_state.get("al_frame_token")
    # Membership is checked against every curated selected frame, not just the page
    # of them on screen: a viewer who picks a frame and then collapses the gallery
    # keeps the panel they opened.
    if not chosen or chosen not in set(ordered["sample_data_token"]):
        return
    frame_row = ordered.loc[ordered["sample_data_token"] == chosen].iloc[0]
    # n_communities from the community table the chart above draws (not the
    # validation JSON's own count), so the panel's "rank N of M" and that chart can
    # never disagree; the validation record is the fallback when the table is absent.
    n_communities = (
        int((communities["community"] >= 0).sum())
        if not communities.empty
        else int(validation.get("n_communities", 0))
    )
    _render_selected_frame(
        frame_row, gt, communities,
        arm=arm, explain=load_al_explain(), n_communities=n_communities,
        night_floor=_night_floor(validation), validation=validation,
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
            visible_gt(gt, token),
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

    gt_rows = visible_gt(gt, token)
    frame_preds = preds.loc[preds["sample_data_token"] == token]
    crop = crop_path(token)
    if crop.is_file():
        st.image(
            draw_overlay(
                Image.open(crop),
                gt_for_render(gt_rows, model),
                frame_preds.loc[frame_preds["model"] == model],
                mode="overlay",
                scale=0.6,
            )
        )
        provenance("recomputed", "per-box claims from predictions.parquet")
        # The legend describes colours this panel just drew, so it renders only
        # where the overlay actually did (final review, Phase 10).
        legend()

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
    loop_breadcrumb(["Mine", "Train", "Evaluate"])
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
    # The strategy comparison draws its own divider only when it drew anything --
    # a package with fewer than two acquisition arms skips the whole section.
    if _render_strategies(arms, n_mined=_mined_count(arm_row, base_row, validation)):
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
