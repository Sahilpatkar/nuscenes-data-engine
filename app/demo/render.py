"""Shared rendering primitives for the demo pages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st
from filters import FilmstripCurve
from PIL import Image, ImageDraw


def metric_cards(
    items: Sequence[tuple[str, str] | tuple[str, str, str]], per_row: int = 4
) -> None:
    """A row of st.metric cards: [(label, value), ...] or, for a card that should
    show a delta alongside a single figure rather than an arrow packed into the
    value string, [(label, value, delta), ...]."""
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        for col, item in zip(st.columns(len(chunk)), chunk, strict=True):
            if len(item) == 3:
                label, value, delta = item
                col.metric(label, value, delta=delta)
            else:
                label, value = item
                col.metric(label, value)


def story_arrows(steps: list[tuple[str, str]]) -> None:
    """The vertical Problem -> ... -> Result narrative (DEMO_PLAN.md §9)."""
    for index, (title, body) in enumerate(steps):
        st.markdown(f"**{title}**  \n{body}")
        if index < len(steps) - 1:
            st.markdown("<div style='text-align:center'>&#8595;</div>", unsafe_allow_html=True)


def recorded_banner(text: str) -> None:
    st.caption(f":material/history: {text}")


# --- GT/prediction overlay renderer (Phase 3) -------------------------------------
#
# Fixed visual language, shared by the Overview hero and the Failure Explorer detail
# view: GT boxes are solid green (dashed orange when unmatched -> a false negative),
# predictions are colored by status (true positive / false positive / low-confidence).
# Pure PIL: the drawing helpers make no Streamlit calls, so they are testable
# without a runtime (``legend()`` below is this section's one Streamlit call).


@dataclass(frozen=True)
class BoxStyle:
    color: tuple[int, int, int]
    width: int
    dash: int | None          # dash segment length in px; None = solid; 1-2 = dotted


STYLE_GT = BoxStyle(color=(46, 204, 64), width=2, dash=None)         # green solid
STYLE_FN = BoxStyle(color=(255, 133, 27), width=3, dash=8)          # orange dashed
STYLE_TP = BoxStyle(color=(255, 255, 255), width=1, dash=None)      # white thin
STYLE_FP = BoxStyle(color=(255, 65, 54), width=2, dash=None)        # red solid
STYLE_LOW_CONF = BoxStyle(color=(255, 220, 0), width=2, dash=2)     # yellow dotted
# Phase 7 (Task 5): a pseudo box on a weak-supervision train-pool frame. Blue solid
# -- a colour no GT/prediction style uses, because a pseudo box is neither: it is a
# baseline-detector proposal that a VLM's per-class counts corroborated (the VLM
# never draws a box), drawn alongside GT so the two can be compared by eye.
# Named STYLE_PSEUDO, not STYLE_VLM (consolidated review C2): the box and the score
# on it are the DETECTOR's, and the old name put the VLM's name on both.
STYLE_PSEUDO = BoxStyle(color=(0, 116, 217), width=2, dash=None)    # blue solid

_PRED_STYLES: dict[str, BoxStyle] = {"tp": STYLE_TP, "fp": STYLE_FP, "low_conf": STYLE_LOW_CONF}

# The same visual language as one line of text, beside the styles it describes so a
# new style cannot ship without a legend entry (the drift guard in
# tests/test_demo_render.py holds the two together). Phase 10 spec sec1: this replaces
# the prose sentence three pages each kept their own copy of.
#
# The badge COLOUR is only a hint -- Streamlit has no white badge, and "dashed" is not
# a colour at all -- so the WORDING carries the on-image description in every item.
LEGEND_ITEMS: tuple[tuple[str, str], ...] = (
    ("green", "green — ground truth"),
    ("orange", "orange dashed — GT box the model missed"),
    ("gray", "white — true positive"),
    ("yellow", "yellow dotted — low-confidence claim (below the hit floor)"),
    ("red", "red — false positive"),
)
# STYLE_PSEUDO deliberately has NO entry here (consolidated review M4): a pseudo box
# is not a prediction status, and the one page that draws them -- Weak Supervision --
# explains in its own prose what a pseudo box IS, which is a different claim. The
# opt-in chip this component shipped with had no caller, and on that page's GT +
# pseudo image it would have claimed prediction colours the image never carries.


def legend_text() -> str:
    """The overlay legend as one markdown line of ``:{colour}-badge[...]`` chips.

    The badge colour is a hint only; the words carry the on-image description
    ("white", "dashed", "dotted").
    """
    return " ".join(f":{color}-badge[{wording}]" for color, wording in LEGEND_ITEMS)


def legend() -> None:
    st.markdown(legend_text())


_VALID_MODES = ("gt", "pred", "overlay")

# A box narrower than this, once scaled onto the display image, is too small for a
# legible label (thumb legibility -- thumbs are scale=0.16, so most boxes land well
# under this).
_MIN_LABEL_WIDTH_PX = 24

_XYXY = tuple[float, float, float, float]


def _draw_dashed_edge(
    draw: ImageDraw.ImageDraw,
    *,
    axis: str,
    fixed: float,
    start: float,
    end: float,
    style: BoxStyle,
    inward: int,
) -> None:
    """Walk one straight, axis-aligned edge in ``style.dash``-length segments with
    equal gaps, each "on" segment drawn as a thin FILLED RECTANGLE rather than a
    ``draw.line`` stroke.

    ``draw.rectangle(..., width=W)`` insets its stroke INWARD from the given
    coordinates by exactly W px (verified against PIL 12: a width-W edge occupies
    pixels [edge, edge + W - 1] toward the box interior). ``draw.line(..., width=W)``
    does NOT behave as a simple centered or fixed-offset version of that: for even
    W its thickening is DIRECTION-DEPENDENT (a vertical width-2 line from (x,10) to
    (x,20) covers columns [x, x+1]; the same line reversed, (x,20) to (x,10), covers
    [x-1, x]) -- verified empirically, and it breaks any single fixed per-edge
    offset, since two of our four edges are necessarily walked in the "reversed"
    direction relative to the other two. Building each dash segment as an explicit
    filled rectangle sidesteps the whole issue: it has the same well-defined inward
    inset as ``draw.rectangle`` by construction, regardless of which direction the
    edge is walked in.

    ``axis``: "x" for a horizontal edge (top/bottom), "y" for a vertical edge (left/
    right). ``fixed``: the edge's constant coordinate (e.g. y_min for a top edge).
    ``start``/``end``: the varying coordinate's range (start < end). ``inward``: +1
    or -1, the direction (in the *other* axis) that grows into the box interior.
    """
    length = end - start
    if length <= 0 or style.dash is None:
        return
    lo, hi = (fixed, fixed + style.width - 1) if inward > 0 else (fixed - style.width + 1, fixed)
    pos, on = 0.0, True
    while pos < length:
        seg_end = min(pos + style.dash, length)
        if on:
            box = (
                (start + pos, lo, start + seg_end, hi)
                if axis == "x"
                else (lo, start + pos, hi, start + seg_end)
            )
            draw.rectangle(box, fill=style.color)
        pos = seg_end
        on = not on


def _draw_rect(draw: ImageDraw.ImageDraw, xyxy: _XYXY, style: BoxStyle) -> None:
    """Draw one box outline: a solid rectangle, or (when ``style.dash`` is set) each
    of its four edges as dashed/dotted segments (see ``_draw_dashed_edge`` for why
    those segments are filled rectangles, not ``draw.line`` strokes) -- inset inward
    exactly like ``draw.rectangle`` does, so a dashed (FN) box and a solid (GT) box
    at identical coordinates occupy the same pixels whenever the box extent is >=
    style.width; degenerate boxes below that may differ by design.
    """
    x_min, y_min, x_max, y_max = xyxy
    if style.dash is None:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=style.color, width=style.width)
        return
    _draw_dashed_edge(draw, axis="x", fixed=y_min, start=x_min, end=x_max, style=style, inward=1)
    _draw_dashed_edge(draw, axis="x", fixed=y_max, start=x_min, end=x_max, style=style, inward=-1)
    _draw_dashed_edge(draw, axis="y", fixed=x_min, start=y_min, end=y_max, style=style, inward=1)
    _draw_dashed_edge(draw, axis="y", fixed=x_max, start=y_min, end=y_max, style=style, inward=-1)


def _validate_xyxy(xyxy: _XYXY, *, kind: str, category: str) -> None:
    """Reject an inverted box before it reaches PIL.

    x_max < x_min (or y_max < y_min) used to behave asymmetrically by style: a
    solid style hits PIL's own ``draw.rectangle`` bounds check and raises an
    unhelpful bare error ("x1 must be greater than or equal to x0"), while a
    dashed style silently vanishes (``_draw_dashed_edge``'s ``length <= 0`` early
    return draws nothing at all). Neither is acceptable for corrupt input data --
    this mirrors ``build.py``'s ``_size_bucket`` guard on a degenerate GT box area:
    corrupt data must not half-render.
    """
    x_min, y_min, x_max, y_max = xyxy
    if x_max < x_min or y_max < y_min:
        raise ValueError(
            f"draw_overlay: inverted {kind} box ({category!r}): "
            f"x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}"
        )


def _draw_label(draw: ImageDraw.ImageDraw, xyxy: _XYXY, text: str, style: BoxStyle) -> None:
    x_min, y_min, x_max, _y_max = xyxy
    if (x_max - x_min) < _MIN_LABEL_WIDTH_PX:
        return
    # Clamp to the visible canvas: a box that extends past the top/left edge
    # (x_min/y_min < 0 -- e.g. an annotation only partially inside the frame)
    # would otherwise draw its label starting off-canvas, clipping it. Collision
    # avoidance between overlapping labels is out of scope (accepted cosmetic) --
    # this only keeps an edge label from being partially invisible.
    draw.text((max(x_min, 0.0), max(y_min, 0.0)), text, fill=style.color)


def draw_overlay(
    image: Image.Image,
    gt_boxes: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    mode: str,
    scale: float,
    pseudo_boxes: pd.DataFrame | None = None,
) -> Image.Image:
    """Render GT and/or predictions onto a copy of ``image``.

    Boxes arrive in native 1600x900 coords; ``scale`` maps them onto this image
    (0.6 for crops, 0.16 for thumbs). ``gt_boxes`` needs x_min..y_max,
    category_group, matched (nullable bool: False renders as the FN style, NA
    renders as plain GT -- "not evaluated" is not the same claim as "missed");
    ``predictions`` needs x_min..y_max, category_group, conf, status (one of
    "tp"/"fp"/"low_conf" -- anything else raises ValueError).
    ``mode``: "gt" | "pred" | "overlay", else raises ValueError. The input image is
    never mutated.

    ``pseudo_boxes`` (Phase 7) is a THIRD, independent layer: the weak-supervision
    pseudo boxes for the frame (x_min..y_max, category_group, score --
    weak_labels.parquet's own schema), drawn last, in every mode. It is
    mode-independent on purpose: the frames that carry pseudo boxes are train-pool
    frames, which have GT but never predictions, so the page draws them in "gt"
    mode and still needs the blue layer. Left as ``None`` (the default) the render
    is byte-identical to the two-layer one this signature had before.

    A pseudo box is the BASELINE DETECTOR's proposal (conf >= 0.5), kept because a
    VLM's per-class counts for the frame agreed with the detector's within a
    tolerance -- the VLM emits counts and never draws a box. The layer, its style
    and its label all say "pseudo" rather than "VLM" for that reason
    (consolidated review C2: ``VLM 0.73`` attributed the detector's own confidence
    to the VLM).
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"draw_overlay: unknown mode {mode!r} — expected one of {_VALID_MODES}")

    out = image.copy()
    draw = ImageDraw.Draw(out)

    if mode in ("gt", "overlay"):
        for row in gt_boxes.itertuples(index=False):
            # matched is a nullable boolean (matched_<model> upstream): NA means "not
            # evaluated" (e.g. a model that hasn't run against this frame yet), not
            # "unmatched" -- rendering it as an FN would paint an unknown as a miss.
            # Only an explicit False is a false negative.
            matched = row.matched
            style = STYLE_GT if (pd.isna(matched) or matched) else STYLE_FN
            xyxy = (row.x_min * scale, row.y_min * scale, row.x_max * scale, row.y_max * scale)
            _validate_xyxy(xyxy, kind="GT", category=str(row.category_group))
            _draw_rect(draw, xyxy, style)
            _draw_label(draw, xyxy, str(row.category_group), style)

    if mode in ("pred", "overlay"):
        for row in predictions.itertuples(index=False):
            if row.status not in _PRED_STYLES:
                raise ValueError(
                    f"draw_overlay: unknown prediction status {row.status!r} — "
                    f"expected one of {sorted(_PRED_STYLES)}"
                )
            style = _PRED_STYLES[row.status]
            xyxy = (row.x_min * scale, row.y_min * scale, row.x_max * scale, row.y_max * scale)
            _validate_xyxy(xyxy, kind="prediction", category=str(row.category_group))
            _draw_rect(draw, xyxy, style)
            label = f"{row.category_group} {row.conf:.2f}"
            _draw_label(draw, xyxy, label, style)

    if pseudo_boxes is not None:
        for row in pseudo_boxes.itertuples(index=False):
            xyxy = (row.x_min * scale, row.y_min * scale, row.x_max * scale, row.y_max * scale)
            _validate_xyxy(xyxy, kind="pseudo", category=str(row.category_group))
            _draw_rect(draw, xyxy, STYLE_PSEUDO)
            # The score, not the category: the category is already on the GT box
            # underneath, and what a viewer needs to judge a pseudo box is how
            # confident the proposal was. The number is the BASELINE DETECTOR's
            # confidence, so the label says "pseudo" -- naming the layer for what
            # the box is, not for the model that merely corroborated it.
            _draw_label(draw, xyxy, f"pseudo {row.score:.2f}", STYLE_PSEUDO)

    return out


# --- Shared altair bar chart (Phase 7) -------------------------------------------
#
# altair is streamlit's OWN hard dependency (st.altair_chart is the API these pages
# call), so importing it at module level adds no wheel to the deployment -- it is
# declared in app/demo/requirements.txt purely to pin the version the specs below
# are written against. The Active Learning and Weak Supervision pages both draw
# their bars through this one helper so the two read as one visual language.

# The accent used for "the thing being explained" -- the same #FF851B the Scenario
# page's on-path graph nodes use.
_ACCENT = "#FF851B"
# Weak-supervision arms: pseudo-label training runs, not night-targeting AL arms.
# Greyed so they read as a different KIND of row rather than a competing result --
# the same #BBBBBB the graph panel fades its context nodes with.
_MUTED = "#BBBBBB"
_DEFAULT_BAR = "#4A90D9"
_ZERO_RULE = "#888888"
_COLOR_COLUMN = "_bar_color"


def _bar_colors(frame: pd.DataFrame, *, x: str, highlight: str | None) -> pd.Series:
    """One literal colour per row: accent for ``highlight``, grey for the weak
    family, the default bar colour otherwise."""
    family = (
        frame["family"] if "family" in frame.columns else pd.Series("", index=frame.index)
    )
    colors = pd.Series(_DEFAULT_BAR, index=frame.index)
    colors = colors.mask(family.eq("weak"), _MUTED)
    if highlight is not None:
        colors = colors.mask(frame[x].eq(highlight), _ACCENT)
    return colors


_MARKS = ("bar", "line")


def bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    highlight: str | None = None,
    color_field: str | None = None,
    title: str = "",
    sort: list[str] | None = None,
    zero_line: bool = True,
    grouped: bool = False,
    y_title: str | None = None,
    label_angle: int | None = None,
    mark: str = "bar",
) -> alt.Chart | alt.LayerChart:
    """A bar chart of ``y`` over the categorical ``x``, ready for
    ``st.altair_chart(chart, width="stretch")``.

    (``width="stretch"``, not the ``use_container_width=True`` the design doc
    wrote: that keyword is deprecated with a removal date already in the past, and
    warns on every render of streamlit 1.59 -- the demo deploys against whatever
    Streamlit Cloud installs, so the page calls the current API and
    app/demo/requirements.txt declares the floor that has it.)

    ``sort`` pins the x-axis category order (the arm chart passes ``round_order``'s
    order, so the arms read as the experiment ran them rather than alphabetically).
    ``highlight`` names the one ``x`` category drawn in the accent colour; rows whose
    ``family`` is "weak" are greyed; everything else takes the default bar colour.
    ``color_field`` overrides all of that with an ordinary categorical colour scale
    on that field (a stacked/grouped chart, e.g. the weak-sup loss decomposition),
    keeping its legend.

    ``grouped`` (with ``color_field``) puts the categories SIDE BY SIDE within each
    ``x`` instead of stacking them, via ``xOffset`` plus an explicit ``stack=None``.
    Stacking claims the parts sum to the whole; two alternative allocations of the
    same budget (the Active Learning page's graph_rate vs graph_rate_night quotas)
    are not summable, and a stacked pair of them reads as a total that does not
    exist (consolidated review, real-browser finding 1).

    ``y_title`` overrides the axis title (default: the ``y`` column name with
    underscores spaced) -- a melted long table's value column is called "value",
    which names nothing. ``label_angle`` tilts the x labels; x labels are never
    truncated (``labelLimit=0``), since a clipped arm name ("weak_graph_rate...")
    is not an identifier, and never DROPPED either (``labelOverlap=False``):
    Streamlit's own Vega theme sets ``labelOverlap: true``, the "parity" strategy
    that thins a crowded categorical axis by hiding every other label -- so a
    13-arm chart silently rendered six unlabelled bars, and a viewer has no way to
    tell an unlabelled bar from a missing one. Both axis defaults are about the
    same promise: every bar on these charts names the arm it belongs to.

    ``mark`` draws the same encoding as a line instead of bars (``"line"``) -- the
    recorded chat agent's ``make_chart`` tool emits either kind (its own enum is
    bar|line, data_engine/chat/agent.py), so the recorded-replay page needs both
    from one helper rather than a second charting path.

    ``zero_line`` layers a rule at y = 0 -- delta charts carry negative values, and
    without the rule a regression reads as just a shorter bar. That layering is why
    the return type is a union: an ``alt.LayerChart`` is not an ``alt.Chart``, and
    ``st.altair_chart`` takes either.
    """
    if grouped and color_field is None:
        raise ValueError("bar_chart: grouped=True needs a color_field")
    if mark not in _MARKS:
        raise ValueError(f"bar_chart: unknown mark {mark!r} — expected one of {list(_MARKS)}")
    data = frame.copy()
    axis_kwargs: dict[str, Any] = {"labelLimit": 0, "labelOverlap": False}
    if label_angle is not None:
        axis_kwargs["labelAngle"] = label_angle
    axis_title = y.replace("_", " ") if y_title is None else y_title
    # stack=None only on the grouped path: passing it unconditionally would also
    # unstack the loss decomposition, whose three components DO sum to the whole.
    y_encoding = (
        alt.Y(f"{y}:Q", title=axis_title, stack=None)
        if grouped
        else alt.Y(f"{y}:Q", title=axis_title)
    )
    encode: dict[str, Any] = {
        "x": alt.X(f"{x}:N", sort=sort, title=None, axis=alt.Axis(**axis_kwargs)),
        "y": y_encoding,
        "tooltip": [c for c in (x, y, color_field, "family") if c and c in data.columns],
    }
    if color_field is not None:
        encode["color"] = alt.Color(f"{color_field}:N", title=color_field.replace("_", " "))
        if grouped:
            encode["xOffset"] = alt.XOffset(f"{color_field}:N")
    else:
        data[_COLOR_COLUMN] = _bar_colors(data, x=x, highlight=highlight)
        # scale=None: the column already holds literal colours, so altair must pass
        # them through instead of building a categorical scale over them.
        encode["color"] = alt.Color(f"{_COLOR_COLUMN}:N", scale=None, legend=None)

    base = alt.Chart(data)
    # point=True on the line: a recorded chart is often 3-5 points, and a bare
    # polyline through that few values hides where the data actually is.
    drawn = base.mark_line(point=True) if mark == "line" else base.mark_bar()
    bars = drawn.encode(**encode)
    chart: alt.Chart | alt.LayerChart = bars
    if zero_line:
        rule = (
            alt.Chart(pd.DataFrame({"zero": [0.0]}))
            .mark_rule(color=_ZERO_RULE)
            .encode(y="zero:Q")
        )
        # alt.layer is typed as returning LayerChart | FacetChart (it facets when
        # given a facet spec, which this never does); the cast keeps this helper's
        # own, narrower return type honest.
        chart = cast("alt.LayerChart", alt.layer(bars, rule))
    if title:
        chart = chart.properties(title=title)
    return chart


def line_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    order: Sequence[str],
    y_title: str,
    selected: str | None = None,
    zero_line: bool = False,
    height: int = 160,
) -> alt.LayerChart:
    """A small line chart of ``y`` over an ORDERED categorical ``x`` -- the event
    viewer's CAN speed/accel over the filmstrip's five keyframe steps.

    Not ``bar_chart(mark="line")``: that helper's x order is optional and its rules
    are about zero, while this one is a sequence chart whose whole meaning is the
    order (``sort=order`` is required, since alphabetically the filmstrip's steps
    would read "current, t+1, t+2, t-1, t-2") and whose defining layer is a vertical
    rule at the step the viewer has selected.

    ``selected`` must name one of ``order``: vega silently drops a rule on an
    unknown band, so a typo'd/stale step would just not draw and the chart would
    quietly stop pointing at anything. ``zero_line`` layers the y = 0 rule (the
    acceleration curve runs mostly negative, and without the rule a deceleration
    reads as just a lower point); ``height`` keeps a pair of these short enough
    that the viewer stays on one screen.

    Axis conventions are ``bar_chart``'s (``labelLimit: 0``, ``labelOverlap:
    False``): five step labels only work as a sequence if all five are drawn, and
    Streamlit's own Vega theme would otherwise thin them. The return type is always
    a LayerChart -- ``st.altair_chart`` takes either kind, and a single, stable type
    keeps the caller from branching on how many rules it asked for.

    There is deliberately no ``title``: what these two charts ARE is said by
    ``y_title`` (which carries the unit) and by ``curve_caption`` under the pair --
    a chart heading on top of those would be a third place to keep in sync (item
    M11, Phase 9b consolidated review, where the parameter had no caller at all).
    """
    if selected is not None and selected not in order:
        raise ValueError(
            f"line_chart: unknown selected step {selected!r} — expected one of {list(order)}"
        )
    sort = list(order)
    line = (
        alt.Chart(frame)
        # point=True: five steps through a bare polyline hide where the readings are.
        .mark_line(point=True)
        .encode(
            x=alt.X(
                f"{x}:N",
                sort=sort,
                axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0, title=None),
            ),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=[c for c in (x, y) if c in frame.columns],
        )
    )
    layers: list[alt.Chart] = [line]
    if selected is not None:
        layers.append(
            alt.Chart(pd.DataFrame({x: [selected]}))
            .mark_rule(color=_ACCENT, size=2)
            .encode(x=alt.X(f"{x}:N", sort=sort, title=None, axis=None))
        )
    if zero_line:
        layers.append(
            alt.Chart(pd.DataFrame({"zero": [0.0]}))
            .mark_rule(color=_ZERO_RULE)
            .encode(y="zero:Q")
        )
    # alt.layer is typed as LayerChart | FacetChart (it facets only when given a
    # facet spec, which this never does) -- same cast bar_chart's zero rule uses.
    return cast("alt.LayerChart", alt.layer(*layers)).properties(height=height)


# --- Phase 9b (Task 4): the event's step curve, drawn the same way twice ----------
#
# The Scenario viewer (with its filmstrip slider as the cursor) and guided-tour step
# 2 (no slider, cursor pinned to the event's own frame) show the SAME two curves, so
# they build them here rather than each assembling its own pair -- the titles and the
# caption below are the whole point: they say what the two series ARE, and a page
# that wrote its own could label an ego-pose reading as CAN.

_CURVE_STEP_COLUMN = "step"
_CURVE_SPEED_COLUMN = "speed_kmh"
_CURVE_ACCEL_COLUMN = "accel_mps2"

# Both titles carry the unit, because neither reading is the one the page's other
# figures use (the filmstrip readout is m/s, the severity caption is g).
#
# The accel title is the SHORT form: at the viewer's chart width vega clipped
# "CAN longitudinal accel (m/s²)" to "CAN longitudinal accel (n", losing the unit
# it exists to carry. The caption under the pair (_CURVE_CAPTION_*) still spells
# out "longitudinal acceleration", so the axis simply stops saying it twice.
CAN_SPEED_TITLE = "CAN speed (km/h)"
EGO_SPEED_TITLE = "ego speed (km/h)"
CAN_ACCEL_TITLE = "CAN accel (m/s²)"

# The x axis is the nominal step label, so the caption is where the viewer learns
# what a step is worth in seconds (~0.5 s between nuScenes keyframes) and where the
# two series came from. The second wording is the pre-0.8 package's: the speed is
# then the GT ego pose converted to km/h, never a CAN reading (spec's honesty rule
# "a CAN curve must be CAN"), and the acceleration is CAN on every package.
_CURVE_CAPTION_CAN = (
    "Keyframes are ~0.5 s apart; speed and longitudinal acceleration from the CAN bus"
)
_CURVE_CAPTION_EGO = (
    "Keyframes are ~0.5 s apart; ego speed from the GT ego pose, longitudinal "
    "acceleration from the CAN bus"
)


def curve_frame(curve: FilmstripCurve) -> pd.DataFrame:
    """One row per filmstrip step: its label and the two readings plotted against
    it. NA readings stay NA -- altair simply breaks the line there, which is the
    honest picture of a step whose CAN row is missing."""
    return pd.DataFrame({
        _CURVE_STEP_COLUMN: [step.label for step in curve.steps],
        _CURVE_SPEED_COLUMN: [step.can_speed_kmh for step in curve.steps],
        _CURVE_ACCEL_COLUMN: [step.accel_mps2 for step in curve.steps],
    })


def curve_caption(curve: FilmstripCurve) -> str:
    """What the pair of curves is, in one line -- CAN wording only when the speed
    really is the CAN reading (``FilmstripCurve.speed_is_can``)."""
    return _CURVE_CAPTION_CAN if curve.speed_is_can else _CURVE_CAPTION_EGO


def curve_charts(
    curve: FilmstripCurve, *, selected: str | None = None, height: int = 150
) -> tuple[alt.LayerChart, alt.LayerChart]:
    """``(speed, acceleration)`` over the filmstrip's steps, both with the vertical
    rule at ``selected`` (the Scenario page passes its slider's step; the tour
    passes the event's own frame).

    The acceleration chart gets the y = 0 rule: the series runs mostly negative and
    without the rule a hard deceleration reads as just a lower point. ``selected``
    must name one of the curve's steps -- ``line_chart`` raises otherwise, rather
    than letting vega drop a rule that points at nothing.
    """
    frame = curve_frame(curve)
    order = [step.label for step in curve.steps]
    speed = line_chart(
        frame,
        x=_CURVE_STEP_COLUMN,
        y=_CURVE_SPEED_COLUMN,
        order=order,
        y_title=CAN_SPEED_TITLE if curve.speed_is_can else EGO_SPEED_TITLE,
        selected=selected,
        height=height,
    )
    accel = line_chart(
        frame,
        x=_CURVE_STEP_COLUMN,
        y=_CURVE_ACCEL_COLUMN,
        order=order,
        y_title=CAN_ACCEL_TITLE,
        selected=selected,
        zero_line=True,
        height=height,
    )
    return speed, accel


# --- Phase 9a (Task 1): trust chrome ----------------------------------------------
#
# The guided tour and every page it visits need two small, recurring pieces of
# chrome: which step of the Diagnose -> Mine -> Train -> Evaluate loop a page (or
# tour step) belongs to, and where a number/claim on screen actually came from
# (recomputed live from the package tables, a recorded experiment output shipped
# as-is, or a reproduced selection re-derived and checked against the run). Both
# are one-line badges/captions, so the wording lives in a pure `_text` function
# (unit-tested without a Streamlit runtime) with a thin Streamlit-calling wrapper
# around it, the same split `parity_caption`/`_render_parity_line` already uses.

LOOP_STAGES: tuple[str, ...] = ("Diagnose", "Mine", "Train", "Evaluate")


def loop_breadcrumb_text(active: Sequence[str] | None) -> str:
    """The loop breadcrumb line: every ``LOOP_STAGES`` entry, in order, each an
    orange badge if it's in ``active`` and a grey one otherwise. ``active=None``
    lights nothing (the Overview page's whole-loop breadcrumb). An entry of
    ``active`` that isn't a real stage raises ``ValueError`` naming it, rather than
    silently lighting nothing -- a typo'd stage name must fail loudly, not draw a
    breadcrumb that quietly lights the wrong (or no) badge.
    """
    lit = set(active or ())
    for stage in lit:
        if stage not in LOOP_STAGES:
            raise ValueError(
                f"loop_breadcrumb_text: unknown stage {stage!r} — expected one of {LOOP_STAGES}"
            )
    return " → ".join(
        f":orange-badge[{stage}]" if stage in lit else f":gray-badge[{stage}]"
        for stage in LOOP_STAGES
    )


def loop_breadcrumb(active: Sequence[str] | None, *, caption: str | None = None) -> None:
    """The breadcrumb markdown, plus an optional caption line under it (e.g. the
    Overview page's "Detect weakness → Find useful data → Retrain → Measure
    impact")."""
    st.markdown(loop_breadcrumb_text(active))
    if caption:
        st.caption(caption)


# {kind: (material icon, fixed sentence)}. The sentence is the honest claim about
# where a number on screen came from -- "recomputed" (this app derived it live from
# the package's own tables), "recorded" (the offline pipeline's own output, shipped
# and shown as-is), "reproduced" (demo al-explain re-derived the selection and this
# app checked it equals the run's). Spec §3 places each on specific numbers/charts
# per page; this module only owns the wording.
PROVENANCE: dict[str, tuple[str, str]] = {
    "recomputed": (":material/calculate:", "recomputed in this app from the package tables"),
    "recorded": (
        ":material/history:",
        "recorded experiment output — shipped as the offline pipeline produced it",
    ),
    "reproduced": (
        ":material/verified:",
        "reproduced selection — re-derived by demo al-explain and validated equal to the run",
    ),
}


def provenance_text(kind: str, detail: str = "") -> str:
    """"{icon} {sentence}", with a non-empty ``detail`` appended as " · {detail}"
    (e.g. a run id or a frame count). Unknown ``kind`` raises ``ValueError`` naming
    it -- a typo'd provenance kind must not silently claim the wrong story about
    where a number came from.
    """
    if kind not in PROVENANCE:
        raise ValueError(
            f"provenance_text: unknown kind {kind!r} — expected one of {sorted(PROVENANCE)}"
        )
    icon, sentence = PROVENANCE[kind]
    text = f"{icon} {sentence}"
    return f"{text} · {detail}" if detail else text


def provenance(kind: str, detail: str = "") -> None:
    st.caption(provenance_text(kind, detail))


def chip_row_text(chips: Sequence[str], *, color: str = "blue") -> str:
    """One markdown line of space-separated ``:{color}-badge[...]`` chips, e.g. the
    tour's "night", "pedestrian" selection tags. Empty input -> empty string, so
    the void wrapper below can no-op rather than render a blank line."""
    return " ".join(f":{color}-badge[{chip}]" for chip in chips)


def chip_row(chips: Sequence[str], *, color: str = "blue") -> None:
    if not chips:
        return
    st.markdown(chip_row_text(chips, color=color))


def learned(text: str) -> None:
    """A bordered "What we learned" callout. Deliberately a plain bordered
    container rather than ``st.info`` -- existing page tests assert ``at.info``'s
    contents for other, unrelated notices, and reusing that widget kind here would
    make those assertions ambiguous about which info box they're reading."""
    with st.container(border=True):
        st.markdown(f":material/school: **What we learned** — {text}")
