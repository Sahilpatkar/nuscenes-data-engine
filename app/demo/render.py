"""Shared rendering primitives for the demo pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


def metric_cards(items: list[tuple[str, str]], per_row: int = 4) -> None:
    """A row of st.metric cards: [(label, value), ...]."""
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        for col, (label, value) in zip(st.columns(len(chunk)), chunk, strict=True):
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
# Pure PIL: no Streamlit calls in this section, so it is testable without a runtime.


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
STYLE_VLM = BoxStyle(color=(0, 116, 217), width=2, dash=None)       # blue solid

_PRED_STYLES: dict[str, BoxStyle] = {"tp": STYLE_TP, "fp": STYLE_FP, "low_conf": STYLE_LOW_CONF}
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
    vlm_boxes: pd.DataFrame | None = None,
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

    ``vlm_boxes`` (Phase 7) is a THIRD, independent layer: the weak-supervision
    pseudo boxes for the frame (x_min..y_max, category_group, score --
    weak_labels.parquet's own schema), drawn last, in every mode. It is
    mode-independent on purpose: the frames that carry pseudo boxes are train-pool
    frames, which have GT but never predictions, so the page draws them in "gt"
    mode and still needs the blue layer. Left as ``None`` (the default) the render
    is byte-identical to the two-layer one this signature had before.
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

    if vlm_boxes is not None:
        for row in vlm_boxes.itertuples(index=False):
            xyxy = (row.x_min * scale, row.y_min * scale, row.x_max * scale, row.y_max * scale)
            _validate_xyxy(xyxy, kind="VLM", category=str(row.category_group))
            _draw_rect(draw, xyxy, STYLE_VLM)
            # The score, not the category: the category is already on the GT box
            # underneath, and what a viewer needs to judge a pseudo box is how
            # confident the proposal was. "VLM" names the LAYER -- the VLM-verified
            # weak-supervision pipeline (spec §4d pins this label) -- while the
            # number is the baseline detector's own confidence, which the page's
            # legend caption spells out in words.
            _draw_label(draw, xyxy, f"VLM {row.score:.2f}", STYLE_VLM)

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

    ``zero_line`` layers a rule at y = 0 -- delta charts carry negative values, and
    without the rule a regression reads as just a shorter bar. That layering is why
    the return type is a union: an ``alt.LayerChart`` is not an ``alt.Chart``, and
    ``st.altair_chart`` takes either.
    """
    data = frame.copy()
    encode: dict[str, Any] = {
        "x": alt.X(f"{x}:N", sort=sort, title=None),
        "y": alt.Y(f"{y}:Q", title=y.replace("_", " ")),
        "tooltip": [c for c in (x, y, color_field, "family") if c and c in data.columns],
    }
    if color_field is not None:
        encode["color"] = alt.Color(f"{color_field}:N", title=color_field.replace("_", " "))
    else:
        data[_COLOR_COLUMN] = _bar_colors(data, x=x, highlight=highlight)
        # scale=None: the column already holds literal colours, so altair must pass
        # them through instead of building a categorical scale over them.
        encode["color"] = alt.Color(f"{_COLOR_COLUMN}:N", scale=None, legend=None)

    bars = alt.Chart(data).mark_bar().encode(**encode)
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
