"""Shared rendering primitives for the demo pages."""

from __future__ import annotations

from dataclasses import dataclass

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

    return out
