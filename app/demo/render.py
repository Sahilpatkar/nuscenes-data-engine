"""Shared rendering primitives for the demo pages."""

from __future__ import annotations

import math
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


def _draw_dashed_segment(
    draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], style: BoxStyle
) -> None:
    """Walk one straight edge in ``style.dash``-length segments with equal gaps."""
    (x0, y0), (x1, y1) = start, end
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0 or style.dash is None:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos, on = 0.0, True
    while pos < length:
        seg_end = min(pos + style.dash, length)
        if on:
            start_px = (x0 + ux * pos, y0 + uy * pos)
            end_px = (x0 + ux * seg_end, y0 + uy * seg_end)
            draw.line([start_px, end_px], fill=style.color, width=style.width)
        pos = seg_end
        on = not on


def _draw_rect(draw: ImageDraw.ImageDraw, xyxy: _XYXY, style: BoxStyle) -> None:
    """Draw one box outline: a solid rectangle, or (when ``style.dash`` is set) each
    edge walked as dashed/dotted segments."""
    x_min, y_min, x_max, y_max = xyxy
    if style.dash is None:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=style.color, width=style.width)
        return
    edges = (
        ((x_min, y_min), (x_max, y_min)),  # top
        ((x_max, y_min), (x_max, y_max)),  # right
        ((x_max, y_max), (x_min, y_max)),  # bottom
        ((x_min, y_max), (x_min, y_min)),  # left
    )
    for edge_start, edge_end in edges:
        _draw_dashed_segment(draw, edge_start, edge_end, style)


def _draw_label(draw: ImageDraw.ImageDraw, xyxy: _XYXY, text: str, style: BoxStyle) -> None:
    x_min, y_min, x_max, _y_max = xyxy
    if (x_max - x_min) < _MIN_LABEL_WIDTH_PX:
        return
    draw.text((x_min, y_min), text, fill=style.color)


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
    category_group, matched (bool: unmatched renders as the FN style);
    ``predictions`` needs x_min..y_max, category_group, conf, status.
    ``mode``: "gt" | "pred" | "overlay". The input image is never mutated.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"draw_overlay: unknown mode {mode!r} — expected one of {_VALID_MODES}")

    out = image.copy()
    draw = ImageDraw.Draw(out)

    if mode in ("gt", "overlay"):
        for row in gt_boxes.itertuples(index=False):
            style = STYLE_GT if bool(row.matched) else STYLE_FN
            xyxy = (row.x_min * scale, row.y_min * scale, row.x_max * scale, row.y_max * scale)
            _draw_rect(draw, xyxy, style)
            _draw_label(draw, xyxy, str(row.category_group), style)

    if mode in ("pred", "overlay"):
        for row in predictions.itertuples(index=False):
            style = _PRED_STYLES[row.status]
            xyxy = (row.x_min * scale, row.y_min * scale, row.x_max * scale, row.y_max * scale)
            _draw_rect(draw, xyxy, style)
            label = f"{row.category_group} {row.conf:.2f}"
            _draw_label(draw, xyxy, label, style)

    return out
