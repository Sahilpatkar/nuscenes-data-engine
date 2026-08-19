"""Overlay-renderer tests: pure PIL, pixel-sampled, no Streamlit runtime needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

APP_DEMO = Path(__file__).resolve().parents[1] / "app" / "demo"
sys.path.insert(0, str(APP_DEMO))

from render import (  # noqa: E402
    STYLE_FN,
    STYLE_FP,
    STYLE_GT,
    STYLE_LOW_CONF,
    STYLE_TP,
    BoxStyle,
    _draw_rect,
    draw_overlay,
)


def _gt(matched: bool) -> pd.DataFrame:
    return pd.DataFrame({
        "x_min": [100.0], "y_min": [100.0], "x_max": [400.0], "y_max": [400.0],
        "category_group": ["pedestrian"], "matched": [matched],
    })


def _preds(status: str) -> pd.DataFrame:
    return pd.DataFrame({
        "x_min": [600.0], "y_min": [200.0], "x_max": [900.0], "y_max": [500.0],
        "category_group": ["car"], "conf": [0.87], "status": [status],
    })


def _blank() -> Image.Image:
    return Image.new("RGB", (960, 540), (10, 10, 10))


def test_draw_overlay_scales_and_colors_gt() -> None:
    out = draw_overlay(_blank(), _gt(matched=True), _preds("tp").iloc[0:0], mode="gt", scale=0.6)
    # GT box at 100..400 scaled by 0.6 -> 60..240; sample the top edge midline
    assert out.getpixel((150, 60)) == STYLE_GT.color
    # FN styling absent for a matched box: bottom edge is GT color too
    assert out.getpixel((150, 240)) == STYLE_GT.color


def test_draw_overlay_fn_is_dashed() -> None:
    out = draw_overlay(_blank(), _gt(matched=False), _preds("tp").iloc[0:0], mode="gt", scale=0.6)
    edge = [out.getpixel((x, 60)) for x in range(61, 240)]
    assert STYLE_FN.color in edge                      # dashes present
    assert (10, 10, 10) in edge                        # gaps present (dashed, not solid)


def test_draw_overlay_pred_styles() -> None:
    for status, style in (("tp", STYLE_TP), ("fp", STYLE_FP), ("low_conf", STYLE_LOW_CONF)):
        out = draw_overlay(_blank(), _gt(True).iloc[0:0], _preds(status), mode="pred", scale=0.6)
        edge = [out.getpixel((x, 120)) for x in range(361, 540)]
        assert style.color in edge, status


def test_draw_overlay_mode_selects_layers() -> None:
    gt, preds = _gt(matched=True), _preds("fp")
    gt_only = draw_overlay(_blank(), gt, preds, mode="gt", scale=0.6)
    pred_only = draw_overlay(_blank(), gt, preds, mode="pred", scale=0.6)
    both = draw_overlay(_blank(), gt, preds, mode="overlay", scale=0.6)
    assert gt_only.getpixel((150, 60)) == STYLE_GT.color
    assert gt_only.getpixel((450, 120)) == (10, 10, 10)      # pred not drawn
    assert pred_only.getpixel((150, 60)) == (10, 10, 10)     # gt not drawn
    assert both.getpixel((150, 60)) == STYLE_GT.color
    assert STYLE_FP.color in [both.getpixel((x, 120)) for x in range(361, 540)]


def test_draw_overlay_empty_inputs_returns_unchanged_pixels() -> None:
    blank = _blank()
    out = draw_overlay(blank, _gt(True).iloc[0:0], _preds("tp").iloc[0:0], mode="overlay", scale=0.6)
    assert list(out.getdata()) == list(blank.getdata())


def test_draw_overlay_does_not_mutate_input() -> None:
    blank = _blank()
    before = list(blank.getdata())
    draw_overlay(blank, _gt(True), _preds("tp"), mode="overlay", scale=0.6)
    assert list(blank.getdata()) == before


def test_draw_overlay_gt_matched_na_renders_as_plain_gt() -> None:
    """matched is a nullable boolean upstream (matched_<model>): NA means "not
    evaluated", not "unmatched" -- it must render as plain GT green, never the FN
    style, and must not raise (bool(pd.NA) itself raises TypeError)."""
    gt = pd.DataFrame({
        "x_min": [100.0], "y_min": [100.0], "x_max": [400.0], "y_max": [400.0],
        "category_group": ["pedestrian"], "matched": pd.array([pd.NA], dtype="boolean"),
    })
    out = draw_overlay(_blank(), gt, _preds("tp").iloc[0:0], mode="gt", scale=0.6)
    assert out.getpixel((150, 60)) == STYLE_GT.color
    assert out.getpixel((150, 240)) == STYLE_GT.color


def test_draw_overlay_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode") as exc_info:
        draw_overlay(_blank(), _gt(True), _preds("tp"), mode="bogus", scale=0.6)
    for allowed in ("gt", "pred", "overlay"):
        assert allowed in str(exc_info.value)


def test_draw_overlay_invalid_status_raises() -> None:
    with pytest.raises(ValueError, match="unknown prediction status") as exc_info:
        draw_overlay(_blank(), _gt(True).iloc[0:0], _preds("ignored"), mode="pred", scale=0.6)
    for allowed in ("tp", "fp", "low_conf"):
        assert allowed in str(exc_info.value)


def test_draw_overlay_label_suppressed_under_min_width() -> None:
    # native/scale chosen so the wide box is exactly at scale=1.0 (100px, well over
    # the 24px floor) and the narrow box scales to 18px (under it); category_group
    # "car" is short enough that its glyphs (verified via ImageDraw.textbbox: bbox
    # (0, 4, 14, 10) for "car" drawn at the origin) land inside the sampled region.
    wide = pd.DataFrame({
        "x_min": [0.0], "y_min": [0.0], "x_max": [100.0], "y_max": [50.0],
        "category_group": ["car"], "matched": [True],
    })
    narrow = pd.DataFrame({
        "x_min": [0.0], "y_min": [0.0], "x_max": [30.0], "y_max": [50.0],
        "category_group": ["car"], "matched": [True],
    })
    empty_preds = _preds("tp").iloc[0:0]

    wide_out = draw_overlay(_blank(), wide, empty_preds, mode="gt", scale=1.0)
    narrow_out = draw_overlay(_blank(), narrow, empty_preds, mode="gt", scale=0.6)  # -> 18px wide

    # A region inside both boxes but clear of the (2px-thick) STYLE_GT outline on
    # every side -- any non-background pixel here can only be the label glyph.
    region = [(x, y) for x in range(2, 14) for y in range(4, 10)]
    assert any(wide_out.getpixel(p) != (10, 10, 10) for p in region)      # label drawn
    assert all(narrow_out.getpixel(p) == (10, 10, 10) for p in region)    # suppressed
def test_draw_overlay_dashed_edges_align_with_solid_rectangle() -> None:
    """Regression pin for the stroke-geometry fix: ``draw.line``'s width-thickening
    is direction-dependent for even widths (verified against PIL 12 -- a vertical
    width-2 line covers different columns depending on whether it's walked top-to-
    bottom or bottom-to-top), which would silently misalign two of a dashed box's
    four edges against an equivalent solid rectangle at the same coordinates unless
    each dash segment is drawn with the same inward inset ``draw.rectangle`` uses.
    Checked directly against ``_draw_rect`` for every width 1-5 (a "solid" style vs.
    the same style with one huge dash segment, so the dashed path draws one
    continuous stroke per edge) so it fails loudly regardless of which BoxStyle
    width the app ends up using, not just the two currently defined dashed styles.
    """
    for width in range(1, 6):
        solid = BoxStyle(color=(255, 0, 0), width=width, dash=None)
        dashed = BoxStyle(color=(255, 0, 0), width=width, dash=10_000)

        solid_img = _blank()
        _draw_rect(ImageDraw.Draw(solid_img), (10.0, 10.0, 40.0, 45.0), solid)

        dashed_img = _blank()
        _draw_rect(ImageDraw.Draw(dashed_img), (10.0, 10.0, 40.0, 45.0), dashed)

        assert list(solid_img.getdata()) == list(dashed_img.getdata()), width
