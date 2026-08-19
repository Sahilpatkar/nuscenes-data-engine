"""Overlay-renderer tests: pure PIL, pixel-sampled, no Streamlit runtime needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PIL")
from PIL import Image

APP_DEMO = Path(__file__).resolve().parents[1] / "app" / "demo"
sys.path.insert(0, str(APP_DEMO))

from render import (  # noqa: E402
    STYLE_FN,
    STYLE_FP,
    STYLE_GT,
    STYLE_LOW_CONF,
    STYLE_TP,
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
