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
    STYLE_VLM,
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


def test_draw_overlay_inverted_box_raises() -> None:
    """x_max < x_min used to behave asymmetrically by style: PIL's draw.rectangle
    raises an unhelpful bare error for a solid style (tp), while a dashed style
    (low_conf) silently vanishes -- _draw_dashed_edge's ``length <= 0`` early
    return draws nothing at all rather than erroring. Both are now a named
    ValueError instead, mirroring build.py's ``_size_bucket`` guard on a
    degenerate GT box area: corrupt data must not half-render."""
    for status in ("tp", "low_conf"):
        preds = pd.DataFrame({
            "x_min": [900.0], "y_min": [200.0], "x_max": [600.0], "y_max": [500.0],
            "category_group": ["car"], "conf": [0.87], "status": [status],
        })
        with pytest.raises(ValueError, match="inverted") as exc_info:
            draw_overlay(_blank(), _gt(True).iloc[0:0], preds, mode="pred", scale=0.6)
        assert "car" in str(exc_info.value)


def test_draw_overlay_label_clamped_at_frame_edge() -> None:
    """A box whose top-left corner is off-canvas (x_min/y_min < 0 -- e.g. an
    annotation only partially inside the frame) must still draw its label ON the
    visible canvas, not clipped away above/left of it (final-review fix)."""
    gt = pd.DataFrame({
        "x_min": [-50.0], "y_min": [-50.0], "x_max": [50.0], "y_max": [0.0],
        "category_group": ["car"], "matched": [True],
    })
    out = draw_overlay(_blank(), gt, _preds("tp").iloc[0:0], mode="gt", scale=1.0)
    # Same "car"-glyph region test_draw_overlay_label_suppressed_under_min_width
    # uses (glyph bbox (0, 4, 14, 10) at the label's drawn origin, verified via
    # ImageDraw.textbbox) -- clear of the GT outline itself. Unclamped, the label
    # would draw at (-50, -50): entirely off-canvas, so this region would stay
    # pure background.
    region = [(x, y) for x in range(2, 14) for y in range(4, 10)]
    assert any(out.getpixel(p) != (10, 10, 10) for p in region)


# --- Phase 7 (Task 4): the shared altair bar chart -------------------------------


def _arms() -> pd.DataFrame:
    return pd.DataFrame({
        "arm": ["baseline", "graph_rate_night", "weak_random", "graph"],
        "delta_night": [0.0, 0.0101, -0.0262, 0.004],
        "family": ["baseline", "graph_rate_night", "weak", "graph"],
        "round_order": [0, 8, 9, 3],
    })


def _colors(chart: object) -> dict[str, str]:
    """{category: color} read out of the compiled spec's own data rows (altair puts
    the frame in a named dataset and points the layer at it by name)."""
    spec = chart.to_dict()
    layer = spec["layer"][0] if "layer" in spec else spec
    data = layer.get("data", spec.get("data", {}))
    rows = data["values"] if "values" in data else spec["datasets"][data["name"]]
    field = layer["encoding"]["color"]["field"]
    x_field = layer["encoding"]["x"]["field"]
    return {row[x_field]: row[field] for row in rows}


def test_bar_chart_colors_highlight_accent_weak_grey_rest_default() -> None:
    """The arm chart's visual claim: the arm being explained is the accent colour,
    the weak-supervision arms are greyed (they are pseudo-label runs, not
    night-targeting arms), everything else is the ordinary bar colour."""
    from render import bar_chart

    chart = bar_chart(
        _arms(), x="arm", y="delta_night", highlight="graph_rate_night",
        sort=["baseline", "graph", "graph_rate_night", "weak_random"],
    )
    colors = _colors(chart)

    assert colors["graph_rate_night"] == "#FF851B"
    assert colors["weak_random"] == "#BBBBBB"
    assert colors["baseline"] == colors["graph"] == "#4A90D9"


def test_bar_chart_sort_order_and_zero_line() -> None:
    """The bars follow the explicit ``sort`` order (round_order, not alphabetical),
    and a zero rule is layered under a chart with negative values so a regression
    reads as one."""
    from render import bar_chart

    ordered = ["baseline", "graph", "graph_rate_night", "weak_random"]
    chart = bar_chart(_arms(), x="arm", y="delta_night", sort=ordered, title="Night")
    spec = chart.to_dict()

    assert "layer" in spec                                   # bars + zero rule
    assert spec["layer"][0]["encoding"]["x"]["sort"] == ordered
    assert spec["title"] == "Night"

    plain = bar_chart(_arms(), x="arm", y="delta_night", zero_line=False)
    assert "layer" not in plain.to_dict()


def test_bar_chart_color_field_keeps_its_own_legend() -> None:
    """With ``color_field`` (Task 5's stacked loss decomposition) the colour encodes
    that field as a normal categorical scale instead of the highlight logic."""
    from render import bar_chart

    stacked = pd.DataFrame({
        "base_arm": ["random", "random", "graph_rate_night", "graph_rate_night"],
        "value": [0.0062, 0.0169, 0.0100, 0.0107],
        "component": ["retained", "dropped-frame cost"] * 2,
    })

    chart = bar_chart(stacked, x="base_arm", y="value", color_field="component")
    encoding = chart.to_dict()["layer"][0]["encoding"]

    assert encoding["color"]["field"] == "component"
    assert encoding["color"]["type"] == "nominal"
    # altair's own categorical scale + legend, not the highlight path's literal
    # pass-through colours (which set scale/legend to None explicitly).
    assert encoding["color"].get("scale", "default") != None    # noqa: E711
    assert encoding["color"].get("legend", "default") != None   # noqa: E711


# --- Phase 7 (Task 5): the VLM pseudo-box layer ----------------------------------


def _vlm() -> pd.DataFrame:
    """One verified VLM pseudo box, in weak_labels.parquet's own schema: no
    ``matched`` and no ``status`` (those are GT/prediction concepts) -- a
    verifier ``score`` instead of a detector confidence."""
    return pd.DataFrame({
        "x_min": [600.0], "y_min": [600.0], "x_max": [900.0], "y_max": [800.0],
        "category_group": ["car"], "score": [0.73],
    })


def test_draw_overlay_vlm_boxes_drawn_in_vlm_style_and_signature_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pseudo boxes are a THIRD, independent layer: they render in every mode
    (a weak-supervision frame is a train-pool frame -- it has GT but no
    predictions, so "gt" is the mode the page actually uses), and the existing
    two-layer signature is untouched.
    """
    gt, preds, vlm = _gt(matched=True), _preds("tp"), _vlm()
    modes = ("gt", "pred", "overlay")
    before = {
        mode: draw_overlay(_blank(), gt, preds, mode=mode, scale=0.6) for mode in modes
    }

    # Backward compatible: omitted / None / empty are all byte-identical to the
    # pre-change two-layer render of the same inputs.
    for mode, baseline in before.items():
        assert (
            draw_overlay(_blank(), gt, preds, mode=mode, scale=0.6, vlm_boxes=None).tobytes()
            == baseline.tobytes()
        )
        assert (
            draw_overlay(
                _blank(), gt, preds, mode=mode, scale=0.6, vlm_boxes=vlm.iloc[0:0]
            ).tobytes()
            == baseline.tobytes()
        )

    # 600..900 x 600..800 scaled by 0.6 -> 360..540 x 360..480; sample the top
    # edge clear of both corners. Solid (no gaps), blue, in every mode.
    for mode, baseline in before.items():
        out = draw_overlay(_blank(), gt, preds, mode=mode, scale=0.6, vlm_boxes=vlm)
        edge = {out.getpixel((x, 360)) for x in range(362, 538)}
        assert edge == {STYLE_VLM.color}, mode
        assert out.tobytes() != baseline.tobytes(), mode

    # Corrupt coordinates raise the same named ValueError the GT/prediction
    # layers do, rather than half-rendering.
    with pytest.raises(ValueError, match="inverted VLM box") as exc_info:
        draw_overlay(
            _blank(), gt, preds, mode="gt", scale=0.6, vlm_boxes=vlm.assign(x_max=[10.0])
        )
    assert "car" in str(exc_info.value)

    # The label is the verifier score, not a category: "VLM 0.73".
    #
    # Patched through ``draw_overlay.__globals__`` rather than
    # ``monkeypatch.setattr(render, ...)``: tests/test_demo_app.py's
    # ``_reset_demo_app_modules`` deletes app/demo's modules from sys.modules
    # between AppTest runs, so a later ``import render`` in THIS process yields a
    # DIFFERENT module object from the one this file's own top-level import bound
    # -- patching that one leaves the ``draw_overlay`` under test calling the
    # original ``_draw_label`` (a real cross-file failure, caught by the full-suite
    # run). A function's ``__globals__`` is always the module namespace it actually
    # resolves names in.
    calls: list[tuple[str, object]] = []
    monkeypatch.setitem(
        draw_overlay.__globals__,
        "_draw_label",
        lambda draw, xyxy, text, style: calls.append((text, style)),
    )
    draw_overlay(_blank(), gt, preds, mode="gt", scale=0.6, vlm_boxes=vlm)
    assert ("VLM 0.73", STYLE_VLM) in calls
