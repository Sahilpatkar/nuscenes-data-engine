"""Derive the story site's data bundle from the committed ``demo_data/`` package.

    uv run python web/build_data.py [--out web]

The React site (``web/src``) contains no numeric result literals: every figure it
renders is written here, from the same package tables and the same
``app/demo/filters.py`` helpers the Streamlit app derives its own numbers with, so
the two front-ends cannot drift apart on a number. Eight JSON files land in
``<out>/src/data/`` (one per story section) and the ~11 pre-rendered overlays and
thumbs in ``<out>/public/story/``; ``tests/test_web_export.py`` re-runs this module
and asserts the committed bundle comes back byte-identical.

Three rules make that guard possible and the bundle honest:

* **Derived, never asserted.** Superlatives and headline claims (the best-night
  arm, the closed-loop headline, the fairness budget clause, the similarity
  caption, the hero-honesty line) are re-derived here with the SAME rules
  ``app/demo/views/tour.py`` applies -- the private helpers over there are the copy
  source of truth, and this module mirrors their logic rather than restating their
  output. Where the tour writes nothing, this writes ``None``.
* **Byte-stable.** No wall-clock timestamp is emitted (the package's own
  ``built_at``/``git_sha`` are the provenance), floats are rounded at source, and
  the JSON writer is canonical (``sort_keys``, 2-space indent, trailing newline).
* **Streamlit markdown is not copy.** The tour's sentences carry ``**bold**``
  markers for Streamlit's renderer; those are stripped here (the site styles its
  own emphasis). Backticked raw arm ids survive verbatim -- the spec's honesty rule
  is that a story label always keeps the identifier it stands for beside it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP_DEMO = ROOT / "app" / "demo"
if str(APP_DEMO) not in sys.path:
    sys.path.insert(0, str(APP_DEMO))

from filters import (  # noqa: E402
    arm_story_label,
    failure_flags,
    filmstrip_steps,
    fixed_boxes,
    frame_quota_before,
    gain_text,
    gt_for_render,
    parity_short,
    rank_events,
    reason_chips,
    relative_gain,
    selection_factors,
    severity_caption,
    strategy_coverage,
    tour_frame_candidates,
    tour_strategies,
    upgrade_callout,
    visible_gt,
    visible_gt_boxes,
)
from render import (  # noqa: E402
    CAN_ACCEL_TITLE,
    CAN_SPEED_TITLE,
    EGO_SPEED_TITLE,
    LEGEND_ITEMS,
    PROVENANCE,
    STYLE_FN,
    STYLE_FP,
    STYLE_GT,
    STYLE_LOW_CONF,
    STYLE_TP,
    curve_caption,
    draw_overlay,
)

# --- constants mirrored from the app ----------------------------------------------
#
# Each of these is a module-private constant over in views/tour.py (or, for the
# conf floor, restated there from configs/active_learning.yaml). They are copied
# rather than imported for the same reason the tour copies them from the deep
# pages: importing views/tour.py would pull in Streamlit page state, and a tour
# screen and this exporter must say exactly the same thing. The copy-contract is
# enforced by tests/test_web_export.py's pinned sentences.

_TOUR_PRESET = "hard_braking_near_pedestrians"
_CONF_HIT_FLOOR = 0.40
_NIGHT_ABSENT_SHARE = 0.005
_FAIRNESS_TAIL = "same training configuration · scored on the same held-out split"

_HELD_OUT_CAPTION = "Held-out validation frame — never in any training set"
_BRIDGE_TO_MINING = (
    "Finding one failure is easy — the hard part is finding the rest of the dataset "
    "where the same thing happens."
)
_MINING_MECHANISM = (
    "The system searches driving context — night, braking, pedestrians near the ego — "
    "not just similar-looking images."
)
_SELECTION_PATH = (
    "failed val frame → similarity community → representative train-pool frames → "
    "selected for retraining"
)
_ARCHITECTURE_STRIP = (
    "System path: nuScenes → validated Parquet → SQL / Neo4j / CAN → failure analysis "
    "→ scenario search → active learning → YOLO retraining → evaluation"
)
_TRAIN_POOL_NOTE = "train-pool frame — no predictions (models never saw it as a test image)"
_RETRAIN_HEADLINE = "We did not just add data — we changed what the model trains on."
_STRATEGY_CHART_TITLE = "Night mAP50-95 vs baseline, by acquisition strategy"
_STRATEGY_CHART_Y_TITLE = "Δ night mAP50-95"
_AFTER_HEADLINE = "Did the targeted retraining fix the kind of failure we started with?"
_PER_BOX_FOLD = "Technical details — per-box claims"
_HERO_HONESTY_LINE = (
    "The hero frame's pedestrian recovery (step 2) is a low-confidence claim and does "
    "not pass this table's confident-detection rule; the hand-approved exemplars do."
)
_CLOSED_LOOP_IMPROVED = "Closed the loop: weakness → targeted data → measurable improvement"
_CLOSED_LOOP_MEASURED = "Closed the loop: weakness → targeted data → measured result"
_HERO_CAPTION = (
    "baseline misses a shadowed car and a pedestrian; the night-targeted "
    "retrain recovers the pedestrian (a low-confidence hit) — the far car "
    "defeats all five models. Explore more in the Failure Explorer."
)

# --- the five story contracts (spec §1) -------------------------------------------
#
# Fixed sentences, shipped as bundle fields rather than hard-coded in a React
# component: the site and the tour must be able to say the same thing, and a
# sentence that lives in the bundle is one a test can pin.

_PROBLEM_SENTENCE = (
    "The detector looked reasonable overall — but performance dropped sharply at "
    "night, especially for pedestrians."
)
_PURPOSE_SENTENCE = (
    "The goal of the system: automatically find failures like this and turn them "
    "into better training data."
)
_LEDE_SENTENCE = (
    "Instead of randomly adding more images, the system searches the training pool "
    "for examples related to the diagnosed failure."
)
_SELECTION_CHAIN = (
    "Failed validation frame",
    "relevant driving scenario",
    "candidate training frames",
    "targeted retraining set",
)
_EXAMPLE_LABEL = "One example"
_EXAMPLE_CAVEAT = "one hand-approved frame — illustrative, not the metric"
_AGGREGATE_LABEL = "The aggregate result"
_CLOSING_THESIS = (
    "Instead of blindly retraining the model, the system diagnoses where it fails, "
    "finds the data that can address the weakness, and measures whether the "
    "intervention actually works."
)

# --- site constants ----------------------------------------------------------------

STREAMLIT_BASE = "https://nuscenes-data-engine-sahil.streamlit.app"
EXPORTER_NAME = "web/build_data.py"

# The pinned `url_path`s from app/demo/main.py, with the tour's own "PageName —
# clause" label shape. No digits in a label: a static string is the one thing on
# the page nothing recomputes, so a count in one would be the only unchecked figure.
_DEEP_LINKS: tuple[tuple[str, str], ...] = (
    ("tour", "Guided tour — the same story, one screen at a time in the live app"),
    ("failures", "Failure Explorer — every miss, filterable by condition"),
    ("scenarios", "Scenario Search — every preset, every matching event"),
    ("active_learning", "Active Learning — the full experiment, all arms"),
    ("weak_supervision", "Weak Supervision — where the rest of the gain went"),
    ("chat_replay", "Ask the Dataset — recorded chat replays"),
)

_ATTRIBUTION = {
    "dataset": "nuScenes",
    "license": "CC BY-NC-SA 4.0",
    "citation": (
        "Caesar et al., nuScenes: A Multimodal Dataset for Autonomous Driving, CVPR 2020"
    ),
    "url": "https://www.nuscenes.org/nuscenes",
}

# The overlay colours are burned into the JPEGs, so the site's legend swatches have
# to be these exact RGBs rather than theme tokens. Keyed off render.LEGEND_ITEMS so
# the wording stays the app's.
_LEGEND_RGB = {
    "green — ground truth": STYLE_GT.color,
    "orange dashed — GT box the model missed": STYLE_FN.color,
    "white — true positive": STYLE_TP.color,
    "yellow dotted — low-confidence claim (below the hit floor)": STYLE_LOW_CONF.color,
    "red — false positive": STYLE_FP.color,
}

_CROP_SCALE = 0.6
_JPEG_QUALITY = 85
_STORY_BYTE_BUDGET = 1_200_000


# --- package loading ---------------------------------------------------------------
#
# The app's own loaders are @st.cache_data-wrapped, which needs a Streamlit script
# run; the file names below are data.py's, read directly.


class Package:
    """Everything the eight sections read, loaded once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.overview: dict[str, Any] = json.loads((root / "overview_metrics.json").read_text())
        self.manifest_meta: dict[str, Any] = json.loads((root / "manifest.json").read_text())
        self.exemplars: dict[str, Any] = json.loads((root / "al_exemplars.json").read_text())
        self.validation: dict[str, Any] = json.loads(
            (root / "al_explain_validation.json").read_text()
        )
        self.subgraph: dict[str, Any] = json.loads(
            (root / "graph_subgraphs" / f"{_TOUR_PRESET}.json").read_text()
        )
        self.arms = pd.read_parquet(root / "active_learning_results.parquet")
        self.manifest = pd.read_parquet(root / "frame_manifest.parquet")
        self.gt = pd.read_parquet(root / "gt_boxes.parquet")
        self.preds = pd.read_parquet(root / "predictions.parquet")
        self.events = pd.read_parquet(root / "scenario_events.parquet")
        self.explain = pd.read_parquet(root / "al_selection_explain.parquet")
        self.communities = pd.read_parquet(root / "al_communities.parquet")
        self.loss = pd.read_parquet(root / "weak_loss_decomposition.parquet")
        self.weaksup = pd.read_parquet(root / "weak_supervision_results.parquet")
        self.baseline = str(self.exemplars["baseline"])
        self.arm = str(self.exemplars["arm"])
        self.hero = str(self.overview["hero_token"])

    def arm_row(self, name: str) -> pd.Series:
        rows = self.arms.loc[self.arms["arm"] == name]
        if rows.empty:
            raise SystemExit(f"build_data: no `{name}` row in active_learning_results.parquet")
        row: pd.Series = rows.iloc[0]
        return row

    def crop(self, token: str) -> Path:
        return self.root / "sample_frames" / "crops" / f"{token}.jpg"

    def thumb(self, token: str) -> Path:
        return self.root / "sample_frames" / "thumbs" / f"{token}.jpg"


# --- formatting (the tour's own format strings) -------------------------------------


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _share(value: float) -> str:
    return f"{value:.0%}"


def _count(value: int) -> str:
    return f"{value:,}"


def _round(value: Any) -> float:
    """Metrics and chart values, rounded at source so the JSON is byte-stable."""
    return round(float(value), 4)


def _optional_round(value: Any) -> float | None:
    return None if not bool(pd.notna(value)) else _round(value)


def _required(row: pd.Series, key: str) -> Any:
    """One arm-table figure the story cannot be told without.

    The tour degrades on an NA here (it draws its stale-package note instead of the
    screen); an exporter has no screen to degrade, so it fails the build rather than
    shipping a bundle with a hole where a number belongs.
    """
    value = row.get(key)
    if not bool(pd.notna(value)):
        raise SystemExit(f"build_data: arm `{row['arm']}` has no {key} — cannot export")
    return value


def _label(name: str) -> dict[str, str]:
    """A story label with the raw id it stands for (the spec's honesty rule)."""
    return {"id": name, "label": arm_story_label(name)}


def _provenance(kind: str, detail: str = "") -> dict[str, str]:
    """One provenance record, in ``render.PROVENANCE``'s own wording (the Streamlit
    material icon is dropped -- it is a widget detail, not copy)."""
    return {"kind": kind, "sentence": PROVENANCE[kind][1], "detail": detail}


def _legend() -> list[dict[str, Any]]:
    return [
        {"wording": wording, "rgb": list(_LEGEND_RGB[wording])} for _color, wording in LEGEND_ITEMS
    ]


# --- output --------------------------------------------------------------------------


class Writer:
    """The canonical JSON writer and the story-image sink."""

    def __init__(self, out: Path) -> None:
        self.json_dir = out / "src" / "data"
        self.story_dir = out / "public" / "story"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.story_dir.mkdir(parents=True, exist_ok=True)

    def section(self, name: str, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        (self.json_dir / f"{name}.json").write_text(text, encoding="utf-8")

    def image(self, image: Image.Image, *, name: str, alt: str) -> dict[str, Any]:
        image.save(self.story_dir / f"{name}.jpg", format="JPEG", quality=_JPEG_QUALITY)
        return {
            "src": f"story/{name}.jpg",
            "width": image.width,
            "height": image.height,
            "alt": alt,
        }

    def copied(self, source: Path, *, name: str, alt: str) -> dict[str, Any]:
        """A package thumb, copied byte-for-byte (re-encoding it would only lose
        quality -- the thumbs are already the size the filmstrip shows)."""
        shutil.copyfile(source, self.story_dir / f"{name}.jpg")
        with Image.open(source) as image:
            width, height = image.size
        return {"src": f"story/{name}.jpg", "width": width, "height": height, "alt": alt}

    def story_bytes(self) -> list[tuple[str, int]]:
        return sorted(
            (path.name, path.stat().st_size) for path in self.story_dir.glob("*.jpg")
        )


def _overlay(
    package: Package, *, token: str, model: str | None, mode: str, gt_rows: pd.DataFrame
) -> Image.Image:
    """One crop with its boxes drawn, at the app's own crop scale."""
    predictions = (
        package.preds.loc[
            (package.preds["sample_data_token"] == token) & (package.preds["model"] == model)
        ]
        if mode == "overlay" and model is not None
        else pd.DataFrame()
    )
    with Image.open(package.crop(token)) as image:
        return draw_overlay(
            image,
            gt_for_render(gt_rows, model),
            predictions,
            mode=mode,
            scale=_CROP_SCALE,
        )


# --- meta ----------------------------------------------------------------------------


def build_meta(package: Package) -> dict[str, Any]:
    return {
        "attribution": _ATTRIBUTION,
        "deep_links": [
            {"url_path": url_path, "url": f"{STREAMLIT_BASE}/{url_path}", "label": label}
            for url_path, label in _DEEP_LINKS
        ],
        "exporter": EXPORTER_NAME,
        "package": {
            "built_at": str(package.manifest_meta["built_at"]),
            "git_sha": str(package.manifest_meta["git_sha"]),
            "version": str(package.manifest_meta["package_version"]),
        },
        "streamlit_base": STREAMLIT_BASE,
    }


# --- 1. the blind spot (contract §1.1) ------------------------------------------------


def _night_miss_counts(package: Package) -> tuple[int, int, int]:
    """``(night val frames, of those with >= 1 baseline miss, val frames)`` --
    ``views/tour.py::_night_miss_counts`` over the same val slice."""
    flags = failure_flags(
        package.manifest,
        visible_gt_boxes(package.gt),
        package.preds,
        model=package.baseline,
    )
    is_night = flags["is_night"].fillna(False).astype(bool)
    return int(is_night.sum()), int((is_night & flags["has_fn"]).sum()), len(flags)


def build_blindspot(package: Package) -> dict[str, Any]:
    row = package.arm_row(package.baseline)
    overall, night = float(row["overall_map5095"]), float(row["night_map5095"])
    n_night, n_night_fn, n_val = _night_miss_counts(package)
    # night_ped_map5095 is the night PEDESTRIAN class -- the mechanism the night arms
    # move, and absent (or NA) on a package whose eval never wrote per-class night
    # metrics. Guarded exactly as `views/tour.py::_render_weakness` guards it: the
    # card AND the sentence's clause are dropped together, because an absent card is
    # better than a card reading "nan" and a sentence must not out-claim its cards.
    night_ped = row.get("night_ped_map5095")
    has_night_ped = night_ped is not None and bool(pd.notna(night_ped))
    cards = [
        {"label": "Overall mAP50-95", "value": _metric(overall)},
        {"label": "Night mAP50-95", "value": _metric(night)},
    ]
    night_ped_clause = ""
    if has_night_ped:
        cards.append({"label": "Night pedestrian mAP50-95", "value": _metric(float(night_ped))})
        night_ped_clause = (
            f" — and {_metric(float(night_ped))} on night pedestrians, the case that matters most"
        )
    return {
        "baseline": _label(package.baseline),
        "cards": cards,
        "derived_sentence": (
            f"`{package.baseline}` scores {_metric(overall)} mAP50-95 overall but "
            f"{_metric(night)} at night{night_ped_clause}."
        ),
        # Site-only editorial, deliberately with no tour twin to pin it to: the tour's
        # step 1 leads with `_PROBLEM_SENTENCE` as its own heading, while a scrolling
        # page wants a section heading that names the finding. The claim stays true and
        # is evidenced by the very cards below it (overall 0.2477 vs night 0.1667), so
        # tests/test_web_export.py asserts it is non-empty rather than tour-bound.
        "headline": "Aggregate accuracy was hiding a night-driving blind spot.",
        "miss_sentence": (
            f"{n_night_fn} of {n_night} night validation frames carry at least one "
            "baseline miss."
        ),
        "problem_sentence": _PROBLEM_SENTENCE,
        "provenance": [
            _provenance("recorded", "mAP from active_learning_results.parquet"),
            _provenance(
                "recomputed",
                f"frame counts from frame_manifest/gt_boxes/predictions ({n_val} curated "
                "val frames)",
            ),
        ],
        "purpose_sentence": _PURPOSE_SENTENCE,
        "takeaway": "Aggregate metrics hide important failure slices.",
    }


# --- 2. the hero frame ------------------------------------------------------------------


def _claim(
    package: Package, *, token: str, model: str, annotation: str
) -> tuple[float, str] | None:
    """``views/tour.py::_claim`` -- the highest-confidence claim ``model`` filed
    against one GT box, or ``None`` when it never claimed it."""
    rows = package.preds.loc[
        (package.preds["sample_data_token"] == token)
        & (package.preds["model"] == model)
        & (package.preds["matched_annotation_token"] == annotation)
    ]
    if rows.empty:
        return None
    row = rows.sort_values("conf", ascending=False).iloc[0]
    return float(row["conf"]), str(row["status"])


def _claim_text(package: Package, *, token: str, model: str, annotation: str) -> str:
    """``views/tour.py::_claim_text``, verbatim."""
    claim = _claim(package, token=token, model=model, annotation=annotation)
    if claim is None:
        return "no claim"
    conf, status = claim
    if status == "low_conf":
        return (
            f"{conf:.3f} (low-confidence — below the {_CONF_HIT_FLOOR:.2f} hit floor; "
            "the matching rule still counts it as a hit)"
        )
    if status == "tp":
        return f"{conf:.3f} (confident detection)"
    return f"{conf:.3f} ({status.replace('_', ' ')})"


def _pedestrian_facts(
    package: Package, *, token: str, gt_rows: pd.DataFrame, models: tuple[str, ...]
) -> list[dict[str, Any]]:
    """``views/tour.py::_pedestrian_facts``, as structured parts: the tour renders
    "**Pedestrian at 11.1 m** — <claims>" as one markdown line, and the site styles
    the subject and the per-model claims separately. Every string below is the
    tour's own."""
    facts: list[dict[str, Any]] = []
    peds = gt_rows.loc[gt_rows["category_group"] == "pedestrian"]
    for row in peds.itertuples(index=False):
        distance = getattr(row, "distance_to_ego_m", None)
        where = ""
        if distance is not None and bool(pd.notna(distance)):
            where = f" at {float(distance):.1f} m"
        facts.append({
            "subject": f"Pedestrian{where}",
            "claims": [
                {
                    **_label(model),
                    "claim": _claim_text(
                        package, token=token, model=model, annotation=str(row.annotation_token)
                    ),
                }
                for model in models
            ],
        })
    return facts


def _pedestrian_miss_to_hit_frames(package: Package) -> set[str]:
    """``views/tour.py::_pedestrian_miss_to_hit_frames``."""
    baseline_column, arm_column = f"matched_{package.baseline}", f"matched_{package.arm}"
    val = package.manifest.loc[package.manifest["split"] == "val"]
    night = set(
        val.loc[val["is_night"].fillna(False).astype(bool), "sample_data_token"].astype(str)
    )
    boxes = visible_gt_boxes(package.gt)
    upgraded = boxes.loc[
        (boxes["category_group"] == "pedestrian")
        & boxes["sample_data_token"].astype(str).isin(night)
        & boxes[baseline_column].eq(False).fillna(False)
        & boxes[arm_column].eq(True).fillna(False)
    ]
    return {str(token) for token in upgraded["sample_data_token"]}


def _miss_to_hit_sentence(package: Package) -> str:
    """``views/tour.py::_miss_to_hit_sentence`` -- counted, never assumed."""
    frames = _pedestrian_miss_to_hit_frames(package)
    if len(frames) == 1 and package.hero in frames:
        return (
            "Across the whole curated val split this is the only night frame where "
            f"`{package.arm}` turns a `{package.baseline}` pedestrian miss into a hit."
        )
    return (
        f"{len(frames)} night val frames carry a pedestrian that `{package.baseline}` "
        f"misses and `{package.arm}` claims."
    )


def _held_out_caption(package: Package, token: str) -> str | None:
    """``views/tour.py::_held_out_caption`` -- written only where the manifest says
    the frame really was held out."""
    rows = package.manifest.loc[package.manifest["sample_data_token"].astype(str) == token]
    if rows.empty or str(rows.iloc[0].get("split")) != "val":
        return None
    return _HELD_OUT_CAPTION


def build_hero_frame(package: Package, writer: Writer) -> dict[str, Any]:
    token = package.hero
    models = (package.baseline, package.arm)
    gt_rows = visible_gt(package.gt, token)
    images = {
        key: writer.image(
            _overlay(package, token=token, model=model, mode="overlay", gt_rows=gt_rows),
            name=f"hero-{key}",
            alt=(
                f"Night driving frame with ground-truth and {arm_story_label(model)} "
                "detection boxes drawn on it."
            ),
        )
        for key, model in (("baseline", package.baseline), ("arm", package.arm))
    }
    return {
        "bridge_sentence": _BRIDGE_TO_MINING,
        "caption": _HERO_CAPTION,
        "facts": _pedestrian_facts(package, token=token, gt_rows=gt_rows, models=models),
        "headline": "Here the baseline misses a pedestrian at night.",
        "held_out_caption": _held_out_caption(package, token),
        "images": images,
        "legend": _legend(),
        "models": {"arm": _label(package.arm), "baseline": _label(package.baseline)},
        "provenance": [
            _provenance("recomputed", "boxes, claims and counts from gt_boxes/predictions")
        ],
        "rarity_sentence": _miss_to_hit_sentence(package),
        "token": token,
    }


# --- 3. the scenario --------------------------------------------------------------------


def build_scenario(package: Package, writer: Writer) -> dict[str, Any]:
    ranked = rank_events(package.events, _TOUR_PRESET)
    row = ranked.iloc[0]
    token = str(row["sample_data_token"])
    scene = str(row["scene_name"])

    n_peds = int(row["n_peds_within_10m"]) if bool(pd.notna(row["n_peds_within_10m"])) else 0
    nearest = row["min_dist_pedestrian_m"]
    facts = f"within 10 m: {n_peds} pedestrian{'' if n_peds == 1 else 's'}"
    if bool(pd.notna(nearest)):
        facts += f" · nearest at {float(nearest):.1f} m"

    curve = filmstrip_steps(row)
    steps = [
        {
            "accel_mps2": _optional_round(step.accel_mps2),
            "can_speed_kmh": _optional_round(step.can_speed_kmh),
            "image": writer.copied(
                package.thumb(step.token),
                name=f"filmstrip-{index}",
                alt=f"Filmstrip thumbnail {step.label} of the {scene} braking event.",
            ),
            "is_current": step.is_current,
            "label": step.label,
            "token": step.token,
        }
        for index, step in enumerate(curve.steps)
    ]
    n_night = int(ranked["is_night"].fillna(False).astype(bool).sum())
    return {
        "event": {
            "chips": [
                "night" if bool(row["is_night"]) else "day",
                *(["rain"] if bool(row["is_rain"]) else []),
            ],
            "facts": facts,
            "scene_name": scene,
            "severity_caption": severity_caption(_TOUR_PRESET, row.to_dict()),
            "token": token,
        },
        "filmstrip": {
            "accel_title": CAN_ACCEL_TITLE,
            "caption": curve_caption(curve),
            "speed_is_can": curve.speed_is_can,
            "speed_title": CAN_SPEED_TITLE if curve.speed_is_can else EGO_SPEED_TITLE,
            "steps": steps,
        },
        "headline": "Is this one bad photo, or a recurring driving scenario?",
        "image": writer.image(
            _overlay(
                package,
                token=token,
                model=None,
                mode="gt",
                gt_rows=visible_gt(package.gt, token),
            ),
            name="scenario-event",
            alt=(
                f"The {scene} hard-braking event with its ground-truth boxes drawn on it."
            ),
        ),
        "mechanism_sentence": _MINING_MECHANISM,
        "night_sentence": f"{n_night} of {len(ranked)} matching events are at night.",
        "parity_caption": parity_short(
            int(package.subgraph["sql_count"]),
            package.subgraph["cypher_count"],
            package.subgraph["parity"],
        ),
        "preset": _TOUR_PRESET,
        "provenance": [
            _provenance(
                "recorded", "event counts computed at build against SQL and the Neo4j graph"
            ),
            _provenance("recomputed", "overlay from gt_boxes.parquet"),
        ],
        "takeaway": "Perception failures must be analyzed in driving context.",
    }


# --- 4. why this frame (contract §1.2) -----------------------------------------------


def _tour_frame(package: Package) -> str:
    """``views/tour.py::_tour_frame`` -- the first ranked candidate with a crop.

    Only the crop branch: a package whose candidates carry thumbs alone would give
    the centrepiece a 256px image, and the exporter should fail loudly rather than
    ship the story's largest visual at thumbnail resolution.
    """
    candidates = tour_frame_candidates(
        package.manifest, package.explain, package.gt, arm=package.arm
    )
    for token in candidates:
        if package.crop(token).is_file():
            return token
    raise SystemExit("build_data: no arm-selected candidate frame has a crop in this package")


def build_why_frame(package: Package, writer: Writer) -> dict[str, Any]:
    token = _tour_frame(package)
    frame_row = package.manifest.loc[package.manifest["sample_data_token"] == token].iloc[0]
    gt_rows = visible_gt(package.gt, token)
    explain_row: dict[str, Any] = (
        package.explain.loc[package.explain["sample_data_token"] == token].iloc[0].to_dict()
    )

    night_floor_value = (package.validation.get("config") or {}).get("night_floor")
    night_floor = int(night_floor_value) if night_floor_value is not None else None
    n_communities = int(package.validation.get("n_communities", len(package.communities)))
    n_selected = int(package.validation.get("n_selected", len(package.explain)))

    floor_text = (
        "a night floor takes a minimum number of night frames first"
        if night_floor is None
        else f"a night floor takes {night_floor} night frames first"
    )
    flagship = rank_events(package.events, _TOUR_PRESET)["sample_data_token"]
    ranked_tokens = [str(value) for value in flagship]
    rank = ranked_tokens.index(token) + 1 if token in ranked_tokens else None

    return {
        "architecture_strip": _ARCHITECTURE_STRIP,
        "chips": reason_chips(
            explain_row,
            gt_rows,
            n_communities=n_communities,
            night_floor=night_floor,
            quota_before=frame_quota_before(package.communities, explain_row, arm=package.arm),
        ),
        "factors": [
            {"flag": flag, "label": label, "value": value}
            for label, value, flag in selection_factors(
                explain_row, n_communities=n_communities, night_floor=night_floor
            )
        ],
        "flagship_rank_sentence": (
            None
            if rank is None
            else (
                f"This frame is itself flagship event #{rank} — the scenario query and "
                "the selection agree on it."
            )
        ),
        "headline": "Thousands of candidate training frames — which are worth labelling?",
        "image": writer.image(
            _overlay(package, token=token, model=package.arm, mode="gt", gt_rows=gt_rows),
            name="selected-frame",
            alt=(
                "A candidate training-pool frame with its ground-truth boxes drawn on it."
            ),
        ),
        "lede_sentence": _LEDE_SENTENCE,
        "mechanism_sentence": (
            "Nothing about this frame on its own selected it: its community carried "
            "failure mass, that mass bought the community a quota, the frame ranked "
            f"high enough inside it by similarity — and {floor_text}."
        ),
        "provenance": [
            _provenance(
                "reproduced",
                f"demo al-explain reproduced the run: {n_selected:,} frames, "
                f"{n_communities} communities",
            )
        ],
        "selection_chain": list(_SELECTION_CHAIN),
        "selection_path": _SELECTION_PATH,
        "token": token,
        "train_pool_note": (
            _TRAIN_POOL_NOTE if str(frame_row.get("split")) == "train_pool" else None
        ),
        "weak_rejected_sentence": (
            "Later, the weak-supervision verifier rejected this frame as too crowded to "
            "label automatically — step 7 shows why that matters."
            if str(frame_row.get("weak_verdict")) == "rejected"
            else None
        ),
    }


# --- 5. the intervention (contract §1.3) ----------------------------------------------


def _night_share_line(package: Package) -> str | None:
    """``views/tour.py::_night_share_line`` -- every arm named by its story label."""
    share = package.arm_row(package.arm).get("night_share")
    if not bool(pd.notna(share)):
        return None
    clauses = [f"{arm_story_label(package.arm)}: {_share(float(share))} night"]
    for name in ("random", "mined"):
        if name == package.arm:
            continue
        rows = package.arms.loc[package.arms["arm"] == name]
        if rows.empty:
            continue
        value = rows.iloc[0].get("night_share")
        if not bool(pd.notna(value)):
            continue
        clauses.append(f"{arm_story_label(name)}: {_share(float(value))} night")
    return " · ".join(clauses)


def _fairness_parts(coverage: pd.DataFrame, *, baseline: str) -> list[str]:
    """``views/tour.py::_fairness_statement``, as its clauses.

    The budget clause names a frame count only when every charted non-baseline arm
    really was trained on the same number of images (NA counting as "not known to
    be equal"); the control clause only when the control arm is charted.
    """
    others = coverage.loc[coverage["arm"] != baseline]
    sizes = others["n_train_images"]
    known = sizes.dropna()
    shared = {int(value) for value in known}
    budget = (
        f"same {_count(shared.pop())}-frame budget"
        if not others.empty and len(known) == len(sizes) and len(shared) == 1
        else "same budget"
    )
    parts = ["Same detector", budget, *_FAIRNESS_TAIL.split(" · ")]
    if "random" in {str(name) for name in coverage["arm"]}:
        parts.append("Random sample is the control")
    return parts


def _fairness_sentence(parts: list[str], *, has_control: bool) -> str:
    """The clauses rejoined exactly as the tour joins them: " · " between the
    held-constant clauses, an em dash before the control clause."""
    if has_control:
        return " · ".join(parts[:-1]) + f" — {parts[-1]}."
    return " · ".join(parts) + "."


def _best_night_arm_sentence(package: Package) -> str | None:
    """``views/tour.py::_best_night_arm_sentence`` -- computed over the whole arm
    table with the baseline dropped, and never promoted into a win it did not earn."""
    ranked = package.arms.dropna(subset=["delta_night"])
    ranked = ranked.loc[ranked["arm"] != package.baseline]
    if ranked.empty:
        return None
    best = ranked.loc[ranked["delta_night"].idxmax()]
    name, delta = str(best["arm"]), float(best["delta_night"])
    if delta <= 0:
        return (
            "No intervention beat the baseline on the diagnosed night weakness — "
            f"the closest was {arm_story_label(name)} (`{name}`, {delta:+.4f} night)."
        )
    return (
        "Best intervention for the diagnosed night weakness: "
        f"{arm_story_label(name)} (`{name}`, {delta:+.4f} night)."
    )


def _similarity_sentence(arms: pd.DataFrame) -> str | None:
    """``views/tour.py::_similarity_sentence`` -- written only where the mined set
    really is a daytime one."""
    rows = arms.loc[arms["arm"] == "mined"]
    if rows.empty:
        return None
    share = rows.iloc[0]["night_share"]
    if not bool(pd.notna(share)) or float(share) >= _NIGHT_ABSENT_SHARE:
        return None
    return (
        "Visual similarity alone concentrated on daytime appearance "
        f"({_share(float(share))} night); the graph + night-floor arm explicitly "
        "preserved night coverage."
    )


def _scene_diversity_sentence(base_row: pd.Series, arm_row: pd.Series) -> str | None:
    """``views/tour.py::_scene_diversity_sentence``."""
    n_scenes = arm_row.get("n_scenes")
    if not bool(pd.notna(n_scenes)):
        return None
    scenes = int(n_scenes)
    base_n, arm_n = base_row.get("n_train_images"), arm_row.get("n_train_images")
    if bool(pd.notna(base_n)) and bool(pd.notna(arm_n)):
        mined = int(arm_n) - int(base_n)
        subject = f"The {_count(mined)} frame{'' if mined == 1 else 's'}"
    else:
        subject = "The mined frames"
    sentence = f"{subject} came from {_count(scenes)} scene{'' if scenes == 1 else 's'}"
    if scenes > 1:
        sentence += " — targeted, but not near-duplicates of one scene"
    return sentence + "."


def build_intervention(package: Package) -> dict[str, Any]:
    base_row, arm_row = package.arm_row(package.baseline), package.arm_row(package.arm)
    coverage = strategy_coverage(
        package.arms,
        strategies=tour_strategies(package.arms, baseline=package.baseline, arm=package.arm),
    )
    highlight = arm_story_label(package.arm)

    cards: list[dict[str, str]] = []
    base_n = int(_required(base_row, "n_train_images"))
    arm_n = int(_required(arm_row, "n_train_images"))
    cards.append({"label": "Frames mined", "value": _count(arm_n - base_n)})
    n_scenes = arm_row.get("n_scenes")
    if bool(pd.notna(n_scenes)):
        cards.append({"label": "Scenes covered", "value": str(int(n_scenes))})
    night_share = arm_row.get("night_share")
    if bool(pd.notna(night_share)):
        cards.append({"label": "Night share", "value": _share(float(night_share))})
    cards.append({
        "label": "Training images",
        "value": _count(arm_n),
        "delta": f"+{_count(arm_n - base_n)} mined frames",
    })

    parts = _fairness_parts(coverage, baseline=package.baseline)
    has_control = parts[-1] == "Random sample is the control"
    return {
        "cards": cards,
        "chart": {
            "bars": [
                {
                    "arm": str(record.arm),
                    "delta_night": _optional_round(record.delta_night),
                    "highlight": str(record.strategy) == highlight,
                    "n_scenes": (
                        None if not bool(pd.notna(record.n_scenes)) else int(record.n_scenes)
                    ),
                    "n_train_images": (
                        None
                        if not bool(pd.notna(record.n_train_images))
                        else int(record.n_train_images)
                    ),
                    "night_share": _optional_round(record.night_share),
                    "strategy": str(record.strategy),
                }
                for record in coverage.itertuples(index=False)
            ],
            "title": _STRATEGY_CHART_TITLE,
            "y_title": _STRATEGY_CHART_Y_TITLE,
        },
        "charted_ids_caption": " · ".join(
            f"{record.strategy} (`{record.arm}`)" for record in coverage.itertuples(index=False)
        ),
        "fairness_parts": parts,
        "fairness_sentence": _fairness_sentence(parts, has_control=has_control),
        "headline": _RETRAIN_HEADLINE,
        "night_share_line": _night_share_line(package),
        "provenance": [_provenance("recorded", "active_learning_results.parquet")],
        "similarity_caption": _similarity_sentence(coverage),
        "spread_sentence": _scene_diversity_sentence(base_row, arm_row),
        "winner_sentence": _best_night_arm_sentence(package),
    }


# --- 6. the verdict (contract §1.4) ----------------------------------------------------


def _night_map_sentence(package: Package) -> str:
    """``views/tour.py::_night_map_sentence``."""
    base_row, arm_row = package.arm_row(package.baseline), package.arm_row(package.arm)
    base_night, arm_night = float(base_row["night_map5095"]), float(arm_row["night_map5095"])
    recorded = arm_row.get("delta_night")
    delta = float(recorded) if bool(pd.notna(recorded)) else arm_night - base_night
    return (
        f"Night mAP50-95 {_metric(base_night)} → {_metric(arm_night)} ({delta:+.4f}) on "
        "the held-out split."
    )


def _night_ped_sentence(package: Package) -> str | None:
    """``views/tour.py::_night_ped_sentence`` -- ``gain_text`` writes both gains."""
    before = package.arm_row(package.baseline).get("night_ped_map5095")
    after = package.arm_row(package.arm).get("night_ped_map5095")
    if not (bool(pd.notna(before)) and bool(pd.notna(after))):
        return None
    return gain_text("Night pedestrian mAP50-95", float(before), float(after))


def _hero_recovery_is_low_conf(package: Package) -> bool:
    """``views/tour.py::_hero_recovery_is_low_conf`` -- derived, never asserted."""
    if package.hero not in _pedestrian_miss_to_hit_frames(package):
        return False
    upgraded = fixed_boxes(
        visible_gt(package.gt, package.hero),
        package.preds.loc[package.preds["sample_data_token"] == package.hero],
        baseline=package.baseline,
        arm=package.arm,
    )
    return bool(upgraded.empty)


def _hero_recovery_conf(package: Package) -> float | None:
    """``views/tour.py::_hero_recovery_conf`` -- the raw confidence behind that line."""
    gt_rows = visible_gt(package.gt, package.hero)
    baseline_column, arm_column = f"matched_{package.baseline}", f"matched_{package.arm}"
    peds = gt_rows.loc[
        (gt_rows["category_group"] == "pedestrian")
        & gt_rows[baseline_column].eq(False).fillna(False)
        & gt_rows[arm_column].eq(True).fillna(False)
    ]
    if peds.empty:
        return None
    claim = _claim(
        package,
        token=package.hero,
        model=package.arm,
        annotation=str(peds.iloc[0]["annotation_token"]),
    )
    return None if claim is None else claim[0]


def build_verdict(package: Package, writer: Writer) -> dict[str, Any]:
    token = str(package.exemplars["tokens"][0])
    gt_rows = visible_gt(package.gt, token)
    frame_preds = package.preds.loc[package.preds["sample_data_token"] == token]
    upgraded = fixed_boxes(gt_rows, frame_preds, baseline=package.baseline, arm=package.arm)
    callout = upgrade_callout(upgraded)

    images = {
        key: writer.image(
            _overlay(package, token=token, model=model, mode="overlay", gt_rows=gt_rows),
            name=f"exemplar-{key}",
            alt=(
                f"A hand-approved validation frame with the {arm_story_label(model)} "
                "detections drawn over the ground truth."
            ),
        )
        for key, model in (("baseline", package.baseline), ("arm", package.arm))
    }
    return {
        "aggregate_label": _AGGREGATE_LABEL,
        "callout": (
            None
            if callout is None
            else {"category": callout[0], "before": callout[1], "after": callout[2]}
        ),
        "callout_labels": {
            "after": f"After — {arm_story_label(package.arm)}",
            "before": f"Before — {arm_story_label(package.baseline)}",
        },
        "example_caveat": _EXAMPLE_CAVEAT,
        "example_label": _EXAMPLE_LABEL,
        "fixed_boxes": [
            {
                "annotation_token": str(record.annotation_token),
                "arm_claim": str(record.arm_claim),
                "baseline_claim": str(record.baseline_claim),
                "category_group": str(record.category_group),
                "distance_to_ego_m": _optional_round(record.distance_to_ego_m),
            }
            for record in upgraded.itertuples(index=False)
        ],
        "headline": _AFTER_HEADLINE,
        "held_out_caption": _held_out_caption(package, token),
        "hero_honesty_line": (
            _HERO_HONESTY_LINE if _hero_recovery_is_low_conf(package) else None
        ),
        "images": images,
        "legend": _legend(),
        "models": {"arm": _label(package.arm), "baseline": _label(package.baseline)},
        "night_map_sentence": _night_map_sentence(package),
        "night_ped_sentence": _night_ped_sentence(package),
        "per_box_fold_label": _PER_BOX_FOLD,
        "provenance": [_provenance("recomputed", "per-box claims from predictions.parquet")],
        "token": token,
    }


# --- 7. the closed loop (contract §1.5) -------------------------------------------------


def _result_headline(arm_row: pd.Series) -> str | None:
    """``views/tour.py::_result_headline`` -- "measurable improvement" only where the
    arm's own recorded night delta really is a gain."""
    delta = arm_row.get("delta_night")
    if not bool(pd.notna(delta)):
        return None
    return _CLOSED_LOOP_IMPROVED if float(delta) > 0 else _CLOSED_LOOP_MEASURED


def _result_cards(package: Package) -> list[dict[str, str]]:
    """``views/tour.py::_result_hero_cards``."""
    base_row, arm_row = package.arm_row(package.baseline), package.arm_row(package.arm)
    cards: list[dict[str, str]] = []
    base_n, arm_n = base_row.get("n_train_images"), arm_row.get("n_train_images")
    if bool(pd.notna(base_n)) and bool(pd.notna(arm_n)):
        cards.append({"label": "Targeted frames added", "value": _count(int(arm_n) - int(base_n))})

    night_share = arm_row.get("night_share")
    if bool(pd.notna(night_share)):
        card = {"label": "Night share of mined data", "value": _share(float(night_share))}
        control_rows = package.arms.loc[package.arms["arm"] == "random"]
        control = control_rows.iloc[0].get("night_share") if not control_rows.empty else None
        if control is not None and bool(pd.notna(control)):
            card["delta"] = f"vs {_share(float(control))} random control"
        cards.append(card)

    before, after = base_row.get("night_ped_map5095"), arm_row.get("night_ped_map5095")
    if bool(pd.notna(before)) and bool(pd.notna(after)):
        delta = float(after) - float(before)
        relative = relative_gain(float(before), float(after))
        if relative is None:
            cards.append({
                "label": "Night-pedestrian mAP50-95",
                "value": f"{delta:+.4f} absolute",
            })
        else:
            cards.append({
                "label": "Night-pedestrian mAP50-95",
                "value": f"{relative:+.1%} relative",
                "delta": f"{delta:+.4f} absolute",
            })
    return cards


def _mined_comparator_clause(package: Package, *, name: str, label: str) -> str | None:
    """``views/tour.py::_mined_comparator_clause`` -- the noun agrees with the count."""
    rows = package.arms.loc[package.arms["arm"] == name]
    if rows.empty:
        return None
    row = rows.iloc[0]
    n_scenes, night_share = row.get("n_scenes"), row.get("night_share")
    if not (bool(pd.notna(n_scenes)) and bool(pd.notna(night_share))):
        return None
    scenes = int(n_scenes)
    return (
        f"{label} covered {scenes} scene{'' if scenes == 1 else 's'} at "
        f"{_share(float(night_share))} night"
    )


def _answer_weakness(package: Package) -> list[str]:
    """``views/tour.py::_result_weakness``."""
    row = package.arm_row(package.baseline)
    sentences = [
        f"Night mAP50-95 {_metric(float(row['night_map5095']))} vs overall "
        f"{_metric(float(row['overall_map5095']))}."
    ]
    night_ped = row.get("night_ped_map5095")
    if bool(pd.notna(night_ped)):
        sentences.append(f"Night pedestrian mAP50-95 {_metric(float(night_ped))}.")
    n_night, n_night_fn, _n_val = _night_miss_counts(package)
    sentences.append(
        f"{n_night_fn} of {n_night} night validation frames carry at least one baseline miss."
    )
    return sentences


def _answer_data_added(package: Package) -> list[str]:
    """``views/tour.py::_result_data_added``."""
    base_row, arm_row = package.arm_row(package.baseline), package.arm_row(package.arm)
    n_mined = int(_required(arm_row, "n_train_images")) - int(
        _required(base_row, "n_train_images")
    )
    scenes = int(_required(arm_row, "n_scenes"))
    sentence = (
        f"`{package.arm}` mined {_count(n_mined)} frame{'' if n_mined == 1 else 's'} across "
        f"{scenes} scene{'' if scenes == 1 else 's'}, "
        f"{_share(float(_required(arm_row, 'night_share')))} at night"
    )
    clauses = [
        clause
        for clause in (
            _mined_comparator_clause(package, name="random", label="random"),
            _mined_comparator_clause(package, name="mined", label="similarity mining"),
        )
        if clause is not None
    ]
    if clauses:
        sentence += " — " + "; ".join(clauses)
    sentences = [sentence + "."]
    similarity = _similarity_sentence(package.arms)
    if similarity is not None:
        sentences.append(similarity)
    return sentences


def _answer_improved(package: Package) -> list[str]:
    """``views/tour.py::_result_improved``."""
    sentences = [_night_map_sentence(package)]
    ped_sentence = _night_ped_sentence(package)
    if ped_sentence is not None:
        sentences.append(ped_sentence)
    delta_overall = package.arm_row(package.arm).get("delta_overall")
    if bool(pd.notna(delta_overall)):
        sentences.append(f"Overall mAP50-95 {float(delta_overall):+.4f} against baseline.")
    ranked = package.arms.dropna(subset=["delta_overall"])
    if not ranked.empty:
        best = ranked.loc[ranked["delta_overall"].idxmax()]
        if str(best["arm"]) != package.arm:
            sentences.append(
                f"The best overall arm is `{best['arm']}` "
                f"({float(best['delta_overall']):+.4f}); the night arm trades overall "
                "gain for night gain."
            )
    return sentences


def _answer_failures(package: Package) -> list[str]:
    """``views/tour.py::_result_failures`` -- the weak-supervision negative result."""
    sentences: list[str] = []
    headline_rows = package.loss.loc[package.loss["headline"]]
    headline_base_arm: str | None = None
    if not headline_rows.empty:
        row = headline_rows.iloc[0]
        headline_base_arm = str(row["base_arm"])
        sentences.append(
            f"Weak (VLM-verified) labels kept {float(row['retention']):.1%} of the "
            f"ground-truth gain for the `{headline_base_arm}` pair: "
            f"{float(row['dropped_frame_share']):.1%} was lost with the frames the "
            f"verifier dropped, {float(row['label_share']):.1%} to label noise on "
            "the ones it kept."
        )

    if headline_base_arm is not None:
        headline_weak = package.weaksup.loc[package.weaksup["arm"] == headline_base_arm]
        if not headline_weak.empty:
            weak_row = headline_weak.iloc[0]
            sentences.append(
                "The verifier kept sparse frames: "
                f"{float(weak_row['gt_boxes_per_accepted_frame']):.2f} vs "
                f"{float(weak_row['gt_boxes_per_rejected_frame']):.2f} GT boxes per "
                f"accepted vs rejected frame ({weak_row['arm']})."
            )

    night_ranked = package.arms.dropna(subset=["delta_night"])
    if not night_ranked.empty:
        worst = night_ranked.loc[night_ranked["delta_night"].idxmin()]
        worst_arm = str(worst["arm"])
        if worst_arm.startswith("weak_") and not worst_arm.endswith("_gt"):
            sentences.append(
                f"`{worst_arm}` is the worst night result of the {len(night_ranked)} arms "
                f"({float(worst['delta_night']):+.4f})."
            )

    hero_conf = _hero_recovery_conf(package) if _hero_recovery_is_low_conf(package) else None
    if hero_conf is not None:
        sentences.append(
            f"The hero frame's pedestrian recovery is a low-confidence claim "
            f"(conf {hero_conf:.3f}) — counted as a hit by the matching rule, not a "
            "confident detection."
        )
    return sentences


def build_closed_loop(package: Package) -> dict[str, Any]:
    return {
        "answers": [
            {"question": "What weakness did we find?", "sentences": _answer_weakness(package)},
            {"question": "What data did we add?", "sentences": _answer_data_added(package)},
            {"question": "Did the model improve?", "sentences": _answer_improved(package)},
            {"question": "What failed along the way?", "sentences": _answer_failures(package)},
        ],
        "cards": _result_cards(package),
        "closing_thesis": _CLOSING_THESIS,
        "headline": _result_headline(package.arm_row(package.arm)),
        "provenance": [
            _provenance(
                "recorded",
                "weak_loss_decomposition.parquet, weak_supervision_results.parquet, "
                "active_learning_results.parquet",
            ),
            _provenance("recomputed", "claim confidence from predictions.parquet"),
        ],
    }


# --- entrypoint -------------------------------------------------------------------------


def build(out: Path, *, demo_data: Path) -> list[tuple[str, int]]:
    """Write the eight sections and the story images; return the image sizes."""
    package = Package(demo_data)
    writer = Writer(out)
    writer.section("meta", build_meta(package))
    writer.section("blindspot", build_blindspot(package))
    writer.section("hero_frame", build_hero_frame(package, writer))
    writer.section("scenario", build_scenario(package, writer))
    writer.section("why_frame", build_why_frame(package, writer))
    writer.section("intervention", build_intervention(package))
    writer.section("verdict", build_verdict(package, writer))
    writer.section("closed_loop", build_closed_loop(package))
    return writer.story_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "web",
        help="destination web/ tree (default: the repo's web/)",
    )
    parser.add_argument(
        "--demo-data",
        type=Path,
        default=ROOT / "demo_data",
        help="the committed package to derive from (default: demo_data/)",
    )
    args = parser.parse_args(argv)

    sizes = build(args.out, demo_data=args.demo_data)
    total = sum(size for _name, size in sizes)
    for name, size in sizes:
        print(f"  {name:<24} {size:>9,} B")
    print(f"  {'total':<24} {total:>9,} B  ({len(sizes)} images)")
    if total >= _STORY_BYTE_BUDGET:
        raise SystemExit(
            f"build_data: story images total {total:,} B, over the "
            f"{_STORY_BYTE_BUDGET:,} B budget"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
