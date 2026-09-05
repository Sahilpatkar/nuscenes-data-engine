"""Guard for the committed web story bundle (`web/src/data/*.json`, `web/public/story/`).

The story site contains no numeric result literals: every figure it renders is
derived at export time from ``demo_data/`` by ``web/build_data.py`` and committed.
These tests are what makes that contract enforceable:

* the **bundle guard** re-runs the exporter into a tmp dir and asserts the eight
  JSON files come back byte-identical to the committed ones (and that a second run
  is byte-stable), so a package change or a helper change that moves a number goes
  red instead of silently drifting the site away from the app;
* the **honesty pins** assert the tour-twin sentences the exporter re-derives are
  the shipped tour's own recorded output on demo_data v0.8 (computed superlatives,
  the fairness budget clause, the closed-loop headline guard, the gain sentence),
  plus the five story-contract fixed sentences from the spec's §1;
* the **schema checks** assert every image reference resolves to a real file with
  the dimensions the bundle claims, and that no missing-value marker (``NaN``,
  ``None``, ``<NA>``, a bare ``nan``) reached a display string;
* the **copy contract** reads the app's own sources and pins the handful of labels
  the React components type out a second time (the seven step titles and stages,
  the selection fold, the fairness lead, the "What we learned" frame) to the app
  literals they mirror -- nothing at runtime can catch those two drifting apart.

Images are compared by name and dimensions rather than by bytes: libjpeg encodes
identically for a given version, but pinning bytes would make the suite hostage to
a Pillow upgrade, and the JSON — where every claim lives — is pinned exactly.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
EXPORTER = WEB / "build_data.py"
COMMITTED_JSON = WEB / "src" / "data"
COMMITTED_PUBLIC = WEB / "public"
COMMITTED_STORY = COMMITTED_PUBLIC / "story"

# The app-side sources the site hand-duplicates a few labels from (the copy contract
# at the bottom of this module reads both halves and pins them against each other).
APP_DEMO = ROOT / "app" / "demo"
TOUR_PY = APP_DEMO / "views" / "tour.py"
RENDER_PY = APP_DEMO / "render.py"
SITE_SRC = WEB / "src"

SECTIONS = (
    "blindspot",
    "closed_loop",
    "hero_frame",
    "intervention",
    "meta",
    "scenario",
    "verdict",
    "why_frame",
)

STREAMLIT_BASE = "https://nuscenes-data-engine-sahil.streamlit.app"

# The five story contracts of the spec (§1), fixed sentences the exporter ships as
# bundle fields rather than the site hard-coding them in a component.
PROBLEM_SENTENCE = (
    "The detector looked reasonable overall — but performance dropped sharply at "
    "night, especially for pedestrians."
)
PURPOSE_SENTENCE = (
    "The goal of the system: automatically find failures like this and turn them "
    "into better training data."
)
LEDE_SENTENCE = (
    "Instead of randomly adding more images, the system searches the training pool "
    "for examples related to the diagnosed failure."
)
SELECTION_CHAIN = [
    "Failed validation frame",
    "relevant driving scenario",
    "candidate training frames",
    "targeted retraining set",
]
EXAMPLE_LABEL = "One example"
EXAMPLE_CAVEAT = "one hand-approved frame — illustrative, not the metric"
AGGREGATE_LABEL = "The aggregate result"
CLOSING_THESIS = (
    "Instead of blindly retraining the model, the system diagnoses where it fails, "
    "finds the data that can address the weakness, and measures whether the "
    "intervention actually works."
)


def _run_exporter(out: Path) -> None:
    subprocess.run(
        [sys.executable, str(EXPORTER), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The bundle, re-derived from the committed package into a scratch tree."""
    out = tmp_path_factory.mktemp("web_export")
    _run_exporter(out)
    return out


@pytest.fixture(scope="session")
def exported_again(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second, independent export — the byte-stability control."""
    out = tmp_path_factory.mktemp("web_export_again")
    _run_exporter(out)
    return out


@pytest.fixture(scope="session")
def exporter() -> ModuleType:
    """``web/build_data.py`` imported rather than run -- the NA guard below is a
    unit test over one section builder, not another whole-bundle export."""
    if str(WEB) not in sys.path:
        sys.path.insert(0, str(WEB))
    import build_data

    return build_data


def _load(section: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((COMMITTED_JSON / f"{section}.json").read_text())
    return payload


@pytest.fixture(scope="session")
def bundle() -> dict[str, dict[str, Any]]:
    """The COMMITTED bundle — what the site actually ships."""
    return {section: _load(section) for section in SECTIONS}


# --- the bundle guard -------------------------------------------------------------


@pytest.mark.parametrize("section", SECTIONS)
def test_committed_json_matches_a_fresh_export(exported: Path, section: str) -> None:
    name = f"{section}.json"
    fresh = (exported / "src" / "data" / name).read_bytes()
    committed = (COMMITTED_JSON / name).read_bytes()
    assert fresh == committed, (
        f"{name} drifted from the package — re-run `uv run python web/build_data.py` "
        "and commit the bundle"
    )


def test_export_is_byte_stable(exported: Path, exported_again: Path) -> None:
    for section in SECTIONS:
        name = f"{section}.json"
        first = (exported / "src" / "data" / name).read_bytes()
        second = (exported_again / "src" / "data" / name).read_bytes()
        assert first == second, f"{name} is not byte-stable across runs"


def test_committed_images_match_a_fresh_export(exported: Path) -> None:
    fresh_dir = exported / "public" / "story"
    fresh = {path.name: Image.open(path).size for path in sorted(fresh_dir.glob("*.jpg"))}
    committed = {path.name: Image.open(path).size for path in sorted(COMMITTED_STORY.glob("*.jpg"))}
    assert fresh == committed


def test_story_images_stay_under_the_size_budget() -> None:
    sizes = [path.stat().st_size for path in COMMITTED_STORY.glob("*.jpg")]
    assert len(sizes) >= 11, f"expected the ~11 story assets, found {len(sizes)}"
    assert sum(sizes) < 1_200_000, f"web/public/story/ is {sum(sizes):,} bytes"


# --- schema ------------------------------------------------------------------------


def _image_refs(node: Any) -> list[dict[str, Any]]:
    """Every ``{src, width, height, alt}`` reference anywhere in a section."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if {"src", "width", "height", "alt"} <= set(node):
            found.append(node)
        for value in node.values():
            found.extend(_image_refs(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_image_refs(value))
    return found


def test_every_image_reference_resolves(bundle: dict[str, dict[str, Any]]) -> None:
    refs = [ref for section in bundle.values() for ref in _image_refs(section)]
    assert len(refs) >= 11, f"expected the ~11 story assets, found {len(refs)} references"
    for ref in refs:
        path = COMMITTED_PUBLIC / str(ref["src"])
        assert path.is_file(), f"missing image {ref['src']}"
        assert Image.open(path).size == (ref["width"], ref["height"])
        assert str(ref["alt"]).strip(), f"{ref['src']} has no alt text"


def _strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [text for value in node.values() for text in _strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _strings(value)]
    return []


# A float NA formatted for display renders lowercase ("nan"), which is the shape an
# unguarded metric read takes -- and the one spelling the capitalised markers below
# miss. Matched as a standalone token so a legitimate word that merely contains the
# letters ("nanosecond", "banana") cannot trip the guard.
_BARE_NAN = re.compile(r"\bnan\b")


def test_no_missing_value_leaks_into_a_display_string(
    bundle: dict[str, dict[str, Any]],
) -> None:
    for section, payload in bundle.items():
        for text in _strings(payload):
            for marker in ("NaN", "None", "<NA>"):
                assert marker not in text, f"{section}: {marker!r} in {text!r}"
            assert not _BARE_NAN.search(text), f"{section}: bare 'nan' in {text!r}"


def test_blindspot_drops_the_night_pedestrian_card_when_the_package_has_none(
    exporter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exporter's twin of ``views/tour.py::_render_weakness``'s NA guard: on a
    package whose eval never wrote per-class night metrics, the card and the
    sentence's clause go away together rather than the bundle shipping "nan"."""
    row = pd.Series(
        {
            "arm": "baseline",
            "overall_map5095": 0.2477,
            "night_map5095": 0.1667,
            "night_ped_map5095": pd.NA,
        }
    )

    class _DoctoredPackage:
        baseline = "baseline"

        def arm_row(self, name: str) -> pd.Series:
            return row

    # The counted miss line reads three frame tables; it is not what is under test.
    monkeypatch.setattr(exporter, "_night_miss_counts", lambda _package: (62, 60, 400))
    section = exporter.build_blindspot(_DoctoredPackage())

    assert [card["label"] for card in section["cards"]] == [
        "Overall mAP50-95",
        "Night mAP50-95",
    ]
    assert section["derived_sentence"] == (
        "`baseline` scores 0.2477 mAP50-95 overall but 0.1667 at night."
    )
    for text in _strings(section):
        assert not _BARE_NAN.search(text), f"bare 'nan' in {text!r}"


def test_deep_links_point_at_the_live_streamlit_app(bundle: dict[str, dict[str, Any]]) -> None:
    links = bundle["meta"]["deep_links"]
    assert len(links) == 6
    assert bundle["meta"]["streamlit_base"] == STREAMLIT_BASE
    for link in links:
        assert str(link["url"]).startswith(STREAMLIT_BASE)
        assert str(link["label"]).strip()


def test_meta_carries_the_package_provenance(bundle: dict[str, dict[str, Any]]) -> None:
    package = bundle["meta"]["package"]
    assert set(package) == {"built_at", "git_sha", "version"}
    assert package["version"] == "0.8"
    assert bundle["meta"]["exporter"] == "web/build_data.py"
    attribution = bundle["meta"]["attribution"]
    assert "CC BY-NC-SA 4.0" in attribution["license"]
    assert "Caesar" in attribution["citation"] and "2020" in attribution["citation"]
    # A wall-clock export timestamp would make the bundle un-guardable.
    assert "exported_at" not in bundle["meta"]


# --- the five story contracts (spec §1) --------------------------------------------


def test_story_contract_sentences_are_shipped_verbatim(
    bundle: dict[str, dict[str, Any]],
) -> None:
    assert bundle["blindspot"]["problem_sentence"] == PROBLEM_SENTENCE
    assert bundle["blindspot"]["purpose_sentence"] == PURPOSE_SENTENCE
    assert bundle["why_frame"]["lede_sentence"] == LEDE_SENTENCE
    assert bundle["why_frame"]["selection_chain"] == SELECTION_CHAIN
    assert bundle["verdict"]["example_label"] == EXAMPLE_LABEL
    assert bundle["verdict"]["example_caveat"] == EXAMPLE_CAVEAT
    assert bundle["verdict"]["aggregate_label"] == AGGREGATE_LABEL
    assert bundle["closed_loop"]["closing_thesis"] == CLOSING_THESIS


# --- the honesty pins (the tour's own recorded output on demo_data v0.8) ------------


def test_blindspot_states_the_three_recorded_metrics(
    bundle: dict[str, dict[str, Any]],
) -> None:
    cards = bundle["blindspot"]["cards"]
    assert [card["value"] for card in cards] == ["0.2477", "0.1667", "0.0826"]
    assert "60 of 62" in bundle["blindspot"]["miss_sentence"]


def test_winner_sentence_is_computed_not_asserted(bundle: dict[str, dict[str, Any]]) -> None:
    winner = bundle["intervention"]["winner_sentence"]
    assert "Graph + night targeting" in winner
    assert "(`graph_rate_night`, +0.0101 night)" in winner


def test_fairness_statement_keeps_its_derived_budget_and_control_clauses(
    bundle: dict[str, dict[str, Any]],
) -> None:
    parts = bundle["intervention"]["fairness_parts"]
    assert "same 8,535-frame budget" in parts
    assert parts[-1] == "Random sample is the control"
    sentence = bundle["intervention"]["fairness_sentence"]
    for part in parts:
        assert part in sentence


def test_similarity_caption_is_written_only_for_a_daytime_mined_set(
    bundle: dict[str, dict[str, Any]],
) -> None:
    caption = bundle["intervention"]["similarity_caption"]
    assert caption is not None
    assert "concentrated on daytime appearance" in caption


def test_verdict_states_both_evidence_tiers(bundle: dict[str, dict[str, Any]]) -> None:
    verdict = bundle["verdict"]
    assert verdict["callout"] == {
        "category": "pedestrian",
        "before": "low-conf 0.30",
        "after": "0.47",
    }
    assert verdict["night_ped_sentence"] == (
        "Night pedestrian mAP50-95 0.0826 → 0.1171 (+0.0345 absolute, +41.8% relative)"
    )
    assert verdict["hero_honesty_line"] is not None
    assert "low-confidence claim" in verdict["hero_honesty_line"]


def test_closed_loop_headline_is_earned(bundle: dict[str, dict[str, Any]]) -> None:
    assert "measurable improvement" in bundle["closed_loop"]["headline"]
    assert len(bundle["closed_loop"]["answers"]) == 4


def test_why_frame_chips_state_only_facts_the_package_carries(
    bundle: dict[str, dict[str, Any]],
) -> None:
    chips = bundle["why_frame"]["chips"]
    assert chips
    # There is no per-frame failure rate in the package, so no chip may say "rate"
    # -- not even by way of the arm name `graph_rate_night`.
    assert not [chip for chip in chips if "rate" in chip]


def test_scenario_filmstrip_is_the_five_can_steps(bundle: dict[str, dict[str, Any]]) -> None:
    filmstrip = bundle["scenario"]["filmstrip"]
    assert filmstrip["speed_is_can"] is True
    steps = filmstrip["steps"]
    assert len(steps) == 5
    assert [step["is_current"] for step in steps].count(True) == 1
    assert [step["label"] for step in steps] == ["t-2", "t-1", "current", "t+1", "t+2"]


def test_hero_frame_carries_both_model_overlays(bundle: dict[str, dict[str, Any]]) -> None:
    images = bundle["hero_frame"]["images"]
    assert set(images) == {"arm", "baseline"}
    assert bundle["hero_frame"]["held_out_caption"].startswith("Held-out validation frame")


# --- the copy contract (site literals ↔ their app twins) ---------------------------
#
# The bundle carries the story's SENTENCES, not the frame around them, so a handful
# of labels are typed out twice: once in a Streamlit call and once in a React
# component. Nothing at runtime can notice those two drifting apart -- a renamed
# expander or a re-worded step title would simply leave the two front-ends calling
# the same thing different names. These read both sources and pin the literals.


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _only(pattern: str, text: str, what: str) -> str:
    """The single capture of ``pattern`` in ``text``. A second match means the
    literal moved and this pin would be reading the wrong one."""
    found = re.findall(pattern, text)
    assert len(found) == 1, f"expected exactly one {what}, found {found}"
    return str(found[0])


def _tour_steps() -> list[tuple[str, str]]:
    """``(title, stage)`` per ``views/tour.py::_STEPS`` entry, in story order.

    The stage is returned as it is WRITTEN -- a quoted name for the six single-stage
    steps, the bare identifier ``LOOP_STAGES`` for the result screen."""
    block = _source(TOUR_PY).split("_STEPS: tuple[_Step, ...] = (", 1)[1].split("\ndef ", 1)[0]
    steps: list[tuple[str, str]] = []
    for chunk in block.split("_Step(")[1:]:
        title = _only(r'title="([^"]+)"', chunk, "step title")
        stage = _only(r"\n\s+stage=([^\n]+),\n", chunk, "step stage")
        steps.append((title, stage.strip('"')))
    return steps


def _site_steps() -> list[tuple[str, str]]:
    """``(title, stage)`` per ``web/src/App.tsx``'s ``SECTIONS`` entry, in order."""
    block = (
        _source(SITE_SRC / "App.tsx")
        .split("const SECTIONS: readonly StorySection[] = [", 1)[1]
        .split("\nconst RAIL_STEPS", 1)[0]
    )
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(r'title: "([^"]+)",[\s\S]*?stage: "([^"]+)",', block)
    ]


def test_site_step_titles_and_stages_are_the_tours_own() -> None:
    tour, site = _tour_steps(), _site_steps()
    assert len(tour) == 7, tour
    assert [title for title, _stage in site] == [title for title, _stage in tour]
    # Steps 1-6 each light exactly one loop stage, named identically on both.
    assert [stage for _title, stage in site[:6]] == [stage for _title, stage in tour[:6]]
    # Step 7 is the one deliberate difference. The tour passes the whole
    # ``render.LOOP_STAGES`` tuple to its breadcrumb (every stage lit at once); the
    # site has no breadcrumb widget, so it says the same thing in words.
    assert tour[6] == ("Closed the loop", "LOOP_STAGES")
    assert site[6] == ("Closed the loop", "The whole loop")


def test_selection_fold_label_is_the_tours_expander_label() -> None:
    label = _only(r'st\.expander\("([^"]+)"\)', _source(TOUR_PY), "literal tour expander label")
    assert label == "How selection works"
    assert f"<summary>{label}</summary>" in _source(SITE_SRC / "sections" / "WhyFrame.tsx")


def test_fairness_lead_is_the_tours_fairness_lead() -> None:
    lead = _only(r'_FAIRNESS_LEAD = "([^"]+)"', _source(TOUR_PY), "_FAIRNESS_LEAD")
    assert lead == "Fair comparison"
    assert f'<p className="eyebrow">{lead}</p>' in _source(
        SITE_SRC / "sections" / "Intervention.tsx"
    )


def test_learned_callout_label_is_render_learneds_own() -> None:
    label = _only(r"\*\*(What we learned)\*\*", _source(RENDER_PY), "learned() label")
    assert f'<span className="learned-label">{label}</span>' in _source(
        SITE_SRC / "components" / "Learned.tsx"
    )


def test_blindspot_headline_is_site_editorial_not_a_tour_twin(
    bundle: dict[str, dict[str, Any]],
) -> None:
    """The one story heading with no app twin, deliberately (see the note beside it
    in ``web/build_data.py``): the tour's step 1 leads with ``_PROBLEM_SENTENCE`` as
    its heading, while a scrolling page wants a heading that names the finding. So
    this is pinned as non-empty rather than bound -- what it claims is evidenced by
    the cards under it, which ``test_blindspot_states_the_three_recorded_metrics``
    pins exactly."""
    headline = str(bundle["blindspot"]["headline"])
    assert headline.strip()
    assert headline not in _source(TOUR_PY), (
        "the tour now says this too — bind the two instead of leaving it site-only"
    )
