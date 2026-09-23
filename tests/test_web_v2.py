"""Copy contract for the report edition of the story site (`web/src/v2/`), and
the layout contract that makes it the site's main edition.

The report edition renders the SAME committed bundle as the original edition, so
the data itself needs no second guard: `tests/test_web_export.py` already re-runs
the exporter and pins every number and sentence the bundle carries. What a second
front-end CAN do is quietly stop agreeing with the first one -- retype a step
title, hard-code a sentence the bundle ships as a field, print a figure as a
literal, or write an `<img src>` that 404s from whichever edition is served off
the root. Nothing at runtime notices any of those, so these tests read the sources
and pin them:

* the seven step titles and stages in `v2/App.tsx` are the live tour's own
  (`app/demo/views/tour.py::_STEPS`), exactly as the original edition's are;
* none of the five story-contract sentences is typed into a v2 component -- each
  is read from the bundle field that ships it, and those field names must appear;
* no recorded result figure appears as a literal anywhere in the v2 sources;
* the selection fold label and the fairness lead are the tour's literals;
* every `<img src>` in EITHER edition routes through `assetUrl`, which is the only
  thing standing between the document-relative bundle srcs and a 404 from `/v1/`;
* the report edition is the document at the site root and the original edition
  the one at `/v1/`, the retired `/v2/` address redirects to the root, and the two
  editions link to each other through Vite's base -- the one thing about the
  layout that a build and a typecheck are both blind to.

They are source-level (regex over the TSX), which is what makes them cheap enough
to run in the default suite: no node, no build, no browser. The helpers and the
contract-sentence constants come from `test_web_export.py` so the two editions are
checked against ONE definition of the story's fixed copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_web_export import (
    AGGREGATE_LABEL,
    CLOSING_THESIS,
    EXAMPLE_CAVEAT,
    EXAMPLE_LABEL,
    LEDE_SENTENCE,
    PROBLEM_SENTENCE,
    PURPOSE_SENTENCE,
    SELECTION_CHAIN,
    SITE_SRC,
    TOUR_PY,
    WEB,
    _only,
    _source,
    _tour_steps,
)

V2_SRC = SITE_SRC / "v2"

# The two editions by their source trees: `v2` is `web/src/v2/**` (the report
# edition, served at the site root); `v1` is everything else under `web/src/` (the
# original scroll-through edition, served at `/v1/`, whose data and pure renderers
# the report edition imports).
_EDITIONS = ("v1", "v2")


def _edition_tsx(edition: str) -> list[Path]:
    """Every component of one edition, in path order."""
    in_v2 = set(V2_SRC.rglob("*.tsx"))
    files = sorted(path for path in SITE_SRC.rglob("*.tsx") if (path in in_v2) == (edition == "v2"))
    assert files, f"no {edition} components found under {SITE_SRC}"
    return files


def _v2_tsx() -> list[Path]:
    """Every component of the report edition, in path order."""
    return _edition_tsx("v2")


# `/* … */` (JSX's `{/* … */}` included) and whole-line `//`. A TRAILING `//`
# comment would survive this, deliberately: eating everything after a `//` would
# also eat the second half of every `https://` URL in the sources, and a comment
# cannot render anything anyway -- the strip exists so that prose ABOUT a literal
# is not mistaken for the literal (see the tier labels quoted in Verdict.tsx's
# header comment), not as a security boundary.
_COMMENT = re.compile(r"/\*[\s\S]*?\*/|^[ \t]*//.*$", re.MULTILINE)


def _code(path: Path) -> str:
    """One source with its comments removed."""
    return _COMMENT.sub("", _source(path))


def _v2_steps() -> list[tuple[str, str]]:
    """``(title, stage)`` per ``web/src/v2/App.tsx``'s ``SECTIONS`` entry, in order.

    Read exactly as the root edition's block is read (`_site_steps` over
    `web/src/App.tsx`): the literal array is why v2 repeats the titles inline
    instead of importing them -- a repeated literal is greppable, and this is what
    greps it."""
    block = (
        _source(V2_SRC / "App.tsx")
        .split("const SECTIONS: readonly StorySection[] = [", 1)[1]
        .split("\nconst CONTENTS", 1)[0]
    )
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(r'title: "([^"]+)",[\s\S]*?stage: "([^"]+)",', block)
    ]


def test_v2_step_titles_and_stages_are_the_tours_own() -> None:
    tour, v2 = _tour_steps(), _v2_steps()
    assert len(tour) == 7, tour
    assert [title for title, _stage in v2] == [title for title, _stage in tour]
    # Steps 1-6 each light exactly one loop stage, named identically on both.
    assert [stage for _title, stage in v2[:6]] == [stage for _title, stage in tour[:6]]
    # Step 7 is the one deliberate difference, and the report edition states it the
    # same way the root edition does. The tour passes the whole ``render.LOOP_STAGES``
    # tuple to its breadcrumb (every stage lit at once); neither web edition has a
    # breadcrumb widget, so both say the same thing in words.
    assert tour[6] == ("Closed the loop", "LOOP_STAGES")
    assert v2[6] == ("Closed the loop", "The whole loop")


# The five fixed sentences of the story contract (spec §1). The exporter ships each
# as a bundle field; a component that types one out instead would keep saying it
# after the exporter stopped.
_CONTRACT_SENTENCES = {
    "problem sentence": PROBLEM_SENTENCE,
    "purpose sentence": PURPOSE_SENTENCE,
    "lede sentence": LEDE_SENTENCE,
    "example caveat": EXAMPLE_CAVEAT,
    "closing thesis": CLOSING_THESIS,
}

# The bundle fields those sentences (and the labels/chain around them) arrive in.
_CONTRACT_FIELDS = (
    "problem_sentence",
    "purpose_sentence",
    "lede_sentence",
    "selection_chain",
    "example_label",
    "example_caveat",
    "aggregate_label",
    "closing_thesis",
)


def test_v2_contract_sentences_come_from_the_bundle_not_literals() -> None:
    sources = {path: _source(path) for path in _v2_tsx()}
    for what, sentence in _CONTRACT_SENTENCES.items():
        typed_out = [
            str(path.relative_to(V2_SRC)) for path, text in sources.items() if sentence in text
        ]
        assert not typed_out, (
            f"the {what} is written out in {typed_out} — render the bundle field instead"
        )

    # The two evidence-tier labels and the four selection-chain steps are checked
    # against the CODE rather than the raw file: Verdict.tsx's header comment names
    # both tier labels while explaining that the tiers ARE the section's structure.
    # Prose about a label is documentation; only a rendered literal is drift.
    code = {path: _COMMENT.sub("", text) for path, text in sources.items()}
    labels = {
        "example label": EXAMPLE_LABEL,
        "aggregate label": AGGREGATE_LABEL,
        **{f"selection chain step {i}": step for i, step in enumerate(SELECTION_CHAIN, 1)},
    }
    for what, literal in labels.items():
        typed_out = [
            str(path.relative_to(V2_SRC)) for path, text in code.items() if literal in text
        ]
        assert not typed_out, (
            f"the {what} is written out in {typed_out} — render the bundle field instead"
        )

    # ... and the fields they come from are actually read somewhere in v2, so the
    # absence above means "read from the bundle", not "dropped from the edition".
    joined = "\n".join(code.values())
    for field in _CONTRACT_FIELDS:
        assert field in joined, f"no v2 component reads the bundle's `{field}`"


# Every recorded figure the story states, as the bundle spells it. These live in
# `web/src/data/*.json` (pinned exactly by test_web_export.py); a copy typed into a
# component would survive the next export.
_RESULT_LITERALS = (
    "0.2477",  # baseline overall mAP50-95
    "0.1667",  # baseline night mAP50-95
    "0.0826",  # baseline night-pedestrian mAP50-95
    "0.1171",  # the retrained arm's night-pedestrian mAP50-95
    "+0.0345",  # the absolute gain
    "41.8",  # ... and the relative one, in percent
    "8,535",  # the equal frame budget every arm trained on
    "0.30",  # the hero frame's before confidence
    "0.47",  # ... and its after confidence
    "60 of 62",  # the missed night pedestrians of the blind-spot sentence
)


def test_v2_has_no_recorded_result_literals() -> None:
    """Bounded on both sides by "not a word character and not a dot", so a CSS or
    SVG number that merely CONTAINS one of these digit strings -- `0.78rem`,
    `0.475`, `translate(41.85px)` -- is not a hit, while the figures as the story
    states them (`0.2477`, `41.8%`, `8,535 frames`) are."""
    patterns = {
        literal: re.compile(rf"(?<![\w.]){re.escape(literal)}(?![\w.])")
        for literal in _RESULT_LITERALS
    }
    offenders: list[str] = []
    for path in _v2_tsx():
        code = _code(path)
        offenders += [
            f"{path.relative_to(V2_SRC)}: {literal}"
            for literal, pattern in patterns.items()
            if pattern.search(code)
        ]
    assert not offenders, f"result literals in v2 sources: {offenders} — read the bundle"


def test_v2_selection_fold_and_fairness_lead_are_the_tours_own() -> None:
    label = _only(r'st\.expander\("([^"]+)"\)', _source(TOUR_PY), "literal tour expander label")
    assert label == "How selection works"
    assert f"<summary>{label}</summary>" in _source(V2_SRC / "sections" / "WhyFrame.tsx")

    lead = _only(r'_FAIRNESS_LEAD = "([^"]+)"', _source(TOUR_PY), "_FAIRNESS_LEAD")
    assert lead == "Fair comparison"
    assert f'<p className="eyebrow">{lead}</p>' in _source(V2_SRC / "sections" / "Intervention.tsx")


@pytest.mark.parametrize("edition", _EDITIONS)
def test_images_route_through_the_base_helper(edition: str) -> None:
    """The bundle's image srcs are document-relative (`"story/hero-arm.jpg"`) —
    correct at the site root, a 404 from `…/nuscenes-data-engine/v1/`. `assetUrl`
    prefixes Vite's build-time base, and it only works if EVERY image goes through
    it, which is not something a build or a typecheck can notice. Both editions
    are held to it: the report edition sits at the root today, but nothing in the
    build would say so if the two swapped places again."""
    offenders: list[str] = []
    for path in _edition_tsx(edition):
        code = _code(path)
        offenders += [
            f"{path.relative_to(SITE_SRC)}: src={{{match.group(1).strip()}"
            for match in re.finditer(r"src=\{([^\n]*)", code)
            if not match.group(1).lstrip().startswith("assetUrl(")
        ]
        # A plain string src would carry a bundle path without passing the helper at
        # all, so it never has a legitimate form here. (Neither edition has one: the
        # cross-edition links are `href`, not `src`, and build on
        # `import.meta.env.BASE_URL` directly — a link to a document, not an asset.)
        if re.search(r"""src=["']""", code):
            offenders.append(f"{path.relative_to(SITE_SRC)}: literal src string")
        # No responsive image sets exist; if one is added, its candidate URLs need
        # the base prefix exactly as `src` does, so this stays a hard stop.
        if "srcSet" in code:
            offenders.append(f"{path.relative_to(SITE_SRC)}: srcSet")
    assert not offenders, f"image srcs bypassing assetUrl: {offenders}"
    # ... and the shared helper itself still prefixes the build-time base. Checked
    # against the code: the file's own doc comment quotes `import.meta.env.BASE_URL`
    # while explaining why it is there, which would keep this green on its own.
    assert "import.meta.env.BASE_URL" in _code(SITE_SRC / "assetUrl.ts")


# ---------------------------------------------------------------------------
# Layout: which document is served where. Vite builds any input map without
# comment, so the swap that made the report edition the site's main edition is
# pinned at the source: the root document, the `/v1/` document, the input map,
# the redirect at the retired address, and the links between the editions.


def test_report_edition_is_the_root_document_and_the_original_is_at_v1() -> None:
    root_doc, v1_doc = WEB / "index.html", WEB / "v1" / "index.html"
    assert v1_doc.exists(), "the original edition's document should be web/v1/index.html"
    assert 'src="/src/v2/main.tsx"' in _source(root_doc), "the root document is the report"
    assert 'src="/src/main.tsx"' in _source(v1_doc), "the /v1/ document is the original"
    # Vite's MPA input map emits each document at its own path: `index.html` at the
    # root, `v1/index.html` at `/v1/`. A `v2/index.html` input would put a second
    # copy of the report at the old address instead of the redirect below.
    config = _source(WEB / "vite.config.ts")
    assert 'new URL("index.html"' in config
    assert 'new URL("v1/index.html"' in config
    assert "v2/index.html" not in config
    assert not (WEB / "v2" / "index.html").exists(), "web/v2/index.html is no longer an entry"


def test_retired_v2_address_redirects_to_the_root() -> None:
    """`…/v2/` was the report edition's published address, and links to it exist
    outside this repo. The stub is a static file under `public/`, so Vite copies
    it to `dist/v2/index.html` untouched and Pages keeps serving the old path."""
    stub = WEB / "public" / "v2" / "index.html"
    assert stub.exists(), "web/public/v2/index.html should redirect the old address"
    text = _source(stub)
    assert re.search(r'http-equiv="refresh"\s+content="0;\s*url=\.\./"', text), text
    assert 'name="robots" content="noindex"' in text


def test_editions_link_to_each_other_through_the_base() -> None:
    """Each edition's door to the other is an `href` built from Vite's base: the
    report edition (at the root) points at `v1/`, the original edition (at `/v1/`)
    points back at the base itself. A hard-coded `/v1/` breaks under the Pages
    base `/nuscenes-data-engine/`; a hard-coded Pages URL breaks the preview."""
    for name in ("Cover.tsx", "Colophon.tsx"):
        assert "${import.meta.env.BASE_URL}v1/" in _code(V2_SRC / "components" / name), name
    assert "import.meta.env.BASE_URL" in _code(SITE_SRC / "components" / "Footer.tsx")
