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
    CLOSING_THESIS,
    COMMITTED_JSON,
    LEDE_SENTENCE,
    PROBLEM_SENTENCE,
    PURPOSE_SENTENCE,
    REPORT_EXPECTED,
    SITE_SRC,
    WEB,
    _only,
    _source,
    _strings,
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


# The report's six sections (V4, 2026-10-01): technical titles of its own. The
# report no longer mirrors the guided tour's seven conversational steps -- the
# tour and the original edition keep those -- so its titles are pinned here.
REPORT_SECTIONS = [
    "Baseline failure analysis",
    "Example failure",
    "Context-aware failure mining",
    "Targeted training-data selection",
    "Training intervention",
    "Results",
]


def _report_titles() -> list[str]:
    """The ``title:`` of each ``SECTIONS`` entry in ``web/src/v2/App.tsx``, in order."""
    block = (
        _source(V2_SRC / "App.tsx")
        .split("const SECTIONS: readonly ReportSection[] = [", 1)[1]
        .split("\n];", 1)[0]
    )
    return re.findall(r'title: "([^"]+)",', block)


def test_report_sections_are_the_six_technical_sections() -> None:
    assert _report_titles() == REPORT_SECTIONS


def test_report_copy_is_read_from_the_bundle_not_typed() -> None:
    """Every sentence and label the report states lives in ``report.json`` (or the
    shared bundle); none is typed into a component, where it would outlive the
    next export. The tour's fixed sentences are not typed out either."""
    code = {path: _code(path) for path in _v2_tsx()}
    joined = "\n".join(code.values())
    # (The project name is also the column head of the design table; the wordmark
    # owns it as chrome, so it is not "report copy".)
    copy = [
        text
        for text in _strings(REPORT_EXPECTED)
        if len(text) > 12 and text != "Perception Data Engine"
    ]
    typed = [text for text in copy if text in joined]
    assert not typed, f"report copy typed into components: {typed}"
    for sentence in (PROBLEM_SENTENCE, PURPOSE_SENTENCE, LEDE_SENTENCE, CLOSING_THESIS):
        assert sentence not in joined
    # ... and the report section is actually what the edition reads.
    assert '"../data/report.json"' in _source(V2_SRC / "App.tsx")
    assert (COMMITTED_JSON / "report.json").exists()


def test_acquisition_details_are_folded() -> None:
    """The per-frame selection facts (community, mass rank, quota, degree rank)
    sit in one closed fold rather than on the page by default."""
    code = _code(V2_SRC / "sections" / "Selection.tsx")
    assert "<summary>Acquisition details</summary>" in code
    assert re.search(r"<details(?![^>]*\bopen\b)[^>]*>", code), "the fold starts closed"


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


def test_only_the_original_edition_links_to_the_other() -> None:
    """The report edition is the site's main edition and carries no link to the
    original one (the cover's "Original edition" cross-reference was removed on
    2026-09-29, the colophon's note before it). The original edition's footer
    still points back at the root, through Vite's base so the link holds under
    the Pages prefix `/nuscenes-data-engine/` and in the preview alike."""
    assert "v1/" not in _code(V2_SRC / "components" / "Cover.tsx")
    assert "v1/" not in _code(V2_SRC / "components" / "Colophon.tsx")
    assert "import.meta.env.BASE_URL" in _code(SITE_SRC / "components" / "Footer.tsx")


def test_report_edition_carries_no_provenance_apparatus() -> None:
    """The report edition sources nothing on the page (decided 2026-09-24): no
    footnote block under a section, and no provenance sentence, package stamp or
    edition note in the end matter -- only the dataset attribution the nuScenes
    licence asks for, and the author line. The bundle still ships every
    provenance entry (the original edition and the app render them), so the
    report's silence is the frame's, not the data's."""
    frame = _code(V2_SRC / "components" / "SectionFrame.tsx")
    assert "provenance" not in frame and "footnote" not in frame
    assert "provenance" not in _code(V2_SRC / "App.tsx")
    colophon = _code(V2_SRC / "components" / "Colophon.tsx")
    for gone in ("exporter", "colophon-stamp", "git_sha", "originalEdition", "Colophon</h2>"):
        assert gone not in colophon, f"{gone} is still in the end matter"
    assert "attribution" in colophon, "the dataset attribution must stay"


def test_cover_dek_is_the_root_documents_description() -> None:
    """The cover's dek is, verbatim, the `<meta name="description">` of the root
    document (and its `og:description`), so what a search result or a link
    preview promises and what the cover says are one sentence. The three copies
    are typed out in two files, which is exactly the kind of drift nothing at
    runtime notices."""
    cover = _code(V2_SRC / "components" / "Cover.tsx")
    literal = _only(r"const DEK =\s*((?:\"[^\"]*\"\s*\+?\s*)+);", cover, "DEK literal")
    dek = "".join(re.findall(r'"([^"]*)"', literal))
    document = _source(WEB / "index.html")
    description = _only(r'name="description"\s+content="([^"]+)"', document, "meta description")
    og_description = _only(
        r'property="og:description"\s+content="([^"]+)"', document, "og:description"
    )
    assert dek == description
    assert og_description == description


def test_report_edition_chrome_has_no_em_dashes() -> None:
    """The edition's own chrome (figure labels, the running head, the contents
    list, the cover's series line, the attribution) separates with a full stop, a
    colon or a middle dot -- never an em dash (decided 2026-09-30). Checked on the
    comment-stripped sources, so prose ABOUT the old separator does not count,
    and on the root document's titles."""
    offenders = [str(path.relative_to(SITE_SRC)) for path in _v2_tsx() if "\u2014" in _code(path)]
    assert not offenders, f"em dashes in the report edition's components: {offenders}"
    document = _source(WEB / "index.html")
    titles = re.findall(r"<title>([^<]*)</title>|og:title\" content=\"([^\"]*)\"", document)
    assert titles and not any("\u2014" in "".join(match) for match in titles), titles


PROJECT_NAME = "Perception Data Engine"
COVER_TITLE = "Improving perception models through targeted data selection"


def test_site_carries_the_project_name_and_the_v4_hero() -> None:
    """The display name (renamed 2026-09-30) and the V4 hero: the running head's
    wordmark and the cover's eyebrow carry the name, the cover's title says what
    the project investigates, and the root document's titles follow. No edition
    still says the old name. Code identifiers are deliberately not renamed."""
    header = _code(V2_SRC / "components" / "Header.tsx")
    cover = _code(V2_SRC / "components" / "Cover.tsx")
    assert f'const WORDMARK = "{PROJECT_NAME}"' in header
    assert f'const SERIES = "{PROJECT_NAME}"' in cover
    assert f'const COVER_TITLE = "{COVER_TITLE}"' in cover
    assert "TAGLINE" not in cover, "the hero states the project once, not three times"
    document = _source(WEB / "index.html")
    assert f"<title>{PROJECT_NAME} · Report edition</title>" in document
    assert f'property="og:title" content="{PROJECT_NAME} · Report edition"' in document
    stale = [
        str(path.relative_to(SITE_SRC))
        for path in SITE_SRC.rglob("*.tsx")
        if "nuScenes Data Engine" in _code(path)
    ]
    assert not stale, f"the old name is still rendered by {stale}"
    for doc in (
        WEB / "index.html",
        WEB / "v1" / "index.html",
        WEB / "public" / "v2" / "index.html",
    ):
        assert "nuScenes Data Engine" not in _source(doc), doc
