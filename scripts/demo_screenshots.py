#!/usr/bin/env python
"""Capture one screenshot per public-demo page (seven, including the guided tour)
for the README gallery.

MANUAL TOOL. Playwright is deliberately not a project dependency and nothing
imports this module: `docs/img/demo-*.png` is refreshed by hand whenever a page
changes visibly, never in CI and never during `demo build`.

Prerequisites
-------------
1. Playwright, installed temporarily into the project venv (`uv pip uninstall
   playwright` again afterwards -- it must not end up in pyproject.toml):

       uv pip install playwright

   The script drives the Google Chrome already installed on the machine
   (`channel="chrome"`), so `playwright install` / a downloaded browser is not
   needed.

2. The demo app, serving the committed package, in another shell:

       uv run streamlit run app/demo/main.py --server.headless true --server.port 8599

Usage
-----
       .venv/bin/python scripts/demo_screenshots.py [--base http://localhost:8599] [--out docs/img]

Each page is loaded, given time to finish its first run, and -- where the page has
a frame gallery -- has its first "View" button clicked so the detail panel is open
in the shot rather than an empty placeholder. The three pages in `NO_CLICK` have no
such gallery on their first screen; those are captured at scroll position 0 instead,
so the title is visible rather than whatever the click would have landed on. Exits
non-zero if any page rendered a Streamlit exception, so a broken page cannot quietly
become a README screenshot.

Any capture over 300 KB is quantized in place to a 256-colour palette PNG
(Pillow, already an app dependency) so the committed screenshots stay small; the
before/after size is printed when this happens.

The PNGs contain nuScenes-derived imagery and are covered by the dataset
attribution section of README.md / docs/DEMO.md.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:  # playwright is not installed in the project venv by default
    from playwright.sync_api import Page

# (url_path, output stem). "" is the default page (Overview); the rest are the
# explicit `url_path=` values each st.Page carries in app/demo/main.py.
PAGES: tuple[tuple[str, str], ...] = (
    ("", "demo-overview"),
    ("tour", "demo-tour"),
    ("failures", "demo-failures"),
    ("scenarios", "demo-scenarios"),
    ("active_learning", "demo-active-learning"),
    ("weak_supervision", "demo-weak-supervision"),
    ("chat_replay", "demo-chat-replay"),
)

# Pages with no frame gallery on their first screen: skip the "View" click and
# capture at scroll position 0 (title visible) instead of clicking blind.
NO_CLICK: frozenset[str] = frozenset({"tour", "active_learning", "weak_supervision"})

VIEWPORT = {"width": 1200, "height": 900}
STATUS_WIDGET = '[data-testid="stStatusWidget"]'  # Streamlit's "Running..." indicator
EXCEPTION_MARKER = '[data-testid="stException"]'
FIRST_RUN_TIMEOUT_MS = 30_000
SETTLE_MS = 2_000
MAX_PNG_BYTES = 300_000


def _compress_if_large(target: Path) -> None:
    """Palette-quantize a screenshot in place if it is over MAX_PNG_BYTES."""
    before = target.stat().st_size
    if before <= MAX_PNG_BYTES:
        return
    Image.open(target).convert("RGB").quantize(
        colors=256,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG,
    ).save(target, optimize=True)
    after = target.stat().st_size
    print(f"  compressed {target.name}: {before / 1024:.0f} KB -> {after / 1024:.0f} KB")


def _capture(page: Page, url: str, target: Path, *, click_view: bool) -> bool:
    """Screenshot one demo page's viewport. Returns False if the page raised."""
    page.goto(url)
    # The status widget is absent entirely on a page that finishes before it is
    # drawn, so a timeout here is a normal outcome, not a failure.
    with contextlib.suppress(Exception):
        page.wait_for_selector(STATUS_WIDGET, state="detached", timeout=FIRST_RUN_TIMEOUT_MS)
    page.wait_for_timeout(SETTLE_MS)
    if click_view:
        view = page.locator('button:has-text("View")').first
        if view.count():
            view.click()
            page.wait_for_timeout(SETTLE_MS)
    else:
        page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(target))
    _compress_if_large(target)
    return page.locator(EXCEPTION_MARKER).count() == 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screenshot the public demo's pages.")
    parser.add_argument(
        "--base", default="http://localhost:8599", help="Base URL of a running demo app."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/img"), help="Directory for the PNGs."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. This script is a manual tool, not a project "
            "dependency:\n    uv pip install playwright\n(and uninstall it afterwards).",
            file=sys.stderr,
        )
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(args.base).rstrip("/")
    broken: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()
        for url_path, stem in PAGES:
            target = out_dir / f"{stem}.png"
            clean = _capture(
                page, f"{base}/{url_path}", target, click_view=url_path not in NO_CLICK
            )
            size_kb = target.stat().st_size / 1024
            note = "" if clean else "   ** Streamlit exception on this page **"
            print(f"{target}  {size_kb:.0f} KB{note}")
            if not clean:
                broken.append(url_path or "(overview)")
        browser.close()

    if broken:
        print(f"pages rendered a Streamlit exception: {', '.join(broken)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
