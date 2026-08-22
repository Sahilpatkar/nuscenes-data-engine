"""The demo's page registry (Phase 9a, spec §1).

``main.py`` builds the ``st.Page`` objects and hands them here once per run; any
view can then deep-link to another page (``st.page_link(nav.page("failures"))``,
``st.switch_page(nav.page("tour"))``) without importing ``main`` -- which it cannot
do anyway, since ``main.py`` is the Streamlit entrypoint and importing it would
re-execute the whole app.

Streamlit is the only import: this module holds page objects (and the one
navigation-adjacent session-state key below), never package data.
"""

from __future__ import annotations

from collections.abc import Iterable

from streamlit.navigation.page import StreamlitPage

# The default page's ``url_path`` property is "" (streamlit reserves the root URL
# for it), so it is keyed under the name every other page uses -- its own filename
# stem -- and callers ask for "overview" like they ask for "failures".
_DEFAULT_PAGE_KEY = "overview"

# Phase 9a (Task 5): the tour's step-index session-state key, shared by
# ``views/tour.py`` (which owns it) and the Overview page's CTA (which resets it to
# 0 before switching pages). Lives here, not in ``views/tour.py``, so the Overview
# module can read it without a views -> views import: ``tour.py`` already imports
# this module, and an ``overview.py`` -> ``tour.py`` import the other way would
# cycle back (``tour.py`` also imports ``views.overview`` for ``HERO_CAPTION``).
TOUR_STEP_KEY = "tour_step"

_REGISTRY: dict[str, StreamlitPage] = {}


def register(pages: Iterable[StreamlitPage]) -> None:
    """Replace the registry with ``pages``, keyed by ``url_path``.

    Cleared first, not merged: ``main.py`` re-runs top to bottom on every rerun and
    builds FRESH ``st.Page`` objects each time, so a merged registry would keep
    handing out stale ones (streamlit only ordains the current run's page objects as
    runnable -- ``StreamlitPage._can_be_called``).
    """
    _REGISTRY.clear()
    for page_object in pages:
        _REGISTRY[page_object.url_path or _DEFAULT_PAGE_KEY] = page_object


def page(url_path: str) -> StreamlitPage:
    """The registered page for ``url_path`` ("overview" for the default page).

    An unregistered path raises ``ValueError`` naming it and what IS registered: a
    deep link whose target was renamed must fail loudly where it is written, rather
    than rendering a link that goes nowhere.
    """
    try:
        return _REGISTRY[url_path]
    except KeyError:
        raise ValueError(
            f"nav.page: no page registered for {url_path!r} — registered: {sorted(_REGISTRY)}"
        ) from None
