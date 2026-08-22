"""Public demo entrypoint — Streamlit Cloud runs this file.

The app reads only the committed demo_data/ package; it never talks to the
FastAPI service, the databases, or src/nuscenes_data_engine.
"""

from __future__ import annotations

import nav
import streamlit as st
from views import (
    active_learning,
    chat_replay,
    failures,
    overview,
    scenarios,
    tour,
    weak_supervision,
)

from data import package_missing

st.set_page_config(page_title="nuScenes Perception Data Engine", layout="wide")

if package_missing():
    st.error(
        "demo_data/ is missing. Build it first:\n\n"
        "```\nuv run nuscenes-data-engine demo build\n```"
    )
    st.stop()

overview_page = st.Page(overview.render, title="Overview", icon=":material/home:", default=True)
# Phase 9a: the guided tour, the app's default 2-3 minute path through the loop.
# url_path="tour" for the same reason every page below pins one -- and specifically
# because tests/test_demo_app.py opens the tour with switch_page("views/tour.py")
# (an in-script st.switch_page is not sticky across AppTest's at.run()).
tour_page = st.Page(
    tour.render, title="Guided tour", icon=":material/explore:", url_path="tour"
)
loop_pages = [
    # url_path is pinned to the module's filename stem ("failures", from
    # views/failures.py) rather than left to the title-derived default: the
    # AppTest smoke in tests/test_demo_app.py switches pages via
    # switch_page("views/failures.py"), which resolves the target page by
    # hashing that same filename-derived name against each page's url_path hash.
    st.Page(failures.render, title="Failure Explorer", icon=":material/search:", url_path="failures"),
    # url_path="scenarios" (same rationale as the Failure Explorer's own explicit
    # url_path above): the AppTest smokes in tests/test_demo_app.py navigate via
    # switch_page("views/scenarios.py"), which resolves the target page by hashing
    # that filename-derived name against each st.Page's url_path hash.
    st.Page(scenarios.render, title="Scenario Search", icon=":material/route:", url_path="scenarios"),
    # url_path="active_learning" (same rationale as the two pages above): the
    # Phase-7 AppTest smokes navigate via switch_page("views/active_learning.py").
    st.Page(
        active_learning.render,
        title="Active Learning",
        icon=":material/model_training:",
        url_path="active_learning",
    ),
    # url_path="weak_supervision" (same rationale as the pages above): the
    # Phase-7 AppTest smokes navigate via switch_page("views/weak_supervision.py").
    st.Page(
        weak_supervision.render,
        title="Weak Supervision",
        icon=":material/fact_check:",
        url_path="weak_supervision",
    ),
]
# url_path="chat_replay" (same rationale as the pages above): the recorded-chat
# AppTest smokes navigate via switch_page("views/chat_replay.py").
chat_page = st.Page(
    chat_replay.render,
    title="Ask the Dataset",
    icon=":material/chat:",
    url_path="chat_replay",
)

# Phase 9a: the pages are registered before st.navigation runs one of them, so any
# view can deep-link to another (the tour's "Go deeper" links, the Overview's CTA)
# through nav.page() instead of importing this entrypoint.
nav.register([overview_page, tour_page, *loop_pages, chat_page])
# Sections are labels only -- they do not change how a page resolves (every
# url_path above is unchanged), they just say what the app is: start here, walk the
# loop, ask it questions.
st.navigation(
    {
        "Start here": [overview_page, tour_page],
        "Explore the loop": loop_pages,
        "Ask": [chat_page],
    }
).run()
