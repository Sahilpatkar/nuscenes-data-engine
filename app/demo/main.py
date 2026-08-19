"""Public demo entrypoint — Streamlit Cloud runs this file.

The app reads only the committed demo_data/ package; it never talks to the
FastAPI service, the databases, or src/nuscenes_data_engine.
"""

from __future__ import annotations

import streamlit as st
from views import failures, overview, stubs

from data import package_missing

st.set_page_config(page_title="nuScenes Perception Data Engine", layout="wide")

if package_missing():
    st.error(
        "demo_data/ is missing. Build it first:\n\n"
        "```\nuv run nuscenes-data-engine demo build\n```"
    )
    st.stop()

pages = [
    st.Page(overview.render, title="Overview", icon=":material/home:", default=True),
    # url_path is pinned to the module's filename stem ("failures", from
    # views/failures.py) rather than left to the title-derived default: the
    # AppTest smoke in tests/test_demo_app.py switches pages via
    # switch_page("views/failures.py"), which resolves the target page by
    # hashing that same filename-derived name against each page's url_path hash.
    st.Page(failures.render, title="Failure Explorer", icon=":material/search:", url_path="failures"),
    st.Page(stubs.scenarios, title="Scenario Search", icon=":material/route:"),
    st.Page(stubs.active_learning, title="Active Learning", icon=":material/model_training:"),
    st.Page(stubs.weak_supervision, title="Weak Supervision", icon=":material/fact_check:"),
    st.Page(stubs.chat_replay, title="Ask the Dataset", icon=":material/chat:"),
]
st.navigation(pages).run()
