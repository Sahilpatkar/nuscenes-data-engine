"""Streamlit demo UI (Phase 4).

Lets a user upload or pick a nuScenes image and view rendered detections from the
serving API. STUB — the real UI is built in Phase 4.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="nuScenes Data Engine — Demo", page_icon="🚗")
    st.title("nuScenes Data Engine — Detection Demo")
    st.info("Demo UI coming in Phase 4. Upload an image and see detections here.")
    # TODO(Phase 4): image upload / picker + call the /predict endpoint + render boxes.


if __name__ == "__main__":
    main()
