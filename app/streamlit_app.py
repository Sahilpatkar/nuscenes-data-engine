"""Streamlit demo UI: detection (Phase 4) + semantic scene search (Phase 6a).

A plain HTTP client of the FastAPI service — it never loads models itself, so
`docker compose up api streamlit` demonstrates the real topology. Each tab degrades
independently when its backend piece (detector, search index) is unavailable.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAMPLES_DIR = Path(__file__).parent / "samples"


def api_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).json()
    except requests.RequestException:
        return None


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x_min, y_min, x_max, y_max = det["bbox"]
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=3)
        draw.text((x_min + 3, y_min + 3), f"{det['label']} {det['confidence']:.2f}", fill="red")
    return annotated


def render_detect(health: dict) -> None:
    if not health.get("model_loaded"):
        st.warning("No detection model loaded (see the API logs) — search may still work.")
        return
    st.caption(f"Serving model version `{health['model_version']}`")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    picked = None
    if samples and uploaded is None:
        choice = st.selectbox("…or pick a bundled sample", ["—", *[s.name for s in samples]])
        if choice != "—":
            picked = SAMPLES_DIR / choice

    if uploaded is not None:
        name, data = uploaded.name, uploaded.getvalue()
    elif picked is not None:
        name, data = picked.name, picked.read_bytes()
    else:
        st.info("Upload an image (or pick a sample) to run detection.")
        return

    server_rendered = st.toggle("Server-rendered annotation", value=False)
    endpoint = "/predict/annotated" if server_rendered else "/predict"
    resp = requests.post(
        f"{API_URL}{endpoint}", files={"file": (name, data, "image/jpeg")}, timeout=60
    )
    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text}")
        return

    if server_rendered:
        st.image(resp.content, caption=name, use_container_width=True)
    else:
        body = resp.json()
        st.image(
            draw_detections(Image.open(BytesIO(data)), body["detections"]),
            caption=f"{name} — {body['n_detections']} detections",
            use_container_width=True,
        )
        if body["detections"]:
            st.dataframe(pd.DataFrame(body["detections"]), use_container_width=True)
        else:
            st.info("No detections above the confidence threshold.")


def _show_results(resp: requests.Response) -> None:
    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text}")
        return
    results = resp.json()["results"]
    if not results:
        st.info("No results.")
        return
    columns = st.columns(3)
    for i, row in enumerate(results):
        with columns[i % 3]:
            st.image(base64.b64decode(row["thumbnail_b64"]), use_container_width=True)
            conditions = ("night" if row["is_night"] else "day") + (
                ", rain" if row["is_rain"] else ""
            )
            st.caption(
                f"**{row['scene_name']}** · {row['channel']} · {conditions} · "
                f"score {row['score']:.3f}\n\n{row['scene_description'][:80]}"
            )
            st.button(
                "Find similar",
                key=f"similar_{i}_{row['sample_data_token']}",
                on_click=lambda tok=row["sample_data_token"]: st.session_state.update(
                    similar_token=tok
                ),
            )


def render_search(health: dict) -> None:
    if not health.get("search_ready"):
        st.warning(
            "Search index unavailable — build it with `nuscenes-data-engine embed` "
            "on the GPU server and rsync `data/lancedb/` here."
        )
        return

    k = st.slider("Results", min_value=3, max_value=24, value=9, step=3)

    if st.session_state.get("similar_token"):
        token = st.session_state["similar_token"]
        st.caption(f"Frames similar to `{token}`")
        if st.button("Clear similar-search"):
            st.session_state["similar_token"] = None
            st.rerun()
        else:
            _show_results(requests.get(f"{API_URL}/search/similar/{token}?k={k}", timeout=60))
            return

    query = st.text_input(
        "Describe a scene", placeholder="construction zone at night · pedestrian crossing in rain"
    )
    example = st.file_uploader(
        "…or search by example image", type=["jpg", "jpeg", "png"], key="search_upload"
    )
    if example is not None:
        _show_results(
            requests.post(
                f"{API_URL}/search/image?k={k}",
                files={"file": (example.name, example.getvalue(), "image/jpeg")},
                timeout=120,
            )
        )
    elif query:
        _show_results(requests.get(f"{API_URL}/search", params={"q": query, "k": k}, timeout=120))
    else:
        st.info("Type a scene description or upload an example image.")


def main() -> None:
    st.set_page_config(page_title="nuScenes Data Engine — Demo", page_icon="🚗", layout="wide")
    st.title("nuScenes Data Engine")

    health = api_health()
    if health is None:
        st.error(f"Serving API unreachable at {API_URL} — start it with `make serve`.")
        st.stop()

    detect_tab, search_tab = st.tabs(["🔍 Detect", "🗂 Scene search"])
    with detect_tab:
        render_detect(health)
    with search_tab:
        render_search(health)


if __name__ == "__main__":
    main()
