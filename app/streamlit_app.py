"""Streamlit demo UI (Phase 4).

Upload an image (or pick a bundled sample from app/samples/) and view detections from
the serving API. The app is a plain HTTP client of the FastAPI service — it never loads
the model itself — so `docker compose up api streamlit` demonstrates the real topology.
"""

from __future__ import annotations

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


def main() -> None:
    st.set_page_config(page_title="nuScenes Data Engine — Demo", page_icon="🚗")
    st.title("nuScenes Data Engine — Detection Demo")

    health = api_health()
    if health is None:
        st.error(f"Serving API unreachable at {API_URL} — start it with `make serve`.")
        st.stop()
    if not health.get("model_loaded"):
        st.error("The API is up but no model is loaded (see the API logs).")
        st.stop()
    st.caption(f"Serving model version `{health['model_version']}` at {API_URL}")

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
        st.stop()

    server_rendered = st.toggle("Server-rendered annotation", value=False)
    endpoint = "/predict/annotated" if server_rendered else "/predict"
    resp = requests.post(
        f"{API_URL}{endpoint}", files={"file": (name, data, "image/jpeg")}, timeout=60
    )
    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text}")
        st.stop()

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


if __name__ == "__main__":
    main()
