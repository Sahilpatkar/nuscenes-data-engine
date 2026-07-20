# Serving image (Phase 4): FastAPI detection API + Streamlit demo (same image, the
# compose service picks the command). Build/run on the INFRA MACHINE.
#
# The full repo is copied and installed editable via `uv sync` (NOT a wheel install):
# training/runtime.py anchors REPO_ROOT/.cache/weights/mlruns at parents[3] of its own
# path, which only resolves correctly with the project laid out under /app.
#
# The linux torch wheel in uv.lock is the CUDA build, so the image is multi-GB but runs
# fine on CPU; a slim CPU-only torch index would conflict with --frozen and is a
# deliberate non-goal this phase.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# opencv-python (a base dep) needs these shared libs on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev --extra serve --extra train
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "nuscenes_data_engine.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
