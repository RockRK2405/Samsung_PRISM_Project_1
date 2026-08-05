# =====================================================================
# Video Detection Dashboard — container image
# =====================================================================
# Build context is the REPO ROOT (see docker-compose.yml) so this can read
# the video module's requirements at build time. The video module CODE +
# CHECKPOINT are mounted at runtime (they're local-only, not baked in).
#
# NOTE: containers run Linux, so there is NO Apple MPS here — the model
# runs on CPU (config device:null auto-resolves to cpu). Slower than
# native, but zero-setup.
# =====================================================================

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY Samsung_PRISM_Video_Dashboard/frontend/package*.json ./
RUN npm ci
COPY Samsung_PRISM_Video_Dashboard/frontend/ ./
RUN npm run build          # -> /fe/dist

# ---- Stage 2: python backend + model deps ----
FROM python:3.11-slim

# System libs the model stack needs:
#   ffmpeg  -> PyAV / video decoding
#   libgl1, libglib2.0-0, libsm6, libxext6 -> opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Samsung_PRISM_Video_Dashboard

# Install deps in two layers so they cache well.
# 1) The video module's model stack (torch/torchvision/timm/facenet/etc.)
COPY Samsung_PRISM_Video_Detection/requirements.txt /tmp/video-req.txt
RUN pip install --no-cache-dir -r /tmp/video-req.txt
# 2) The dashboard's own web deps
COPY Samsung_PRISM_Video_Dashboard/requirements.txt /tmp/dash-req.txt
RUN pip install --no-cache-dir -r /tmp/dash-req.txt

# Dashboard source + config
COPY Samsung_PRISM_Video_Dashboard/backend ./backend
COPY Samsung_PRISM_Video_Dashboard/config ./config

# Built frontend from stage 1 (FastAPI serves it at :8000)
COPY --from=frontend /fe/dist ./frontend/dist

# PYTHONPATH so `backend` and (mounted) `src` import cleanly.
ENV PYTHONPATH=/app/Samsung_PRISM_Video_Dashboard

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
