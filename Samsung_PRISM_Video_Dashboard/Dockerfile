# =====================================================================
# Video Detection Dashboard — container image
# =====================================================================
# Build context is the REPO ROOT (see docker-compose.yml). The image bakes
# in a LEAN, CPU-only dep set (requirements-docker.txt — only what inference
# imports); the video module CODE + CHECKPOINT are mounted at runtime
# (they're local-only, not baked in).
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

# System libs opencv/ffmpeg need at runtime (headless opencv keeps this small).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Samsung_PRISM_Video_Dashboard

# Be patient/retry on flaky networks (the big torch wheel otherwise times out).
ENV PIP_DEFAULT_TIMEOUT=180

# 1) CPU-only torch + torchvision from the PyTorch CPU index. Small wheels,
#    no CUDA packages, no version backtracking. facenet-pytorch pins
#    torch<2.3, so use the same 2.2.2 / 0.17.2 pair as the native venv.
RUN pip install --no-cache-dir --retries 5 \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else (lean list — only what inference imports + web layer).
COPY Samsung_PRISM_Video_Dashboard/requirements-docker.txt /tmp/req.txt
RUN pip install --no-cache-dir --retries 5 -r /tmp/req.txt

# Dashboard source + config
COPY Samsung_PRISM_Video_Dashboard/backend ./backend
COPY Samsung_PRISM_Video_Dashboard/config ./config

# Built frontend from stage 1 (FastAPI serves it at :8000)
COPY --from=frontend /fe/dist ./frontend/dist

# PYTHONPATH so `backend` and (mounted) `src` import cleanly.
ENV PYTHONPATH=/app/Samsung_PRISM_Video_Dashboard

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
