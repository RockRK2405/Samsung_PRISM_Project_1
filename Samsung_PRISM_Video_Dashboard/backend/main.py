"""FastAPI application entrypoint.

Run (dev):
    uvicorn backend.main:app --reload --port 8000
API docs at http://127.0.0.1:8000/docs

Serves the JSON API and (if built) the React frontend from frontend/dist.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config_loader import DASHBOARD_ROOT, get_config, outputs_dir
from backend.database import db
from backend.services import model_service
from backend.api.routes import detection, evaluation, experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Synthetic Media Detection — Research Testing Dashboard",
    description="Local dashboard for testing the Video Deepfake Detection module.",
    version="1.0.0",
)

_cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg["server"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router)
app.include_router(experiments.router)
app.include_router(evaluation.router)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    model_service.init_model()  # loads real model or falls back to mock
    status = model_service.model_status()
    banner = "MOCK MODEL — inference model NOT connected" if status["mock"] else "REAL model connected"
    logger.info("Startup complete. %s", banner)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve GradCAM overlay images / timeline PNGs produced by the real model.
# The model writes them under the video module's outputs/explainability/.
# We expose the dashboard's own outputs dir here; the frontend requests
# explainability images by experiment via /api/experiments/{id} which
# includes the heatmap_dir path.
app.mount("/outputs", StaticFiles(directory=str(outputs_dir())), name="outputs")

# Serve the built React frontend if present (production single-command run).
_frontend_dist = DASHBOARD_ROOT / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = _frontend_dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_frontend_dist / "index.html")
