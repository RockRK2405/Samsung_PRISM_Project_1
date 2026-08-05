"""Experiment history, dashboard summary, model info, and export routes."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.database import db
from backend.services import model_service

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/dashboard/summary")
def dashboard_summary():
    """Summary cards + recent results + model status."""
    status = model_service.model_status()
    return {
        "summary": db.summary_stats(),
        "model_version": status["version"],
        "mock": status["mock"],
        "recent": db.list_experiments(limit=10),
    }


@router.get("/model/status")
def model_info():
    """Full model information panel."""
    return model_service.model_status()


@router.get("/experiments")
def experiments(limit: int | None = None):
    return {"experiments": db.list_experiments(limit=limit)}


@router.get("/experiments/{exp_id}")
def experiment_detail(exp_id: str):
    row = db.get_experiment(exp_id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return row


@router.delete("/experiments/{exp_id}")
def experiment_delete(exp_id: str):
    if not db.delete_experiment(exp_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"deleted": exp_id}


@router.get("/experiments/{exp_id}/export.json")
def export_json(exp_id: str):
    row = db.get_experiment(exp_id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    buf = io.BytesIO(json.dumps(row, indent=2).encode())
    return StreamingResponse(
        buf, media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}.json"'},
    )


@router.get("/experiments/export/all.csv")
def export_all_csv():
    rows = db.list_experiments()
    buf = io.StringIO()
    fields = [
        "id", "created_at", "filename", "ground_truth", "generator",
        "prediction", "confidence", "synthetic_probability", "real_probability",
        "threshold", "processing_time", "num_frames_used", "model_version", "mock",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    out = io.BytesIO(buf.getvalue().encode())
    return StreamingResponse(
        out, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="experiments.csv"'},
    )
