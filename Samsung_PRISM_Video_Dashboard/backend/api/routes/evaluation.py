"""Evaluation + threshold-testing routes. All computed from stored rows."""

from __future__ import annotations

from fastapi import APIRouter

from backend.database import db
from backend.services import evaluation_service

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("")
def evaluation(threshold: float | None = None):
    """Full evaluation at the stored threshold, or a candidate one."""
    rows = db.list_for_evaluation()
    return {
        "metrics": evaluation_service.metrics(rows, threshold=threshold),
        "per_generator": evaluation_service.per_generator(rows, threshold=threshold),
        "score_distributions": evaluation_service.score_distributions(rows),
        "threshold_used": threshold,
    }


@router.get("/threshold-sweep")
def threshold_sweep(values: str = "0.40,0.45,0.50,0.55,0.60"):
    """Recompute metrics across candidate thresholds (no model reruns)."""
    try:
        thresholds = [float(v) for v in values.split(",") if v.strip()]
    except ValueError:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    rows = db.list_for_evaluation()
    return {"sweep": evaluation_service.threshold_sweep(rows, thresholds)}
