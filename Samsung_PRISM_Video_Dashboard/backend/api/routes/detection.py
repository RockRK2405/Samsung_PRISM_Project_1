"""Detection routes: single-video test + batch test."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config_loader import get_config, uploads_dir
from backend.database import db
from backend.services import model_service, video_service
from backend.services.video_service import VideoValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["detection"])


def _save_upload(upload: UploadFile) -> Path:
    """Stream an upload to disk (never load the whole video into memory)."""
    dest = uploads_dir() / f"{uuid.uuid4().hex[:8]}_{Path(upload.filename).name}"
    with dest.open("wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    return dest


def _run_one(path: Path, ground_truth: str | None, generator: str | None, batch_id: str | None) -> dict:
    """Probe metadata, run the model, persist, return the enriched payload."""
    metadata = video_service.probe_metadata(path)
    payload = model_service.predict_video(path)
    payload["frame_predictions"] = video_service.add_frame_timestamps(
        payload["frame_predictions"], metadata["duration_sec"]
    )
    gt = ground_truth if ground_truth in ("Real", "Synthetic", "Unknown") else None
    version = get_config()["model"]["version"]
    exp_id = db.insert_experiment(
        filename=metadata["filename"],
        payload=payload,
        metadata=metadata,
        ground_truth=gt,
        generator=generator if gt == "Synthetic" else None,
        model_version=version,
        batch_id=batch_id,
    )
    result = _evaluate_correctness(payload, gt)
    return {"experiment_id": exp_id, "metadata": metadata, "result": payload, "evaluation": result}


def _evaluate_correctness(payload: dict, ground_truth: str | None) -> dict:
    if ground_truth not in ("Real", "Synthetic"):
        return {"ground_truth": ground_truth, "correct": None}
    correct = payload["prediction"] == ground_truth
    return {
        "ground_truth": ground_truth,
        "prediction": payload["prediction"],
        "correct": correct,
        "result": "CORRECT" if correct else "INCORRECT",
    }


@router.post("/detect")
async def detect_single(
    file: UploadFile = File(...),
    ground_truth: str | None = Form(None),
    generator: str | None = Form(None),
):
    """Single-video detection. Ground-truth/generator are eval-only metadata."""
    contents_len = 0
    try:
        # Read size from the SpooledTemporaryFile without loading fully.
        file.file.seek(0, 2)
        contents_len = file.file.tell()
        file.file.seek(0)
        video_service.validate_upload(file.filename, contents_len)
    except VideoValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    path = _save_upload(file)
    try:
        return _run_one(path, ground_truth, generator, batch_id=None)
    except VideoValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("Detection failed for %s", path.name)
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


@router.post("/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    dataset_label: str | None = Form(None),   # 'Real Dataset' | 'Synthetic Dataset' | 'Mixed Dataset'
    generator: str | None = Form(None),
):
    """Batch detection. Runs sequentially; returns one row per video.

    dataset_label maps to a default ground truth per video:
      Real Dataset -> Real, Synthetic Dataset -> Synthetic, Mixed -> Unknown.
    """
    default_gt = {
        "Real Dataset": "Real",
        "Synthetic Dataset": "Synthetic",
        "Mixed Dataset": "Unknown",
    }.get(dataset_label or "", None)

    batch_id = uuid.uuid4().hex[:12]
    results = []
    errors = []
    for idx, file in enumerate(files):
        try:
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            video_service.validate_upload(file.filename, size)
            path = _save_upload(file)
            gen = generator if default_gt == "Synthetic" else None
            results.append(_run_one(path, default_gt, gen, batch_id=batch_id))
        except Exception as e:  # noqa: BLE001 - one bad file must not sink the batch
            logger.warning("Batch item %s failed: %s", file.filename, e)
            errors.append({"filename": file.filename, "error": str(e)})

    return {
        "batch_id": batch_id,
        "total": len(files),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
