"""Video validation + metadata probing (OpenCV). Model-independent.

Handles: extension/size validation, corrupt-file detection, and probing
duration / resolution / fps / codec / frame count without loading the
whole video into memory (requirement 19).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from backend.config_loader import get_config

logger = logging.getLogger(__name__)


class VideoValidationError(Exception):
    """Raised for an unusable upload (bad extension, too big, corrupt)."""


def validate_upload(filename: str, size_bytes: int) -> None:
    """Check extension + size before we bother saving/decoding."""
    cfg = get_config()["uploads"]
    ext = Path(filename).suffix.lower()
    if ext not in cfg["allowed_extensions"]:
        raise VideoValidationError(
            f"Unsupported format '{ext}'. Allowed: {', '.join(cfg['allowed_extensions'])}"
        )
    max_bytes = int(cfg["max_size_mb"]) * 1024 * 1024
    if size_bytes > max_bytes:
        raise VideoValidationError(
            f"File too large ({size_bytes / 1e6:.1f} MB). Max: {cfg['max_size_mb']} MB"
        )


def probe_metadata(video_path: str | Path) -> dict[str, Any]:
    """Return video metadata via OpenCV. Raises VideoValidationError if unreadable."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoValidationError(f"Could not open/decode video: {video_path.name}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        codec = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0))

        # Sanity: a valid video should read at least one frame.
        ok, _ = cap.read()
        if not ok:
            raise VideoValidationError(f"Video has no readable frames: {video_path.name}")

        size_bytes = video_path.stat().st_size if video_path.is_file() else 0
        return {
            "filename": video_path.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1e6, 2),
            "duration_sec": round(duration, 2),
            "resolution": f"{width}x{height}" if width and height else "unknown",
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "codec": codec,
            "frame_count": frame_count,
        }
    finally:
        cap.release()


def add_frame_timestamps(frame_predictions: list[dict], duration_sec: float) -> list[dict]:
    """Fill in approximate timestamps for uniformly-sampled frames.

    The model samples N frames uniformly across the clip but does not
    store true timestamps, so we derive them as (i + 0.5)/N * duration.
    Marked approximate ("timestamp_approx") so the UI can label them "≈".
    """
    n = len(frame_predictions)
    if n == 0 or duration_sec <= 0:
        for fp in frame_predictions:
            fp["timestamp_approx"] = None
        return frame_predictions
    for i, fp in enumerate(frame_predictions):
        fp["timestamp_approx"] = round((i + 0.5) / n * duration_sec, 2)
    return frame_predictions


def _fourcc_to_str(fourcc: int) -> str:
    if not fourcc:
        return "unknown"
    try:
        chars = [chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)]
        s = "".join(c for c in chars if c.isprintable()).strip()
        return s or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"
