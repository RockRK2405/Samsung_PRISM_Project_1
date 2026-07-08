"""Single-video prediction API — the fusion engine's entrypoint."""

from src.inference.api import (
    MODALITY,
    SCHEMA_VERSION,
    DetectionResult,
    VideoDetector,
    predict,
)

__all__ = [
    "MODALITY",
    "SCHEMA_VERSION",
    "DetectionResult",
    "VideoDetector",
    "predict",
]
