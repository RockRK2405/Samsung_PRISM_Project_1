"""Single-video prediction API — the fusion engine's entrypoint."""

from src.inference.predictor import DetectionResult, VideoDetector

__all__ = ["DetectionResult", "VideoDetector"]
