"""Neural network architectures for the video detection module."""

from src.models.baseline import BaselineDetector, build_baseline_from_config

__all__ = ["BaselineDetector", "build_baseline_from_config"]
