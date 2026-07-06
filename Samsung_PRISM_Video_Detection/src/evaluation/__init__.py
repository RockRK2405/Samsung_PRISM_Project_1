"""Metrics + evaluation harness for the video detection module."""

from src.evaluation.metrics import (
    ClassificationReport,
    compute_report,
    find_threshold_for_fpr,
)

__all__ = ["ClassificationReport", "compute_report", "find_threshold_for_fpr"]
