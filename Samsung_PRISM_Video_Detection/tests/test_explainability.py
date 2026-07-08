"""Unit tests for the pure-python explainability helpers.

GradCAM itself needs a real backbone and heavy imports so it's covered
by a separate smoke test in scripts/predict.py, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.explainability.heatmaps import overlay_cam
from src.explainability.scorer import face_localisation_score, video_explainability_score
from src.explainability.timeline import plot_timeline


def test_localisation_score_peak_in_centre_returns_one() -> None:
    """Peak inside the central 70% window → score 1.0."""
    h = np.zeros((224, 224), dtype=np.float32)
    h[112, 112] = 1.0
    assert face_localisation_score(h) == 1.0


def test_localisation_score_peak_in_corner_returns_zero() -> None:
    """Peak in the corner → score 0.0."""
    h = np.zeros((224, 224), dtype=np.float32)
    h[3, 3] = 1.0
    assert face_localisation_score(h) == 0.0


def test_localisation_score_all_zero_returns_zero() -> None:
    """A flat-zero CAM has no meaningful peak."""
    h = np.zeros((224, 224), dtype=np.float32)
    assert face_localisation_score(h) == 0.0


def test_video_score_is_fraction_of_frames_with_peak_in_centre() -> None:
    good = np.zeros((224, 224), dtype=np.float32)
    good[100, 110] = 1.0
    bad = np.zeros((224, 224), dtype=np.float32)
    bad[5, 5] = 1.0
    cams = np.stack([good, good, good, bad])  # 3 of 4 frames pass
    assert abs(video_explainability_score(cams) - 0.75) < 1e-6


def test_overlay_cam_shape_and_dtype() -> None:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    cam = np.random.rand(224, 224).astype(np.float32)
    out = overlay_cam(rgb, cam)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_overlay_cam_resizes_when_shapes_differ() -> None:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    cam = np.random.rand(7, 7).astype(np.float32)        # backbone spatial size
    out = overlay_cam(rgb, cam)
    assert out.shape == rgb.shape


def test_timeline_writes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "timeline.png"
    plot_timeline([0.1, 0.2, 0.8, 0.9, 0.7], threshold=0.5, out_path=out, title="test")
    assert out.is_file() and out.stat().st_size > 100
