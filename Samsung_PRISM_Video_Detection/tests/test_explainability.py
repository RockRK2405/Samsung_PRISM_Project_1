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


def test_localisation_score_mass_in_centre_scores_high() -> None:
    """A heatmap that lights up only the centre should score ~ 1.0."""
    h = np.zeros((224, 224), dtype=np.float32)
    h[80:144, 80:144] = 1.0             # a small central square, well within 60%
    assert face_localisation_score(h) >= 0.95


def test_localisation_score_mass_in_corners_scores_low() -> None:
    """A heatmap that lights up only the corners should score ~ 0.0."""
    h = np.zeros((224, 224), dtype=np.float32)
    h[0:32, 0:32] = 1.0
    h[-32:, -32:] = 1.0
    assert face_localisation_score(h) <= 0.05


def test_localisation_score_uniform_scores_36_percent() -> None:
    """A uniform CAM's central-fraction score should equal the area ratio (0.6² ≈ 0.36)."""
    h = np.ones((224, 224), dtype=np.float32)
    assert abs(face_localisation_score(h) - 0.36) < 0.01


def test_video_score_averages_across_frames() -> None:
    cams = np.stack([
        np.zeros((224, 224), dtype=np.float32),           # score 0.0
        np.ones((224, 224), dtype=np.float32),            # score ~0.36
    ])
    assert abs(video_explainability_score(cams) - 0.18) < 0.02


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
