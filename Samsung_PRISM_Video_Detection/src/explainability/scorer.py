"""Explainability quality metric.

The worklet target is **Explainability Score ≥ 85%**. We define it as:

    For each frame's GradCAM heatmap, compute the *fraction of activation
    mass that falls inside the central 60% × 60% window of the crop*. A
    well-behaved detector concentrates attention on the face; a broken
    one distributes it toward the edges/background. Per-video score is
    the mean over that video's frames.

We use **75%** as the default central-window fraction because MTCNN
crops with a 20-pixel margin around a tight face box, so the face +
neck + hair fringe end up occupying ~70–80% of the 224×224 tile. A
tighter window (60%) undercounts CAM mass that legitimately falls on
the jaw / neck / hairline; a looser one (85%+) starts admitting
background artefacts near the corners.

The metric is intentionally simple and dataset-agnostic. It does *not*
require face landmarks at eval time, so it can be run on any cached
face crop without extra dependencies.
"""

from __future__ import annotations

import numpy as np


def face_localisation_score(cam: np.ndarray, central_frac: float = 0.75) -> float:
    """Fraction of ``cam`` mass inside a central rectangle.

    Args:
        cam: ``(H, W)`` float heatmap in ``[0, 1]``. Non-negative expected.
        central_frac: Side length of the central window, as a fraction of
            ``H`` and ``W``. Defaults to ``0.6``.

    Returns:
        A score in ``[0, 1]``. Returns ``0.0`` if the CAM has no mass.
    """
    if cam.ndim != 2:
        raise ValueError(f"cam must be 2D, got shape {cam.shape}")
    h, w = cam.shape
    ch = int(round(h * central_frac))
    cw = int(round(w * central_frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    central = cam[y0 : y0 + ch, x0 : x0 + cw]
    total = float(cam.sum())
    if total <= 1e-6:
        return 0.0
    return float(central.sum() / total)


def video_explainability_score(cams: np.ndarray, central_frac: float = 0.75) -> float:
    """Mean per-frame localisation score across a video.

    Args:
        cams: ``(T, H, W)`` heatmaps for T frames.

    Returns:
        Mean score in ``[0, 1]``.
    """
    if cams.ndim != 3:
        raise ValueError(f"cams must be TxHxW, got shape {cams.shape}")
    scores = [face_localisation_score(cams[i], central_frac) for i in range(cams.shape[0])]
    return float(np.mean(scores)) if scores else 0.0
