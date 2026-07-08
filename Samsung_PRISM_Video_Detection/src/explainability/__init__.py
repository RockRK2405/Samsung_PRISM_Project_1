"""GradCAM + per-frame timeline + explainability scoring."""

from src.explainability.gradcam import FrameGradCAM
from src.explainability.heatmaps import overlay_cam, save_overlay
from src.explainability.scorer import face_localisation_score, video_explainability_score
from src.explainability.timeline import plot_timeline

__all__ = [
    "FrameGradCAM",
    "face_localisation_score",
    "overlay_cam",
    "plot_timeline",
    "save_overlay",
    "video_explainability_score",
]
