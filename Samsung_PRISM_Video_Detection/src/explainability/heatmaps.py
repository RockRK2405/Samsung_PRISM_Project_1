"""Heatmap overlay helpers — colourise a grayscale CAM and blend it onto a face crop."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


def overlay_cam(
    rgb_uint8: np.ndarray,
    cam_gray: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a jet-colored heatmap onto an RGB face crop.

    Args:
        rgb_uint8: ``(H, W, 3)`` uint8 RGB face crop.
        cam_gray: ``(H, W)`` float32 heatmap in ``[0, 1]``. If it doesn't
            match the crop's spatial size it will be resized bilinearly.
        alpha: Blending weight of the heatmap onto the RGB image.

    Returns:
        ``(H, W, 3)`` uint8 RGB overlay.
    """
    if rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 3:
        raise ValueError(f"rgb_uint8 must be HxWx3 uint8, got shape={rgb_uint8.shape} dtype={rgb_uint8.dtype}")

    h, w = rgb_uint8.shape[:2]
    if cam_gray.shape != (h, w):
        cam_gray = cv2.resize(cam_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.clip(cam_gray * 255.0, 0, 255).astype(np.uint8)
    cmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cmap_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB)

    blended = (1.0 - alpha) * rgb_uint8.astype(np.float32) + alpha * cmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_overlay(rgb_uint8: np.ndarray, cam_gray: np.ndarray, out_path: Path) -> None:
    """Overlay + write a heatmap PNG to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_cam(rgb_uint8, cam_gray)).save(out_path, format="PNG")
