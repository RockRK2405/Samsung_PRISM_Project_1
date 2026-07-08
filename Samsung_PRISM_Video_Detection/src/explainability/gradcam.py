"""GradCAM wrapper for the video-detection RGB backbone.

Provides per-frame class-activation heatmaps that the QC dashboard and
mentor report can display to answer *"why did the model call this
synthetic?"* The heatmap is computed on the CNN backbone's last block
(7×7 spatial for a 224×224 input) and upsampled bilinearly to 224×224.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from src.models import BaselineDetector, DualStreamDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _target_layer(model: nn.Module) -> nn.Module:
    """Return the last EfficientNet block to hook CAM onto.

    For dual-stream models we explain the RGB stream only — the FFT
    stream produces heatmaps in the frequency domain that don't overlay
    back onto pixel space meaningfully.
    """
    if isinstance(model, BaselineDetector):
        return model.backbone.blocks[-1]
    if isinstance(model, DualStreamDetector):
        return model.rgb_backbone.blocks[-1]
    raise ValueError(
        f"GradCAM support not yet implemented for {type(model).__name__}. "
        "Baseline and DualStream are supported; TemporalDetector uses "
        "attention weights instead — see docs/architecture/ADR-003."
    )


def _backbone_only_wrapper(model: nn.Module) -> nn.Module:
    """Return a callable that maps ``(B, 3, H, W)`` to per-frame logits.

    For :class:`BaselineDetector` and :class:`DualStreamDetector` the
    class's ``forward(images)`` already does this, so we return the model
    itself. Kept as a seam so richer architectures can wrap here later.
    """
    return model


class FrameGradCAM:
    """Compute per-frame GradCAM heatmaps for the *synthetic* class.

    Attributes:
        threshold: Decision threshold. Used to label individual frames
            when writing metadata alongside the heatmap.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        threshold: float = 0.5,
    ) -> None:
        # Grads are required; MPS has intermittent issues with pytorch-grad-cam
        # (hooks + activation-checkpointing quirks), so we default to CPU.
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.threshold = threshold
        self._cam = GradCAM(
            model=_backbone_only_wrapper(self.model),
            target_layers=[_target_layer(self.model)],
        )

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Batched GradCAM for the ``synthetic`` (class-index 1) target.

        Args:
            images: ``(B, 3, 224, 224)`` normalised float tensor.

        Returns:
            ``(B, 224, 224)`` float32 heatmap in ``[0, 1]``.
        """
        images = images.to(self.device)
        targets = [ClassifierOutputTarget(1) for _ in range(images.shape[0])]
        cam = self._cam(input_tensor=images, targets=targets)
        return np.asarray(cam, dtype=np.float32)
