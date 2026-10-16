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


def _last_spatial_layer(backbone: nn.Module) -> nn.Module:
    """Last layer of a timm backbone that still emits a spatial feature map.

    Different families name their top-level stage container differently --
    EfficientNet uses ``blocks``, ConvNeXt and Swin use ``stages`` -- so this
    tries the known names and then falls back to the final ``Conv2d``, which
    is a valid CAM target for any convolutional backbone. Hardcoding
    ``blocks`` used to raise ``'ConvNeXt' object has no attribute 'blocks'``
    for every non-EfficientNet model, which is every backbone the project has
    trained since EXP-005.
    """
    for attr in ("blocks", "stages"):
        container = getattr(backbone, attr, None)
        if container is not None and len(container) > 0:
            return container[-1]

    last_conv = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        logger.info("GradCAM: no blocks/stages on %s, hooking its last Conv2d.",
                    type(backbone).__name__)
        return last_conv

    raise ValueError(
        f"Cannot find a spatial layer to hook GradCAM onto in "
        f"{type(backbone).__name__}: it exposes neither 'blocks' nor 'stages' "
        f"nor any Conv2d. Transformer backbones need attention rollout "
        f"instead of CAM."
    )


def _target_layer(model: nn.Module) -> nn.Module:
    """Return the layer to hook CAM onto, whatever backbone is in use.

    For dual-stream models we explain the RGB stream only — the FFT
    stream produces heatmaps in the frequency domain that don't overlay
    back onto pixel space meaningfully.
    """
    if isinstance(model, BaselineDetector):
        return _last_spatial_layer(model.backbone)
    if isinstance(model, DualStreamDetector):
        return _last_spatial_layer(model.rgb_backbone)
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

    def __call__(
        self,
        images: torch.Tensor,
        target_classes: list[int] | None = None,
    ) -> np.ndarray:
        """Batched GradCAM for the model's decision.

        Args:
            images: ``(B, 3, 224, 224)`` normalised float tensor.
            target_classes: Optional per-frame class index to explain
                (``0`` real, ``1`` synthetic). When ``None`` (default),
                each frame is explained for its *own predicted class* —
                the honest interpretation of "why did the model decide
                this?" For a real prediction this shows what looked real;
                for a synthetic prediction it shows what looked
                synthetic. Hard-coding class 1 was a bug in the first
                cut of this code — it produced diffuse noise on
                correctly-called reals and cratered the aggregate
                explainability score.

        Returns:
            ``(B, 224, 224)`` float32 heatmap in ``[0, 1]``.
        """
        images = images.to(self.device)
        if target_classes is None:
            with torch.no_grad():
                logits = self.model(images)
                target_classes = [int(c) for c in logits.argmax(dim=-1).cpu().tolist()]
        cam_targets = [ClassifierOutputTarget(int(c)) for c in target_classes]
        cam = self._cam(input_tensor=images, targets=cam_targets)
        return np.asarray(cam, dtype=np.float32)
