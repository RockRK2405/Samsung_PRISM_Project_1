"""Baseline video-deepfake detector — EfficientNet-B0 + mean-pool.

Architecture
------------
For each of the 32 face crops of a video we run an ImageNet-pretrained
EfficientNet-B0 backbone with a 2-class classifier head. At video level
we mean-pool the 32 per-frame logits, softmax the result, and read off
``prob_synthetic``.

Why this baseline
-----------------
* **Small (~5M params)** — trains comfortably on a laptop MPS in a few
  epochs per hour.
* **ImageNet-pretrained** — huge head start; we are fine-tuning, not
  training from scratch.
* **Mirrors the team's Image module choice** (EfficientNet-B0) so the
  overall project has one consistent story for its per-modality backbone.
* **Mean-pool aggregation** is the dumbest possible temporal head. If
  the baseline beats the ≥0.92 F1 target with mean-pool alone we know
  the per-frame signal is strong; if it does not, we know Milestone 5
  (LSTM / attention temporal head) is buying us the missing accuracy —
  which is exactly the isolated experiment we want.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineDetector(nn.Module):
    """Per-frame EfficientNet-B0 with a mean-pool video head.

    Attributes:
        backbone_name: ``timm`` model identifier.
        num_classes: Fixed at ``2`` (real / synthetic).
    """

    num_classes: int = 2

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
            drop_rate=dropout,
        )
        logger.info(
            "BaselineDetector built with %s (pretrained=%s, params=%.2fM)",
            backbone_name,
            pretrained,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Per-frame forward — used during training.

        Args:
            images: Batch of face crops, shape ``(B, 3, 224, 224)``.

        Returns:
            Logits, shape ``(B, num_classes)``.
        """
        return self.backbone(images)

    def forward_video(self, video_stack: torch.Tensor) -> torch.Tensor:
        """Video-level forward with mean-pool over frames — used at eval.

        Args:
            video_stack: Stack of ``T`` face crops for one or more videos,
                shape ``(B, T, 3, 224, 224)``.

        Returns:
            Video-level logits, shape ``(B, num_classes)``, obtained by
            averaging the per-frame logits along the ``T`` dimension.
        """
        b, t, c, h, w = video_stack.shape
        flat = video_stack.reshape(b * t, c, h, w)
        frame_logits = self.backbone(flat)                   # (B*T, 2)
        return frame_logits.reshape(b, t, -1).mean(dim=1)   # (B, 2)


def build_baseline_from_config(model_cfg: dict) -> BaselineDetector:
    """Instantiate :class:`BaselineDetector` from a ``configs/model.yaml`` dict.

    Args:
        model_cfg: The ``model`` sub-tree of ``configs/model.yaml``.

    Returns:
        A ready-to-train :class:`BaselineDetector`.
    """
    return BaselineDetector(
        backbone_name=str(model_cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("classifier", {}).get("dropout", 0.2)),
    )
