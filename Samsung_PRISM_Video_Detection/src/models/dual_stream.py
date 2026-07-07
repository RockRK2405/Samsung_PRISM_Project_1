"""Dual-stream detector — RGB + frequency-domain streams share the classifier.

Motivation
----------
GAN-based generators leave characteristic fingerprints in the
frequency spectrum that are invisible to the eye but detectable by a
CNN operating on the FFT log-magnitude of the same face crop
(F3-Net, SPSL, and follow-up papers). Crucially these fingerprints
tend to persist even when the RGB output has been polished to look
correct — which is exactly the failure mode Milestone 5B exposed on
Celeb-DF. The bet: adding a frequency stream lifts cross-dataset AUC.

Architecture
------------
* **RGB stream** — EfficientNet-B0 backbone, ImageNet-pretrained,
  optionally warm-started from Milestone 4's ``best.pt`` and frozen.
* **Frequency stream** — a *separate* EfficientNet-B0 backbone
  (ImageNet init, no shared weights) whose input is the
  ``fftshift(log(1 + |FFT2(x)|))`` of the RGB crop, z-scored
  per image per channel to keep the input distribution in a range
  the ImageNet-pretrained convs can handle.
* **Classifier** — LayerNorm + dropout + linear on the concatenated
  (RGB, FFT) feature vector.
* **Temporal aggregation** — mean-pool over frames, matching the
  Milestone 4 baseline (5A confirmed a heavier temporal head is not
  worth the training budget on FF++ c23).

Training strategy (locked in ``configs/model.yaml``)
----------------------------------------------------
* RGB backbone: **loaded from ``checkpoints/best.pt`` and frozen** —
  no need to re-learn what the baseline already knows.
* FFT backbone + classifier: trained end-to-end.

This keeps the trainable parameter count around 4M and makes the
experiment a clean test of "does the frequency stream add signal on
top of the frozen RGB baseline."
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DualStreamDetector(nn.Module):
    """Two backbones (RGB + FFT), concatenated features, linear classifier.

    Attributes:
        num_classes: Fixed at ``2`` (real / synthetic).
        feat_dim: Per-stream feature dimension (from the backbone).
    """

    num_classes: int = 2

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_rgb: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_rgb = freeze_rgb

        self.rgb_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.freq_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.feat_dim = int(self.rgb_backbone.num_features)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim * 2, self.num_classes),
        )

        if freeze_rgb:
            for p in self.rgb_backbone.parameters():
                p.requires_grad = False
            self.rgb_backbone.eval()   # freeze BN stats too

        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            "DualStreamDetector: backbone=%s feat_dim=%d params=%.2fM (trainable=%.2fM, rgb_frozen=%s)",
            backbone_name, self.feat_dim, total, trainable, freeze_rgb,
        )

    # ------------------------------------------------------------------
    # FFT preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _log_fft_magnitude(images: torch.Tensor) -> torch.Tensor:
        """Compute ``fftshift(log(1 + |FFT2(x)|))`` per image per channel, z-scored.

        Args:
            images: ``(B, 3, H, W)`` float tensor. Input is already
                ImageNet-normalised — the FFT is linear so that's fine.

        Returns:
            Same-shape float tensor ready to feed to an ImageNet-pretrained CNN.
        """
        fft = torch.fft.fft2(images.float())
        mag = torch.abs(fft)
        log_mag = torch.log1p(mag)
        centered = torch.fft.fftshift(log_mag, dim=(-2, -1))
        # Per-image, per-channel z-score keeps the input distribution roughly in the
        # range an ImageNet-pretrained backbone was optimised for.
        mean = centered.mean(dim=(-1, -2), keepdim=True)
        std = centered.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return (centered - mean) / std

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Per-frame forward.

        Args:
            images: ``(B, 3, 224, 224)`` float tensor.

        Returns:
            Logits ``(B, num_classes)``.
        """
        if self.freeze_rgb:
            # No grads on the RGB stream when frozen — saves memory and a bit of time.
            with torch.no_grad():
                rgb_feat = self.rgb_backbone(images)
        else:
            rgb_feat = self.rgb_backbone(images)

        freq_input = self._log_fft_magnitude(images)
        freq_feat = self.freq_backbone(freq_input)

        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        return self.classifier(combined)

    def forward_video(self, video_stack: torch.Tensor) -> torch.Tensor:
        """Video-level forward — mean-pool over frames.

        Args:
            video_stack: ``(B, T, 3, 224, 224)``.

        Returns:
            Video-level logits ``(B, num_classes)``.
        """
        b, t, c, h, w = video_stack.shape
        flat = video_stack.reshape(b * t, c, h, w)
        frame_logits = self.forward(flat)
        return frame_logits.reshape(b, t, -1).mean(dim=1)


# ---------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------

def build_dual_stream_from_config(model_cfg: dict) -> DualStreamDetector:
    """Instantiate :class:`DualStreamDetector` from ``configs/model.yaml``."""
    spatial = model_cfg.get("spatial_stream") or {}
    return DualStreamDetector(
        backbone_name=str(model_cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float((model_cfg.get("classifier") or {}).get("dropout", 0.2)),
        freeze_rgb=bool(spatial.get("freeze", False)),
    )


# ---------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------

def warm_start_rgb_from_baseline(
    model: DualStreamDetector, checkpoint_path: Path
) -> None:
    """Copy a :class:`~src.models.BaselineDetector`'s backbone into the RGB stream.

    The baseline's ``state_dict`` uses ``backbone.*`` keys (plus a two-class
    ``backbone.classifier.*`` head we throw away). Ours uses
    ``rgb_backbone.*``. This helper renames on load so the training loop
    can start from a proven-good RGB feature extractor.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict") or {}
    if not state:
        raise ValueError(f"Checkpoint {checkpoint_path} has no state_dict")

    renamed: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not k.startswith("backbone."):
            continue
        tail = k[len("backbone."):]
        # timm creates a classifier module even at num_classes=0 (identity),
        # but the baseline used num_classes=2 so its classifier weights exist.
        # We discard them — the dual-stream detector has its own classifier
        # over concatenated features.
        if tail.startswith("classifier."):
            continue
        renamed[f"rgb_backbone.{tail}"] = v

    if not renamed:
        raise ValueError(
            f"No 'backbone.*' keys found in {checkpoint_path} — "
            "is this really a BaselineDetector checkpoint?"
        )

    result = model.load_state_dict(renamed, strict=False)
    logger.info(
        "Warm-started RGB from %s: copied %d tensors, %d missing (freq + classifier — expected), %d unexpected",
        checkpoint_path.name, len(renamed), len(result.missing_keys), len(result.unexpected_keys),
    )
