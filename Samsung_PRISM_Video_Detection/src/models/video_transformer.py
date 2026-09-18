"""Pretrained video foundation models (ADR-007 Phase 3, Parts 8-9).

Wraps HuggingFace video-classification transformers behind the SAME
interface as :class:`~src.models.video_spatial.VideoSpatialNet`:

    forward(clips) : (B, T, 3, H, W) -> logits (B, 2)

so the trainer and eval code are model-agnostic — swapping
``model.family`` in configs/video_training.yaml between ``frame_baseline``
and ``videomae`` / ``timesformer`` is the only change (Part 24 ablation).

Supported families
------------------
* ``videomae``    -> VideoMAEForVideoClassification (default 16 frames, 224)
* ``timesformer`` -> TimesformerForVideoClassification (default 8 frames, 224)

Both are true spatio-temporal transformers (joint/divided space-time
attention), so unlike the frame-baseline they model temporal artefacts
directly — the reason they're worth the extra compute (Part 8 Stage 4).

Frame count MUST match the pretrained model's expected ``num_frames``
(VideoMAE base = 16, TimeSformer k400 = 8); the loader checks and warns.

MPS note (Part 27): these run on CUDA and CPU. On Apple MPS some ops in
the HF video encoders fall back to CPU or error on older torch; if you
hit an MPS error, train these on CUDA (or CPU for a smoke test) and keep
the frame-baseline for MPS. This is documented, not silently worked around.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class VideoTransformerNet(nn.Module):
    """HF video-classification transformer with a 2-class head.

    Attributes:
        family: "videomae" or "timesformer".
        num_frames: Frames the pretrained backbone expects.
    """

    num_classes: int = 2

    def __init__(
        self,
        family: str = "videomae",
        pretrained_id: str | None = None,
        num_frames: int = 16,
        image_size: int = 224,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.family = family.lower()
        self.num_frames = int(num_frames)

        try:
            from transformers import (
                TimesformerForVideoClassification,
                VideoMAEForVideoClassification,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Phase 3 video transformers need `transformers`. "
                "Install it: pip install 'transformers>=4.40'."
            ) from exc

        if self.family == "videomae":
            pid = pretrained_id or "MCG-NJU/videomae-base"
            self.model = VideoMAEForVideoClassification.from_pretrained(
                pid, num_labels=self.num_classes, ignore_mismatched_sizes=True,
                hidden_dropout_prob=dropout,
            )
            expected = getattr(self.model.config, "num_frames", 16)
        elif self.family == "timesformer":
            pid = pretrained_id or "facebook/timesformer-base-finetuned-k400"
            self.model = TimesformerForVideoClassification.from_pretrained(
                pid, num_labels=self.num_classes, ignore_mismatched_sizes=True,
            )
            expected = getattr(self.model.config, "num_frames", 8)
        else:
            raise ValueError(f"Unknown video-transformer family: {family}")

        if int(expected) != self.num_frames:
            logger.warning(
                "%s expects num_frames=%s but config set %d. Set video.num_frames=%s "
                "to match, or the model will error/underperform.",
                pid, expected, self.num_frames, expected,
            )
        logger.info(
            "VideoTransformerNet: family=%s id=%s num_frames=%d params=%.1fM",
            self.family, pid, self.num_frames, sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze everything except the classification head (Part 26)."""
        for name, p in self.model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = trainable

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> logits (B, 2). HF video models take pixel_values
        in exactly this (B, T, C, H, W) layout."""
        return self.model(pixel_values=clips).logits


def build_video_transformer_from_config(cfg: dict) -> VideoTransformerNet:
    m = cfg.get("model", {})
    v = cfg.get("video", {})
    vt = m.get("video_transformer", {}) or {}
    return VideoTransformerNet(
        family=str(m.get("family", "videomae")),
        pretrained_id=vt.get("pretrained_id"),
        num_frames=int(v.get("num_frames", 16)),
        image_size=int(v.get("image_size", 224)),
        dropout=float(m.get("dropout", 0.2)),
    )
