"""Temporal detector — EfficientNet-B0 features + transformer aggregator.

Motivation
----------
The mean-pool baseline (:class:`~src.models.baseline.BaselineDetector`)
treats every frame's per-frame logit independently, which is exactly
the wrong aggregator for manipulation methods whose artefacts live in
the *temporal* dimension. NeuralTextures at 94% recall was the visible
weak spot after Milestone 4 — its GAN-based reenactment produces frames
that individually look fine but flicker/jitter as a sequence.

Architecture
------------
* **Per-frame backbone** — EfficientNet-B0 (or any timm backbone) with
  the classifier stripped, so we get a pooled feature vector per frame.
* **Frame projection** — linear projection to a smaller hidden dim
  (default 256) to keep the transformer cheap.
* **CLS token + learned positional embeddings** for T frames.
* **Transformer encoder** — a few layers of standard multi-head
  self-attention over the (CLS + T frame) sequence.
* **Classifier head** — LayerNorm → dropout → linear on the CLS token.

The transformer's self-attention weights over the frame tokens are
recoverable and will be reused as an explainability signal in
Milestone 7 (they show which frames drove the decision).

Padding
-------
A handful of FF++ videos have <32 face crops because MTCNN missed a
few frames. The model accepts a boolean ``padding_mask`` (``True``
where the corresponding frame is padding) so those get zero attention
weight; short videos are still handled correctly.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TemporalDetector(nn.Module):
    """Per-frame backbone + transformer temporal head + binary classifier.

    Attributes:
        num_classes: Fixed at ``2`` (real / synthetic).
        feat_dim: Backbone output dimension (populated at construction).
        hidden_dim: Transformer working dimension.
    """

    num_classes: int = 2

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
        pretrained: bool = True,
        num_frames: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # num_classes=0 strips the classifier; forward() returns pooled features.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.feat_dim = int(self.backbone.num_features)

        self.frame_proj = nn.Linear(self.feat_dim, hidden_dim)

        # Learned CLS token (aggregates video-level info) and positional
        # embeddings for CLS + T frame slots.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # `enable_nested_tensor=False` silences a cosmetic UserWarning when
        # `norm_first=True` — the nested-tensor fast path is not compatible
        # with norm-first anyway, so we explicitly opt out.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

        logger.info(
            "TemporalDetector: backbone=%s (feat_dim=%d) hidden=%d layers=%d heads=%d params=%.2fM",
            backbone_name, self.feat_dim, hidden_dim, num_layers, num_heads,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def _frame_features(self, video_stack: torch.Tensor) -> torch.Tensor:
        """Backbone forward over ``B×T`` frames → ``(B, T, hidden_dim)``."""
        b, t, c, h, w = video_stack.shape
        flat = video_stack.reshape(b * t, c, h, w)
        feats = self.backbone(flat)                          # (B*T, feat_dim)
        feats = self.frame_proj(feats)                       # (B*T, hidden_dim)
        return feats.reshape(b, t, self.hidden_dim)

    def forward_video(
        self,
        video_stack: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Video-level forward.

        Args:
            video_stack: ``(B, T, 3, 224, 224)`` float tensor.
            padding_mask: Optional ``(B, T)`` bool tensor — ``True`` where
                the frame is padding and should be ignored by attention.

        Returns:
            Logits ``(B, 2)``.
        """
        b, t, _, _, _ = video_stack.shape
        feats = self._frame_features(video_stack)            # (B, T, hidden)

        cls = self.cls_token.expand(b, -1, -1)               # (B, 1, hidden)
        x = torch.cat([cls, feats], dim=1)                   # (B, T+1, hidden)
        x = x + self.pos_embed[:, : t + 1]

        src_key_padding_mask: torch.Tensor | None = None
        if padding_mask is not None:
            # CLS is never padding.
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=video_stack.device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = x[:, 0]                                    # (B, hidden)
        return self.classifier(cls_out)                      # (B, 2)

    # Kept for API compatibility with training-loop code written for the
    # baseline — the temporal detector is always trained video-level.
    def forward(self, video_stack: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Alias for :meth:`forward_video` — the temporal head has no per-frame mode."""
        return self.forward_video(video_stack)


def build_temporal_from_config(model_cfg: dict) -> TemporalDetector:
    """Instantiate :class:`TemporalDetector` from ``configs/model.yaml``."""
    th = model_cfg.get("temporal_head", {}) or {}
    return TemporalDetector(
        backbone_name=str(model_cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_frames=int(model_cfg.get("input", {}).get("frames_per_clip", 32)),
        hidden_dim=int(th.get("hidden_dim") or 256),
        num_layers=int(th.get("num_layers") or 2),
        num_heads=int(th.get("num_heads") or 4),
        dropout=float(model_cfg.get("classifier", {}).get("dropout", 0.2)),
    )
