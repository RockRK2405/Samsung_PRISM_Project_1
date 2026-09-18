"""Strong spatial video baseline (ADR-007 Phase 2, Parts 6-8).

A per-frame timm backbone (ConvNeXt / Xception / EfficientNet / ViT /
Swin — anything timm exposes) followed by a configurable temporal
aggregator over the frame features:

    mean_pool        -- average frame features (Stage 1 baseline)
    attention_pool   -- learned attention-weighted average (Stage 1+)
    transformer      -- CLS-token transformer encoder over frames (Stage 3)

This is deliberately heavier and more flexible than the Milestone-4
EfficientNet-B0 baseline: the refreshed 26TS08 priorities allow heavy
models, and cross-generator generalization is the target. The temporal
head is swappable so we can ablate spatial-only vs spatial+temporal
(Part 24) with one config change.

Input:  (B, T, 3, H, W)      Output: logits (B, 2)
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class _AttentionPool(nn.Module):
    """Learned attention-weighted pooling over the T frame features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, D)
        w = torch.softmax(self.score(feats).squeeze(-1), dim=1)  # (B, T)
        return torch.einsum("btd,bt->bd", feats, w)


class _TransformerPool(nn.Module):
    """CLS-token transformer encoder over frame features."""

    def __init__(self, dim: int, num_frames: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_frames + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, D)
        b, t, _ = feats.shape
        x = torch.cat([self.cls.expand(b, -1, -1), feats], dim=1)
        x = x + self.pos[:, : t + 1]
        x = self.encoder(x)
        return x[:, 0]


class VideoSpatialNet(nn.Module):
    """Per-frame backbone + temporal aggregator + binary classifier.

    Attributes:
        num_classes: Fixed at 2 (real / synthetic).
        feat_dim: Backbone feature dimension.
    """

    num_classes: int = 2

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.fb_in22k",
        pretrained: bool = True,
        num_frames: int = 16,
        temporal: str = "attention_pool",
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_frames = num_frames
        self.temporal = temporal

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feat_dim = int(self.backbone.num_features)

        if temporal == "mean_pool":
            self.pool: nn.Module = nn.Identity()
            head_dim = self.feat_dim
        elif temporal == "attention_pool":
            self.pool = _AttentionPool(self.feat_dim)
            head_dim = self.feat_dim
        elif temporal == "transformer":
            self.proj = nn.Linear(self.feat_dim, hidden_dim)
            self.pool = _TransformerPool(hidden_dim, num_frames, num_layers, num_heads, dropout)
            head_dim = hidden_dim
        else:
            raise ValueError(f"Unknown temporal aggregator: {temporal}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Dropout(dropout), nn.Linear(head_dim, self.num_classes),
        )
        logger.info(
            "VideoSpatialNet: backbone=%s feat_dim=%d temporal=%s params=%.2fM",
            backbone_name, self.feat_dim, temporal, sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> logits (B, 2)."""
        b, t, c, h, w = clips.shape
        feats = self.backbone(clips.reshape(b * t, c, h, w))     # (B*T, D)
        feats = feats.reshape(b, t, self.feat_dim)               # (B, T, D)
        if self.temporal == "mean_pool":
            pooled = feats.mean(dim=1)
        elif self.temporal == "transformer":
            pooled = self.pool(self.proj(feats))
        else:  # attention_pool
            pooled = self.pool(feats)
        return self.classifier(pooled)


def build_video_spatial_from_config(cfg: dict) -> VideoSpatialNet:
    """Instantiate from the ``model`` + ``video`` sub-trees of video_training.yaml."""
    m = cfg.get("model", {})
    v = cfg.get("video", {})
    t = m.get("temporal", {}) or {}
    return VideoSpatialNet(
        backbone_name=str(m.get("backbone", "convnext_tiny.fb_in22k")),
        pretrained=bool(m.get("pretrained", True)),
        num_frames=int(v.get("num_frames", 16)),
        temporal=str(t.get("type", "attention_pool")),
        hidden_dim=int(t.get("hidden_dim", 512)),
        num_layers=int(t.get("num_layers", 2)),
        num_heads=int(t.get("num_heads", 8)),
        dropout=float(m.get("dropout", 0.2)),
    )
