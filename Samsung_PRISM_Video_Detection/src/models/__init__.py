"""Neural network architectures for the video detection module."""

from typing import Any

from torch import nn

from src.models.baseline import BaselineDetector, build_baseline_from_config
from src.models.dual_stream import (
    DualStreamDetector,
    build_dual_stream_from_config,
    warm_start_rgb_from_baseline,
)
from src.models.temporal import TemporalDetector, build_temporal_from_config

__all__ = [
    "BaselineDetector",
    "DualStreamDetector",
    "TemporalDetector",
    "build_baseline_from_config",
    "build_dual_stream_from_config",
    "build_model_from_config",
    "build_temporal_from_config",
    "warm_start_rgb_from_baseline",
]


def build_model_from_config(model_cfg: dict[str, Any]) -> nn.Module:
    """Factory — dispatch on which streams and head the config selects.

    Precedence:
      1. ``frequency_stream.enabled: true`` → :class:`DualStreamDetector`
         (Milestone 6). Temporal head is always mean-pool in this variant.
      2. ``temporal_head.type: transformer|attention`` → :class:`TemporalDetector`
         (Milestone 5).
      3. Otherwise → :class:`BaselineDetector` (Milestone 4).

    Unknown ``temporal_head.type`` values raise ``ValueError`` — a config
    typo should surface loudly, not silently fall back to the baseline.
    """
    freq_stream = model_cfg.get("frequency_stream") or {}
    if bool(freq_stream.get("enabled", False)):
        return build_dual_stream_from_config(model_cfg)

    head_type = str((model_cfg.get("temporal_head") or {}).get("type", "mean_pool"))
    if head_type == "mean_pool":
        return build_baseline_from_config(model_cfg)
    if head_type in {"transformer", "attention"}:
        return build_temporal_from_config(model_cfg)
    raise ValueError(
        f"Unknown temporal_head.type '{head_type}' — "
        "expected one of: mean_pool, transformer, attention."
    )
