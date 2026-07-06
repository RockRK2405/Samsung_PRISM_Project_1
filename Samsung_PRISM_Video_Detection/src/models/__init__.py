"""Neural network architectures for the video detection module."""

from typing import Any

from torch import nn

from src.models.baseline import BaselineDetector, build_baseline_from_config
from src.models.temporal import TemporalDetector, build_temporal_from_config

__all__ = [
    "BaselineDetector",
    "TemporalDetector",
    "build_baseline_from_config",
    "build_model_from_config",
    "build_temporal_from_config",
]


def build_model_from_config(model_cfg: dict[str, Any]) -> nn.Module:
    """Factory — dispatch on ``model.temporal_head.type``.

    * ``mean_pool`` → :class:`BaselineDetector` (Milestone 4)
    * ``transformer`` / ``attention`` → :class:`TemporalDetector` (Milestone 5)

    Any other value raises ``ValueError`` — a config typo should surface
    loudly, not silently fall back to the baseline.
    """
    head_type = str((model_cfg.get("temporal_head") or {}).get("type", "mean_pool"))
    if head_type == "mean_pool":
        return build_baseline_from_config(model_cfg)
    if head_type in {"transformer", "attention"}:
        return build_temporal_from_config(model_cfg)
    raise ValueError(
        f"Unknown temporal_head.type '{head_type}' — "
        "expected one of: mean_pool, transformer, attention."
    )
