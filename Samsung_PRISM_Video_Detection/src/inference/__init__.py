"""Single-video prediction API — the fusion engine's entrypoint."""

from src.inference.api import (
    MODALITY,
    SCHEMA_VERSION,
    DetectionResult,
    VideoDetector,
    predict,
)

# Opt-in, additive (ADR-006) -- requires checkpoints/general.pt to exist.
# The fusion engine's default entrypoint above (predict / VideoDetector)
# is unchanged; switching to this is a deliberate follow-up decision.
from src.inference.multi_target_predictor import MultiTargetVideoDetector

# Recommended entry point: general AI-video + face-swap branches, fused.
from src.inference.unified_detector import UnifiedVideoDetector

__all__ = [
    "MODALITY",
    "SCHEMA_VERSION",
    "DetectionResult",
    "VideoDetector",
    "predict",
    "MultiTargetVideoDetector",
    "UnifiedVideoDetector",
]
