"""Public API — everything the fusion engine imports from us.

Example
-------

>>> from src.inference import predict, MODALITY, SCHEMA_VERSION
>>> result = predict("path/to/clip.mp4")
>>> result.prediction
'synthetic'
>>> result.modality
'video'

Or for repeated calls (loads the checkpoint once):

>>> from src.inference import VideoDetector
>>> detector = VideoDetector.load_default()
>>> r1 = detector.predict("clip_a.mp4")
>>> r2 = detector.predict("clip_b.mp4")
"""

from __future__ import annotations

from pathlib import Path

from src.inference.predictor import (
    MODALITY,
    SCHEMA_VERSION,
    DetectionResult,
    VideoDetector,
)

# Cache the default detector so repeated ``predict()`` calls don't
# reload the checkpoint every invocation.
_default_detector: VideoDetector | None = None


def predict(
    video_path: str | Path,
    produce_explanation: bool = True,
    output_dir: str | Path | None = None,
) -> DetectionResult:
    """One-shot prediction API — the fusion engine's simplest entry point.

    Uses the project's canonical production detector (loaded on first
    call, cached thereafter). For advanced use cases — a custom
    checkpoint, a different threshold, a specific device — construct
    :class:`VideoDetector` directly instead.

    Args:
        video_path: Path to any video OpenCV can decode.
        produce_explanation: When ``True`` (default), also compute
            GradCAM overlays + per-frame timeline + explainability score.
            Set ``False`` when the fusion engine only wants the raw score
            and doesn't need to display artefacts.
        output_dir: Where to write explanation artefacts. When ``None``
            defaults to ``outputs/explainability/<video-stem>/``.

    Returns:
        A populated :class:`DetectionResult`.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = VideoDetector.load_default()
    return _default_detector.predict(
        video_path,
        produce_explanation=produce_explanation,
        output_dir=output_dir,
    )


__all__ = [
    "MODALITY",
    "SCHEMA_VERSION",
    "DetectionResult",
    "VideoDetector",
    "predict",
]
