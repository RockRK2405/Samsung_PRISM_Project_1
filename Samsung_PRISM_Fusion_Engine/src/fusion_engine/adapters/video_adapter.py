"""Adapter for the Video module — EfficientNet-B0 + mean-pool baseline.

Imports ``src.inference`` from ``Samsung_PRISM_Video_Detection/`` (added
to ``sys.path`` at import time). That module's top-level package is
literally named ``src`` — a side effect of how it was scaffolded — so
this adapter adds the *video module's root*, not a `src/` subfolder, to
``sys.path``, matching how the video module's own ``conftest.py`` makes
itself importable for its test suite.

Uses ``VideoDetector.load_default()``, which points at
``checkpoints/best.pt`` (the Milestone-4 mean-pool baseline — the
un-mixed-training version, F1=0.9862 / FPR=0.03 on FaceForensics++ c23;
see that module's README and ``docs/experiments/`` for the full
methodology and the documented cross-dataset caveats).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fusion_engine.schema import ModalityResult

_VIDEO_MODULE_ROOT = Path(__file__).resolve().parents[4] / "Samsung_PRISM_Video_Detection"

if str(_VIDEO_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEO_MODULE_ROOT))


class VideoAdapter:
    """Loads the Video module's default detector once; call :meth:`predict` per video."""

    def __init__(self) -> None:
        from src.inference import VideoDetector

        self.detector = VideoDetector.load_default()

    def predict(self, video_path: str | Path, produce_explanation: bool = False) -> ModalityResult:
        """Run the video detector on a single video file.

        Args:
            video_path: Path to a video file OpenCV can decode.
            produce_explanation: When ``True``, also runs GradCAM + timeline
                generation (slower — several seconds extra per video).
                Defaults to ``False`` for the fusion engine's hot path;
                the QC dashboard can request explanations on-demand for a
                specific flagged item instead of on every submission.

        Returns:
            A :class:`ModalityResult` with ``modality="video"``. On any
            failure (no face detected, bad file), returns
            ``available=False`` rather than raising.
        """
        t0 = time.perf_counter()
        try:
            result = self.detector.predict(video_path, produce_explanation=produce_explanation)

            latency_ms = (time.perf_counter() - t0) * 1000
            return ModalityResult(
                modality="video",
                prob_synthetic=float(result.prob_synthetic),
                confidence=float(result.confidence),
                available=True,
                latency_ms=latency_ms,
                raw=result.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 - must not crash the fusion pipeline
            return ModalityResult(
                modality="video",
                prob_synthetic=0.0,
                confidence=0.0,
                available=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
