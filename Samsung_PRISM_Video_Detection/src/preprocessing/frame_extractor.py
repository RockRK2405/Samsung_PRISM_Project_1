"""Decode a video into a fixed number of sampled RGB frames.

Sampling knobs are set in ``configs/dataset.yaml``. For Milestone 3 the
locked values are:

    frames_per_video : 32
    strategy         : "uniform"

See ``docs/architecture/ADR-002-frame-sampling-and-crop-size.md``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FrameExtractor:
    """Extract ``num_frames`` frames from a video file.

    Attributes:
        num_frames: Number of frames to return per video.
        strategy: Sampling strategy. Only ``"uniform"`` is implemented today.
    """

    def __init__(self, num_frames: int, strategy: str = "uniform") -> None:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if strategy != "uniform":
            raise NotImplementedError(
                f"Sampling strategy '{strategy}' is not implemented yet."
            )
        self.num_frames = num_frames
        self.strategy = strategy

    def extract(self, video_path: Path) -> list[np.ndarray]:
        """Return a list of RGB frames sampled from ``video_path``.

        Args:
            video_path: Path to an ``.mp4`` (or any codec OpenCV supports).

        Returns:
            A list of ``self.num_frames`` NumPy arrays with shape
            ``(H, W, 3)`` and dtype ``uint8`` in RGB order. If the video
            has fewer physical frames than requested, indices wrap by
            clamping — the earliest and latest frames are duplicated as
            necessary so downstream code can assume a fixed length.

        Raises:
            RuntimeError: If OpenCV cannot open the video.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise RuntimeError(f"Video reports zero frames: {video_path}")

            indices = self._uniform_indices(total, self.num_frames)
            frames = self._read_at_indices(cap, indices, str(video_path))
        finally:
            cap.release()

        return frames

    @staticmethod
    def _uniform_indices(total: int, n: int) -> list[int]:
        """Compute ``n`` evenly-spaced integer frame indices in ``[0, total-1]``."""
        if total >= n:
            # Uniform without duplication.
            return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
        # Very short clip — clamp indices so we still return exactly n frames.
        return [min(i, total - 1) for i in range(n)]

    @staticmethod
    def _read_at_indices(
        cap: cv2.VideoCapture, indices: list[int], src: str
    ) -> list[np.ndarray]:
        """Seek to each requested frame index and return the decoded RGB frame."""
        frames: list[np.ndarray] = []
        last_good: np.ndarray | None = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                if last_good is None:
                    raise RuntimeError(
                        f"Failed to decode any frame from {src} (first bad index {idx})"
                    )
                logger.debug("Frame %d unreadable in %s — reusing previous frame", idx, src)
                frames.append(last_good.copy())
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            last_good = rgb
        return frames
