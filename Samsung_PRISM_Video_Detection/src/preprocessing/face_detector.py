"""Face detection + cropping using facenet-pytorch's MTCNN.

Why MTCNN
---------
MTCNN is the canonical face detector for deepfake-detection pipelines
in the literature (DFDC baseline, FaceForensics++ tooling, most
follow-up papers). ``facenet-pytorch``'s implementation is:

* small enough to run on MPS / CPU without ceremony,
* returns an already-cropped and resized tensor when ``keep_all=False``,
* battles the two hard cases we care about here — multiple faces per
  frame (return the highest-confidence one) and no face detected
  (return ``None``).

ADR-003 will revisit MediaPipe vs MTCNN once we have Milestone-3
throughput numbers to compare.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MTCNNFaceDetector:
    """Detect the largest face in an RGB frame and return a fixed-size crop.

    Attributes:
        image_size: Side length (in pixels) of the returned crop.
        margin: Extra pixels of margin around the tight bounding box,
            in the resized coordinate system.
        device: PyTorch device the MTCNN runs on.
    """

    def __init__(
        self,
        image_size: int = 224,
        margin: int = 20,
        device: torch.device | str = "cpu",
    ) -> None:
        self.image_size = image_size
        self.margin = margin
        self.device = torch.device(device)
        self._mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,             # return only the top-confidence face
            post_process=False,         # keep uint8 output; we handle normalisation later
            device=self.device,
        )

    def detect_and_crop(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Return an ``(image_size, image_size, 3)`` uint8 crop, or ``None`` if no face.

        Args:
            frame_rgb: HxWx3 uint8 RGB frame as produced by
                :class:`~src.preprocessing.frame_extractor.FrameExtractor`.

        Returns:
            A uint8 NumPy array shaped ``(image_size, image_size, 3)`` if a
            face was found; ``None`` otherwise.
        """
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3:
            raise ValueError(
                f"Expected HxWx3 uint8 RGB frame, got shape={frame_rgb.shape} dtype={frame_rgb.dtype}"
            )

        pil = Image.fromarray(frame_rgb)
        with torch.no_grad():
            face = self._mtcnn(pil)  # tensor (3, H, W) float in [0, 255], or None

        if face is None:
            return None

        # (3, H, W) float -> (H, W, 3) uint8
        arr = face.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return arr

    def crop_video_frames(
        self, frames_rgb: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[int]]:
        """Batch helper — detect+crop every frame of a video.

        Args:
            frames_rgb: List of RGB frames from one video.

        Returns:
            Tuple ``(crops, missed_indices)`` where ``crops`` contains only
            the frames in which a face was detected (in original order) and
            ``missed_indices`` lists the input positions that produced
            ``None``. Order is preserved so callers can zip them back to
            source frame indices if needed.
        """
        crops: list[np.ndarray] = []
        missed: list[int] = []
        for i, frame in enumerate(frames_rgb):
            crop = self.detect_and_crop(frame)
            if crop is None:
                missed.append(i)
            else:
                crops.append(crop)
        return crops, missed


def save_crop(crop: np.ndarray, out_path: Path) -> None:
    """Persist a face crop to disk as a JPEG.

    Args:
        crop: HxWx3 uint8 RGB array.
        out_path: Destination file path. Parent dirs are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(out_path, format="JPEG", quality=95)
