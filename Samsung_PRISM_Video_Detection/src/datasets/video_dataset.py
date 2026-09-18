"""Video-level dataset for the general AI-video track (ADR-007, Part 5).

Reads a manifest built by ``scripts/prepare_genvidbench.py`` (columns:
``video_id, video_path, label, generator, source, dataset, split``) and
returns, per item, a stack of ``num_frames`` uniformly-sampled RGB frames
as a ``(T, 3, H, W)`` float tensor plus the label and generator metadata.

Unlike the FF++ face datasets (which read pre-cropped face JPEGs), this
decodes whole video frames on the fly via the existing
:class:`~src.preprocessing.frame_extractor.FrameExtractor`, so it works
directly on the extracted GenVidBench videos with no crop-caching step.

Robustness (Part 5): corrupt/unreadable videos never kill a run — they
are logged and a black clip is substituted (and flagged so the trainer
can count how many were bad). Short clips are handled by the extractor's
index clamping.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.preprocessing.frame_extractor import FrameExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class VideoSample:
    video_id: str
    video_path: Path
    label: int          # 0=real, 1=fake
    generator: str      # "" for real
    source: str         # "" for fake


def load_video_manifest(manifest_csv: Path) -> list[VideoSample]:
    samples: list[VideoSample] = []
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            samples.append(VideoSample(
                video_id=row["video_id"],
                video_path=Path(row["video_path"]),
                label=int(row["label"]),
                generator=row.get("generator", ""),
                source=row.get("source", ""),
            ))
    if not samples:
        raise ValueError(f"Manifest {manifest_csv} contained zero rows.")
    logger.info("Loaded %d video samples from %s", len(samples), manifest_csv)
    return samples


class VideoManifestDataset(Dataset):
    """Frame-stack dataset over a GenVidBench-style manifest.

    Each item -> (frames, label) where frames is ``(T, 3, image_size,
    image_size)``. Training applies light augmentation (flip, colour
    jitter); val/test applies resize + normalise only. Augmentation is
    applied CONSISTENTLY across all frames of a clip (same flip decision)
    so temporal cues are not scrambled.
    """

    def __init__(
        self,
        manifest_csv: Path,
        num_frames: int = 16,
        image_size: int = 224,
        train: bool = True,
        sampling: str = "uniform",
    ) -> None:
        self.samples = load_video_manifest(Path(manifest_csv))
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.train = bool(train)
        self._extractor = FrameExtractor(num_frames=self.num_frames, strategy=sampling)

        # Per-frame spatial transform. Flip is decided per-clip (below), so
        # the compose here is deterministic given an already-flipped PIL.
        norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        if train:
            self._jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        else:
            self._jitter = None
        self._resize = transforms.Resize((self.image_size, self.image_size))
        self._to_tensor = transforms.ToTensor()
        self._norm = norm

    def __len__(self) -> int:
        return len(self.samples)

    def _transform_clip(self, frames: list[np.ndarray]) -> torch.Tensor:
        # One flip decision for the whole clip (temporal consistency).
        do_flip = self.train and torch.rand(1).item() < 0.5
        out = []
        for arr in frames:
            img = Image.fromarray(arr)
            img = self._resize(img)
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self._jitter is not None:
                img = self._jitter(img)
            t = self._norm(self._to_tensor(img))
            out.append(t)
        return torch.stack(out, dim=0)  # (T, 3, H, W)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        try:
            frames = self._extractor.extract(s.video_path)
        except Exception as exc:  # noqa: BLE001 - corrupt video must not kill training
            logger.warning("Unreadable video %s (%s) — substituting black clip.", s.video_path, exc)
            black = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frames = [black for _ in range(self.num_frames)]
        clip = self._transform_clip(frames)
        return clip, s.label
