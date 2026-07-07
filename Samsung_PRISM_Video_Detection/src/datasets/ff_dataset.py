"""PyTorch datasets that read cached FF++ face crops from disk.

Two datasets, one filesystem layout:

    data/processed/faces/ff_c23/<split>/<label>/<video_folder>/face_NN.jpg

* :class:`FaceFrameDataset` yields one crop at a time — used for
  **training**, where shuffling across frames gives us more gradient
  diversity per batch than shuffling whole clips would.
* :class:`FaceVideoDataset` yields the full 32-crop stack for one video
  — used for **evaluation**, where we mean-pool per-frame predictions
  into one video-level score.

Splitting into two dataset classes keeps ``__getitem__`` return shapes
deterministic, which pleases DataLoader collation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Image normalisation — ImageNet stats (all our backbones are pretrained
# on ImageNet, so aligning input statistics matters).
# ---------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a normalised ``(3, H, W)`` float tensor."""
    # np.asarray on a PIL image returns a read-only view of the JPEG
    # buffer. PyTorch warns loudly on read-only inputs to `from_numpy`,
    # so copy once to make it writable — negligible cost, silences the
    # spam and future-proofs against undefined-behaviour warnings.
    arr = np.array(img, dtype=np.uint8, copy=True)       # HxWx3 uint8 (writable)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


# ---------------------------------------------------------------------
# Index record — one row per (video, crop file)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FaceSample:
    """One face-crop file on disk.

    Attributes:
        video_dir: Absolute path to the video's crop folder.
        frame_path: Absolute path to a single ``face_NN.jpg``.
        label: ``0`` (real) or ``1`` (synthetic).
        video_id: Stable identifier — the video's folder name.
        manipulation: e.g. ``"original"``, ``"Deepfakes"``, ``"Face2Face"``.
    """

    video_dir: Path
    frame_path: Path
    label: int
    video_id: str
    manipulation: str


def _manipulation_from_folder(folder_name: str, label: int) -> str:
    """Recover the manipulation / source tag from the video-folder name.

    Naming conventions:
      * FF++ reals:  ``"000"``, ``"001"``, …           → ``"original"``
      * FF++ fakes:  ``"Deepfakes_000_003"``           → ``"Deepfakes"``
      * Celeb-DF reals: ``"Celeb-real_id0_0000"``      → ``"Celeb-real"``
      * Celeb-DF reals: ``"YouTube-real_00170"``       → ``"YouTube-real"``
      * Celeb-DF fakes: ``"Celeb-synthesis_id0_..."``  → ``"Celeb-synthesis"``

    A folder name without an underscore (FF++ real) always maps to
    ``"original"`` regardless of label — preserves Milestone 4/5A
    behaviour on FF++.
    """
    if "_" not in folder_name:
        return "original"
    return folder_name.split("_", 1)[0]


def _scan_split(root: Path, split: str) -> list[FaceSample]:
    """Enumerate every ``face_NN.jpg`` under one split directory."""
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder missing: {split_dir}")

    samples: list[FaceSample] = []
    label_map = {"real": 0, "synthetic": 1}
    for label_name, label_int in label_map.items():
        label_dir = split_dir / label_name
        if not label_dir.is_dir():
            logger.warning("Label folder missing: %s", label_dir)
            continue
        for video_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            manip = _manipulation_from_folder(video_dir.name, label_int)
            for jpg in sorted(video_dir.glob("face_*.jpg")):
                samples.append(
                    FaceSample(
                        video_dir=video_dir,
                        frame_path=jpg,
                        label=label_int,
                        video_id=video_dir.name,
                        manipulation=manip,
                    )
                )
    return samples


# ---------------------------------------------------------------------
# Frame-level dataset (training)
# ---------------------------------------------------------------------

class FaceFrameDataset(Dataset[dict]):
    """Yield one face crop at a time — for training.

    Each sample is a dict with keys ``"image"``, ``"label"``, ``"video_id"``,
    ``"manipulation"``. Ordering is deterministic; shuffling is the
    DataLoader's job.
    """

    def __init__(self, root: Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.samples = _scan_split(self.root, split)
        if not self.samples:
            raise RuntimeError(f"No face crops found under {self.root / split}")
        logger.info(
            "FaceFrameDataset[%s]: %d crops across %d videos",
            split,
            len(self.samples),
            len({s.video_id for s in self.samples}),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        with Image.open(s.frame_path) as img:
            tensor = _to_tensor(img.convert("RGB"))
        return {
            "image": tensor,
            "label": s.label,
            "video_id": s.video_id,
            "manipulation": s.manipulation,
        }


# ---------------------------------------------------------------------
# Video-level dataset (evaluation)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _VideoIndex:
    video_id: str
    label: int
    manipulation: str
    frames: tuple[Path, ...]


def _group_by_video(samples: list[FaceSample]) -> list[_VideoIndex]:
    """Collapse frame-level samples into one entry per video."""
    by_video: dict[str, list[FaceSample]] = {}
    for s in samples:
        by_video.setdefault(s.video_id, []).append(s)

    videos: list[_VideoIndex] = []
    for vid_id, sample_list in by_video.items():
        sample_list.sort(key=lambda s: s.frame_path.name)
        videos.append(
            _VideoIndex(
                video_id=vid_id,
                label=sample_list[0].label,
                manipulation=sample_list[0].manipulation,
                frames=tuple(s.frame_path for s in sample_list),
            )
        )
    videos.sort(key=lambda v: v.video_id)
    return videos


def pad_video_collate(batch: list[dict], pad_to: int) -> dict:
    """Custom collate function for :class:`FaceVideoDataset`.

    Pads every clip in the batch to ``pad_to`` frames by repeating a
    zero-frame at the end, and emits a boolean ``padding_mask`` so
    transformer heads can zero-out attention on padded positions.

    Args:
        batch: List of dicts from :meth:`FaceVideoDataset.__getitem__`.
        pad_to: Target sequence length (e.g. 32).

    Returns:
        A dict with keys ``images`` ``(B, pad_to, 3, H, W)``,
        ``padding_mask`` ``(B, pad_to)`` bool (True where padding),
        ``label`` ``(B,)`` long, ``video_id`` list[str],
        ``manipulation`` list[str].
    """
    images_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    labels: list[int] = []
    video_ids: list[str] = []
    manips: list[str] = []
    for item in batch:
        clip = item["images"]                       # (T, 3, H, W)
        t = clip.shape[0]
        if t > pad_to:
            clip = clip[:pad_to]
            t = pad_to
        pad_len = pad_to - t
        if pad_len > 0:
            pad_shape = (pad_len, *clip.shape[1:])
            padding = torch.zeros(pad_shape, dtype=clip.dtype)
            clip = torch.cat([clip, padding], dim=0)
        mask = torch.zeros(pad_to, dtype=torch.bool)
        mask[t:] = True
        images_list.append(clip)
        masks_list.append(mask)
        labels.append(int(item["label"]))
        video_ids.append(str(item["video_id"]))
        manips.append(str(item["manipulation"]))
    return {
        "images": torch.stack(images_list, dim=0),
        "padding_mask": torch.stack(masks_list, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
        "video_id": video_ids,
        "manipulation": manips,
    }


class FaceVideoDataset(Dataset[dict]):
    """Yield the full stack of face crops for one video — for evaluation.

    Each sample is a dict with keys ``"images"`` (T×3×224×224 float tensor),
    ``"label"``, ``"video_id"``, ``"manipulation"``.

    A video's ``T`` may be less than ``frames_per_video`` (32) if MTCNN
    missed some frames during extraction. Callers should handle variable
    ``T`` — either by padding, by using a custom collate_fn, or by
    running with ``batch_size=1``.
    """

    def __init__(self, root: Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.videos = _group_by_video(_scan_split(self.root, split))
        if not self.videos:
            raise RuntimeError(f"No videos found under {self.root / split}")
        logger.info(
            "FaceVideoDataset[%s]: %d videos (%d total crops)",
            split,
            len(self.videos),
            sum(len(v.frames) for v in self.videos),
        )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        v = self.videos[idx]
        tensors = []
        for p in v.frames:
            with Image.open(p) as img:
                tensors.append(_to_tensor(img.convert("RGB")))
        stack = torch.stack(tensors, dim=0)  # (T, 3, 224, 224)
        return {
            "images": stack,
            "label": v.label,
            "video_id": v.video_id,
            "manipulation": v.manipulation,
        }
