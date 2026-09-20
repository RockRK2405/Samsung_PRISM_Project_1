"""Dataset for the general (non-face) AI-generated content path — ADR-006.

Unlike :mod:`src.datasets.ff_dataset` (which reads pre-cropped face
JPEGs from a fixed FF++ folder layout), this reads a plain
real/fake-labeled image folder — the layout GenImage-style Kaggle
datasets ship in. Because :class:`~src.models.baseline.BaselineDetector`
is architecturally a generic per-image classifier (see ADR-006), this
dataset can feed the exact same model class with no code changes there.

Expects a manifest CSV built by ``scripts/prepare_general_dataset.py``
with columns: ``path,label`` (label: 0=real, 1=synthetic) — deliberately
NOT a raw ``torchvision.datasets.ImageFolder`` over the dataset root,
because GenImage-family Kaggle mirrors use inconsistent folder-naming
conventions (``real``/``fake``, ``nature``/``ai``, ``0_real``/``1_fake``,
...); the prep script normalises all of that into one manifest format
once, so this dataset class stays trivial and layout-agnostic.
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

import io
import random as _random

from src.utils.logging import get_logger

logger = get_logger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class ForensicDegrade:
    """Random real-world video degradations for face-forensics robustness
    (ADR-008 Phase 5, Part 18): JPEG recompression, Gaussian blur, and
    resolution down/up-scaling. These mimic the compression/quality shifts
    that make a model trained on one dataset (e.g. FF++ c23) collapse on
    another (Celeb-DF/DFDC). Applied on the PIL image before ToTensor.
    Each degradation fires independently with probability ``p``.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: "Image.Image") -> "Image.Image":
        if _random.random() < self.p:  # JPEG recompression
            q = _random.randint(40, 90)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if _random.random() < self.p:  # resolution degradation
            w, h = img.size
            scale = _random.uniform(0.4, 0.9)
            small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        if _random.random() < self.p * 0.6:  # mild blur
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=_random.uniform(0.3, 1.2)))
        return img


@dataclass(frozen=True)
class GeneralImageSample:
    path: Path
    label: int  # 0 = real, 1 = synthetic


def load_manifest(manifest_csv: Path) -> list[GeneralImageSample]:
    """Read a ``path,label`` manifest built by ``prepare_general_dataset.py``."""
    samples: list[GeneralImageSample] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(GeneralImageSample(path=Path(row["path"]), label=int(row["label"])))
    if not samples:
        raise ValueError(f"Manifest {manifest_csv} contained zero rows.")
    logger.info("Loaded %d samples from %s", len(samples), manifest_csv)
    return samples


class GeneralImageDataset(Dataset):
    """One real/synthetic image per item — trains the general-content path.

    Attributes:
        samples: Parsed manifest rows.
        train: When ``True``, applies light augmentation (flip, color
            jitter) matching the Image module's own CIFAKE training
            recipe, for consistency across the project's two
            image-classifier training pipelines. When ``False``
            (validation/test), only resize + normalise.
    """

    def __init__(self, manifest_csv: Path, image_size: int = 224, train: bool = True,
                 forensic_aug: bool = False) -> None:
        self.samples = load_manifest(Path(manifest_csv))
        self.train = train
        if train:
            aug = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
            # Forensic degradations go BEFORE ToTensor (they operate on PIL).
            if forensic_aug:
                aug.insert(1, ForensicDegrade(p=0.5))
            aug += [
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
            self.transform = transforms.Compose(aug)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        try:
            img = Image.open(sample.path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - corrupt file must not kill the run
            logger.warning("Corrupt image %s (%s) — substituting a black frame.", sample.path, exc)
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), sample.label
