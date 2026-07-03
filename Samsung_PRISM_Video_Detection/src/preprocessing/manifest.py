"""Dataset scanning and leak-free train/val/test splitting.

Scans the FaceForensics++ (c23) directory tree on disk, produces one
record per video (real + all manipulation types), and emits a JSON
manifest split into train / val / test.

Split-safety
------------
FF++ fakes are named ``AAA_BBB.mp4`` where ``AAA`` is the *source*
real-video identity. If ``original/AAA.mp4`` ends up in one split but
``Deepfakes/AAA_BBB.mp4`` ends up in another, the model can trivially
memorise identity — a well-known leakage bug.

We prevent this by treating ``AAA`` as a *group id* and splitting
groups (not individual videos). All clips derived from real video
``AAA`` — the original itself and every manipulation whose filename
starts with ``AAA`` — always land in the same split.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Manipulation folders included in the FF++ baseline split.
# FaceShifter and DeepFakeDetection are present on disk but deliberately
# excluded here — see docs/dataset_notes_ff_c23.md.
FF_MANIPULATIONS: tuple[str, ...] = (
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)
FF_REAL_FOLDER: str = "original"


@dataclass(frozen=True)
class VideoRecord:
    """One row in the dataset manifest.

    Attributes:
        video_id: Stable identifier — the video filename without extension.
        group_id: Source real-video id used for leak-free grouping. Equal to
            ``video_id`` for reals, and to the ``AAA`` prefix for fakes
            named ``AAA_BBB.mp4``.
        label: ``"real"`` or ``"synthetic"``.
        manipulation: Manipulation folder name (``"original"`` for reals,
            otherwise e.g. ``"Deepfakes"``).
        rel_path: Path relative to the dataset root (posix-style).
    """

    video_id: str
    group_id: str
    label: str
    manipulation: str
    rel_path: str


def _scan_folder(folder: Path, manipulation: str, label: str) -> list[VideoRecord]:
    """Enumerate ``.mp4`` files under ``folder`` and return one record per file."""
    if not folder.is_dir():
        logger.warning("Expected folder is missing: %s", folder)
        return []

    records: list[VideoRecord] = []
    for mp4 in sorted(folder.glob("*.mp4")):
        video_id = mp4.stem
        # For fakes named AAA_BBB, the source real-video id is AAA.
        group_id = video_id.split("_", 1)[0]
        records.append(
            VideoRecord(
                video_id=video_id,
                group_id=group_id,
                label=label,
                manipulation=manipulation,
                rel_path=str(mp4.as_posix()),
            )
        )
    return records


def scan_ff_c23(dataset_root: Path) -> list[VideoRecord]:
    """Enumerate every FF++ video (real + 4 baseline manipulations).

    Args:
        dataset_root: Path to ``.../FaceForensics++_C23`` — the directory
            that contains ``original/`` and the manipulation folders.

    Returns:
        List of :class:`VideoRecord`, sorted deterministically by
        ``(label, manipulation, video_id)``.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records: list[VideoRecord] = []
    records.extend(_scan_folder(dataset_root / FF_REAL_FOLDER, FF_REAL_FOLDER, "real"))
    for manip in FF_MANIPULATIONS:
        records.extend(_scan_folder(dataset_root / manip, manip, "synthetic"))

    records.sort(key=lambda r: (r.label, r.manipulation, r.video_id))
    logger.info("Scanned %d videos from %s", len(records), dataset_root)
    return records


def group_split(
    records: list[VideoRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[VideoRecord]]:
    """Split records into train/val/test by ``group_id`` (no identity leakage).

    Args:
        records: All records returned by :func:`scan_ff_c23`.
        ratios: Three floats that sum to ``1.0`` — train / val / test.
        seed: RNG seed applied to the *shuffled group order*.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    unique_groups = sorted({r.group_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    # Test gets the remainder — avoids losing samples to int() truncation.

    splits: dict[str, list[VideoRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        if r.group_id in train_groups:
            splits["train"].append(r)
        elif r.group_id in val_groups:
            splits["val"].append(r)
        else:
            splits["test"].append(r)

    for name, items in splits.items():
        logger.info("Split %-5s: %5d videos (%d unique groups)", name, len(items),
                    len({r.group_id for r in items}))
    return splits


def write_manifest(splits: dict[str, list[VideoRecord]], out_path: Path) -> None:
    """Serialise a splits dictionary to a single JSON manifest.

    Args:
        splits: Output of :func:`group_split`.
        out_path: Destination JSON file. Parent directories are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits": {name: [asdict(r) for r in items] for name, items in splits.items()},
        "meta": {
            "num_train": len(splits["train"]),
            "num_val": len(splits["val"]),
            "num_test": len(splits["test"]),
            "manipulations": [FF_REAL_FOLDER, *FF_MANIPULATIONS],
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote manifest: %s", out_path)
