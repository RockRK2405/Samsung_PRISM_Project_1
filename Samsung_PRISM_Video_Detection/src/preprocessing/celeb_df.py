"""Celeb-DF v2 dataset scanning.

Celeb-DF v2 layout (as distributed):

    Celeb-DF-v2/
    ├── Celeb-real/           # 590 real "celebrity" videos
    ├── YouTube-real/         # 300 additional real videos (in-the-wild)
    ├── Celeb-synthesis/      # 5639 deepfake videos
    └── List_of_testing_videos.txt

The ``List_of_testing_videos.txt`` file defines the standard test
split (518 videos) used by every published cross-dataset benchmark.
Each line has the form ``<label> <relative_path>`` where **Celeb-DF's
convention has 1 = real and 0 = fake — the OPPOSITE of ours**. We flip
the label on read so downstream code (metrics, DataLoader, model)
never needs to know about the inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CelebDFRecord:
    """One Celeb-DF video entry (after label flip).

    Attributes:
        label: ``0`` real, ``1`` synthetic (our project-wide convention).
        video_path: Absolute path to the ``.mp4`` on disk.
        video_id: Filename stem, e.g. ``id0_id16_0000``.
        manipulation: Top-level folder — one of ``Celeb-real``,
            ``YouTube-real``, ``Celeb-synthesis``.
    """

    label: int
    video_path: Path
    video_id: str
    manipulation: str


def parse_test_list(list_path: Path, dataset_root: Path) -> list[CelebDFRecord]:
    """Parse Celeb-DF's ``List_of_testing_videos.txt`` into typed records.

    Args:
        list_path: Path to ``List_of_testing_videos.txt``.
        dataset_root: Path to the folder that contains ``Celeb-real/``
            etc. (the paths inside the list file are relative to this).

    Returns:
        A list of :class:`CelebDFRecord`, in the order the list file
        defines them. Non-existent files are skipped with a warning.
    """
    if not list_path.exists():
        raise FileNotFoundError(f"Celeb-DF test list not found: {list_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Celeb-DF dataset root not found: {dataset_root}")

    records: list[CelebDFRecord] = []
    missing: list[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logger.warning("Malformed line in %s: %r", list_path.name, line)
                continue
            celebdf_label_str, rel = parts
            try:
                celebdf_label = int(celebdf_label_str)
            except ValueError:
                logger.warning("Unparseable label in %s: %r", list_path.name, line)
                continue

            # Celeb-DF convention: 1=real, 0=fake. Flip to ours.
            our_label = 0 if celebdf_label == 1 else 1
            video_path = (dataset_root / rel).resolve()
            if not video_path.is_file():
                missing.append(rel)
                continue

            manipulation = rel.split("/", 1)[0] if "/" in rel else "unknown"
            records.append(
                CelebDFRecord(
                    label=our_label,
                    video_path=video_path,
                    video_id=video_path.stem,
                    manipulation=manipulation,
                )
            )

    if missing:
        logger.warning(
            "Celeb-DF test list references %d files that don't exist on disk (first 3: %s)",
            len(missing), missing[:3],
        )
    logger.info(
        "Parsed %d Celeb-DF test videos: %d real / %d synthetic",
        len(records),
        sum(1 for r in records if r.label == 0),
        sum(1 for r in records if r.label == 1),
    )
    return records
