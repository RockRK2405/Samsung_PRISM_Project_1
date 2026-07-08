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


# ---------------------------------------------------------------------
# Training-set scanning (for Option A — mixed FF++ + Celeb-DF training)
# ---------------------------------------------------------------------

CELEB_REAL_FOLDER = "Celeb-real"
CELEB_YOUTUBE_REAL_FOLDER = "YouTube-real"
CELEB_SYNTHESIS_FOLDER = "Celeb-synthesis"


def _identity_of(rel_path: str) -> str:
    """Return an identity-group key for split-safe splitting.

    * ``Celeb-real/id0_0000.mp4``      → ``"celeb_id0"``
    * ``Celeb-synthesis/id0_id16_0000.mp4`` → ``"celeb_id0"``  (source identity)
    * ``YouTube-real/00170.mp4``       → ``"yt_00170"``        (each YT video is its own subject)
    """
    parts = rel_path.split("/", 1)
    if len(parts) != 2:
        return rel_path
    folder, name = parts
    stem = Path(name).stem
    if folder == CELEB_YOUTUBE_REAL_FOLDER:
        return f"yt_{stem}"
    # Both Celeb-real and Celeb-synthesis start with idN — use the first token.
    return f"celeb_{stem.split('_', 1)[0]}"


def scan_celeb_df_training(
    dataset_root: Path,
    test_list_path: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_per_source: int | None = None,
) -> dict[str, list[CelebDFRecord]]:
    """Enumerate all Celeb-DF videos not in the test list, split into train/val.

    The split is **identity-safe**: no ``celeb_idN`` group appears in both
    train and val, and the test-set identities are excluded from both
    train and val entirely.

    Args:
        dataset_root: Folder containing ``Celeb-real/`` etc.
        test_list_path: Path to ``List_of_testing_videos.txt``.
        val_ratio: Fraction of the (identity-shuffled) training pool held
            out for val. Default ``0.15``.
        seed: RNG seed applied to the identity-order shuffle.
        max_per_source: Optional per-folder cap. Applied *before* identity
            grouping so a smoke run can complete quickly (e.g. ``100``).

    Returns:
        Mapping ``{"train": [...], "val": [...]}``.
    """
    import random

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Celeb-DF root not found: {dataset_root}")

    test_records = parse_test_list(test_list_path, dataset_root)
    test_identities = {_identity_of(str(r.video_path.relative_to(dataset_root))) for r in test_records}
    test_paths = {r.video_path for r in test_records}
    logger.info(
        "Excluding %d test identities (from %d test videos) from training pool",
        len(test_identities), len(test_paths),
    )

    folders = [
        (CELEB_REAL_FOLDER, 0),
        (CELEB_YOUTUBE_REAL_FOLDER, 0),
        (CELEB_SYNTHESIS_FOLDER, 1),
    ]

    all_records: list[CelebDFRecord] = []
    for folder_name, label in folders:
        folder = dataset_root / folder_name
        if not folder.is_dir():
            logger.warning("Celeb-DF folder missing: %s", folder)
            continue
        candidates: list[CelebDFRecord] = []
        for mp4 in sorted(folder.glob("*.mp4")):
            if mp4 in test_paths:
                continue
            rel = f"{folder_name}/{mp4.name}"
            if _identity_of(rel) in test_identities:
                continue
            candidates.append(
                CelebDFRecord(
                    label=label,
                    video_path=mp4.resolve(),
                    video_id=mp4.stem,
                    manipulation=folder_name,
                )
            )
        if max_per_source is not None and len(candidates) > max_per_source:
            rng = random.Random(seed + hash(folder_name) % 1000)
            rng.shuffle(candidates)
            candidates = candidates[:max_per_source]
        logger.info("  %s: %d videos kept", folder_name, len(candidates))
        all_records.extend(candidates)

    # Split by identity
    groups: dict[str, list[CelebDFRecord]] = {}
    for rec in all_records:
        rel = f"{rec.manipulation}/{rec.video_path.name}"
        key = _identity_of(rel)
        groups.setdefault(key, []).append(rec)

    rng = random.Random(seed)
    ids = sorted(groups.keys())
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])

    train: list[CelebDFRecord] = []
    val: list[CelebDFRecord] = []
    for key, recs in groups.items():
        (val if key in val_ids else train).extend(recs)

    logger.info(
        "Celeb-DF training pool split: train=%d videos (%d identities), val=%d videos (%d identities)",
        len(train), len(ids) - n_val, len(val), n_val,
    )
    return {"train": train, "val": val}
