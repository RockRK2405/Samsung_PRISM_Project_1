"""Extract face crops for Celeb-DF v2's TRAINING pool (non-test videos).

Mirrors ``prepare_celeb_df.py`` but points at the non-test half of the
dataset. Test videos and any videos sharing their identity are excluded
so we don't leak subjects across train / val / test.

Output layout matches :class:`FaceVideoDataset`'s expectations:

    data/processed/faces/celeb_df_v2/{train,val}/{real,synthetic}/<video>/face_NN.jpg

so training code can point ``FaceFrameDataset`` at
``data/processed/faces/celeb_df_v2`` and get train/val splits directly.

Idempotent — videos with an already-populated output dir are skipped
unless ``--overwrite`` is passed.

Warning
-------
Full extraction is **~3-5 hours on M5 CPU MTCNN** (5000+ videos). Use
``--max-per-source`` for smoke tests. ``--max-per-source 200`` gives you
600 videos and finishes in ~20 minutes.

Example
-------
::

    # Smoke test on a small pool
    python scripts/prepare_celeb_df_train.py \\
        --dataset-root data/raw/celeb_df_v2/Celeb-DF-v2 --max-per-source 50

    # Full extraction
    python scripts/prepare_celeb_df_train.py \\
        --dataset-root data/raw/celeb_df_v2/Celeb-DF-v2
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from tqdm import tqdm

from src.preprocessing import (
    FrameExtractor,
    MTCNNFaceDetector,
    save_crop,
    scan_celeb_df_training,
)
from src.utils import get_logger, load_paths, load_yaml, project_root, select_device

app = typer.Typer(add_completion=False, help="Extract Celeb-DF v2 training-pool face crops.")
logger = get_logger(__name__)


@app.command()
def main(
    dataset_root: Path = typer.Option(
        ..., "--dataset-root",
        help="Path to the unzipped Celeb-DF-v2 folder (holds Celeb-real/ etc.).",
    ),
    test_list: Path | None = typer.Option(
        None, help="Path to List_of_testing_videos.txt (default: <dataset-root>/…).",
    ),
    val_ratio: float = typer.Option(
        0.15, help="Fraction of the training pool held out for val.",
    ),
    max_per_source: int | None = typer.Option(
        None,
        help="Cap each of Celeb-real / YouTube-real / Celeb-synthesis to N videos. "
        "Useful for smoke tests; leave unset for the full run.",
    ),
    overwrite: bool = typer.Option(
        False, help="Re-extract even if the per-video output dir is non-empty.",
    ),
    device: str | None = typer.Option(
        None, help="Force MTCNN device: mps | cuda | cpu (default: cpu).",
    ),
    seed: int = typer.Option(42, help="Seed for the identity-order shuffle."),
) -> None:
    """Extract face crops for the training + val splits of Celeb-DF v2."""
    root = project_root()
    ds_cfg = load_yaml(root / "configs" / "dataset.yaml")["dataset"]
    paths = load_paths()

    if test_list is None:
        test_list = dataset_root / "List_of_testing_videos.txt"

    splits = scan_celeb_df_training(
        dataset_root=dataset_root,
        test_list_path=test_list,
        val_ratio=val_ratio,
        seed=seed,
        max_per_source=max_per_source,
    )

    sampling = ds_cfg["sampling"]
    extractor = FrameExtractor(
        num_frames=int(sampling["frames_per_video"]),
        strategy=str(sampling["strategy"]),
    )

    dev = "cpu" if device is None else select_device(device)
    logger.info("Face detector device: %s", dev)
    detector = MTCNNFaceDetector(
        image_size=int(sampling["face_crop_size"]),
        margin=20,
        device=dev,
    )

    faces_root = paths["data"]["processed"] / "faces" / "celeb_df_v2"
    faces_root.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        logger.info("Processing split '%s' (%d videos)", split_name, len(records))
        total_ok = 0
        total_missed = 0
        n_skipped_existing = 0
        for rec in tqdm(records, desc=split_name, unit="vid"):
            label_name = "real" if rec.label == 0 else "synthetic"
            out_dir = (
                faces_root / split_name / label_name / f"{rec.manipulation}_{rec.video_id}"
            )
            if (not overwrite) and out_dir.is_dir() and any(out_dir.glob("*.jpg")):
                n_skipped_existing += 1
                continue

            try:
                frames = extractor.extract(rec.video_path)
            except RuntimeError as exc:
                logger.warning("Skipping %s: %s", rec.video_path, exc)
                continue

            crops, missed = detector.crop_video_frames(frames)
            total_ok += len(crops)
            total_missed += len(missed)

            out_dir.mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(crops):
                save_crop(crop, out_dir / f"face_{i:02d}.jpg")

        logger.info(
            "  '%s' done. crops=%d missed_frames=%d already_had_crops=%d",
            split_name, total_ok, total_missed, n_skipped_existing,
        )

    logger.info("Face-crop cache: %s", faces_root)


if __name__ == "__main__":
    app()
