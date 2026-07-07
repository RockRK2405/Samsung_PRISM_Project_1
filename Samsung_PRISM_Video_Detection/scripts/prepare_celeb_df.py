"""Extract face crops for Celeb-DF v2's official test split.

Reads Celeb-DF's ``List_of_testing_videos.txt`` to select the standard
518-video test set (published cross-dataset benchmarks report on this
same split, so our numbers stay directly comparable to literature),
then runs the same frame-sampling + MTCNN face-cropping pipeline used
for FF++.

Face crops land under a layout compatible with
:class:`~src.datasets.FaceVideoDataset` so the evaluator can be pointed
at ``data/processed/faces/celeb_df_v2`` without any code change:

    data/processed/faces/celeb_df_v2/test/{real,synthetic}/<video>/face_NN.jpg

Idempotent — videos whose output dir already contains a crop are
skipped unless ``--overwrite`` is passed.

Example
-------
::

    python scripts/prepare_celeb_df.py \\
        --dataset-root ~/Downloads/Celeb-DF-v2 \\
        --limit 5              # smoke test first

    python scripts/prepare_celeb_df.py \\
        --dataset-root ~/Downloads/Celeb-DF-v2
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
    parse_test_list,
    save_crop,
)
from src.utils import get_logger, load_paths, load_yaml, project_root, select_device

app = typer.Typer(add_completion=False, help="Extract Celeb-DF v2 test-split face crops.")
logger = get_logger(__name__)


@app.command()
def main(
    dataset_root: Path = typer.Option(
        ..., "--dataset-root",
        help="Path to the unzipped Celeb-DF-v2 folder (holds Celeb-real/ etc.).",
    ),
    test_list: Path | None = typer.Option(
        None,
        help="Path to List_of_testing_videos.txt "
        "(default: <dataset-root>/List_of_testing_videos.txt).",
    ),
    limit: int | None = typer.Option(
        None, help="Process at most N videos (for smoke tests).",
    ),
    overwrite: bool = typer.Option(
        False, help="Re-extract even if the per-video output dir is non-empty.",
    ),
    device: str | None = typer.Option(
        None, help="Force MTCNN device: mps | cuda | cpu (default: cpu, per FF++ workaround).",
    ),
) -> None:
    """Extract 32 face crops per Celeb-DF test video to the shared cache root."""
    root = project_root()
    ds_cfg = load_yaml(root / "configs" / "dataset.yaml")["dataset"]
    paths = load_paths()

    if test_list is None:
        test_list = dataset_root / "List_of_testing_videos.txt"

    records = parse_test_list(test_list, dataset_root)
    if limit is not None:
        records = records[:limit]
        logger.info("Smoke-test mode: processing only %d videos", len(records))

    sampling = ds_cfg["sampling"]
    extractor = FrameExtractor(
        num_frames=int(sampling["frames_per_video"]),
        strategy=str(sampling["strategy"]),
    )

    # Same MPS workaround as FF++ — MTCNN's image pyramid breaks
    # adaptive pooling on Apple Silicon.
    dev = "cpu" if device is None else select_device(device)
    logger.info("Face detector device: %s", dev)
    detector = MTCNNFaceDetector(
        image_size=int(sampling["face_crop_size"]),
        margin=20,
        device=dev,
    )

    faces_root = paths["data"]["processed"] / "faces" / "celeb_df_v2"
    faces_root.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_missed = 0
    n_skipped_existing = 0
    for rec in tqdm(records, desc="celeb-df", unit="vid"):
        label_name = "real" if rec.label == 0 else "synthetic"
        out_dir = faces_root / "test" / label_name / f"{rec.manipulation}_{rec.video_id}"
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
        "Done. crops_written=%d frames_with_no_face=%d already_had_crops=%d",
        total_ok, total_missed, n_skipped_existing,
    )
    logger.info("Face-crop cache: %s", faces_root)


if __name__ == "__main__":
    app()
