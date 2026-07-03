"""End-to-end preprocessing CLI: FF++ videos → per-frame face crops.

Runs two ordered stages:

    1. ``manifest`` — scan ``data/raw/ff_c23/FaceForensics++_C23/`` and
       emit ``data/metadata/ff_c23_manifest.json`` with a leak-free
       80 / 10 / 10 split.
    2. ``faces``    — decode 32 uniform frames per video and cache the
       largest detected face at 224×224 under
       ``data/processed/faces/ff_c23/<split>/<label>/<video_id>/``.

Frame extraction and face cropping are fused in a single pass to avoid
writing ~5,000 × 32 intermediate frame PNGs to disk — only face crops
are persisted.

Idempotency: any video whose output directory already contains at least
one face crop is skipped, so the script is safe to re-run after crashes
or interrupts. Use ``--overwrite`` to force re-extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from tqdm import tqdm

from src.preprocessing import (
    FrameExtractor,
    MTCNNFaceDetector,
    VideoRecord,
    group_split,
    save_crop,
    scan_ff_c23,
    write_manifest,
)
from src.utils import get_logger, load_paths, load_yaml, project_root, select_device, set_seed

app = typer.Typer(add_completion=False, help="FaceForensics++ (c23) preprocessing pipeline.")
logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Helpers — pure functions callable from anywhere
# ---------------------------------------------------------------------

def _load_configs() -> tuple[dict, dict]:
    root = project_root()
    dataset = load_yaml(root / "configs" / "dataset.yaml")["dataset"]
    paths = load_paths()
    return dataset, paths


def _dataset_root(paths: dict, dataset_cfg: dict) -> Path:
    """Return the folder that holds ``original/``, ``Deepfakes/`` etc."""
    raw_root = paths["data"]["raw"] / Path(dataset_cfg["root"]).name
    return raw_root / "FaceForensics++_C23"


def _video_output_dir(faces_root: Path, split: str, record: VideoRecord) -> Path:
    """Return the per-video face-crop output directory."""
    if record.label == "synthetic":
        # Include manipulation so ``Deepfakes/000_003`` and ``Face2Face/000_003``
        # do not collide on disk.
        sub = f"{record.manipulation}_{record.video_id}"
    else:
        sub = record.video_id
    return faces_root / split / record.label / sub


def _build_manifest(seed: int) -> Path:
    """Scan the dataset root and write the JSON manifest. Returns the path."""
    dataset_cfg, paths = _load_configs()
    set_seed(seed)

    ds_root = _dataset_root(paths, dataset_cfg)
    logger.info("Scanning dataset root: %s", ds_root)
    records = scan_ff_c23(ds_root)

    ratios = tuple(dataset_cfg["splits"]["split_ratio"])
    splits = group_split(records, ratios, seed=dataset_cfg["splits"]["split_seed"])

    out_path = paths["data"]["metadata"] / "ff_c23_manifest.json"
    write_manifest(splits, out_path)
    return out_path


def _run_face_extraction(
    limit: int | None,
    split: str | None,
    overwrite: bool,
    device: str | None,
) -> None:
    """Full frame-extraction + face-detection pass."""
    dataset_cfg, paths = _load_configs()

    manifest_path = paths["data"]["metadata"] / "ff_c23_manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(
            f"Manifest not found: {manifest_path}. Run `prepare_dataset manifest` first."
        )
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    sampling = dataset_cfg["sampling"]
    extractor = FrameExtractor(
        num_frames=int(sampling["frames_per_video"]),
        strategy=str(sampling["strategy"]),
    )

    dev = select_device(device)
    logger.info("Face detector device: %s", dev)
    detector = MTCNNFaceDetector(
        image_size=int(sampling["face_crop_size"]),
        margin=20,
        device=dev,
    )

    faces_root = paths["data"]["processed"] / "faces" / "ff_c23"
    faces_root.mkdir(parents=True, exist_ok=True)

    split_names = [split] if split else ["train", "val", "test"]
    for name in split_names:
        if name not in manifest["splits"]:
            raise typer.BadParameter(f"Unknown split '{name}'")
        items_dicts = manifest["splits"][name]
        if limit is not None:
            items_dicts = items_dicts[:limit]

        logger.info("Processing split '%s' (%d videos)", name, len(items_dicts))
        _process_split(items_dicts, extractor, detector, faces_root, name, overwrite)


def _process_split(
    items: list[dict],
    extractor: FrameExtractor,
    detector: MTCNNFaceDetector,
    faces_root: Path,
    split: str,
    overwrite: bool,
) -> None:
    total_missed = 0
    total_ok = 0
    for item in tqdm(items, desc=f"{split:5s}", unit="vid"):
        record = VideoRecord(**item)
        out_dir = _video_output_dir(faces_root, split, record)
        if (not overwrite) and out_dir.is_dir() and any(out_dir.glob("*.jpg")):
            continue

        try:
            frames = extractor.extract(Path(record.rel_path))
        except RuntimeError as exc:
            logger.warning("Skipping %s: %s", record.rel_path, exc)
            continue

        crops, missed = detector.crop_video_frames(frames)
        total_missed += len(missed)
        total_ok += len(crops)

        out_dir.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(crops):
            save_crop(crop, out_dir / f"face_{i:02d}.jpg")

    logger.info(
        "Split '%s' done. face crops written: %d | frames with no face: %d",
        split, total_ok, total_missed,
    )


# ---------------------------------------------------------------------
# CLI commands — thin wrappers around the helpers above
# ---------------------------------------------------------------------

@app.command("manifest")
def cmd_manifest(
    seed: int = typer.Option(42, help="RNG seed for the group shuffle."),
) -> None:
    """Scan the FF++ tree and write a train/val/test manifest JSON."""
    _build_manifest(seed=seed)


@app.command("faces")
def cmd_faces(
    limit: int | None = typer.Option(
        None, help="Process at most N videos per split (smoke-test)."
    ),
    split: str | None = typer.Option(
        None, help='Run only one split — "train", "val", or "test". Default: all.'
    ),
    overwrite: bool = typer.Option(
        False, help="Re-extract even if a video's output directory is non-empty."
    ),
    device: str | None = typer.Option(
        None, help="Force device: mps | cuda | cpu. Default: auto-detect."
    ),
) -> None:
    """Decode each video and save 32 face crops (224×224) per video to disk."""
    _run_face_extraction(limit=limit, split=split, overwrite=overwrite, device=device)


@app.command("all")
def cmd_all(
    limit: int | None = typer.Option(None),
    overwrite: bool = typer.Option(False),
    device: str | None = typer.Option(None),
    seed: int = typer.Option(42),
) -> None:
    """Run manifest + faces stages back-to-back."""
    _build_manifest(seed=seed)
    _run_face_extraction(limit=limit, split=None, overwrite=overwrite, device=device)


if __name__ == "__main__":
    app()
