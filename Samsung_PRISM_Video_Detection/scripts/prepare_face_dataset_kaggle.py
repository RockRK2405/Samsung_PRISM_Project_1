"""Build a unified face-crop training set from FF++ + Celeb-DF + DFDC on Kaggle.

Samples K frames per video, runs MTCNN, saves 224x224 face crops as JPGs,
and emits a manifest CSV in the SAME format ``GeneralImageDataset`` reads
(``path,label``). That lets us reuse ``train_general.py`` unchanged for
the face-path — same BaselineDetector architecture, different data, so
the produced checkpoint slots straight into ``VideoDetector`` (``best.pt``).

Handles each dataset's real/fake convention:
    * FaceForensics++ C23: ``original_sequences/`` (real) vs ``manipulated_sequences/`` (fake)
    * Celeb-DF v2: ``Celeb-real/`` + ``YouTube-real/`` (real) vs ``Celeb-synthesis/`` (fake)
    * DFDC: per-folder ``metadata.json`` maps each ``*.mp4`` to ``REAL``/``FAKE``

Usage
-----
::

    python scripts/prepare_face_dataset_kaggle.py \\
        --ff-root  /kaggle/input/datasets/<owner>/faceforensics-c23 \\
        --celeb-root /kaggle/input/datasets/<owner>/celeb-df-v2 \\
        --dfdc-root /kaggle/input/deepfake-detection-challenge \\
        --out-dir /kaggle/working/face_crops \\
        --manifest-dir /kaggle/working/manifests \\
        --frames-per-video 8 \\
        --videos-per-class 800

Missing dataset roots are skipped with a warning rather than failing —
so you can run with only the roots you have available.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.preprocessing import FrameExtractor, MTCNNFaceDetector

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_videos(root: Path, patterns: list[str]) -> list[Path]:
    """Find every video file under `root` matching any of the glob patterns."""
    if not root.is_dir():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return [f for f in files if f.suffix.lower() in _VIDEO_EXTS]


def collect_ff(root: Path) -> tuple[list[Path], list[Path]]:
    # Canonical FF++ layout (original_sequences / manipulated_sequences) AND
    # the flat-named layout most Kaggle mirrors ship with (original + one folder
    # per manipulation method).
    real_patterns = [
        "original_sequences/**/*.mp4", "**/original_sequences/**/*.mp4",
        "**/original/**/*.mp4",
    ]
    fake_patterns = [
        "manipulated_sequences/**/*.mp4", "**/manipulated_sequences/**/*.mp4",
    ]
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"):
        fake_patterns.append(f"**/{method}/**/*.mp4")
    real = collect_videos(root, real_patterns)
    fake = collect_videos(root, fake_patterns)
    return real, fake


def collect_celeb(root: Path) -> tuple[list[Path], list[Path]]:
    real = collect_videos(root, ["**/Celeb-real/**/*.mp4", "**/YouTube-real/**/*.mp4"])
    fake = collect_videos(root, ["**/Celeb-synthesis/**/*.mp4"])
    return real, fake


def collect_dfdc(root: Path) -> tuple[list[Path], list[Path]]:
    """DFDC uses per-part metadata.json to label each mp4 as REAL/FAKE."""
    real: list[Path] = []
    fake: list[Path] = []
    for meta_path in root.rglob("metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        base = meta_path.parent
        for video_name, info in meta.items():
            v = base / video_name
            if not v.is_file():
                continue
            label = str(info.get("label", "")).upper()
            if label == "REAL":
                real.append(v)
            elif label == "FAKE":
                fake.append(v)
    # Some DFDC mirrors omit metadata.json for the "test" folder — those go unlabeled.
    return real, fake


def extract_crops(
    videos: list[Path],
    label: int,
    label_dir: Path,
    extractor: FrameExtractor,
    detector: MTCNNFaceDetector,
    tag: str,
    seen_names: set[str],
) -> list[tuple[Path, int]]:
    """Run frame-sample -> MTCNN -> save crops. Returns (crop_path, label) rows."""
    rows: list[tuple[Path, int]] = []
    for video in tqdm(videos, desc=f"{tag} ({'real' if label == 0 else 'fake'})", unit="vid"):
        stem = f"{tag}__{video.stem}"
        # Guard against duplicate stems across datasets (e.g. DFDC "abcd.mp4"
        # colliding with a Celeb-DF file of the same name).
        if stem in seen_names:
            stem = f"{stem}__{len(seen_names)}"
        seen_names.add(stem)
        try:
            frames = extractor.extract(video)
        except Exception:
            continue
        for i, frame in enumerate(frames):
            try:
                crop = detector.detect_and_crop(frame)
            except Exception:
                crop = None
            if crop is None:
                continue
            out_path = label_dir / f"{stem}_f{i:02d}.jpg"
            Image.fromarray(crop).save(out_path, quality=90)
            rows.append((out_path, label))
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device for MTCNN: {device}")
    extractor = FrameExtractor(num_frames=args.frames_per_video, strategy="uniform")
    detector = MTCNNFaceDetector(image_size=224, margin=20, device=device)

    all_real: list[Path] = []
    all_fake: list[Path] = []
    for root_str, collector, tag in [
        (args.ff_root, collect_ff, "ff"),
        (args.celeb_root, collect_celeb, "celeb"),
        (args.dfdc_root, collect_dfdc, "dfdc"),
    ]:
        if not root_str:
            continue
        root = Path(root_str)
        if not root.is_dir():
            print(f"[WARN] {tag} root not found, skipping: {root}")
            continue
        r, f = collector(root)
        print(f"{tag}: {len(r)} real, {len(f)} fake videos")
        random.shuffle(r)
        random.shuffle(f)
        if args.videos_per_class:
            r = r[: args.videos_per_class]
            f = f[: args.videos_per_class]
        all_real.append((tag, r))
        all_fake.append((tag, f))

    seen: set[str] = set()
    real_rows: list[tuple[Path, int]] = []
    fake_rows: list[tuple[Path, int]] = []
    for tag, vids in all_real:
        real_rows.extend(extract_crops(vids, 0, out_dir / "real", extractor, detector, tag, seen))
    for tag, vids in all_fake:
        fake_rows.extend(extract_crops(vids, 1, out_dir / "fake", extractor, detector, tag, seen))

    print(f"\nExtracted {len(real_rows)} real crops, {len(fake_rows)} fake crops")
    # Balance to avoid the model shortcutting to the majority class.
    n = min(len(real_rows), len(fake_rows))
    if len(real_rows) != len(fake_rows):
        print(f"Balancing to {n} per class")
    random.shuffle(real_rows)
    random.shuffle(fake_rows)
    rows = real_rows[:n] + fake_rows[:n]
    random.shuffle(rows)

    split = int(len(rows) * (1 - args.val_fraction))
    train_rows, val_rows = rows[:split], rows[split:]

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_dir / "face_train.csv", train_rows)
    _write_manifest(manifest_dir / "face_val.csv", val_rows)
    print(f"\nWrote {len(train_rows)} train rows -> {manifest_dir / 'face_train.csv'}")
    print(f"Wrote {len(val_rows)} val rows   -> {manifest_dir / 'face_val.csv'}")


def _write_manifest(path: Path, rows: list[tuple[Path, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for p, label in rows:
            w.writerow([str(p.resolve()), label])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a unified face-crop dataset for face-path retraining.")
    p.add_argument("--ff-root", default=None, help="FaceForensics++ dataset root (optional).")
    p.add_argument("--celeb-root", default=None, help="Celeb-DF v2 dataset root (optional).")
    p.add_argument("--dfdc-root", default=None, help="DFDC competition root (optional).")
    p.add_argument("--out-dir", default="/kaggle/working/face_crops", help="Where cropped face JPGs go.")
    p.add_argument("--manifest-dir", default="/kaggle/working/manifests", help="Where face_train.csv / face_val.csv go.")
    p.add_argument("--frames-per-video", type=int, default=8, help="How many frames to sample from each video.")
    p.add_argument("--videos-per-class", type=int, default=800, help="Cap videos per class per dataset (None = all).")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
