"""Build train/val manifests for the general (non-face) detection path.

Normalises a GenImage-family Kaggle download into the ``path,label``
manifest format :mod:`src.datasets.general_image_dataset` expects.
Handles the handful of folder-naming conventions these mirrors actually
ship with, rather than assuming one specific layout — GenImage-derived
Kaggle datasets are not consistent with each other.

Recognised "real" folder names (case-insensitive):
    real, nature, 0_real, original, natural
Recognised "synthetic" folder names (case-insensitive):
    fake, ai, 1_fake, synthetic, generated, sdv1.4, ai_gen

If your download uses different names, pass --real-name / --fake-name
explicitly rather than relying on the auto-detected guess.

Usage
-----
::

    # 1. Download (run on the training machine, NOT here):
    kaggle datasets download -d vtphatt2/genimage-stable-diffusion-v1-4 \\
        --unzip -p data/raw/genimage_sdv14

    # 2. Build manifests:
    python scripts/prepare_general_dataset.py \\
        --data-root data/raw/genimage_sdv14 \\
        --out-dir data/metadata \\
        --val-fraction 0.15

Output
------
``data/metadata/general_train.csv`` and ``general_val.csv``, each with
columns ``path,label`` (0=real, 1=synthetic).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

_REAL_NAMES = {"real", "nature", "0_real", "original", "natural"}
_FAKE_NAMES = {"fake", "ai", "1_fake", "synthetic", "generated", "sdv1.4", "ai_gen", "imagenet_ai_sdv14"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_label_dirs(root: Path, real_name: str | None, fake_name: str | None) -> tuple[list[Path], list[Path]]:
    """Locate every real-labeled and fake-labeled subdirectory under root.

    Searches recursively (GenImage mirrors often nest train/val/generator
    subfolders above the real/fake split) rather than assuming a fixed depth.
    """
    real_dirs: list[Path] = []
    fake_dirs: list[Path] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if real_name:
            if name == real_name.lower():
                real_dirs.append(d)
            continue
        if fake_name:
            if name == fake_name.lower():
                fake_dirs.append(d)
            continue
        if name in _REAL_NAMES:
            real_dirs.append(d)
        elif name in _FAKE_NAMES:
            fake_dirs.append(d)
    return real_dirs, fake_dirs


def collect_images(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        files.extend(p for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    return files


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"--data-root does not exist or is not a directory: {root}")

    real_dirs, fake_dirs = find_label_dirs(root, args.real_name, args.fake_name)
    if not real_dirs or not fake_dirs:
        raise SystemExit(
            f"Could not auto-detect real/fake folders under {root}.\n"
            f"Found real dirs: {[str(d) for d in real_dirs]}\n"
            f"Found fake dirs: {[str(d) for d in fake_dirs]}\n"
            "Pass --real-name / --fake-name explicitly (run `find <root> -type d` "
            "to see the actual folder names in your download)."
        )

    real_files = collect_images(real_dirs)
    fake_files = collect_images(fake_dirs)
    if not real_files or not fake_files:
        raise SystemExit(
            f"Found label directories but no images inside them "
            f"(real={len(real_files)}, fake={len(fake_files)}). "
            f"Check --data-root points at the actual unzipped dataset."
        )
    print(f"Found {len(real_files)} real images across {len(real_dirs)} dir(s)")
    print(f"Found {len(fake_files)} fake images across {len(fake_dirs)} dir(s)")

    # Balance classes by downsampling the larger one -- an unbalanced
    # real/fake ratio would let the model shortcut to predicting the
    # majority class and still score well on accuracy.
    random.seed(args.seed)
    random.shuffle(real_files)
    random.shuffle(fake_files)
    if args.max_per_class:
        real_files = real_files[: args.max_per_class]
        fake_files = fake_files[: args.max_per_class]
    n = min(len(real_files), len(fake_files))
    if len(real_files) != len(fake_files):
        print(f"Balancing classes to {n} images each (was real={len(real_files)}, fake={len(fake_files)})")
    real_files, fake_files = real_files[:n], fake_files[:n]

    rows = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    random.shuffle(rows)

    split_idx = int(len(rows) * (1 - args.val_fraction))
    train_rows, val_rows = rows[:split_idx], rows[split_idx:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(out_dir / "general_train.csv", train_rows)
    _write_manifest(out_dir / "general_val.csv", val_rows)

    print(f"\nWrote {len(train_rows)} train rows -> {out_dir / 'general_train.csv'}")
    print(f"Wrote {len(val_rows)} val rows   -> {out_dir / 'general_val.csv'}")
    print(f"Class balance: {n} real + {n} synthetic per split (before split), 0=real 1=synthetic")


def _write_manifest(path: Path, rows: list[tuple[Path, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for p, label in rows:
            writer.writerow([str(p.resolve()), label])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifests for the general-content detection path.")
    parser.add_argument("--data-root", required=True, help="Root of the unzipped GenImage-family download.")
    parser.add_argument("--out-dir", default="data/metadata", help="Where to write general_train.csv / general_val.csv.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap images per class (for a quick smoke run).")
    parser.add_argument("--real-name", default=None, help="Exact real-folder name, if auto-detection fails.")
    parser.add_argument("--fake-name", default=None, help="Exact fake-folder name, if auto-detection fails.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
