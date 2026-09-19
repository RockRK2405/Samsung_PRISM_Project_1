"""Export the videos referenced by the current manifests into one folder,
preserving the ``Pair1/<generator>/<file>.mp4`` layout, so the subset can be
zipped and uploaded to Kaggle/Colab for GPU training (ADR-007 Plan B).

The full GenVidBench Pair1 is ~74GB — too big for a Kaggle notebook's disk
once you add extraction. But a 10k/2k/2k run only touches ~14k videos
(~20-25GB), which fits. This copies exactly those videos plus a filtered
label file, so ``prepare_genvidbench.py`` can rebuild identical-protocol
manifests on the GPU box.

Usage
-----
::

    python scripts/export_subset.py \\
        --manifest-dir data/manifests \\
        --data-root data/raw/genvidbench/GenVidBench \\
        --out data/genvidbench_subset

Then zip and upload ``data/genvidbench_subset`` as a private Kaggle dataset.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def _read_paths(manifest: Path) -> list[Path]:
    if not manifest.is_file():
        return []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        return [Path(row["video_path"]) for row in csv.DictReader(f)]


def main() -> None:
    p = argparse.ArgumentParser(description="Export manifest-referenced videos into a portable subset.")
    p.add_argument("--manifest-dir", default="data/manifests")
    p.add_argument("--data-root", default="data/raw/genvidbench/GenVidBench",
                   help="Root the label paths (Pair1/...) are relative to.")
    p.add_argument("--out", default="data/genvidbench_subset")
    p.add_argument("--labels", nargs="*",
                   default=["data/raw/genvidbench/GenVidBench/Pair1_labels.txt"])
    args = p.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path(args.manifest_dir)
    all_paths: set[Path] = set()
    for split in ("train.csv", "val.csv", "test.csv"):
        paths = _read_paths(manifest_dir / split)
        all_paths.update(paths)
        print(f"{split}: {len(paths)} videos")
    print(f"Unique videos to copy: {len(all_paths)}")

    copied = 0
    missing = 0
    total_bytes = 0
    for src in sorted(all_paths):
        if not src.is_file():
            missing += 1
            continue
        # Preserve the Pair1/<gen>/<file> tail so the label paths still resolve.
        try:
            rel = src.relative_to(data_root.resolve())
        except ValueError:
            # Fall back to last 3 components (Pair1/<gen>/<file>).
            rel = Path(*src.parts[-3:])
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            total_bytes += dst.stat().st_size
        copied += 1
        if copied % 1000 == 0:
            print(f"  copied {copied}/{len(all_paths)} ({total_bytes/1e9:.1f} GB)")

    # Copy the label file(s) so manifests can be rebuilt on the GPU box.
    for lf in args.labels:
        lfp = Path(lf)
        if lfp.is_file():
            shutil.copy2(lfp, out / lfp.name)
            print(f"copied label file: {lfp.name}")

    print(f"\nDone. Copied {copied} videos ({total_bytes/1e9:.1f} GB), {missing} missing, to {out}")
    print(f"Next: zip '{out}' and upload as a private Kaggle dataset.")


if __name__ == "__main__":
    main()
