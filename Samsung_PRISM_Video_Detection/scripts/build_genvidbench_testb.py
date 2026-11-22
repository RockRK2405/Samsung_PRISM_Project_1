#!/usr/bin/env python3
"""GenVidBench Pair2 labels -> test_b.csv (unseen generators + unseen real source).

Pair2 is never trained on. Each line of ``Pair2_labels.txt`` is
``<rel_path> <label>``; the generator/source is the 2nd path component
(e.g. ``Pair2/svd/xxx.mp4``). ``--per-class`` caps videos per folder so the
whole of test-B need not be downloaded and sharded.

    python scripts/build_genvidbench_testb.py --labels Pair2_labels.txt \
        --out manifests/test_b.csv --per-class 2000
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

REAL_FOLDERS = {"hd_vg_130m", "vript"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_folder: dict[str, list[tuple[str, int]]] = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        parts = line.strip().rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        rel, label = parts
        p = Path(rel).parts
        folder = p[1] if len(p) > 2 else p[0]
        by_folder.setdefault(folder, []).append((rel, int(label)))

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "video_path", "label", "generator", "source", "dataset", "split"])
        for folder, items in sorted(by_folder.items()):
            rng.shuffle(items)
            items = items[: args.per_class]
            for rel, _ in items:
                real = folder in REAL_FOLDERS
                # Label comes from the folder, not the file's class index (which may be multi-class).
                w.writerow([f"{folder}__{Path(rel).stem}", f"GenVidBench/{rel}", 0 if real else 1,
                            "" if real else folder, folder if real else "", "genvidbench", "test_b"])
            print(f"{folder:12s} {len(items):6d}  ({'real' if folder in REAL_FOLDERS else 'fake'})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
