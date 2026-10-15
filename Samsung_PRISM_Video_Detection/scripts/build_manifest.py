#!/usr/bin/env python3
"""Build a labeled test manifest CSV from directories of real/fake media.

    # one real dir, one fake dir
    python scripts/build_manifest.py --out test.csv \
        --real /data/vript --fake /data/t2vz

    # several fake generators, each its own reported group
    python scripts/build_manifest.py --out test.csv \
        --real /data/vript:vript \
        --fake /data/t2vz:text2video-zero --fake /data/pika:pika \
        --per-group 250

Each --real/--fake takes `PATH` or `PATH:group_label`. The group label is what
`score_manifest.py` breaks results down by, which is how you read "recall on the
unseen generator" instead of one number that averages it away. Without an
explicit label the directory name is used.

Output: `path,label,group` with label 1=fake, 0=real. Sampling is seeded, so
the same flags always produce the same manifest.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".webp"}


def _spec(s: str) -> tuple[Path, str | None]:
    """Parse `PATH` or `PATH:group`, tolerating Windows drive letters."""
    if ":" in s[2:]:                      # skip a leading "C:" style prefix
        head, _, tail = s.rpartition(":")
        if head and tail and "/" not in tail and "\\" not in tail:
            return Path(head).expanduser(), tail
    return Path(s).expanduser(), None


def _collect(root: Path, cap: int, rng: random.Random) -> list[Path]:
    if not root.is_dir():
        print(f"  WARNING: not a directory, skipped: {root}")
        return []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXT and p.is_file())
    if not files:
        print(f"  WARNING: no media files under {root}")
        return []
    if cap and len(files) > cap:
        files = rng.sample(files, cap)
        files.sort()
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="append", default=[], metavar="PATH[:group]")
    ap.add_argument("--fake", action="append", default=[], metavar="PATH[:group]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=0,
                    help="Cap files taken per directory (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.real and not args.fake:
        print("ERROR: give at least one --real or --fake directory.")
        return 2

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for specs, label in ((args.real, 0), (args.fake, 1)):
        for s in specs:
            root, group = _spec(s)
            group = group or root.name
            files = _collect(root, args.per_group, rng)
            for p in files:
                rows.append({"path": str(p.resolve()), "label": label, "group": group})
            print(f"  {'fake' if label else 'real'}  {group:22s} {len(files):5d} files  <- {root}")

    if not rows:
        print("\nERROR: zero files collected — check your paths.")
        return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group"])
        w.writeheader()
        w.writerows(rows)

    n_fake = sum(r["label"] for r in rows)
    print(f"\nwrote {out}  ({len(rows)} rows: {len(rows) - n_fake} real / {n_fake} fake)")
    if n_fake == 0 or n_fake == len(rows):
        print("  NOTE: only one class present. ROC-AUC will be undefined (n/a);")
        print("        you will get recall or specificity instead. Add the other class")
        print("        for a usable AUC.")
    print(f"\nNext:\n  python scripts/score_manifest.py --manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
