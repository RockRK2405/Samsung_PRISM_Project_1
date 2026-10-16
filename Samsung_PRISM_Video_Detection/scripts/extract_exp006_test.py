#!/usr/bin/env python3
"""Pull the exact 182 EXP-006 test videos out of a local Celeb-DF v2 copy.

    python extract_exp006_test.py <celeb-df-v2-root> <manifest.csv> <dest-dir>

Uses the committed test manifest as the source of truth, so the set is
identical to what ROC-AUC 0.9644 was measured on — no hand-picking.
"""
import csv, shutil, sys
from pathlib import Path

src_root = Path(sys.argv[1]).expanduser()
manifest = Path(sys.argv[2]).expanduser()
dest     = Path(sys.argv[3]).expanduser()

# Celeb-DF's three folders can sit at any depth depending on how it unzipped.
anchors = {}
for name in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
    hits = [p for p in src_root.rglob(name) if p.is_dir()]
    if hits:
        anchors[name] = hits[0]
if not anchors:
    sys.exit(f"ERROR: none of Celeb-real / YouTube-real / Celeb-synthesis found under {src_root}")
print("dataset folders found:")
for k, v in anchors.items():
    print(f"  {k:18s} {len(list(v.glob('*.mp4'))):5d} mp4   {v}")

want = {}
for r in csv.DictReader(manifest.open(encoding="utf-8")):
    parts = Path(r["video_id"]).parts          # celeb-df-v2/<Folder>/<file>.mp4
    folder, fname = parts[-2], parts[-1]
    want[(folder, fname)] = int(r["label"])
print(f"\nmanifest lists {len(want)} unique videos")

ok = miss = 0
missing = []
for (folder, fname), label in sorted(want.items()):
    base = anchors.get(folder)
    srcf = (base / fname) if base else None
    out  = dest / ("fake" if label else "real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    if srcf and srcf.is_file():
        shutil.copy2(srcf, out); ok += 1
    else:
        miss += 1; missing.append(f"{folder}/{fname}")

print(f"\ncopied {ok}, missing {miss}")
for m in missing[:5]:
    print("   missing:", m)
nr = len(list((dest/'real').glob('*.mp4'))) if (dest/'real').is_dir() else 0
nf = len(list((dest/'fake').glob('*.mp4'))) if (dest/'fake').is_dir() else 0
print(f"\n  {dest}/real : {nr:4d}  (expect 156)")
print(f"  {dest}/fake : {nf:4d}  (expect  26)")
print("\nMATCH -- this is the exact EXP-006 test set." if (nr, nf) == (156, 26)
      else "\nCOUNTS DIFFER -- check the dataset root path above before scoring.")
