#!/usr/bin/env python3
"""Registry-driven dataset prep: N deepfake datasets -> train/val/test manifests.

Generalises `prepare_face_dataset_kaggle.py` (hardcoded to FF++/Celeb-DF/DFDC)
to every dataset in `configs/dataset_registry.yaml`, which encodes the owner's
spreadsheet. Split roles come from the sheet's "Recommended Evaluation Use":

  train_pool    -> pooled, then split train/val VIDEO-DISJOINTLY
  external_test -> never trained on; one manifest (and one metric) EACH

Typical use (datasets as subdirectories named by registry key):

    python scripts/prepare_from_registry.py \
        --data-root /content/data \
        --out-dir /content/crops --manifest-dir /content/manifests \
        --priority P0 --videos-per-class 400

Explicit roots, when directory names differ:

    python scripts/prepare_from_registry.py \
        --root faceforensics++=/content/ffpp \
        --root celeb-df-v2=/content/celeb

Before trusting any dataset whose registry entry says `verified: false`,
print its real structure and fix the globs:

    python scripts/prepare_from_registry.py --root kodf=/content/kodf \
        --inspect kodf

Outputs `path,label,video_id,dataset,group` — extra columns are ignored by
`GeneralImageDataset`, so these manifests drop straight into the existing
`train_general.py` and `eval_face_crops.py`.

Leakage rules enforced here (the run ABORTS on violation, it does not warn):
  1. No video contributes crops to both train and val.
  2. No external_test dataset contributes to train or val.
  3. A dataset marked as overlapping another is refused unless --allow-overlap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except ImportError:
    raise SystemExit("Needs pyyaml. pip install pyyaml")

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Datasets that overlap another and must not be pooled naively.
_OVERLAP = {
    "celeb-df": "celeb-df-v2",
    "celeb-df++": "celeb-df-v2",
    "dfdc-preview": "dfdc",
    "faceforensics": "faceforensics++",
}


# ------------------------------------------------------------------ discovery
def inspect(root: Path, depth: int = 3, per_dir: int = 6) -> None:
    """Print the real tree so unverified globs can be corrected by hand."""
    print(f"\nINSPECT {root}")
    if not root.is_dir():
        print("  NOT A DIRECTORY"); return
    counts: dict[str, int] = defaultdict(int)
    for p in root.rglob("*"):
        if p.is_file():
            counts[p.suffix.lower()] += 1
    print("  file types:", dict(sorted(counts.items(), key=lambda kv: -kv[1])[:10]) or "none")

    def walk(d: Path, level: int) -> None:
        if level > depth:
            return
        try:
            kids = sorted(d.iterdir())
        except PermissionError:
            return
        dirs = [k for k in kids if k.is_dir()]
        files = [k for k in kids if k.is_file()]
        pad = "  " * (level + 1)
        for k in dirs[:per_dir]:
            n = sum(1 for _ in k.rglob("*") if _.is_file())
            print(f"{pad}{k.name}/   ({n} files)")
            walk(k, level + 1)
        if len(dirs) > per_dir:
            print(f"{pad}... +{len(dirs) - per_dir} more dirs")
        for k in files[:3]:
            print(f"{pad}{k.name}")
        if len(files) > 3:
            print(f"{pad}... +{len(files) - 3} more files")

    walk(root, 0)
    print("\n  -> Put the correct patterns into real_globs / fake_globs in "
          "configs/dataset_registry.yaml, then set verified: true.\n")


def _glob_many(root: Path, patterns: list[str], exts: set[str]) -> list[Path]:
    seen: dict[Path, None] = {}
    for pat in patterns or []:
        for p in root.glob(pat):
            if p.is_file() and p.suffix.lower() in exts:
                seen.setdefault(p.resolve(), None)
    return sorted(seen)


def collect(name: str, root: Path, entry: dict) -> tuple[list[Path], list[Path]]:
    """Return (real_files, fake_files) for one dataset, per its layout mode."""
    layout = entry.get("layout") or {}
    mode = layout.get("mode", "globs")
    exts = _VIDEO_EXT | _IMAGE_EXT

    if mode == "metadata_json":
        reals: list[Path] = []
        fakes: list[Path] = []
        metas = list(root.glob(layout.get("metadata_glob", "**/metadata.json")))
        if not metas:
            print(f"  [{name}] no metadata.json found under {root}")
            return [], []
        for meta in metas:
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] unreadable {meta}: {exc}")
                continue
            part = meta.parent
            for fname, info in (data or {}).items():
                lbl = str((info or {}).get("label", "")).upper()
                f = part / fname
                if not f.is_file():
                    continue
                (fakes if lbl == "FAKE" else reals if lbl == "REAL" else []).append(f)
        return sorted(set(reals)), sorted(set(fakes))

    if mode == "label_file":
        print(f"  [{name}] layout mode 'label_file' needs a label_file path in the "
              f"registry — skipping. Run --inspect {name}.")
        return [], []

    reals = _glob_many(root, layout.get("real_globs", []), exts)
    fakes = _glob_many(root, layout.get("fake_globs", []), exts)
    return reals, fakes


# ------------------------------------------------------------------ extraction
def _video_id(name: str, path: Path, root: Path) -> str:
    """Stable, collision-resistant per-video id: dataset + relative path."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(path.name)
    return f"{name}/{rel.as_posix()}"


def _identities(video_id: str, regex: str | None) -> frozenset[str]:
    """All identity tokens in a filename, e.g. id0_id16_0000.mp4 -> {id0, id16}."""
    if not regex:
        return frozenset()
    return frozenset(m.group(0) for m in re.finditer(regex, Path(video_id).name))


class Extractor:
    """Samples frames and (optionally) MTCNN-crops faces. Resumable."""

    def __init__(self, frames: int, image_size: int, face_crop: bool, device: str | None):
        self.frames = frames
        self.image_size = image_size
        self.face_crop = face_crop
        from src.preprocessing import FrameExtractor  # noqa: PLC0415
        self._fx = FrameExtractor(num_frames=frames, strategy="uniform")
        self._fd = None
        if face_crop:
            from src.preprocessing import MTCNNFaceDetector  # noqa: PLC0415
            import torch  # noqa: PLC0415
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            # MTCNN returns None for every frame on MPS -- pin to CPU there.
            if str(dev).startswith("mps"):
                dev = "cpu"
            self._fd = MTCNNFaceDetector(image_size=image_size, margin=20, device=dev)
            print(f"  MTCNN on {dev}")

    def run(self, src: Path, dest_dir: Path) -> list[Path]:
        """Extract `src` into `dest_dir`; returns written image paths."""
        existing = sorted(dest_dir.glob("crop_*.jpg"))
        if len(existing) >= self.frames:
            return existing[: self.frames]           # resume: already done

        from PIL import Image  # noqa: PLC0415

        # An image-only dataset (e.g. WildDeepfake) needs no decoding.
        if src.suffix.lower() in _IMAGE_EXT:
            dest_dir.mkdir(parents=True, exist_ok=True)
            out = dest_dir / "crop_00.jpg"
            Image.open(src).convert("RGB").resize(
                (self.image_size, self.image_size)).save(out, quality=95)
            return [out]

        frames = self._fx.extract(src)
        if not frames:
            return []
        if self._fd is not None:
            crops, _missed = self._fd.crop_video_frames(frames)
            if not crops:
                return []                            # no face anywhere: skip video
            imgs = crops
        else:
            imgs = frames

        dest_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, arr in enumerate(imgs[: self.frames]):
            out = dest_dir / f"crop_{i:02d}.jpg"
            im = Image.fromarray(arr).convert("RGB")
            if im.size != (self.image_size, self.image_size):
                im = im.resize((self.image_size, self.image_size))
            im.save(out, quality=95)
            written.append(out)
        return written


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(_ROOT / "configs" / "dataset_registry.yaml"))
    ap.add_argument("--data-root", default=None,
                    help="Parent dir; each dataset in a subdir named by registry key.")
    ap.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                    help="Explicit root for one dataset. Repeatable.")
    ap.add_argument("--out-dir", default="data/processed/registry_crops")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--only", default=None, help="Comma-separated registry keys.")
    ap.add_argument("--priority", default=None, help="Comma-separated, e.g. P0,P1.")
    ap.add_argument("--frames-per-video", type=int, default=None)
    ap.add_argument("--videos-per-class", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-face-crop", action="store_true",
                    help="Keep whole frames instead of MTCNN face crops.")
    ap.add_argument("--device", default=None, help="cuda | cpu (MTCNN).")
    ap.add_argument("--inspect", default=None, help="Print one dataset's tree and exit.")
    ap.add_argument("--allow-overlap", action="store_true",
                    help="Permit datasets that overlap another (e.g. Celeb-DF v1+v2).")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="Use datasets whose layout is not verified (default: skip).")
    ap.add_argument("--role", action="append", default=[], metavar="NAME=ROLE",
                    help="Override a dataset's split_role (train_pool|external_test). Repeatable.")
    ap.add_argument("--test-fraction", type=float, default=0.0,
                    help="Carve a TEST split out of train_pool too (for when no external "
                         "test set is available). Disjoint by --split-key.")
    ap.add_argument("--split-key", choices=["video", "identity"], default="video",
                    help="What train/val/test must be disjoint on. 'identity' uses the "
                         "dataset's identity_regex -- essential for Celeb-DF, where a "
                         "video-level split leaks the same celebrity into both sides.")
    ap.add_argument("--allow-imbalance", action="store_true",
                    help="Proceed despite a pathological train class balance (default: abort).")
    ap.add_argument("--min-minority-videos", type=int, default=20,
                    help="Abort if the rarer train class has fewer videos than this.")
    ap.add_argument("--min-minority-frac", type=float, default=0.20,
                    help="Abort if the rarer train class is below this fraction of train videos.")
    args = ap.parse_args()

    reg = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    defaults = reg.get("defaults", {})
    datasets: dict[str, dict] = reg["datasets"]

    frames = args.frames_per_video or defaults.get("frames_per_video", 8)
    cap = args.videos_per_class or defaults.get("videos_per_class", 600)
    val_frac = args.val_fraction if args.val_fraction is not None else defaults.get("val_fraction", 0.15)
    image_size = args.image_size or defaults.get("image_size", 224)
    seed = args.seed if args.seed is not None else defaults.get("seed", 42)
    rng = random.Random(seed)

    # ---- resolve roots
    roots: dict[str, Path] = {}
    for spec in args.root:
        if "=" not in spec:
            print(f"ERROR: --root wants NAME=PATH, got {spec!r}"); return 2
        k, _, v = spec.partition("=")
        roots[k.strip().lower()] = Path(v).expanduser()
    if args.data_root:
        base = Path(args.data_root).expanduser()
        for key in datasets:
            if key not in roots and (base / key).is_dir():
                roots[key] = base / key

    if args.inspect:
        key = args.inspect.lower()
        if key not in roots:
            print(f"No root for {key!r}. Pass --root {key}=/path or --data-root.")
            return 2
        inspect(roots[key])
        return 0

    # ---- select
    want = {s.strip().lower() for s in args.only.split(",")} if args.only else None
    prio = {s.strip().upper() for s in args.priority.split(",")} if args.priority else None

    selected: list[tuple[str, dict, Path]] = []
    for key, entry in datasets.items():
        if want and key not in want:
            continue
        if prio and str(entry.get("priority", "")).upper() not in prio:
            continue
        if key not in roots:
            continue
        verified = bool((entry.get("layout") or {}).get("verified", False))
        if not verified and not args.allow_unverified:
            print(f"  SKIP {key}: layout not verified. "
                  f"Run --inspect {key}, fix the globs, or pass --allow-unverified.")
            continue
        selected.append((key, entry, roots[key]))

    if not selected:
        print("\nNothing selected. Datasets present on disk and selectable:")
        for key in datasets:
            mark = "root OK" if key in roots else "no root"
            print(f"  {key:28s} {mark}")
        return 2

    # ---- role overrides (lets a normally-held-out set serve as the train pool)
    role_override: dict[str, str] = {}
    for spec in args.role:
        if "=" not in spec:
            print(f"ERROR: --role wants NAME=ROLE, got {spec!r}"); return 2
        k, _, v = spec.partition("=")
        v = v.strip().lower()
        if v not in ("train_pool", "external_test"):
            print(f"ERROR: role must be train_pool or external_test, got {v!r}"); return 2
        role_override[k.strip().lower()] = v
    for key, entry, _ in selected:
        if key in role_override and entry.get("split_role") != role_override[key]:
            print(f"  ROLE OVERRIDE {key}: {entry.get('split_role')} -> {role_override[key]}")
            entry["split_role"] = role_override[key]

    # ---- overlap guard
    chosen = {k for k, _, _ in selected}
    for k in sorted(chosen):
        other = _OVERLAP.get(k)
        if other and other in chosen and not args.allow_overlap:
            print(f"\nABORT: {k} overlaps {other}; using both risks contamination "
                  f"and double-counts evidence. Drop one, or pass --allow-overlap.")
            return 2

    out_dir = Path(args.out_dir).expanduser()
    man_dir = Path(args.manifest_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    ex = Extractor(frames, image_size, not args.no_face_crop, args.device)

    pool_rows: list[dict] = []                      # train_pool crops
    test_rows: dict[str, list[dict]] = defaultdict(list)
    summary: dict[str, dict] = {}

    for key, entry, root in selected:
        role = entry.get("split_role", "train_pool")
        print(f"\n=== {entry.get('display_name', key)}  [{role}]  {root}")
        reals, fakes = collect(key, root, entry)
        print(f"  found: {len(reals)} real / {len(fakes)} fake videos")
        if not reals and not fakes:
            summary[key] = {"role": role, "videos": 0, "note": "nothing matched the globs"}
            continue
        if cap:
            if len(reals) > cap:
                reals = rng.sample(reals, cap)
            if len(fakes) > cap:
                fakes = rng.sample(fakes, cap)

        n_crop = 0
        n_skip = 0
        for label, files in ((0, reals), (1, fakes)):
            for i, src in enumerate(sorted(files), 1):
                vid = _video_id(key, src, root)
                # Stable across processes -- hash() is salted per-run, which
                # silently defeated the resume check on every new session.
                digest = hashlib.sha1(vid.encode("utf-8")).hexdigest()[:10]
                dest = out_dir / key / ("real" if label == 0 else "fake") / \
                    f"{src.stem[:70]}_{digest}"
                try:
                    imgs = ex.run(src, dest)
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop prep
                    n_skip += 1
                    print(f"    ! {src.name}: {type(exc).__name__}: {exc}")
                    continue
                if not imgs:
                    n_skip += 1
                    continue
                rows = [{"path": str(p), "label": label, "video_id": vid,
                         "dataset": key, "group": f"{key}:{'real' if label == 0 else 'fake'}"}
                        for p in imgs]
                (pool_rows if role == "train_pool" else test_rows[key]).extend(rows)
                n_crop += len(rows)
                if i % 100 == 0:
                    print(f"    {('real', 'fake')[label]}: {i}/{len(files)} videos, {n_crop} crops")
        print(f"  -> {n_crop} crops, {n_skip} videos skipped (no face / unreadable)")
        summary[key] = {"role": role, "priority": entry.get("priority"),
                        "videos_real": len(reals), "videos_fake": len(fakes),
                        "crops": n_crop, "skipped": n_skip,
                        "sheet_use": entry.get("sheet_use")}

    if not pool_rows and not test_rows:
        print("\nERROR: zero crops produced. Check roots and globs (--inspect).")
        return 2

    # ---- train/val[/test] split, disjoint on --split-key (never frame-level)
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in pool_rows:
        by_video[r["video_id"]].append(r)

    pool_test_rows: list[dict] = []
    dropped_span = 0

    if args.split_key == "identity":
        # Split IDENTITIES, then place a video only if ALL its identities land on
        # one side. Videos spanning sides are dropped -- that is what makes the
        # split honest rather than nearly-disjoint.
        regex_for = {k: (e.get("layout") or {}).get("identity_regex")
                     for k, e, _ in selected}
        missing = sorted(k for k, e, _ in selected
                         if e.get("split_role") == "train_pool" and not regex_for.get(k))
        if missing:
            print(f"\nABORT: --split-key identity needs an identity_regex in the registry "
                  f"for: {missing}")
            return 2
        vid_ids = {v: _identities(v, regex_for.get(rows[0]['dataset']))
                   for v, rows in by_video.items()}
        # Some files carry no identity token -- e.g. Celeb-DF v2's YouTube-real
        # clips (00109.mp4). Each is a distinct, unlabelled person, so give it an
        # identity of its own rather than aborting or lumping them together.
        n_fallback = 0
        for v, ids in list(vid_ids.items()):
            if not ids:
                vid_ids[v] = frozenset({f"solo:{v}"})
                n_fallback += 1
        if n_fallback:
            print(f"  {n_fallback} video(s) had no identity token -> each treated as "
                  f"its own identity")
        all_ids = sorted({i for ids in vid_ids.values() for i in ids})
        rng.shuffle(all_ids)
        n_t = int(len(all_ids) * args.test_fraction)
        n_v = int(len(all_ids) * val_frac)
        test_ids, val_ids = set(all_ids[:n_t]), set(all_ids[n_t:n_t + n_v])
        train_ids = set(all_ids[n_t + n_v:])
        print(f"\n  identities: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

        buckets = {"train": [], "val": [], "test": []}
        for v, rows in by_video.items():
            ids = vid_ids[v]
            where = [n for n, s in (("train", train_ids), ("val", val_ids), ("test", test_ids))
                     if ids & s]
            if len(where) == 1:
                buckets[where[0]].extend(rows)
            else:
                dropped_span += 1
        train, val, pool_test_rows = buckets["train"], buckets["val"], buckets["test"]
        print(f"  dropped {dropped_span} videos spanning splits (kept the split clean)")
        key_of = lambda rows: {i for r in rows for i in vid_ids[r["video_id"]]}
        train_keys, val_keys, test_keys = key_of(train), key_of(val), key_of(pool_test_rows)
    else:
        vids = sorted(by_video)
        rng.shuffle(vids)
        n_t = int(len(vids) * args.test_fraction)
        n_v = int(len(vids) * val_frac)
        test_v, val_v = set(vids[:n_t]), set(vids[n_t:n_t + n_v])
        train_v = set(vids[n_t + n_v:])
        train = [r for v in train_v for r in by_video[v]]
        val = [r for v in val_v for r in by_video[v]]
        pool_test_rows = [r for v in test_v for r in by_video[v]]
        train_keys, val_keys, test_keys = train_v, val_v, test_v

    if pool_test_rows:
        src = sorted({r["dataset"] for r in pool_test_rows})
        test_rows[f"{'-'.join(src)}_heldout_{args.split_key}"] = pool_test_rows

    # ---- leakage assertions (abort, not warn)
    assert not (train_keys & val_keys), \
        f"LEAKAGE: {len(train_keys & val_keys)} {args.split_key} key(s) in train AND val"
    assert not (train_keys & test_keys), \
        f"LEAKAGE: {len(train_keys & test_keys)} {args.split_key} key(s) in train AND test"
    assert not (val_keys & test_keys), \
        f"LEAKAGE: {len(val_keys & test_keys)} {args.split_key} key(s) in val AND test"
    ext_names = {k for k, e, _ in selected if e.get("split_role") == "external_test"}
    bad = {r["dataset"] for r in train + val} & ext_names
    assert not bad, f"LEAKAGE: external_test dataset(s) {bad} leaked into train/val"

    # ---- class-balance guard
    # A tiny or lopsided training set trains a constant predictor, which then
    # scores ~0.5 AUC with recall 0.0 on the test set -- a plausible-looking
    # number that is really "the data was broken". Catch it here, not after
    # an hour of training.
    def _video_labels(rows: list[dict]) -> dict[int, set[str]]:
        out: dict[int, set[str]] = {0: set(), 1: set()}
        for r in rows:
            out[r["label"]].add(r["video_id"])
        return out

    tv = _video_labels(train)
    n_real, n_fake = len(tv[0]), len(tv[1])
    n_tot = n_real + n_fake
    minority = min(n_real, n_fake)
    frac = (minority / n_tot) if n_tot else 0.0
    print(f"\n=== TRAIN CLASS BALANCE ===")
    print(f"  videos: {n_real} real / {n_fake} fake   minority share: {frac:.1%}")
    print(f"  crops : {sum(1 for r in train if r['label']==0)} real / "
          f"{sum(1 for r in train if r['label']==1)} fake")

    problems = []
    if minority < args.min_minority_videos:
        problems.append(f"rarer train class has only {minority} videos "
                        f"(need >= {args.min_minority_videos})")
    if frac < args.min_minority_frac:
        problems.append(f"rarer train class is {frac:.1%} of train "
                        f"(need >= {args.min_minority_frac:.0%})")
    if problems:
        msg = ("\nABORT -- the training set cannot produce a meaningful model:\n  - "
               + "\n  - ".join(problems)
               + "\n\nA model trained on this will predict one class for everything, and"
                 "\nthe test metrics will look like a real (bad) result rather than a"
                 "\nbroken run. Fix the data first: check that both classes actually"
                 "\ndownloaded, then re-run. Override with --allow-imbalance only if you"
                 "\nintend to train on this and will say so in the write-up.")
        if args.allow_imbalance:
            print(msg.replace("ABORT --", "WARNING (--allow-imbalance) --"))
        else:
            print(msg)
            return 2

    fields = ["path", "label", "video_id", "dataset", "group"]

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        n_fake = sum(r["label"] for r in rows)
        print(f"  {path.name:34s} {len(rows):7d} crops  "
              f"({len(rows) - n_fake} real / {n_fake} fake)  "
              f"{len({r['video_id'] for r in rows})} videos")

    print("\n=== MANIFESTS ===")
    write(man_dir / "train.csv", train)
    write(man_dir / "val.csv", val)
    for key, rows in sorted(test_rows.items()):
        write(man_dir / f"test_{key}.csv", rows)

    meta = {
        "registry": str(args.registry),
        "seed": seed, "frames_per_video": frames, "videos_per_class": cap,
        "val_fraction": val_frac, "image_size": image_size,
        "face_crop": not args.no_face_crop,
        "train_pool_datasets": sorted({r["dataset"] for r in pool_rows}),
        "external_test_datasets": sorted(test_rows),
        "counts": {"train": len(train), "val": len(val),
                   **{f"test_{k}": len(v) for k, v in test_rows.items()}},
        "train_keys": len(train_keys), "val_keys": len(val_keys),
        "test_keys": len(test_keys),
        "split_key": args.split_key,
        "test_fraction": args.test_fraction,
        "videos_dropped_spanning_splits": dropped_span,
        "leakage_checks": (f"passed: {args.split_key}-disjoint train/val/test; "
                           "no external_test dataset in train"),
        "train_balance": {"videos_real": n_real, "videos_fake": n_fake,
                          "minority_share": round(frac, 4),
                          "guard": "bypassed (--allow-imbalance)" if problems else "passed"},
        "per_dataset": summary,
        "evaluation_matrix": reg.get("evaluation_matrix", {}),
    }
    (man_dir / "splits_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {man_dir / 'splits_summary.json'}")

    if not test_rows:
        print("\nNOTE: no external_test dataset was prepared. You will only have an "
              "in-distribution val number, which historically runs ~0.99 AUC and says "
              "nothing about generalization. Add Celeb-DF v2 or WildDeepfake.")
    print("\nNext:\n"
          f"  python scripts/train_general.py --train-manifest {man_dir / 'train.csv'} "
          f"--val-manifest {man_dir / 'val.csv'} --backbone convnext_tiny.fb_in22k_ft_in1k "
          f"--forensic-aug --epochs 8 --out checkpoints/face_registry.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
