"""Build generator-balanced manifests for GenVidBench (ADR-007, Parts 3-4).

GenVidBench is a ~6.8M-video AI-generated-video benchmark with a
deliberate *cross-source-cross-generator* design: the point is to train
on one set of generators and test on UNSEEN generators, so the detector
learns "this content has synthetic characteristics" rather than "this is
generator X" (the exact failure mode we want to avoid — see the FF++ ->
Celeb-DF generalization collapse in EXP-001).

Generators (from the GenVidBench paper):
    real sources:            Vript, HD-VG-130M
    fake pair 1 (VidProM):   Pika, VideoCrafter2, ModelScope, Text2Video-Zero
    fake pair 2 (HD-VG):     SVD, MuseV, Mora, CogVideo

This script does NOT download the dataset (it is far too large — Part 32).
It scans an already-downloaded subtree, classifies each video by matching
a known generator / real-source name against its path components
(case-insensitive, punctuation-normalised), and writes three manifests:

    data/manifests/train.csv
    data/manifests/val.csv
    data/manifests/test.csv

Each row: video_id, video_path, label, generator, source, dataset, split
(label: 0=real, 1=fake). Sampling is generator-balanced and deterministic
(seeded), and the real/fake split within train and test is drawn from the
generator/source lists in configs/video_training.yaml so train and test
never share a generator or a real source.

Usage
-----
::

    python scripts/prepare_genvidbench.py --config configs/video_training.yaml

    # or override the data root / sizes on the CLI:
    python scripts/prepare_genvidbench.py \\
        --data-root /path/to/genvidbench \\
        --train-fake 10000 --train-real 10000 \\
        --val-fake 2000 --val-real 2000 \\
        --test-fake 2000 --test-real 2000

If auto-classification misses videos (unexpected folder names), the
script prints the unmatched path samples so you can extend the name maps
or pass --strict to fail loudly instead of silently dropping them.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}

# Canonical generator / source name -> the set of aliases that may appear
# as a path component. Matching is done on a normalised token (lowercased,
# stripped of - _ . spaces), so "Text2Video-Zero" and "text2video_zero"
# both collapse to "text2videozero".
# Canonical name -> aliases. Includes the GenVidBench archive/folder
# abbreviations confirmed from the official label files: ms, vc2, t2vz,
# hd_vg_130m.
_REAL_SOURCES: dict[str, set[str]] = {
    "vript": {"vript"},
    "hd-vg-130m": {"hdvg", "hdvg130m", "hdvg130", "hdvideo", "hdvg130mreal"},
}
_FAKE_GENERATORS: dict[str, set[str]] = {
    "pika": {"pika"},
    "videocrafter2": {"videocrafter2", "videocrafterv2", "videocrafter", "vc2"},
    "modelscope": {"modelscope", "modelscopet2v", "ms"},
    "text2video-zero": {"text2videozero", "t2vzero", "t2v0", "t2vz"},
    "svd": {"svd", "stablevideodiffusion"},
    "musev": {"musev"},
    "mora": {"mora"},
    "cogvideo": {"cogvideo", "cogvideox"},
}


def _norm(token: str) -> str:
    return token.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


def _build_alias_index(mapping: dict[str, set[str]]) -> dict[str, str]:
    """Flatten {canonical: {aliases}} into {normalised_alias: canonical}."""
    index: dict[str, str] = {}
    for canonical, aliases in mapping.items():
        index[_norm(canonical)] = canonical
        for a in aliases:
            index[_norm(a)] = canonical
    return index


_REAL_INDEX = _build_alias_index(_REAL_SOURCES)
_FAKE_INDEX = _build_alias_index(_FAKE_GENERATORS)


def classify(video_path: Path, data_root: Path) -> tuple[int, str, str] | None:
    """Classify a video by its path components.

    Returns (label, generator_or_source, kind) where:
        label: 0=real, 1=fake
        generator_or_source: canonical name (e.g. "pika", "vript")
        kind: "real" or "fake"
    or None if no known generator/source name is found in the path.
    """
    rel = video_path.relative_to(data_root)
    # Check each path component (most specific first) against both indexes.
    for part in rel.parts:
        tok = _norm(part)
        if tok in _FAKE_INDEX:
            return 1, _FAKE_INDEX[tok], "fake"
        if tok in _REAL_INDEX:
            return 0, _REAL_INDEX[tok], "real"
    return None


def scan_from_labels(
    data_root: Path,
    label_files: list[Path],
    require_exists: bool,
    strict: bool,
) -> tuple[list[dict], list[dict]]:
    """Read GenVidBench's official label files (authoritative).

    Each line is ``<relative/path with spaces>.mp4 <label>`` where label
    1=fake, 0=real. The generator/source is the 2nd path component
    (e.g. ``Pair1/ms/xxx.mp4`` -> ``ms`` -> ModelScope). We split on the
    LAST space because filenames themselves contain spaces.

    With ``require_exists`` (default), rows whose video is not on disk are
    skipped — so you can extract only some archives and still build a
    valid manifest over what you have.
    """
    real_rows: list[dict] = []
    fake_rows: list[dict] = []
    total = 0
    missing = 0
    unknown: list[str] = []
    for lf in label_files:
        if not lf.is_file():
            raise SystemExit(f"Label file not found: {lf}")
        print(f"Reading labels: {lf}", flush=True)
        with lf.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    rel, lab = line.rsplit(" ", 1)
                    label = int(lab)
                except ValueError:
                    continue
                total += 1
                parts = rel.split("/")
                folder = parts[1] if len(parts) > 1 else parts[0]
                tok = _norm(folder)
                if label == 0:
                    canonical = _REAL_INDEX.get(tok)
                    kind = "real"
                else:
                    canonical = _FAKE_INDEX.get(tok)
                    kind = "fake"
                if canonical is None:
                    unknown.append(folder)
                    continue
                abs_path = (data_root / rel).resolve()
                if require_exists and not abs_path.is_file():
                    missing += 1
                    continue
                row = {
                    "video_id": Path(rel).stem,
                    "video_path": str(abs_path),
                    "label": label,
                    "generator": canonical if kind == "fake" else "",
                    "source": canonical if kind == "real" else "",
                    "dataset": "genvidbench",
                }
                (real_rows if kind == "real" else fake_rows).append(row)

    print(f"\nLabel rows: {total:,} | on-disk kept: {len(real_rows)+len(fake_rows):,} "
          f"(real={len(real_rows):,} fake={len(fake_rows):,}) | missing-on-disk skipped: {missing:,}")
    if unknown:
        uc = Counter(unknown)
        print(f"Unknown generator/source folders skipped: {dict(uc)}")
        if strict:
            raise SystemExit(f"--strict: {len(unknown)} rows had unknown generator/source folders.")
    if require_exists and (real_rows == [] or fake_rows == []):
        raise SystemExit(
            "No videos found on disk for one class. Extract the archives under "
            f"{data_root} (Pair1/, Pair2/), or pass --no-require-exists to build "
            "a manifest of all label rows regardless of extraction."
        )
    return real_rows, fake_rows


def scan(data_root: Path, strict: bool) -> tuple[list[dict], list[dict]]:
    """Walk data_root, classify every video. Returns (real_rows, fake_rows)."""
    real_rows: list[dict] = []
    fake_rows: list[dict] = []
    unmatched: list[Path] = []
    scanned = 0
    print(f"Scanning {data_root} for videos (this can take a while on 6M-scale trees)...", flush=True)
    for p in data_root.rglob("*"):
        if p.suffix.lower() not in _VIDEO_EXTS or not p.is_file():
            continue
        scanned += 1
        if scanned % 50000 == 0:
            print(f"  ...scanned {scanned:,} videos (real={len(real_rows):,} fake={len(fake_rows):,})", flush=True)
        result = classify(p, data_root)
        if result is None:
            unmatched.append(p)
            continue
        label, name, kind = result
        row = {
            "video_id": p.stem,
            "video_path": str(p.resolve()),
            "label": label,
            "generator": name if kind == "fake" else "",
            "source": name if kind == "real" else "",
            "dataset": "genvidbench",
        }
        (real_rows if kind == "real" else fake_rows).append(row)

    print(f"\nScanned {scanned:,} videos: {len(real_rows):,} real, {len(fake_rows):,} fake, {len(unmatched):,} unmatched")
    if unmatched:
        print("Sample unmatched paths (first 10) — extend the name maps or pass --strict:")
        for u in unmatched[:10]:
            print(f"  {u}")
        if strict:
            raise SystemExit(f"--strict: {len(unmatched)} videos could not be classified.")
    return real_rows, fake_rows


def _balanced_sample(rows: list[dict], key: str, cap: int | None, rng: random.Random) -> list[dict]:
    """Sample up to `cap` rows, balanced across distinct values of `key`.

    Ensures no single generator/source dominates (Part 4): draws round-robin
    from each group until the cap is hit or all groups are exhausted.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key] or "_"].append(r)
    for g in groups.values():
        rng.shuffle(g)
    if cap is None:
        out = [r for g in groups.values() for r in g]
        rng.shuffle(out)
        return out
    # Round-robin across groups for balance.
    out: list[dict] = []
    order = list(groups.keys())
    idx = {k: 0 for k in order}
    while len(out) < cap:
        progressed = False
        for k in order:
            if idx[k] < len(groups[k]):
                out.append(groups[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(out) >= cap:
                    break
        if not progressed:
            break
    return out


def _select_split(
    real_rows: list[dict],
    fake_rows: list[dict],
    real_sources: list[str],
    generators: list[str],
    real_cap: int | None,
    fake_cap: int | None,
    rng: random.Random,
) -> list[dict]:
    """Pick real rows from `real_sources` and fake rows from `generators`,
    balanced within each class and capped."""
    real_norm = {_norm(s) for s in real_sources}
    gen_norm = {_norm(g) for g in generators}
    real_pool = [r for r in real_rows if _norm(r["source"]) in real_norm]
    fake_pool = [r for r in fake_rows if _norm(r["generator"]) in gen_norm]
    real_sel = _balanced_sample(real_pool, "source", real_cap, rng)
    fake_sel = _balanced_sample(fake_pool, "generator", fake_cap, rng)
    return real_sel + fake_sel


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    dcfg = cfg.get("dataset", {}) if cfg else {}

    data_root = Path(args.data_root or dcfg.get("data_root", "data/raw/genvidbench"))
    if not data_root.is_dir():
        raise SystemExit(f"--data-root does not exist: {data_root}")
    manifest_dir = Path(args.manifest_dir or dcfg.get("manifest_dir", "data/manifests"))
    seed = args.seed if args.seed is not None else int(dcfg.get("seed", 42))
    rng = random.Random(seed)

    train_gen = dcfg.get("train_generators", ["pika", "videocrafter2", "modelscope", "text2video-zero"])
    test_gen = dcfg.get("test_generators", ["svd", "musev", "mora", "cogvideo"])
    train_src = dcfg.get("train_real_sources", ["vript"])
    test_src = dcfg.get("test_real_sources", ["hd-vg-130m"])

    def _sz(cli, key, default):
        return cli if cli is not None else dcfg.get(key, default)

    train_real = _sz(args.train_real, "train_real", 10000)
    train_fake = _sz(args.train_fake, "train_fake", 10000)
    val_real = _sz(args.val_real, "val_real", 2000)
    val_fake = _sz(args.val_fake, "val_fake", 2000)
    test_real = _sz(args.test_real, "test_real", 2000)
    test_fake = _sz(args.test_fake, "test_fake", 2000)

    # Prefer the official label files (authoritative). Fall back to
    # path-scanning only if no label files are given/found.
    label_files: list[Path] = []
    if args.labels:
        label_files = [Path(p) for p in args.labels]
    else:
        for cand in (
            data_root / "Pair1_labels.txt", data_root / "Pair2_labels.txt",
            data_root / "GenVidBench" / "Pair1_labels.txt",
            data_root / "GenVidBench" / "Pair2_labels.txt",
        ):
            if cand.is_file():
                label_files.append(cand)

    if label_files:
        print(f"Using {len(label_files)} official label file(s).")
        real_rows, fake_rows = scan_from_labels(
            data_root, label_files, require_exists=not args.no_require_exists, strict=args.strict
        )
    else:
        print("No label files found — falling back to path-scan classification.")
        real_rows, fake_rows = scan(data_root, args.strict)
    if not real_rows or not fake_rows:
        raise SystemExit(f"Need both classes; got real={len(real_rows)} fake={len(fake_rows)}. Check --data-root.")

    print("\nGenerator distribution (fake):")
    for g, c in sorted(Counter(r["generator"] for r in fake_rows).items()):
        print(f"  {g:20s} {c:,}")
    print("Source distribution (real):")
    for s, c in sorted(Counter(r["source"] for r in real_rows).items()):
        print(f"  {s:20s} {c:,}")

    # TRAIN: train generators/sources. VAL: held out FROM the train
    # generators (same generators, disjoint videos) so val tracks training
    # fit. TEST: the unseen generators/sources (true cross-generator).
    train_all = _select_split(real_rows, fake_rows, train_src, train_gen,
                              None, None, rng)
    rng.shuffle(train_all)

    # Carve val off the train pool BEFORE capping, so val videos never
    # appear in train (video-level disjointness — Part 12).
    val_needed_real = val_real if val_real else 0
    val_needed_fake = val_fake if val_fake else 0
    train_real_all = [r for r in train_all if r["label"] == 0]
    train_fake_all = [r for r in train_all if r["label"] == 1]
    val_rows = train_real_all[:val_needed_real] + train_fake_all[:val_needed_fake]
    remain_real = train_real_all[val_needed_real:]
    remain_fake = train_fake_all[val_needed_fake:]
    train_rows = (_balanced_sample(remain_real, "source", train_real, rng)
                  + _balanced_sample(remain_fake, "generator", train_fake, rng))
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    test_rows = _select_split(real_rows, fake_rows, test_src, test_gen,
                              test_real, test_fake, rng)
    rng.shuffle(test_rows)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write(manifest_dir / "train.csv", train_rows, "train")
    _write(manifest_dir / "val.csv", val_rows, "val")
    _write(manifest_dir / "test.csv", test_rows, "test")

    print(f"\nWrote manifests to {manifest_dir}/")
    print(f"  train.csv: {len(train_rows):,}  (generators={train_gen}, real={train_src})")
    print(f"  val.csv:   {len(val_rows):,}  (held-out videos from train generators)")
    print(f"  test.csv:  {len(test_rows):,}  (UNSEEN generators={test_gen}, real={test_src})")
    print("\nCross-generator protocol active: test generators are disjoint from train.")


def _write(path: Path, rows: list[dict], split: str) -> None:
    fields = ["video_id", "video_path", "label", "generator", "source", "dataset", "split"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "split": split})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build generator-balanced GenVidBench manifests (ADR-007).")
    p.add_argument("--config", default="configs/video_training.yaml", help="Reads dataset.* defaults from here.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--train-real", type=int, default=None)
    p.add_argument("--train-fake", type=int, default=None)
    p.add_argument("--val-real", type=int, default=None)
    p.add_argument("--val-fake", type=int, default=None)
    p.add_argument("--test-real", type=int, default=None)
    p.add_argument("--test-fake", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--strict", action="store_true", help="Fail if any video can't be classified.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Official GenVidBench label .txt files (Pair1_labels.txt Pair2_labels.txt). "
                        "If omitted, the script looks for them under --data-root.")
    p.add_argument("--no-require-exists", action="store_true",
                   help="Include label rows even if the video file is not on disk yet "
                        "(useful to preview the manifest before extracting archives).")
    return p.parse_args()


if __name__ == "__main__":
    main()
