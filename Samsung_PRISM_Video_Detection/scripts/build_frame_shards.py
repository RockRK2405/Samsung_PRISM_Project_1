#!/usr/bin/env python3
"""Video -> compact, resumable frame shards (Phase 1 data prep).

For every manifest row whose video belongs to the archive just extracted,
sample ``--frames`` uniform frames (same sampler the models were trained
with), resize so the SHORT side is ``--short-side`` (aspect kept, so both the
CNN's squash-resize and CLIP's resize+center-crop can be applied later),
JPEG-encode, and pack ``--shard-size`` videos per ``.tar`` shard:

    <out>/<group>/shard-00000.tar      # <video_id>/00.jpg ... 15.jpg
    <out>/<group>/shard-00000.jsonl    # one row per video: id, label, split, ...

A shard is COMPLETE only when its .jsonl exists; the .jsonl is written last,
after the .tar is fully copied. On restart, videos already listed in any
.jsonl are skipped and orphan .tar files (no .jsonl) are deleted. Shards are
built on local disk and copied to ``--out`` (Google Drive) in one go, because
Drive is slow at many small writes.

Typical use, archive by archive (raw video never needs to live on Drive):

    python scripts/build_frame_shards.py \
        --manifest manifests/train.csv manifests/val.csv manifests/test.csv \
        --video-root /content/raw/GenVidBench --folder t2vz \
        --out /content/drive/MyDrive/prism_video/frames
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
try:
    from src.preprocessing.frame_extractor import FrameExtractor  # noqa: E402
except ImportError:  # package __init__ pulls torch; the extractor itself only needs cv2
    import importlib.util, logging, types
    _u = types.ModuleType("src.utils.logging"); _u.get_logger = logging.getLogger
    sys.modules.update({"src": types.ModuleType("src"), "src.utils": types.ModuleType("src.utils"),
                        "src.utils.logging": _u})
    _spec = importlib.util.spec_from_file_location(
        "frame_extractor", _ROOT / "src/preprocessing/frame_extractor.py")
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    FrameExtractor = _m.FrameExtractor

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def rel_after(path: str, anchor: str = "GenVidBench") -> Path:
    """'.../GenVidBench/Pair1/t2vz/x.mp4' -> 'Pair1/t2vz/x.mp4' (manifest paths are absolute)."""
    parts = Path(path.replace("\\", "/")).parts
    if anchor in parts:
        return Path(*parts[parts.index(anchor) + 1:])
    return Path(*parts[-3:])  # best effort: PairX/<folder>/<file>


def load_rows(manifests: list[Path], folder: str | None) -> list[dict]:
    rows = []
    for m in manifests:
        split_default = m.stem  # train / val / test / test_b
        with m.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rel = rel_after(r["video_path"])
                if folder and (len(rel.parts) < 2 or rel.parts[1] != folder):
                    continue
                rows.append({
                    "video_id": r["video_id"],
                    "rel": str(rel),
                    "label": int(r["label"]),
                    "generator": r.get("generator", "") or "",
                    "source": r.get("source", "") or "",
                    "split": r.get("split") or split_default,
                })
    return rows


def group_of(row: dict) -> str:
    return row["generator"] if row["label"] == 1 else f"real-{row['source'] or 'unknown'}"


def done_ids(group_dir: Path) -> set[str]:
    ids: set[str] = set()
    for idx in group_dir.glob("shard-*.jsonl"):
        for line in idx.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["video_id"])
    # Orphans: a .tar without its .jsonl was interrupted mid-copy.
    for tar in group_dir.glob("shard-*.tar"):
        if not tar.with_suffix(".jsonl").exists():
            tar.unlink()
            print(f"  removed orphan {tar.name}")
    return ids


def next_shard_no(group_dir: Path) -> int:
    nums = [int(p.stem.split("-")[1]) for p in group_dir.glob("shard-*.jsonl")]
    return max(nums) + 1 if nums else 0


def resize_short(rgb: np.ndarray, short: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = short / min(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def encode_video(path: Path, extractor: FrameExtractor, n: int, short: int, q: int) -> list[bytes]:
    frames = extractor.extract(path)
    if not frames:
        raise RuntimeError("no frames decoded")
    while len(frames) < n:  # pad short decodes so every video has exactly n frames
        frames.append(frames[-1])
    out = []
    for rgb in frames[:n]:
        bgr = cv2.cvtColor(resize_short(rgb, short), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        out.append(buf.tobytes())
    return out


def flush(shard: list[tuple[dict, list[bytes]]], group_dir: Path, no: int) -> None:
    name = f"shard-{no:05d}"
    with tempfile.TemporaryDirectory() as td:
        tmp_tar = Path(td) / f"{name}.tar"
        with tarfile.open(tmp_tar, "w") as tf:
            for row, jpgs in shard:
                for i, data in enumerate(jpgs):
                    info = tarfile.TarInfo(f"{row['video_id']}/{i:02d}.jpg")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        shutil.copyfile(tmp_tar, group_dir / f"{name}.tar")
    # Index last: its presence marks the shard complete.
    tmp_idx = group_dir / f"{name}.jsonl.part"
    tmp_idx.write_text("".join(json.dumps({**r, "n_frames": len(j)}) + "\n" for r, j in shard),
                       encoding="utf-8")
    tmp_idx.rename(group_dir / f"{name}.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, type=Path)
    ap.add_argument("--video-root", required=True, type=Path,
                    help="Directory that plays the role of 'GenVidBench/' (contains Pair1/, Pair2/).")
    ap.add_argument("--folder", default=None,
                    help="Only rows from this archive folder, e.g. t2vz, vript, pika.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--shard-size", type=int, default=500)
    args = ap.parse_args()

    rows = load_rows(args.manifest, args.folder)
    print(f"{len(rows)} manifest rows for folder={args.folder or 'ALL'}")
    if not rows:
        return 1

    # Archives sometimes extract with an extra nesting level; index by filename as a fallback.
    by_name: dict[str, Path] = {}
    for p in args.video_root.rglob("*"):
        if p.suffix.lower() in _VIDEO_EXT:
            by_name.setdefault(p.name, p)
    print(f"{len(by_name)} video files found under {args.video_root}")

    extractor = FrameExtractor(num_frames=args.frames, strategy="uniform")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_of(r), []).append(r)

    failures = []
    t0 = time.time()
    for g, grows in groups.items():
        gdir = args.out / g
        gdir.mkdir(parents=True, exist_ok=True)
        done = done_ids(gdir)
        todo = [r for r in grows if r["video_id"] not in done]
        print(f"\n[{g}] {len(grows)} rows, {len(done)} already sharded, {len(todo)} to do")
        no, shard = next_shard_no(gdir), []
        for i, r in enumerate(todo, 1):
            path = args.video_root / r["rel"]
            if not path.is_file():
                path = by_name.get(Path(r["rel"]).name)
            if path is None or not path.is_file():
                failures.append({**r, "error": "missing"})
                continue
            try:
                shard.append((r, encode_video(path, extractor, args.frames,
                                              args.short_side, args.jpeg_quality)))
            except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
                failures.append({**r, "error": f"{type(exc).__name__}: {exc}"})
            if len(shard) >= args.shard_size:
                flush(shard, gdir, no); no += 1; shard = []
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f} videos/s  failures={len(failures)}")
        if shard:
            flush(shard, gdir, no)

    if failures:
        fpath = args.out / f"failures_{args.folder or 'all'}.jsonl"
        with fpath.open("a", encoding="utf-8") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")
        print(f"\n{len(failures)} failures logged to {fpath}")
        missing = sum(1 for r in failures if r["error"] == "missing")
        if missing > 0.05 * len(rows):
            print(f"WARNING: {missing} videos missing -- check that the archive extracted "
                  f"under --video-root as Pair*/{args.folder}/*.mp4")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
