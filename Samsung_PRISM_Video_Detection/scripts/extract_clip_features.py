#!/usr/bin/env python3
"""Frame shards -> frozen CLIP ViT-L/14 per-frame features (Phase 1, candidate B).

For every complete shard (``shard-NNNNN.tar`` + ``.jsonl``) under ``--frames``
writes, under ``--out`` with the same <group>/ layout:

    shard-NNNNN.npy        float16, shape (n_videos, n_frames, 768)
    shard-NNNNN.ids.json   video_ids in row order (copied from the .jsonl)

Already-done shards are skipped (resumable); each .npy is written to a temp
name and renamed, so a disconnect never leaves a half file that looks done.

    python scripts/extract_clip_features.py \
        --frames /content/drive/MyDrive/prism_video/frames \
        --out    /content/drive/MyDrive/prism_video/features
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "openai/clip-vit-large-patch14"


def read_shard(tar_path: Path, ids: list[str], n_frames: int) -> list[list[Image.Image]]:
    """Return frames per video, in the order of ``ids``."""
    imgs: dict[str, dict[int, Image.Image]] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            vid, fname = m.name.rsplit("/", 1)
            data = tf.extractfile(m).read()
            imgs.setdefault(vid, {})[int(fname.split(".")[0])] = Image.open(io.BytesIO(data)).convert("RGB")
    out = []
    for vid in ids:
        frames = imgs[vid]
        out.append([frames[i] for i in sorted(frames)][:n_frames])
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--groups", nargs="*", default=None, help="Only these group folders (default: all).")
    ap.add_argument("--batch", type=int, default=256, help="Frames per forward pass.")
    ap.add_argument("--n-frames", type=int, default=16)
    args = ap.parse_args()

    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = CLIPImageProcessor.from_pretrained(MODEL_ID)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    print(f"{MODEL_ID} on {device}")

    shards = sorted(args.frames.glob("*/shard-*.jsonl"))
    if args.groups:
        shards = [s for s in shards if s.parent.name in set(args.groups)]
    print(f"{len(shards)} complete frame shards")

    t0, done_videos = time.time(), 0
    for idx in shards:
        group = idx.parent.name
        out_dir = args.out / group
        out_npy = out_dir / f"{idx.stem}.npy"
        if out_npy.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = [json.loads(l)["video_id"] for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()]

        with tempfile.TemporaryDirectory() as td:
            local_tar = Path(td) / "s.tar"
            shutil.copyfile(idx.with_suffix(".tar"), local_tar)  # one sequential read from Drive
            videos = read_shard(local_tar, ids, args.n_frames)

        flat = [img for v in videos for img in v]
        feats = []
        for i in range(0, len(flat), args.batch):
            px = proc(images=flat[i:i + args.batch], return_tensors="pt")["pixel_values"].to(device, dtype)
            feats.append(model(pixel_values=px).image_embeds.float().cpu())
        feats = torch.cat(feats).numpy().astype(np.float16)
        arr = feats.reshape(len(videos), args.n_frames, -1)

        tmp = out_dir / f"{idx.stem}.tmp.npy"
        np.save(tmp, arr)
        (out_dir / f"{idx.stem}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
        tmp.rename(out_npy)  # presence of .npy marks the shard done
        done_videos += len(ids)
        rate = done_videos / (time.time() - t0)
        print(f"  {group}/{idx.stem}: {arr.shape}  ({rate:.1f} videos/s)")

    print(f"done: {done_videos} videos in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
