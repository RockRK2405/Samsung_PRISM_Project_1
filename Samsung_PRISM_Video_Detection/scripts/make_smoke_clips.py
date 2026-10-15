#!/usr/bin/env python3
"""Generate PIPELINE edge-case clips. These are NOT deepfake test data.

    python scripts/make_smoke_clips.py --out data/smoke

!!! READ THIS !!!
These clips are procedurally drawn (gradients, noise, shapes). They are NOT
AI-generated video and they are NOT real video. They carry **no ground-truth
label** and must NEVER appear in an accuracy number or a report. A model
scoring them "fake" or "real" tells you nothing about detection quality.

What they ARE for: proving the plumbing survives awkward input. Each clip
targets a specific failure mode that has actually bitten this project:

  faceless_*        -- no face in any frame. The face path raises
                       RuntimeError("No face detected...") on these. Known
                       defect (work log §7.6): it should degrade, not crash.
  short_4frames     -- fewer frames than num_frames (16/32). Tests the frame
                       sampler's padding/repeat behaviour.
  single_frame      -- a 1-frame video. Boundary case for temporal pooling.
  tiny_64x64        -- below the 224 input size. Tests upscaling.
  tall_portrait     -- 9:16. Tests non-square aspect handling.
  static_identical  -- every frame identical. Temporal variance = 0; a
                       temporal head should not divide by it.
  still_image.png   -- not a video at all. Tests the image path, which is
                       currently rejected at upload (work log §7.8).

Requires opencv-python + numpy (both already in requirements.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Needs opencv-python and numpy ({exc}). pip install -r requirements.txt"
    )

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def _write(path: Path, frames: list["np.ndarray"], fps: float = 24.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), _FOURCC, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter refused {path} ({w}x{h}) — codec unavailable?")
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"  {path.name:24s} {len(frames):3d} frames  {w}x{h}")


def _gradient(w: int, h: int, t: float) -> "np.ndarray":
    """A drifting colour gradient — smooth, faceless, deterministic."""
    xs = np.linspace(0, 255, w, dtype=np.float32)
    ys = np.linspace(0, 255, h, dtype=np.float32)
    g = ys[:, None] * 0.5 + xs[None, :] * 0.5
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = (g + t * 40) % 256
    img[..., 1] = (g * 0.7 + t * 25) % 256
    img[..., 2] = (g * 0.4 + t * 60) % 256
    return img.astype(np.uint8)


def _noise(w: int, h: int, seed: int) -> "np.ndarray":
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _shapes(w: int, h: int, t: int) -> "np.ndarray":
    """Moving shapes — gives the frame sampler real motion to chew on."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cx = int((w * 0.5) + (w * 0.3) * np.sin(t * 0.3))
    cy = int((h * 0.5) + (h * 0.25) * np.cos(t * 0.21))
    cv2.circle(img, (cx, cy), max(8, min(w, h) // 8), (200, 120, 60), -1)
    cv2.rectangle(img, (int(w * 0.1), int(h * 0.7)),
                  (int(w * 0.1) + 40 + t, int(h * 0.7) + 30), (60, 200, 160), -1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/smoke")
    ap.add_argument("--frames", type=int, default=40, help="Frames in normal clips.")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing pipeline smoke clips to {out}/\n")
    n = args.frames

    _write(out / "faceless_gradient.mp4", [_gradient(640, 480, i) for i in range(n)])
    _write(out / "faceless_noise.mp4", [_noise(640, 480, i) for i in range(n)])
    _write(out / "faceless_shapes.mp4", [_shapes(640, 480, i) for i in range(n)])
    _write(out / "short_4frames.mp4", [_shapes(640, 480, i) for i in range(4)])
    _write(out / "single_frame.mp4", [_gradient(640, 480, 0)])
    _write(out / "tiny_64x64.mp4", [_shapes(64, 64, i) for i in range(n)])
    _write(out / "tall_portrait.mp4", [_shapes(360, 640, i) for i in range(n)])
    _write(out / "static_identical.mp4", [_gradient(640, 480, 7)] * n)

    img = out / "still_image.png"
    cv2.imwrite(str(img), _shapes(640, 480, 5))
    print(f"  {img.name:24s} (single image — tests the image path)")

    print(f"""
Done. Run each through the detector and record WHAT HAPPENS, not the score:

    python scripts/selftest_detector.py --video {out}/faceless_gradient.mp4

Expected today (per the work log):
  * faceless_* via the face-only path -> RuntimeError. That is defect §7.6.
  * still_image.png via the dashboard -> rejected at upload. That is §7.8.
  * everything else -> should return a score without crashing.

A crash here is a plumbing bug worth fixing. A score here means nothing about
accuracy — these clips have no ground truth. Do not put them in a manifest.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
