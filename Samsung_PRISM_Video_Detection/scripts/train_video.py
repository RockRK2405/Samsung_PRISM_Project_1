"""Train the general AI-video detector (ADR-007 Phase 2).

    python scripts/train_video.py --config configs/video_training.yaml

All hyperparameters live in the config. CLI flags override the common
ones for quick experiments without editing the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from src.training.video_trainer import train


def main() -> None:
    p = argparse.ArgumentParser(description="Train the general AI-video detector (ADR-007).")
    p.add_argument("--config", default="configs/video_training.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--temporal", default=None, help="mean_pool | attention_pool | transformer")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--name", default=None, help="Experiment name (experiments/<name>/).")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # CLI overrides.
    if args.epochs is not None: cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None: cfg["training"]["learning_rate"] = args.lr
    if args.backbone is not None: cfg["model"]["backbone"] = args.backbone
    if args.temporal is not None: cfg["model"].setdefault("temporal", {})["type"] = args.temporal
    if args.num_frames is not None: cfg["video"]["num_frames"] = args.num_frames
    if args.name is not None: cfg.setdefault("experiment", {})["name"] = args.name

    train(cfg)


if __name__ == "__main__":
    main()
