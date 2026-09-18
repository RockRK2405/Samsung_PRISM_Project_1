"""Evaluate a trained AI-video detector on the test manifest (ADR-007).

Reports the full metric suite (Part 22) AND a per-generator breakdown
(Part 25) so seen-vs-unseen generator performance is never hidden behind
one headline number. Because the test manifest holds UNSEEN generators
(cross-generator protocol), the per-generator recall here IS the
generalization result.

    python scripts/evaluate_video.py --checkpoint checkpoints/best_model.pt

Writes reports/video_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.video_dataset import VideoManifestDataset, load_video_manifest
from src.models.video_spatial import build_video_model_from_config
from src.training.video_trainer import _metric_suite
from src.utils import get_logger, select_device

logger = get_logger(__name__)


@torch.no_grad()
def _score_all(model, loader, device) -> np.ndarray:
    model.eval()
    scores: list[float] = []
    for clips, _ in loader:
        clips = clips.to(device)
        probs = torch.softmax(model(clips), dim=1)[:, 1].detach().cpu().numpy()
        scores.extend(probs.tolist())
    return np.array(scores)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the AI-video detector (ADR-007).")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--config", default="configs/video_training.yaml")
    p.add_argument("--split", default="test", help="Which manifest split (default: test = unseen generators).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="reports/video_results.json")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = select_device(cfg.get("hardware", {}).get("device"))
    logger.info("Device: %s", device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Prefer the config embedded in the checkpoint (matches trained arch).
    model_cfg = ckpt.get("config", cfg)
    model = build_video_model_from_config(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    manifest = Path(cfg["dataset"]["manifest_dir"]) / f"{args.split}.csv"
    samples = load_video_manifest(manifest)
    ds = VideoManifestDataset(manifest, cfg["video"]["num_frames"], cfg["video"]["image_size"], train=False)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=int(cfg["training"].get("num_workers", 4)),
                        pin_memory=(device.type == "cuda"))

    y_true = np.array([s.label for s in samples])
    y_score = _score_all(model, loader, device)

    overall = _metric_suite(y_true, y_score, args.threshold)

    # Per-generator (fakes) and per-source (reals) breakdown.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        key = s.generator if s.label == 1 else f"real:{s.source}"
        groups[key].append(i)

    per_group = {}
    for key, idxs in sorted(groups.items()):
        idxs = np.array(idxs)
        gt = y_true[idxs]
        sc = y_score[idxs]
        pred = (sc >= args.threshold).astype(int)
        if (gt == 1).all():
            per_group[key] = {"recall": float((pred == 1).mean()), "n": int(len(idxs)), "mean_score": float(sc.mean())}
        else:
            per_group[key] = {"specificity": float((pred == 0).mean()), "n": int(len(idxs)), "mean_score": float(sc.mean())}

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "threshold": args.threshold,
        "overall": overall,
        "per_generator": per_group,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    logger.info("=== OVERALL (%s) ===", args.split)
    for k in ("accuracy", "precision", "recall", "f1", "auc", "pr_auc", "fpr", "fnr"):
        logger.info("  %-10s %.4f", k, overall[k])
    logger.info("=== PER-GENERATOR (unseen-generator generalization) ===")
    for key, m in per_group.items():
        metric = "recall" if "recall" in m else "specificity"
        logger.info("  %-22s %s=%.4f  n=%d  mean_score=%.3f", key, metric, m[metric], m["n"], m["mean_score"])
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
