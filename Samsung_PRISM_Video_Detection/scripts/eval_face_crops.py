"""Evaluate a face-forensics checkpoint on a crop manifest (ADR-008 Phase 5).

Used for CROSS-DATASET evaluation: train on FF++ + Celeb-DF, then run this
on the held-out DFDC ``face_test.csv`` to get the true generalization
number (the metric the FF++-only model failed — EXP-001: Celeb-DF AUC 0.71).

Reports the full metric suite (Part 22). Optionally tunes the decision
threshold on a val manifest for a target FPR, then applies it to test, so
the FPR figure is honest rather than an artefact of the default 0.5.

    python scripts/eval_face_crops.py \\
        --checkpoint checkpoints/face_convnext.pt \\
        --test-manifest data/manifests/face_test.csv \\
        --val-manifest data/manifests/face_val.csv --target-fpr 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from src.datasets.general_image_dataset import GeneralImageDataset
from src.models.baseline import BaselineDetector
from src.utils import get_logger, select_device

logger = get_logger(__name__)


@torch.no_grad()
def _scores(model, manifest, image_size, device, bs, nw):
    ds = GeneralImageDataset(manifest, image_size=image_size, train=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=(device.type == "cuda"))
    y, s = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
        s.extend(probs.tolist()); y.extend(labels.numpy().tolist())
    return np.array(y), np.array(s)


def _suite(y, s, thr):
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = {"threshold": float(thr),
           "accuracy": float(accuracy_score(y, p)),
           "precision": float(precision_score(y, p, zero_division=0)),
           "recall": float(recall_score(y, p, zero_division=0)),
           "f1": float(f1_score(y, p, zero_division=0)),
           "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
           "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
           "n": int(len(y))}
    try: out["auc"] = float(roc_auc_score(y, s))
    except ValueError: out["auc"] = float("nan")
    try: out["pr_auc"] = float(average_precision_score(y, s))
    except ValueError: out["pr_auc"] = float("nan")
    return out


def _threshold_for_fpr(y, s, target):
    """Lowest threshold whose FPR on (y,s) is <= target."""
    reals = s[y == 0]
    if reals.size == 0:
        return 0.5
    thr = float(np.quantile(reals, 1.0 - target))
    return min(max(thr, 0.0), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset face-forensics eval (ADR-008).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--val-manifest", default=None, help="Tune threshold here for --target-fpr.")
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="reports/face_crossdataset.json")
    args = ap.parse_args()

    device = select_device(None)
    logger.info("Device: %s", device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})
    model = BaselineDetector(
        backbone_name=cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k"),
        pretrained=False, dropout=cfg.get("classifier", {}).get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    thr = 0.5
    if args.val_manifest and args.target_fpr:
        yv, sv = _scores(model, args.val_manifest, args.image_size, device, args.batch_size, args.num_workers)
        thr = _threshold_for_fpr(yv, sv, args.target_fpr)
        logger.info("Tuned threshold=%.4f for FPR<=%.2f on val", thr, args.target_fpr)

    yt, st = _scores(model, args.test_manifest, args.image_size, device, args.batch_size, args.num_workers)
    report = {"checkpoint": args.checkpoint, "test_manifest": args.test_manifest,
              "at_0.5": _suite(yt, st, 0.5)}
    if thr != 0.5:
        report["at_tuned"] = _suite(yt, st, thr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    logger.info("=== CROSS-DATASET (%s) @0.5 ===", Path(args.test_manifest).name)
    for k in ("accuracy", "precision", "recall", "f1", "auc", "pr_auc", "fpr", "fnr"):
        logger.info("  %-10s %.4f", k, report["at_0.5"][k])
    if "at_tuned" in report:
        logger.info("=== @tuned threshold %.4f (FPR<=%.2f) ===", thr, args.target_fpr)
        for k in ("accuracy", "f1", "auc", "fpr"):
            logger.info("  %-10s %.4f", k, report["at_tuned"][k])
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
