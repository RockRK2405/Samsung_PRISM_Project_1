"""Train the general (non-face) AI-content detection path — ADR-006.

Reuses :class:`src.models.baseline.BaselineDetector` UNCHANGED — it is
already a generic per-image classifier (see ADR-006's reasoning). Only
the data is different (GenImage-style general images instead of MTCNN
face crops), so this script is a plain image-classification training
loop, simpler than ``train.py`` (no video-level mean-pool needed since
each training example already IS one image).

Produces ``checkpoints/general.pt`` in the SAME state_dict format
``VideoDetector``/adapters already expect (a dict with ``state_dict`` +
embedded ``config``), so it slots into the existing checkpoint-loading
code with no changes there either.

Example
-------
::

    python scripts/train_general.py \\
        --train-manifest data/metadata/general_train.csv \\
        --val-manifest data/metadata/general_val.csv \\
        --epochs 5 --out checkpoints/general.pt

Run this on a GPU (Colab). On CPU it will be very slow for anything
beyond a --max-per-class smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.general_image_dataset import GeneralImageDataset
from src.models.baseline import BaselineDetector
from src.utils import get_logger, select_device

logger = get_logger(__name__)


def main() -> None:
    args = parse_args()

    # A previous run of this script was pointed at checkpoints/best.pt and
    # silently destroyed the Milestone-4 face baseline that EXP-001 and EXP-002
    # were evaluated on -- it is unrecoverable. Never clobber weights by
    # default; the cost of a stray --out is someone else's published result.
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"\nREFUSING to overwrite an existing checkpoint:\n  {out_path.resolve()}\n\n"
            f"Weights are git-ignored, so this would be unrecoverable. Either:\n"
            f"  - choose a new --out name (preferred; keeps both), or\n"
            f"  - pass --overwrite if you are certain this file is disposable.\n\n"
            f"Record whatever you keep in checkpoints/MANIFEST.md with its sha256.\n"
        )

    torch.manual_seed(args.seed)

    device = select_device(args.device)
    logger.info("Device: %s", device)

    train_ds = GeneralImageDataset(args.train_manifest, image_size=args.image_size, train=True, forensic_aug=args.forensic_aug)
    val_ds = GeneralImageDataset(args.val_manifest, image_size=args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logger.info("Train: %d images | Val: %d images", len(train_ds), len(val_ds))

    model = BaselineDetector(backbone_name=args.backbone, pretrained=True, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_metrics: dict = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_loss = running_loss / len(train_ds)

        metrics = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - start
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_acc=%.4f  val_f1=%.4f  val_auc=%.4f  (%.1fs)",
            epoch + 1, args.epochs, train_loss, metrics["accuracy"], metrics["f1"], metrics["auc"], elapsed,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            _save_checkpoint(model, args, out_path)
            logger.info("  -> new best (val_f1=%.4f), saved to %s", best_f1, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(best_metrics, indent=2))
    logger.info("Best val_f1=%.4f. Metrics saved to %s", best_f1, metrics_path)


@torch.no_grad()
def evaluate(model: BaselineDetector, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in tqdm(loader, desc="  [val]", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": auc,
        "n": len(all_labels),
    }


def _save_checkpoint(model: BaselineDetector, args: argparse.Namespace, out_path: Path) -> None:
    """Save in the SAME format VideoDetector.__init__ expects:
    a dict with 'state_dict' and an embedded 'config' -> {'model': {...}}.
    This is what lets the existing checkpoint-loading code (predictor.py)
    and the dashboard's model_service load this with zero changes.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "model": {
                    "backbone": args.backbone,
                    "pretrained": True,
                    "temporal_head": {"type": "mean_pool"},
                    "frequency_stream": {"enabled": False},
                    "classifier": {"dropout": args.dropout},
                }
            },
            # Record what was ACTUALLY trained. This block used to hardcode
            # "GenImage-family / still images" regardless of input, which
            # mislabelled every face-crop model trained with this script --
            # see checkpoints/MANIFEST.md.
            "meta": {
                "train_manifest": str(args.train_manifest),
                "val_manifest": str(args.val_manifest),
                "backbone": args.backbone,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "image_size": args.image_size,
                "forensic_aug": bool(args.forensic_aug),
                "seed": args.seed,
                "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": ("Per-image classifier. The data is whatever the manifests "
                         "point at -- read train_manifest above, do not assume."),
            },
        },
        out_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the general (non-face) AI-content detector.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out", default="checkpoints/general.pt")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow --out to replace an existing checkpoint "
                             "(refused by default; see checkpoints/MANIFEST.md).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forensic-aug", action="store_true",
                        help="Enable JPEG/blur/resolution degradations for cross-dataset robustness (Phase 5).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
