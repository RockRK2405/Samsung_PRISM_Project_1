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

from src.datasets.general_image_dataset import GeneralImageDataset
from src.models.baseline import BaselineDetector
from src.utils import get_logger, select_device

logger = get_logger(__name__)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    logger.info("Device: %s", device)

    train_ds = GeneralImageDataset(args.train_manifest, image_size=args.image_size, train=True)
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
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
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
    for images, labels in loader:
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
            "meta": {
                "task": "general_ai_content_detection",
                "dataset": "GenImage-family (see ADR-006)",
                "note": "Trained on still images, not video -- see ADR-006 honest scope note.",
            },
        },
        out_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the general (non-face) AI-content detector.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out", default="checkpoints/general.pt")
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
