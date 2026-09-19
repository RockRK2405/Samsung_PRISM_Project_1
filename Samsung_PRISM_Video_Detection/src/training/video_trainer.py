"""Training loop for the general AI-video detector (ADR-007 Phase 2).

Heavy-model friendly (Parts 26-27):
* device auto-select (CUDA > MPS > CPU)
* AMP mixed precision on CUDA (auto-disabled on MPS — 2.2 op gaps)
* gradient accumulation (effective batch = batch_size * accum)
* staged unfreezing: train the head for ``freeze_backbone_epochs``, then
  unfreeze the backbone and continue at the same LR schedule
* early stopping on val AUC, best + last checkpoints with full state
* full metric suite (Part 22) + per-generator recall (Part 25)

Everything is driven by ``configs/video_training.yaml``. Metrics and the
resolved config are written into an ``experiments/<name>/`` directory for
reproducibility (Part 33).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.video_dataset import VideoManifestDataset, load_video_manifest
from src.models.video_spatial import build_video_model_from_config
from src.utils import get_logger, select_device

logger = get_logger(__name__)


def _metric_suite(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Full Part-22 metric suite from video-level scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def _amp_enabled(device: torch.device, want: bool) -> bool:
    # AMP only on CUDA; MPS autocast in torch 2.2 has correctness gaps.
    return bool(want) and device.type == "cuda"


def train(cfg: dict) -> dict:
    dcfg = cfg["dataset"]
    vcfg = cfg["video"]
    tcfg = cfg["training"]
    hcfg = cfg.get("hardware", {})
    ecfg = cfg.get("experiment", {})

    device = select_device(hcfg.get("device"))
    logger.info("Device: %s", device)

    manifest_dir = Path(dcfg["manifest_dir"])
    train_ds = VideoManifestDataset(manifest_dir / "train.csv", vcfg["num_frames"], vcfg["image_size"], train=True, sampling=vcfg.get("sampling", "uniform"))
    val_ds = VideoManifestDataset(manifest_dir / "val.csv", vcfg["num_frames"], vcfg["image_size"], train=False, sampling=vcfg.get("sampling", "uniform"))

    nw = int(tcfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))

    model = build_video_model_from_config(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    use_amp = _amp_enabled(device, tcfg.get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum = int(tcfg.get("gradient_accumulation", 1))
    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 0))

    exp_dir = Path(ecfg.get("output_root", "experiments")) / ecfg.get("name", "exp_video")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(json.dumps(cfg, indent=2))
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)

    if freeze_epochs > 0:
        model.set_backbone_trainable(False)
        logger.info("Backbone frozen for first %d epoch(s).", freeze_epochs)

    best_auc = -1.0
    best_epoch = -1
    patience = int(tcfg.get("early_stopping", {}).get("patience", 4))
    since_improved = 0
    history = []

    for epoch in range(tcfg["epochs"]):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.set_backbone_trainable(True)
            logger.info("Unfroze backbone at epoch %d.", epoch + 1)

        model.train()
        running = 0.0
        start = time.perf_counter()
        optimizer.zero_grad()
        for step, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)
            # torch 2.2 rejects autocast(device_type='mps') even with
            # enabled=False, so only enter the autocast context on CUDA
            # (the only device where AMP is used); no-op elsewhere.
            amp_ctx = (
                torch.autocast(device_type="cuda", enabled=True)
                if use_amp else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = model(clips)
                loss = criterion(logits, labels) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running += loss.item() * accum * clips.size(0)
        scheduler.step()
        train_loss = running / len(train_ds)

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - start
        logger.info(
            "Epoch %d/%d  loss=%.4f  val_acc=%.4f  val_f1=%.4f  val_auc=%.4f  val_pr_auc=%.4f  (%.0fs)",
            epoch + 1, tcfg["epochs"], train_loss, val_metrics["accuracy"], val_metrics["f1"],
            val_metrics["auc"], val_metrics["pr_auc"], elapsed,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch + 1,
            "val_metrics": val_metrics,
            "config": cfg,
        }
        torch.save(state, ckpt_dir / "last_model.pt")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            since_improved = 0
            torch.save(state, ckpt_dir / "best_model.pt")
            logger.info("  -> new best (val_auc=%.4f), saved best_model.pt", best_auc)
        else:
            since_improved += 1
            if tcfg.get("early_stopping", {}).get("enabled", True) and since_improved >= patience:
                logger.info("Early stopping at epoch %d (no val_auc improvement for %d epochs).", epoch + 1, patience)
                break

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))
    result = {"best_val_auc": best_auc, "best_epoch": best_epoch, "history": history}
    (exp_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    logger.info("Training done. Best val_auc=%.4f @ epoch %d. Artifacts in %s", best_auc, best_epoch, exp_dir)
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    scores: list[float] = []
    labels_all: list[int] = []
    for clips, labels in loader:
        clips = clips.to(device)
        logits = model(clips)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        scores.extend(probs.tolist())
        labels_all.extend(labels.numpy().tolist())
    return _metric_suite(np.array(labels_all), np.array(scores), threshold)
