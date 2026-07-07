"""Training loop for the baseline video-deepfake detector.

Design notes:
* **Frame-level batching** during training (each item is one crop) — gives
  more gradient diversity than clip-level batching.
* **Video-level evaluation** on val/test — mean-pool the 32 frame probs
  into one score per video before applying the decision threshold.
* **No mixed precision on MPS** — PyTorch ``autocast('mps')`` in 2.2 has
  known correctness bugs. We just train in fp32; it's fast enough on an
  M5 for this model size.
* **MLflow** logs metrics per epoch under ``logs/mlruns/``. Checkpoints
  land under ``checkpoints/`` with the best-val-F1 model saved as
  ``best.pt``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from functools import partial

from src.datasets import FaceFrameDataset, FaceVideoDataset, pad_video_collate
from src.evaluation import compute_report, find_threshold_for_fpr
from src.models import (
    BaselineDetector,
    TemporalDetector,
    build_baseline_from_config,
    build_model_from_config,
)
from src.utils import get_logger, select_device, set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Config dataclass — the subset of configs/train.yaml we actually use
# ---------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Runtime configuration for one training run."""

    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    label_smoothing: float
    seed: int
    device_pref: str | None
    log_every_steps: int
    checkpoint_dir: Path
    mlflow_dir: Path
    experiment_name: str


def _train_config_from_dict(cfg: dict, paths: dict) -> TrainConfig:
    t = cfg["train"]
    return TrainConfig(
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        num_workers=int(t["num_workers"]),
        lr=float(t["optimizer"]["lr"]),
        weight_decay=float(t["optimizer"]["weight_decay"]),
        scheduler=str(t["scheduler"]["name"]),
        warmup_epochs=int(t["scheduler"]["warmup_epochs"]),
        label_smoothing=float(t["loss"]["label_smoothing"]),
        seed=int(t["seed"]),
        device_pref=t.get("device"),
        log_every_steps=int(t["logging"]["log_every"]),
        checkpoint_dir=Path(paths["checkpoints"]),
        mlflow_dir=Path(paths["logs"]) / "mlruns",
        experiment_name="baseline_efficientnet_b0",
    )


# ---------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------

def _make_loaders_frame(
    faces_root: Path, cfg: TrainConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Frame-level loaders — used by the mean-pool baseline (Milestone 4)."""
    train_ds = FaceFrameDataset(faces_root, "train")
    val_ds = FaceVideoDataset(faces_root, "val")
    test_ds = FaceVideoDataset(faces_root, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


def _make_loaders_video(
    faces_root: Path, cfg: TrainConfig, num_frames: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Video-level loaders — used by the temporal detector (Milestone 5).

    Train uses the padded collate + shuffling. Val/test still run with
    batch_size=1 so we can keep evaluation code paths shared with the
    baseline (padding mask happens to be present but the evaluator
    handles that transparently).
    """
    train_ds = FaceVideoDataset(faces_root, "train")
    val_ds = FaceVideoDataset(faces_root, "val")
    test_ds = FaceVideoDataset(faces_root, "test")
    collate = partial(pad_video_collate, pad_to=num_frames)
    single_collate = partial(pad_video_collate, pad_to=num_frames)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=single_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=single_collate,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
    epoch: int,
) -> float:
    """Run one epoch of frame-level training and return the mean loss."""
    model.train()
    losses: list[float] = []
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            logger.info("epoch %d step %d/%d loss=%.4f", epoch, step + 1, len(loader), recent)

    return float(np.mean(losses))


def _train_one_epoch_video(
    model: TemporalDetector,
    loader: DataLoader,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
    epoch: int,
) -> float:
    """Video-level training epoch — feeds whole clips through the temporal head."""
    model.train()
    losses: list[float] = []
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)               # (B, T, 3, H, W)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)   # (B, T)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model.forward_video(images, padding_mask=padding_mask)
        loss = criterion(logits, labels)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            logger.info("epoch %d step %d/%d loss=%.4f", epoch, step + 1, len(loader), recent)

    return float(np.mean(losses))


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Video-level evaluation. Returns metrics + raw arrays for logging.

    Works transparently with both :class:`BaselineDetector` (ignores
    padding_mask) and :class:`TemporalDetector` (uses it) — the loader
    controls whether ``padding_mask`` is present in the batch dict.
    """
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_manips: list[str] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        # `label` may be a Python int (batch_size=1, plain dataset) or a
        # tensor (padded collate). Normalise via item() extraction.
        label = int(batch["label"][0])
        manip = str(batch["manipulation"][0])

        if "padding_mask" in batch and isinstance(model, TemporalDetector):
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            logits = model.forward_video(images, padding_mask=padding_mask)
        else:
            logits = model.forward_video(images)          # (1, 2)
        prob_synth = torch.softmax(logits, dim=-1)[0, 1].item()
        all_scores.append(prob_synth)
        all_labels.append(label)
        all_manips.append(manip)

    report = compute_report(
        y_true=np.asarray(all_labels),
        y_score=np.asarray(all_scores),
        manipulations=all_manips,
    )
    return {
        "report": report,
        "scores": all_scores,
        "labels": all_labels,
        "manipulations": all_manips,
    }


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def run_training(
    train_yaml: dict, model_yaml: dict, paths: dict, faces_root: Path
) -> Path:
    """Train the baseline detector end-to-end.

    Args:
        train_yaml: Parsed ``configs/train.yaml``.
        model_yaml: Parsed ``configs/model.yaml``.
        paths: Resolved paths dict from :func:`src.utils.load_paths`.
        faces_root: Absolute path to ``data/processed/faces/ff_c23``.

    Returns:
        Path to the best-val-F1 checkpoint on disk.
    """
    cfg = _train_config_from_dict(train_yaml, paths)
    set_seed(cfg.seed)

    device = select_device(cfg.device_pref)
    logger.info("Training device: %s", device)

    model = build_model_from_config(model_yaml["model"]).to(device)
    is_temporal = isinstance(model, TemporalDetector)
    logger.info(
        "Training mode: %s", "video-level (temporal)" if is_temporal else "frame-level (baseline)"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimiser = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimiser, cfg)

    if is_temporal:
        num_frames = int(model_yaml["model"]["input"]["frames_per_clip"])
        train_loader, val_loader, _test_loader = _make_loaders_video(
            faces_root, cfg, num_frames=num_frames
        )
        experiment_name = "temporal_efficientnet_b0"
    else:
        train_loader, val_loader, _test_loader = _make_loaders_frame(faces_root, cfg)
        experiment_name = cfg.experiment_name

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Name the checkpoint after the model — keeps Milestone-4 baseline and
    # Milestone-5 temporal checkpoints from clobbering each other.
    ckpt_name = "best.pt" if not is_temporal else "best_temporal.pt"
    best_ckpt = cfg.checkpoint_dir / ckpt_name

    # MLflow deprecated the file-store backend in 3.x; use SQLite instead.
    # A single mlflow.db under logs/mlruns/ keeps everything file-based
    # (no server to run), while satisfying the new "database backend" rule.
    cfg.mlflow_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.mlflow_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")
    mlflow.set_experiment(experiment_name)

    best_f1 = -1.0
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "backbone": model_yaml["model"].get("backbone"),
                "device": str(device),
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            if is_temporal:
                train_loss = _train_one_epoch_video(
                    model, train_loader, optimiser, criterion, device,
                    log_every=cfg.log_every_steps, epoch=epoch,
                )
            else:
                train_loss = _train_one_epoch(
                    model, train_loader, optimiser, criterion, device,
                    log_every=cfg.log_every_steps, epoch=epoch,
                )
            scheduler.step()
            elapsed = time.time() - t0

            eval_result = _evaluate(model, val_loader, device)
            report = eval_result["report"]

            logger.info(
                "epoch %d done in %.1fs — train_loss=%.4f | val F1=%.4f AUC=%.4f FPR=%.4f",
                epoch, elapsed, train_loss, report.f1, report.auc, report.fpr,
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_f1": report.f1,
                    "val_auc": report.auc,
                    "val_accuracy": report.accuracy,
                    "val_precision": report.precision,
                    "val_recall": report.recall,
                    "val_fpr": report.fpr,
                    "epoch_seconds": elapsed,
                },
                step=epoch,
            )

            if report.f1 > best_f1:
                best_f1 = report.f1
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": {
                            "model": model_yaml["model"],
                            "backbone": model_yaml["model"].get("backbone"),
                            "head_type": (
                                (model_yaml["model"].get("temporal_head") or {}).get("type")
                            ),
                        },
                        "epoch": epoch,
                        "val_f1": best_f1,
                    },
                    best_ckpt,
                )
                logger.info("Saved new best checkpoint (val F1=%.4f) → %s", best_f1, best_ckpt)

        mlflow.log_metric("best_val_f1", best_f1)

    logger.info("Training complete. Best val F1 = %.4f", best_f1)
    return best_ckpt


def _build_scheduler(
    optimiser: optim.Optimizer, cfg: TrainConfig
) -> optim.lr_scheduler.LRScheduler:
    """Cosine schedule with optional linear warmup at the start."""
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(cfg.epochs, 1))
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimiser, step_size=max(cfg.epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")


# ---------------------------------------------------------------------
# Standalone evaluation entry point (used by scripts/evaluate.py)
# ---------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: Path,
    model_yaml: dict,
    paths: dict,
    faces_root: Path,
    split: str,
    device_pref: str | None,
    tune_threshold_fpr: float | None = None,
    tune_faces_root: Path | None = None,
    dataset_name: str = "ff_c23",
) -> dict:
    """Load a checkpoint and evaluate it on the given split.

    Args:
        tune_threshold_fpr: If set, tune the threshold on val so its FPR is
            ``<= tune_threshold_fpr``, then apply that threshold to ``split``.
        tune_faces_root: If tuning, which dataset root to tune on. Defaults
            to ``faces_root`` (in-distribution tuning). For cross-dataset
            evaluation, pass the source dataset's root (e.g. FF++) here so
            the threshold is calibrated on a distribution we've actually
            trained on.
        dataset_name: Short identifier used in the output-metrics filename
            (e.g. ``ff_c23``, ``celeb_df_v2``).

    Returns the same dict shape as ``_evaluate`` plus the write-path of the
    JSON metrics file.
    """
    device = select_device(device_pref)
    model = build_model_from_config(model_yaml["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info("Loaded checkpoint %s (epoch=%d, val_f1=%.4f)",
                checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_f1", float("nan")))

    is_temporal = isinstance(model, TemporalDetector)
    num_frames = int(model_yaml["model"]["input"]["frames_per_clip"])
    collate_fn = partial(pad_video_collate, pad_to=num_frames) if is_temporal else None

    def _build_loader(root: Path, name: str) -> DataLoader:
        ds = FaceVideoDataset(root, name)
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    threshold = 0.5
    if tune_threshold_fpr is not None:
        tune_root = tune_faces_root if tune_faces_root is not None else faces_root
        val_result = _evaluate(model, _build_loader(tune_root, "val"), device)
        threshold = find_threshold_for_fpr(
            np.asarray(val_result["labels"]),
            np.asarray(val_result["scores"]),
            max_fpr=tune_threshold_fpr,
        )
        logger.info(
            "Threshold tuned on val (%s): %.4f (target FPR ≤ %.2f%%)",
            tune_root.name, threshold, tune_threshold_fpr * 100,
        )

    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    result = _evaluate(model, _build_loader(faces_root, split), device)

    # Recompute the report with the tuned threshold (has no effect if 0.5).
    if tune_threshold_fpr is not None:
        result["report"] = compute_report(
            y_true=np.asarray(result["labels"]),
            y_score=np.asarray(result["scores"]),
            manipulations=result["manipulations"],
            threshold=threshold,
        )

    metrics_dir = Path(paths["outputs"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tuned" if tune_threshold_fpr is not None else ""
    prefix = "temporal" if is_temporal else "baseline"
    ds_tag = "" if dataset_name == "ff_c23" else f"_{dataset_name}"
    out_path = metrics_dir / f"{prefix}{ds_tag}_{split}{suffix}.json"
    payload = result["report"].to_dict()
    payload["threshold"] = threshold
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote metrics to %s", out_path)

    result["metrics_path"] = str(out_path)
    result["threshold"] = threshold
    return result
