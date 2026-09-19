# EXP-004 — GenVidBench general-video baseline (cross-generator)

**Date:** 2026-09-19
**Status:** Done — first real general-video result (ADR-007 Phase 1-2).

## Setup

- **Model:** `VideoSpatialNet` — ConvNeXt-Tiny (`convnext_tiny.fb_in22k`,
  28.1M params) + attention-pool temporal head + linear classifier.
- **Data:** GenVidBench Option B (Pair1 only, ~74GB), cross-generator split:
  - train: pika + videocrafter2 + modelscope (fake) + vript (real), 10,000 videos
  - val: held-out videos from the SAME train generators, 2,000
  - **test: text2video-zero (UNSEEN generator) + disjoint vript, 2,000**
  - 16 frames/video, 224×224, video-level splits, zero leakage (verified).
- **Training:** 1 epoch, backbone FROZEN (head-only), batch 8, AdamW lr 1e-4,
  MPS (Apple M5 Pro). Checkpoint = `best_model.pt` (val AUC 0.9632).
  > Note: only epoch 1 completed. Epoch 2 (backbone unfrozen) thrashed MPS
  > memory (~6× slower); training was stopped and the epoch-1 checkpoint
  > evaluated. A memory-lighter config (fewer frames / no grad-accum) is the
  > follow-up for a fuller fine-tune.

## Results — test split (UNSEEN generator t2vz)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8675 |
| Precision | 0.9016 |
| Recall | 0.8250 |
| **F1** | **0.8616** |
| **ROC-AUC** | **0.9426** |
| PR-AUC | 0.9422 |
| FPR | 0.0900 |
| FNR | 0.1750 |

Per-group (threshold 0.5):

| Group | Metric | Value | n | mean score |
|-------|--------|-------|---|-----------|
| text2video-zero (unseen fake) | recall | 0.8250 | 1000 | 0.688 |
| vript (real) | specificity | 0.9100 | 1000 | 0.222 |

Val (seen generators, epoch 1): acc 0.8975, F1 0.8982, AUC 0.9632.

## Interpretation

The detector, trained on pika/videocrafter2/modelscope, generalizes to the
**completely unseen** text2video-zero generator at **AUC 0.94 / 82.5% recall /
9% FPR** — with only a frozen-backbone single epoch.

Contrast with the face track's cross-*dataset* generalization (EXP-001,
FF++→Celeb-DF: AUC 0.71, FPR 0.72). The general-video approach generalizes to
an unseen generator far better than the face model generalized across datasets.
This is the empirical basis for the module's core thesis: general synthetic-
video detection generalizes, where face-artifact-specific detection did not.

## Next

- Memory-lighter fine-tune (unfreeze backbone) for a stronger number.
- Head-to-head vs VideoMAE / TimeSformer (Phase 3) on the SAME test split.
- Scale to Option A (cross-source, Pair2) once the pipeline is proven.
- Robustness (compression/resolution/FPS) per the master plan.

Artifact: `reports/exp01_baseline.json`.
