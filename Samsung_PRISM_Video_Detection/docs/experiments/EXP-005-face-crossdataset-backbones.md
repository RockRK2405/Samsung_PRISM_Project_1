# EXP-005 — Face-forensics cross-dataset generalization (Xception vs ConvNeXt)

**Date:** 2026-09-20
**Status:** Done — **negative result** (cross-dataset generalization remains weak). Phase 5 step 1.

## Question

Does retraining the face branch on **two** datasets (FaceForensics++ c23 +
Celeb-DF v2) with **forensic augmentation** (JPEG/blur/resolution degradation,
`ForensicDegrade`, Part 18) close the cross-dataset generalization gap — and
does a modern backbone (ConvNeXt-Tiny) beat the classic deepfake-forensics
backbone (Xception)? Both are measured on a **held-out dataset never seen in
training** (DFDC), which is the metric the FF++-only model failed (EXP-001).

## Setup

- **Models:** `BaselineDetector` (timm backbone + 2-class head), two variants:
  - **Xception** (`legacy_xception`, 20.8M params), lr 3e-4
  - **ConvNeXt-Tiny** (`convnext_tiny.fb_in22k_ft_in1k`, 27.8M params), lr 2e-4
- **Data (face crops via MTCNN, 224×224):**
  - **train/val:** FF++ c23 + Celeb-DF v2 — 600 videos/class/dataset, 6 frames/video
    → 12,233 train + 2,159 val crops, balanced 50/50 real/fake.
  - **test (held out):** DFDC (`dfdc-10` mirror, labels from per-part
    `metadata.json`) — 7,158 crops. **Never seen in training.**
- **Training:** 6 epochs, batch 64, AdamW, cosine LR, label smoothing 0.05,
  `--forensic-aug` ON. Kaggle T4 GPU, committed (headless) run.
- **Eval:** `eval_face_crops.py`, threshold 0.5 plus a val-tuned threshold
  targeting FPR ≤ 0.10.

## Results

### In-distribution (FF++ + Celeb-DF validation) — both strong

| Backbone | val Accuracy | val F1 | val ROC-AUC |
|----------|--------------|--------|-------------|
| Xception | 0.9680 | 0.9691 | 0.9913 |
| ConvNeXt-Tiny | 0.9597 | 0.9615 | 0.9890 |

### Cross-dataset (held-out DFDC, unseen) — both collapse

Threshold 0.5:

| Backbone | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|----------|----------|-----------|--------|----|---------|--------|-----|-----|
| Xception | 0.5827 | 0.6971 | 0.2904 | 0.4100 | 0.6407 | 0.6504 | 0.1258 | 0.7096 |
| **ConvNeXt-Tiny** | 0.5870 | 0.7063 | 0.2960 | 0.4172 | **0.6564** | 0.6605 | 0.1228 | 0.7040 |

At a val-tuned threshold (target FPR ≤ 0.10):

| Backbone | tuned thr | Accuracy | F1 | ROC-AUC | FPR (on DFDC) |
|----------|-----------|----------|----|---------|----|
| Xception | 0.1088 | 0.6084 | 0.5808 | 0.6407 | 0.3267 |
| ConvNeXt-Tiny | 0.1921 | 0.6119 | 0.5373 | 0.6564 | 0.2280 |

(The tuned threshold, calibrated on the FF++/Celeb val set, does **not** hold on
DFDC — the FPR blows past the 0.10 target to 0.23–0.33. The score
distributions differ across datasets, so a val-calibrated operating point does
not transfer either.)

## Interpretation (honest)

1. **In-distribution is near-solved; cross-dataset is not.** Both backbones hit
   ~0.99 val AUC but drop to **~0.64–0.66 AUC on unseen DFDC**. The gap between
   in-distribution and cross-dataset performance is the entire story.
2. **The failure is recall, not false alarms.** Recall on DFDC fakes is ~0.29 —
   the model calls ~71% of DFDC fakes "real." FPR stays low (~0.12), so forensic
   augmentation kept the model from panicking on real faces; it simply does not
   recognize DFDC's manipulation types. The FF++/Celeb artifacts it learned do
   not transfer.
3. **ConvNeXt marginally > Xception cross-dataset** (0.656 vs 0.641 AUC), but
   both are weak. Neither is promoted to a production "best" checkpoint on the
   strength of a 0.65.
4. **Forensic augmentation helped calibration, not transfer.** It bought a low,
   stable FPR but did not raise recall on unseen manipulations.

### Comparison caveat — NOT a clean regression vs EXP-001

EXP-001 reported AUC 0.71 with **Celeb-DF as the cross-dataset test**. Here
Celeb-DF is **in training** and the held-out test is **DFDC** — a harder, more
diverse benchmark. So the two numbers are **not apples-to-apples**, and this is
**not** evidence that the model got worse. The defensible claim is narrower and
still important:

> Even with two training datasets and forensic augmentation, generalization to
> an *unseen* dataset (DFDC) is only ~0.65 AUC. The cross-dataset gap that
> motivated the whole general-video thesis is real and persists for the face
> branch.

This reinforces EXP-004's contrast: the general-video branch generalized to an
**unseen generator** at 0.94 AUC, whereas the face branch generalizes to an
**unseen dataset** at only ~0.65. Face-artifact detection stays dataset-bound.

## Next (candidate directions, none claimed to work yet)

- Add DFDC into the training mix and hold out a *different* dataset (e.g.
  DeeperForensics / WildDeepfake) — measure whether 3-dataset training raises
  the floor, keeping the held-out set truly unseen.
- Stronger/frequency-aware backbone or the dual-stream FFT head (EXP-002) on
  this exact protocol.
- More frames/videos per class (this run was capped at 600×6 to fit a committed
  Kaggle run) — check whether the gap is data-limited or method-limited.
- Per-manipulation-type breakdown on DFDC to see which fakes are missed.

Artifacts: `reports/EXP-005-face-crossdataset.json` (metrics),
Kaggle output `checkpoints/face_xception.pt`, `checkpoints/face_convnext.pt`,
`face_crops.zip` (extracted crops + manifests, saved so the 3h MTCNN
extraction need not be repeated).
