# EXP-006 — Celeb-DF v2, identity-disjoint face forensics

**Date:** 2026-10-15
**Status:** Done — **first positive result, and the project's first genuine
end-to-end trained-and-tested run.**
**Track:** Face branch (ADR-008 Phase 5 / enablement-plan Phase 2)

---

## 1. What this establishes — and what it does not

**Establishes:** a face-forensics model trained on Celeb-DF v2 detects
manipulations of **celebrities it has never seen**, at **ROC-AUC 0.9644**
(95% CI ±0.05), catching 91% of fakes at a 7.3% false-positive rate.

**Does NOT establish:** cross-*dataset* generalization. Training and test both
come from Celeb-DF v2; only the *identities* are disjoint.

> **Do not compare this number to EXP-001 (0.7076) or EXP-005 (0.6564).**
> Those held out an entire unseen dataset. Unseen-identity is a strictly
> easier task, so 0.96 vs 0.71 is **not** an improvement — it is a different
> question. Any claim of progress against those figures would be false.
>
> Note also that `eval_face_crops.py` prints its results under the header
> `=== CROSS-DATASET ===`. That string is hardcoded from EXP-005 and is
> **wrong here**. The split is cross-identity.

---

## 2. Setup

- **Model:** `BaselineDetector` + ConvNeXt-Tiny
  (`convnext_tiny.fb_in22k_ft_in1k`, 27.82M params) — the backbone that won
  EXP-005's comparison (0.6564 vs Xception 0.6407).
- **Augmentation:** `ForensicDegrade` (`--forensic-aug`) — JPEG recompression
  q40-90, resolution 0.4-0.9, mild blur.
- **Training:** 8 epochs, batch 64, AdamW lr 3e-4, 224x224, CUDA (Colab T4).
- **Data:** Celeb-DF v2 (Kaggle mirror, 6529 mp4 verified), MTCNN face crops,
  6 frames/video, cap 700 videos/class, seed 42.
- **Checkpoint:** `checkpoints/face_celebdf_identity.pt`

### Protocol — identity-disjoint, not video-disjoint

Celeb-DF filenames carry the celebrity ids (`id0_0000.mp4` real;
`id0_id16_0000.mp4` synthesis, source id0 -> target id16). A *video-level*
split puts the same celebrity on both sides, the model memorises identity, and
the AUC inflates toward 0.99. So the split is on **identities**: identities are
partitioned first, and a video is kept only if **all** its identities land on
one side. Videos spanning sides are **dropped**.

| | Identity keys | Crops | Videos |
|---|:-:|:-:|:-:|
| train | 185 | 4,607 | 768 (436 real / 332 fake) |
| val | 42 | 747 | ~125 |
| **test** | **56** | **1,091** | **182 (156 real / 26 fake)** |
| dropped (spanning) | — | 1,950 | 325 |

Extraction: 700 real + 700 fake videos sampled (cap 700/class) → **8,395
crops, 0 videos skipped** (MTCNN found a face in every sampled video).
Accounting checks out: 8,395 = 6,445 kept (4,607 + 747 + 1,091) + 1,950
dropped (325 spanning videos × 6 frames).

`splits_summary.json` records `leakage_checks: passed: identity-disjoint
train/val/test`, and train balance 436 real / 332 fake videos (minority share
43.2%, balance guard **passed**). Note train ends up *real*-heavy despite a
balanced 700/700 sample, because synthesis clips involve two identities each
and so are dropped far more often.

### "56 identities" is misleading — it is really 14 people

Celeb-DF v2's YouTube-real clips carry no id token (`00109.mp4`), so each is
given its own `solo:` identity rather than being pooled. Decomposing the test
split's 56 keys:

| Key type | Count | Contributes |
|---|:-:|---|
| Real celebrity identities (`id5`, `id19`, `id22`, …) | **14** | 114 Celeb-real + **all 26 fakes** |
| `solo:` YouTube-real clips | 42 | 42 real videos, 1 each |

So the fake side of the test set draws on **14 celebrities / 26 videos**, and
two of those 14 (`id11`, `id12`) contribute zero fakes — effectively **12
identities**. Per-identity fake counts run 0-9. Quote the result as "unseen
identities" but do not imply 56 people's worth of diversity on the detection
side.

### Deviation from the source spreadsheet — stated deliberately

`Video_V1_Synthetic_Deepfake_Datasets.xlsx` designates Celeb-DF v2 as
**"External Test"**, and the registry carries that as its `split_role`. This
run overrode it to `train_pool` (`--role celeb-df-v2=train_pool`), which
`splits_summary.json` preserves as `sheet_use: "External Test"` alongside
`role: "train_pool"` — the provenance of the override is on the record.

The override was necessary because no second dataset could be obtained (§6),
and it is the reason this is an identity-disjoint rather than a cross-dataset
result. **Consequence:** Celeb-DF v2 can no longer serve as a clean external
test set for any model trained here. Once FF++ or another corpus is available,
Celeb-DF v2 should return to its intended held-out role and this checkpoint
must not be evaluated on it as if unseen.

---

## 3. Results — held-out identities (n = 1,091 crops)

| Metric | @ 0.5 | @ tuned 0.1433 |
|---|:-:|:-:|
| **ROC-AUC** | **0.9644** | 0.9644 |
| PR-AUC | 0.8285 | 0.8285 |
| Accuracy | 0.9248 | 0.9028 |
| Precision | 0.6762 | 0.6016 |
| Recall | 0.9103 | 0.9487 |
| F1 | 0.7760 | 0.7363 |
| FPR | 0.0727 | **0.1048** |
| FNR | 0.0897 | 0.0513 |

Metrics JSON: `reports/EXP-006-celebdf-identity.json`.

### 3.0 Validation vs test — the generalization gap is ~0.02

| | Val (seen identities, n=747) | Test (unseen identities, n=1,091) | Δ |
|---|:-:|:-:|:-:|
| **ROC-AUC** | **0.9829** | **0.9644** | **−0.019** |
| Accuracy | 0.9572 | 0.9248 | −0.032 |
| F1 | 0.8462 | 0.7760 | −0.070 |

**Holding out identities costs almost nothing.** The 0.019 AUC drop is *inside*
the test set's own ±0.05 confidence interval, so it is not even a resolvable
degradation — the model's ranking ability barely notices that the faces are
new people.

Put that beside the cross-dataset result and the contrast is stark:

| Shift | Val AUC | Test AUC | Δ |
|---|:-:|:-:|:-:|
| **Unseen identity** (EXP-006, within Celeb-DF v2) | 0.9829 | **0.9644** | **−0.02** |
| **Unseen dataset** (EXP-005, → DFDC) | ~0.99 | **0.6564** | **−0.33** |

A **16x larger** collapse for a change of corpus than for a change of person.
This is the quantitative form of §4's claim: the face branch learns
manipulation artifacts perfectly well and generalizes across *identities*
nearly for free — what destroys it is a change of **dataset/domain**
(codec, compression, capture pipeline, synthesis method), not unfamiliar faces.

Caveat on the val figures: the checkpoint was selected on **best val F1**, so
these are best-epoch values and mildly optimistic by construction. The test
column is the honest estimate. The F1 column is also not directly comparable
across val and test, because the two splits have different real/fake ratios —
**compare the AUC row.**

Source: `reports/EXP-006-train-val-metrics.json` (written by
`train_general.py` beside the checkpoint).

### 3.1 The headline is AUC, not F1

F1 is 0.776 despite 0.91 recall because the test set is **real-heavy**:
counted directly from the test manifest, **156 fake and 935 real crops
(1:6.0)** — 26 fake and 156 real videos. (This was first back-solved from the
reported precision/recall/FPR and then confirmed exactly against the CSV.)
The identity split dropped 325 spanning videos,
most of them synthesis clips (each involves two identities and so is far more
likely to straddle the partition), which inverted Celeb-DF's usual fake-heavy
balance. With 6x more reals, even a 7% FPR produces enough false positives to
hold precision near 0.68. This is the same trap flagged in EXP-001, where
F1 0.81 coexisted with FPR 0.72 — **quote AUC.**

### 3.2 Sample size — the main limitation

1,091 crops sounds ample, but crops from one video are highly correlated, so
the effective sample size is the **video** count: exactly **26 fake** and
**156 real** videos (5.99 crops/video), drawn from **14** celebrity
identities on the fake side.

| Statistic | Value | 95% CI (on videos) |
|---|:-:|:-:|
| ROC-AUC | 0.9644 | **±0.050** |
| Recall | 0.9103 | **[0.74, 0.97]** |
| FPR | 0.0727 | [0.041, 0.125] |

Computed naively on crops the AUC interval would read ±0.021 — **about half
the honest width.** Report ±0.05.

So the defensible claim is "AUC in the low-to-mid 0.9s"; the second decimal is
not resolved. **Recall rests on 26 fake videos from 12-14 identities** and
could plausibly be anywhere from 0.74 to 0.97. Raising `--videos-per-class` or
`--frames-per-video`, or relaxing the spanning-drop rule, would tighten this
and is the obvious follow-up.

### 3.3 Threshold tuning did not transfer

The decision threshold was tuned on **val** for FPR <= 0.05 and produced
**0.1048 on test** — roughly **2x the target**. Val is optimistic about the
operating point even when the shift is only to new identities. Consequence: a
threshold calibrated on a validation split should not be trusted as a
deployment operating point without a held-out calibration set. This supports
the enablement plan's Phase 3 (calibration) being a separate, measured step
rather than a value copied from config.

---

## 4. Interpretation

The face branch works well **within a dataset, across identities**, which the
project had never actually measured — EXP-001/005 only ever reported the much
harder cross-dataset setting. Taken with those, the picture is:

| Generalization axis | Result | Source |
|---|:-:|---|
| Unseen **identity** (same dataset) | **0.9644** | EXP-006 |
| Unseen **dataset** | 0.6564-0.7076 | EXP-005 / EXP-001 |
| Unseen **generator** (general video) | 0.9426 | EXP-004 |

So the face branch's weakness is specifically a **dataset/domain shift**
problem, not an inability to learn manipulation artifacts. It learns them
well; they just do not survive a change of corpus. §3.0 quantifies this
directly: the val→test drop is **0.019** for unseen identities against
**0.33** for an unseen dataset — the same model, the same training recipe, a
16x difference in how far it falls. That refines, rather than
overturns, the Phase 6 question ("fix or de-scope"): the branch is useful
in-domain and unreliable out-of-domain.

---

## 5. Honest caveats

1. **Not cross-dataset.** Section 1. The single most likely misreading.
2. **26 fake test videos from 14 identities** (12 contributing). AUC ±0.05,
   recall [0.74, 0.97]. This is the binding limitation.
3. **Celeb-DF v2 was repurposed from its spreadsheet role** of External Test,
   so it is spent as a held-out set for this checkpoint.
4. **Threshold did not transfer** (0.05 target -> 0.105 actual).
5. **One backbone, one seed.** No Xception arm, no seed variance.
6. **Crop-level metrics**, not video-level aggregated scores.
7. **DFDC and FF++ were attempted and abandoned** — see §6.

---

## 6. Why no second dataset (recorded so it is not retried blindly)

Both intended cross-dataset partners proved unobtainable in Colab:

- **DFDC** (`pranay22077/dfdc-10`): 96.5 GB total vs ~75 GB of Colab disk, so
  bulk download does not fit. The mirror ships **loose mp4s, no per-part
  archives**, so `-f` works only per *file* — and Kaggle **rate-limits after
  ~80 file downloads** (observed twice: `ok=80` frozen while failures climbed
  past 600). Three independent walls.
- **FF++** (`ahmedelbanby/...deepfakebench-structure`): ships pre-extracted
  **PNG frames**, not mp4 — and **contains Celeb-DF** under `frames/Celeb-DF/`.
  Training on it while testing on Celeb-DF v2 would be silent leakage that the
  registry's name-based checks cannot detect.

A real cross-dataset number needs the **official FF++ EULA**
(`github.com/ondyari/FaceForensics`) or another verified mirror.

---

## 7. Reproduction

```bash
python scripts/prepare_from_registry.py \
    --root celeb-df-v2=<path> --role celeb-df-v2=train_pool \
    --split-key identity --test-fraction 0.20 --val-fraction 0.15 \
    --videos-per-class 700 --frames-per-video 6 --device cuda --seed 42

python scripts/train_general.py \
    --train-manifest manifests/train.csv --val-manifest manifests/val.csv \
    --backbone convnext_tiny.fb_in22k_ft_in1k --forensic-aug \
    --epochs 8 --batch-size 64 --lr 3e-4 --device cuda \
    --out checkpoints/face_celebdf_identity.pt

python scripts/eval_face_crops.py \
    --checkpoint checkpoints/face_celebdf_identity.pt \
    --test-manifest manifests/test_celeb-df-v2_heldout_identity.csv \
    --val-manifest manifests/val.csv --target-fpr 0.05
```

## 8. Next

1. **Tighten the interval** — more videos/frames per class; ~26 fake test
   videos is the binding constraint.
2. **Video-level aggregation** — mean/max-pool crop scores per video and report
   video-level metrics, which is what a deployment actually decides on.
3. **Xception arm + a second seed** — EXP-005 compared backbones; this did not.
4. **Fix the `CROSS-DATASET` header** in `eval_face_crops.py` to read the
   protocol from the manifest instead of hardcoding it.
5. **Pursue official FF++** for the cross-dataset number this cannot provide.
